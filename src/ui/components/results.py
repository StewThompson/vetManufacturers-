import re
import streamlit as st
import pandas as pd
from src.ui.components.violations import render_violation_dashboard


def render_results(assessment, vetting_agent):
    if assessment.risk_score < 15:
        badge_label, badge_bg, badge_color = "Low Risk", "#D1F0DC", "#1A6B3A"
    elif assessment.risk_score < 45:
        badge_label, badge_bg, badge_color = "Moderate Risk", "#FEF3CD", "#7A5500"
    else:
        badge_label, badge_bg, badge_color = "High Risk", "#FDE0E3", "#8B1A24"

    st.markdown(
        f'<div style="margin:6px 0 18px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        f'<div style="font-size:1.25rem;font-weight:800;color:#0F2240">{assessment.manufacturer.name}</div>'
        f'<div style="display:inline-block;padding:4px 14px;border-radius:20px;'
        f'font-size:0.77rem;font-weight:700;background:{badge_bg};color:{badge_color}">{badge_label}</div>'
        f'<div style="font-size:0.77rem;color:#9BAFC5">Assessment complete</div></div>',
        unsafe_allow_html=True,
    )

    n_rec = len(assessment.records)
    if n_rec > 1:
        if assessment.aggregation_warning:
            st.info(
                f"**Aggregated analysis** — {assessment.aggregation_warning} "
                f"Use the per-site breakdown and violation details below to identify "
                f"facility-specific patterns.",
                icon="ℹ️",
            )
        else:
            st.info(
                f"**Aggregated analysis** — this score combines **{n_rec} inspections** across "
                f"likely related establishments and may not represent a single physical site. "
                f"Use the violation breakdown below to identify facility-specific patterns.",
                icon="ℹ️",
            )

    if assessment.concentration_warning:
        st.warning(assessment.concentration_warning, icon="⚠️")

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Risk Score", f"{assessment.risk_score} / 100")
    k2.metric("Recommendation", assessment.recommendation)
    if assessment.establishment_count > 1:
        k3.metric("Establishments", assessment.establishment_count)
    else:
        k3.metric("Records Found", len(assessment.records))
    if not assessment.missing_naics and assessment.industry_group:
        k4.metric("Industry Percentile", f"{assessment.industry_percentile}%")
    else:
        k4.metric("Pop. Percentile", f"{assessment.percentile_rank}%")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _render_industry_comparison(assessment)
    _render_site_breakdown(assessment)
    _render_explanation_sections(assessment)
    render_violation_dashboard(assessment, llm_breakdown_text=None)
    _render_feature_importance(assessment)
    _render_penalty_timeline(assessment)
    _render_accident_details(assessment)
    _render_raw_records(assessment)

    st.divider()
    _render_chat(assessment, vetting_agent)


def _render_industry_comparison(assessment):
    _show_industry = (
        not assessment.missing_naics
        and assessment.industry_group
    )
    if _show_industry:
        _ind_pct = assessment.industry_percentile
        _ind_lbl = assessment.industry_label
        if _ind_lbl == "Unknown Industry":
            _ind_lbl = ""
        _ind_grp = f"NAICS {assessment.industry_group}" if assessment.industry_group else ""
        _risk_score = assessment.risk_score

        _display_pct = _ind_pct

        if _display_pct >= 75:
            _ind_bg, _ind_color, _ind_icon = "#FDE0E3", "#8B1A24", "⚠️"
            _ind_rank_text = f"Riskier than {_display_pct:.0f}% of peers"
        elif _display_pct >= 50:
            _ind_bg, _ind_color, _ind_icon = "#FEF3CD", "#7A5500", "🔶"
            _ind_rank_text = f"Riskier than {_display_pct:.0f}% of peers"
        else:
            _ind_bg, _ind_color, _ind_icon = "#D1F0DC", "#1A6B3A", "✅"
            _ind_rank_text = f"Safer than {100 - _display_pct:.0f}% of peers"

        _discrepancy_html = ""
        if _ind_pct < 50 and _risk_score >= 30:
            _discrepancy_note = (
                "Violation rate is below industry average, but overall risk remains elevated "
                "by other factors (severity, penalties, incidents, or willful/repeat violations)."
            )
            _discrepancy_html = (
                f"<div style='margin-top:8px;font-size:0.82rem;color:{_ind_color};"
                f"font-style:italic'>ℹ️ {_discrepancy_note}</div>"
            )

        _cmp_bullets = "".join(
            f"<li style='margin:2px 0'>{c}</li>"
            for c in (assessment.industry_comparison or [])
        )
        _cmp_html = (
            f"<ul style='margin:6px 0 0 0;padding-left:18px;font-size:0.82rem;color:#444'>{_cmp_bullets}</ul>"
            if _cmp_bullets else ""
        )

        st.markdown(
            f'<div style="background:{_ind_bg};border-radius:10px;padding:14px 18px;margin-bottom:10px">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span style="font-size:1.2rem">{_ind_icon}</span>'
            f'<div><div style="font-weight:700;font-size:0.95rem;color:{_ind_color}">{_ind_rank_text}</div>'
            f'<div style="font-size:0.82rem;color:#555">{_ind_lbl}'
            f'{"(" + _ind_grp + ")" if _ind_grp else ""}</div></div></div>'
            f'{_cmp_html}{_discrepancy_html}</div>',
            unsafe_allow_html=True,
        )
    elif assessment.missing_naics:
        st.info("No NAICS code available — industry peer comparison not possible.", icon="ℹ️")


def _render_site_breakdown(assessment):
    """Show per-establishment risk scores when there are multiple sites."""
    if assessment.establishment_count <= 1 or not assessment.site_scores:
        return
    with st.expander(
        f"🏭 Per-Establishment Breakdown ({assessment.establishment_count} sites)",
        expanded=True,
    ):
        rows = []
        for s in assessment.site_scores:
            score_val = s["score"]
            if score_val >= 60:
                tag = "🔴 High"
            elif score_val >= 30:
                tag = "🟡 Moderate"
            else:
                tag = "🟢 Low"
            loc_parts = [p for p in (s.get("city"), s.get("state")) if p]
            location = ", ".join(loc_parts) if loc_parts else "—"
            rows.append({
                "Establishment": s["name"],
                "Location": location,
                "Risk Score": score_val,
                "Risk Level": tag,
                "Inspections": s["n_inspections"],
                "NAICS": s.get("naics_code") or "—",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if assessment.systemic_risk_flag:
            st.error(
                "⚠️ **Systemic risk detected** — high-risk conditions are present "
                "across multiple establishments, not isolated to a single site.",
                icon="🚩",
            )


def _render_explanation_sections(assessment):
    _VB_PATTERN = re.compile(
        r'(?:^|\n)(#{1,3}\s*Violation Breakdown.*|'
        r'\*{1,2}Violation Breakdown[^*\n]*\*{0,2})',
        re.IGNORECASE,
    )
    _vb_match = _VB_PATTERN.search(assessment.explanation)
    if _vb_match:
        _summary_text = assessment.explanation[:_vb_match.start()].strip()
    else:
        _summary_text = assessment.explanation.strip()

    with st.expander("📝 Executive Summary", expanded=False):
        st.markdown(_summary_text)


def _render_feature_importance(assessment):
    with st.expander("📌 Key Risk Signals & Feature Importances", expanded=False):
        if assessment.feature_weights:
            _labels = {
                "log_inspections": "Inspections (log)",
                "log_violations": "Violations (log)",
                "serious_violations": "Serious Viols.",
                "willful_violations": "Willful Viols.",
                "repeat_violations": "Repeat Viols.",
                "log_penalties": "Total Penalties (log)",
                "avg_penalty": "Avg Penalty",
                "max_penalty": "Max Penalty",
                "recent_ratio": "Recency (1yr)",
                "severe_incidents": "Fat/Cat Inspections",
                "violations_per_inspection": "Viols/Inspection",
                "accident_count": "Linked Accidents",
                "fatality_count": "Fatalities",
                "injury_count": "Injuries",
                "avg_gravity": "Avg Gravity",
                "penalties_per_inspection": "Penalty/Inspection",
                "clean_ratio": "Clean Ratio",
            }
            fw_display = {_labels.get(k, k): v for k, v in assessment.feature_weights.items()}
            fw_df = pd.DataFrame({"Feature": fw_display.keys(), "Importance": fw_display.values()})
            fw_df = fw_df.sort_values("Importance", ascending=True).tail(8)
            st.caption("Top ML feature importances — factors most influential to this risk score.")
            st.bar_chart(fw_df.set_index("Feature"), height=280)
        else:
            st.caption("No feature weight data available.")
        if st.session_state.get("sb_show_raw") and assessment.features:
            st.divider()
            fv_df = pd.DataFrame({
                "Feature": assessment.features.keys(),
                "Value": [round(v, 3) for v in assessment.features.values()],
            })
            st.caption("Raw feature values")
            st.dataframe(fv_df, width='stretch', hide_index=True)


def _render_penalty_timeline(assessment):
    if not assessment.records:
        return
    with st.expander("📊 Penalty Timeline", expanded=False):
        st.caption("Inspection penalties over time — higher bars indicate costlier enforcement actions.")
        data = [
            {"Date": pd.to_datetime(r.date_opened), "Penalty ($)": r.total_penalties}
            for r in assessment.records
        ]
        df = pd.DataFrame(data)
        if not df.empty and df["Penalty ($)"].sum() > 0:
            st.bar_chart(df.set_index("Date")["Penalty ($)"], height=180)
        else:
            st.caption("No penalty data to chart.")


def _render_accident_details(assessment):
    all_accidents = [a for r in assessment.records for a in r.accidents]
    if not all_accidents:
        return
    with st.expander(f"🚨 Accident & Injury Details — {len(all_accidents)} incident(s)", expanded=False):
        for acc in all_accidents:
            fat_label = "  🔴 **FATALITY**" if acc.fatality else ""
            st.markdown(f"**Accident {acc.summary_nr}** &nbsp;({acc.event_date or 'unknown date'}){fat_label}")
            if acc.event_desc:
                st.write(acc.event_desc)
            if acc.injuries:
                inj_rows = [
                    {
                        "Degree": inj.get("degree", ""),
                        "Nature": inj.get("nature", ""),
                        "Body Part": inj.get("body_part", ""),
                        "Age": inj.get("age", ""),
                    }
                    for inj in acc.injuries
                ]
                st.dataframe(pd.DataFrame(inj_rows), width='stretch', hide_index=True)
            st.divider()


def _render_raw_records(assessment):
    with st.expander("🗂️ Raw OSHA Records", expanded=False):
        st.json([r.model_dump() for r in assessment.records])


def _render_chat(assessment, vetting_agent):
    st.markdown("**💬 AI Assistant**")
    st.caption("Ask follow-up questions about this company's compliance record.")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask about specific risks, violations, or recommendations…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = vetting_agent.discuss_assessment(assessment, prompt)
                st.write(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
