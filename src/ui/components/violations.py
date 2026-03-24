import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from src.ui.styles import map_standard_to_theme, theme_description, THEME_DEFS


def _build_violations_df(assessment) -> pd.DataFrame:
    rows = []
    for rec in assessment.records:
        # Only mark fatality-linked when an actual fatal accident is tied to this inspection.
        # (severe_injury_or_fatality is True for *any* linked accident, which is too broad.)
        fat_linked = any(a.fatality for a in rec.accidents)
        for v in rec.violations:
            theme, icon = map_standard_to_theme(v.category)
            try:
                gravity_val = float(v.gravity) if v.gravity else 0.0
            except (ValueError, TypeError):
                gravity_val = 0.0
            priority = (
                (100 if fat_linked else 0)
                + (50 if v.is_willful else 0)
                + (30 if v.is_repeat else 0)
                + (10 if v.severity == "Serious" else 0)
                + min(gravity_val, 10)
                + min(v.penalty_amount / 10_000, 20)
            )
            _loc_parts = [p for p in (rec.site_city, rec.site_state) if p]
            _location = ", ".join(_loc_parts) if _loc_parts else ""
            rows.append({
                "inspection_id": rec.inspection_id,
                "inspection_date": rec.date_opened,
                "estab_name": rec.estab_name or "Unknown",
                "location": _location,
                "standard_code": v.category or "Unknown",
                "hazard_theme": theme,
                "theme_icon": icon,
                "hazard_label": (v.description or v.category or "")[:180],
                "severity": v.severity,
                "gravity": gravity_val,
                "penalty": v.penalty_amount,
                "is_willful": bool(v.is_willful),
                "is_repeat": bool(v.is_repeat),
                "fatality_linked": fat_linked,
                "nr_exposed": float(v.nr_exposed or 0),
                "_priority": priority,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["inspection_date"] = pd.to_datetime(df["inspection_date"])
    return df.sort_values("_priority", ascending=False).reset_index(drop=True)


def _render_theme_summary_cards(theme_agg: pd.DataFrame):
    themes = theme_agg.sort_values(
        ["fatality_linked_count", "willful_count", "repeat_count", "serious_count", "total_penalty"],
        ascending=False,
    ).head(6)
    cols = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(themes.iterrows()):
        is_critical = row["fatality_linked_count"] > 0 or row["willful_count"] > 0
        is_warn = row["repeat_count"] > 0 and not is_critical
        bg = "#F5E8E8" if is_critical else ("#FFFBEA" if is_warn else "#EEF3FA")
        fg = "#8B1A24" if is_critical else ("#7A5500" if is_warn else "#1A3558")
        border = "#E8B4B8" if is_critical else ("#F3D77A" if is_warn else "#C8D9EE")
        severe_badge = (
            f"<span style='background:#F8D7DA;border-radius:6px;padding:2px 8px;"
            f"font-size:0.71rem;font-weight:600;color:#8B1A24'>⚠ {int(row['severe_count'])} severe</span> "
        ) if row["severe_count"] > 0 else ""
        fat_badge = (
            "<span style='background:#F8D7DA;border-radius:6px;padding:2px 8px;"
            "font-size:0.71rem;font-weight:600;color:#8B1A24'>☠ fatality-linked</span>"
        ) if row["fatality_linked_count"] > 0 else ""
        pen_line = (
            f"<div style='margin-top:6px;font-size:0.73rem;font-weight:600;color:{fg}'>"
            f"${row['total_penalty']:,.0f} in penalties</div>"
        ) if row["total_penalty"] > 0 else ""
        cite_count = int(row['total_citations'])
        cite_word = 's' if cite_count != 1 else ''
        card_html = (
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:16px 18px;margin-bottom:8px">'
            f'<div style="font-size:1.3rem;margin-bottom:5px">{row["icon"]}</div>'
            f'<div style="font-size:0.87rem;font-weight:700;color:{fg};margin-bottom:7px">{row["theme"]}</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:7px">'
            f'<span style="background:rgba(255,255,255,0.65);border-radius:6px;padding:2px 8px;font-size:0.71rem;font-weight:600;color:#444">{cite_count} citation{cite_word}</span>'
            f'{severe_badge}{fat_badge}</div>'
            f'<div style="font-size:0.75rem;color:#6B7E99;line-height:1.45">{row["description"]}</div>'
            f'{pen_line}</div>'
        )
        with cols[i % 3]:
            st.markdown(card_html, unsafe_allow_html=True)


def _render_top_findings(df: pd.DataFrame):
    _sev_color = {
        "Willful": ("#FDE0E3", "#8B1A24"),
        "Repeat":  ("#FEF3CD", "#7A5500"),
        "Serious": ("#FFFAE6", "#6B5300"),
        "Other":   ("#F4F7FB", "#4A5568"),
    }
    for _, row in df.head(8).iterrows():
        if row["fatality_linked"]:
            icon, (bg, fg) = "☠️", ("#FDE0E3", "#8B1A24")
        elif row["is_willful"]:
            icon, (bg, fg) = "🔴", ("#FDE0E3", "#8B1A24")
        elif row["is_repeat"]:
            icon, (bg, fg) = "🟠", ("#FEF3CD", "#7A5500")
        else:
            icon = "🟡" if row["severity"] == "Serious" else "⚪"
            bg, fg = _sev_color.get(row["severity"], _sev_color["Other"])
        date_str = row["inspection_date"].strftime("%b %Y") if pd.notna(row["inspection_date"]) else "—"
        penalty_str = f"${row['penalty']:,.0f}" if row["penalty"] > 0 else "—"
        fat_note = (
            "&nbsp;&nbsp;<b style='color:#8B1A24'>☠ Fatality-linked</b>"
            if row["fatality_linked"] else ""
        )
        html = (
            f"<div style='display:flex;align-items:flex-start;gap:12px;"
            f"background:{bg};border:1px solid #DDE4EF;"
            f"border-left:4px solid {fg};"
            f"border-radius:9px;padding:12px 16px;margin-bottom:7px'>"
            f"<div style='font-size:1.1rem;margin-top:1px'>{icon}</div>"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px'>"
            f"<span style='font-size:0.82rem;font-weight:700;color:{fg}'>{row['severity']}</span>"
            f"<span style='font-size:0.78rem;color:#4A5568'>{row['hazard_theme']}</span>"
            f"<code style='font-size:0.71rem;color:#7B8FA8'>{row['standard_code']}</code>"
            f"{fat_note}</div>"
            f"<div style='font-size:0.8rem;color:#2D3748;line-height:1.4'>{row['hazard_label'][:160]}</div>"
            f"</div>"
            f"<div style='text-align:right;min-width:80px;flex-shrink:0'>"
            f"<div style='font-size:0.8rem;font-weight:700;color:#1A3558'>{penalty_str}</div>"
            f"<div style='font-size:0.72rem;color:#9BAFC5'>{date_str}</div>"
            f"</div></div>"
        )
        st.markdown(html, unsafe_allow_html=True)


def render_violation_dashboard(assessment, llm_breakdown_text: str = None):
    # Build once per assessment and cache — avoids rebuilding on every Streamlit rerun
    # (e.g. every "Select" button click) which is expensive for large record sets.
    _cache_key = f"_violations_df_{id(assessment)}"
    if _cache_key not in st.session_state:
        st.session_state[_cache_key] = _build_violations_df(assessment)
    df = st.session_state[_cache_key]

    with st.expander("📋 Violation Breakdown", expanded=False):
        if df.empty:
            st.caption("No structured violation data available.")
            if llm_breakdown_text:
                st.markdown(llm_breakdown_text)
            return

        theme_agg = (
            df.groupby(["hazard_theme", "theme_icon"]).agg(
                total_citations=("standard_code", "count"),
                severe_count=("severity", lambda x: x.isin(["Serious", "Willful", "Repeat"]).sum()),
                willful_count=("is_willful", "sum"),
                repeat_count=("is_repeat", "sum"),
                serious_count=("severity", lambda x: (x == "Serious").sum()),
                fatality_linked_count=("fatality_linked", "sum"),
                total_penalty=("penalty", "sum"),
                locations=("estab_name", "nunique"),
            )
            .reset_index()
            .rename(columns={"hazard_theme": "theme", "theme_icon": "icon"})
        )
        theme_agg["description"] = theme_agg["theme"].map(theme_description)

        total_cit = len(df)
        serious_ct = int((df["severity"] == "Serious").sum())
        willful_ct = int(df["is_willful"].sum())
        repeat_ct = int(df["is_repeat"].sum())
        # Count distinct inspections with a fatal accident — not violation rows, because
        # a fatality inspection might have zero formal citations (vacated, compliance-only, etc.)
        fat_ct = sum(1 for r in assessment.records if any(a.fatality for a in r.accidents))
        total_pen = df["penalty"].sum()
        n_themes = len(theme_agg)

        m1, m2, m3, m4, m5 = st.columns(5, gap="small")
        m1.metric("Hazard Themes", n_themes)
        m2.metric("Total Citations", total_cit)
        m3.metric("Serious / Willful / Repeat", f"{serious_ct} / {willful_ct} / {repeat_ct}")
        m4.metric("Fatality-Linked", fat_ct)
        m5.metric("Total Penalties", f"${total_pen:,.0f}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("**Key Hazard Themes**")
        st.caption("Citations grouped by recognized OSHA hazard category, ranked by severity.")
        _render_theme_summary_cards(theme_agg)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        with st.expander("🚨 Most Critical Findings (top 8)", expanded=False):
            st.caption("Ranked by fatality linkage, willfulness, repeat pattern, and penalty size.")
            _render_top_findings(df)

        with st.expander("📂 Violations by Hazard Theme", expanded=False):
            for theme_name in (
                theme_agg.sort_values(
                    ["fatality_linked_count", "willful_count", "repeat_count",
                     "serious_count", "total_penalty"],
                    ascending=False,
                )["theme"]
            ):
                tdf = df[df["hazard_theme"] == theme_name]
                row_meta = theme_agg[theme_agg["theme"] == theme_name].iloc[0]
                label = (
                    f"{row_meta['icon']} {theme_name} "
                    f"({len(tdf)} citation{'s' if len(tdf) != 1 else ''})"
                )
                with st.expander(label, expanded=False):
                    st.caption(theme_description(theme_name))
                    deduped = (
                        tdf.groupby("standard_code").agg(
                            severity=("severity", "first"),
                            occurrences=("standard_code", "count"),
                            total_penalty=("penalty", "sum"),
                            first_date=("inspection_date", "min"),
                            last_date=("inspection_date", "max"),
                            willful=("is_willful", "any"),
                            repeat=("is_repeat", "any"),
                        )
                        .reset_index()
                        .sort_values("total_penalty", ascending=False)
                    )
                    deduped["Date Range"] = deduped.apply(
                        lambda r: (
                            r["last_date"].strftime("%b %Y")
                            if r["first_date"] == r["last_date"]
                            else f"{r['first_date'].strftime('%b %Y')} – {r['last_date'].strftime('%b %Y')}"
                        ),
                        axis=1,
                    )
                    deduped["Flags"] = deduped.apply(
                        lambda r: ", ".join(
                            x for x in (
                                ("Willful" if r["willful"] else ""),
                                ("Repeat" if r["repeat"] else ""),
                            ) if x
                        ),
                        axis=1,
                    )
                    st.dataframe(
                        deduped.rename(columns={
                            "standard_code": "OSHA Code",
                            "severity": "Severity",
                            "occurrences": "Count",
                            "total_penalty": "Penalty ($)",
                        })[["OSHA Code", "Severity", "Flags", "Count", "Penalty ($)", "Date Range"]],
                        width='stretch',
                        hide_index=True,
                    )

        # Per-location violation chart (only when multiple establishments)
        if "estab_name" in df.columns and df["estab_name"].nunique() > 1:
            with st.expander("🏭 Violations by Location", expanded=True):
                st.caption("Citation count and penalties per establishment.")
                # Build a display label that includes the physical location
                df["_loc_label"] = df.apply(
                    lambda r: f"{r['estab_name']} ({r['location']})" if r.get("location") else r["estab_name"],
                    axis=1,
                )
                loc_agg = (
                    df.groupby("_loc_label").agg(
                        Citations=("standard_code", "count"),
                        Serious=("severity", lambda x: (x == "Serious").sum()),
                        Willful=("is_willful", "sum"),
                        Repeat=("is_repeat", "sum"),
                        Penalties=("penalty", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"_loc_label": "Establishment"})
                    .sort_values("Penalties", ascending=False)
                )
                _gb2 = GridOptionsBuilder.from_dataframe(loc_agg)
                _gb2.configure_default_column(resizable=True, sortable=True)
                _gb2.configure_column("Establishment", minWidth=200)
                _gb2.configure_column("Penalties", type=["numericColumn"])
                AgGrid(
                    loc_agg,
                    gridOptions=_gb2.build(),
                    height=min(500, 56 + len(loc_agg) * 42),
                    fit_columns_on_grid_load=True,
                    theme="streamlit",
                    enable_enterprise_modules=False,
                )
                chart_data = loc_agg.set_index("Establishment")[["Citations", "Serious", "Willful", "Repeat"]]
                st.bar_chart(chart_data, height=250)

        with st.expander("🗃️ Detailed Citation Evidence", expanded=False):
            st.caption("All citations — click a column header to sort.")
            detail = df[[
                "inspection_id", "estab_name", "location", "inspection_date", "standard_code", "hazard_theme",
                "severity", "gravity", "penalty", "fatality_linked", "is_willful", "is_repeat",
            ]].copy()
            detail["inspection_date"] = detail["inspection_date"].dt.strftime("%Y-%m-%d")
            for col in ("fatality_linked", "is_willful", "is_repeat"):
                detail[col] = detail[col].map({True: "Yes", False: ""})
            _detail_renamed = detail.rename(columns={
                "inspection_id": "Inspection ID",
                "estab_name": "Establishment",
                "location": "City, State",
                "inspection_date": "Date",
                "standard_code": "OSHA Code",
                "hazard_theme": "Theme",
                "severity": "Severity",
                "gravity": "Gravity",
                "penalty": "Penalty ($)",
                "fatality_linked": "Fatality Linked",
                "is_willful": "Willful",
                "is_repeat": "Repeat",
            })
            _gb = GridOptionsBuilder.from_dataframe(_detail_renamed)
            _gb.configure_default_column(resizable=True, sortable=True, filter=True, minWidth=80)
            _gb.configure_column("Inspection ID", minWidth=110)
            _gb.configure_column("Establishment", minWidth=160)
            _gb.configure_column("OSHA Code", minWidth=100)
            _gb.configure_column("Penalty ($)", type=["numericColumn"], minWidth=90)
            _gb.configure_column("Gravity", type=["numericColumn"], minWidth=80)
            _gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
            AgGrid(
                _detail_renamed,
                gridOptions=_gb.build(),
                height=500,
                fit_columns_on_grid_load=False,
                theme="streamlit",
                enable_enterprise_modules=False,
            )

        with st.expander("🗓️ Inspection-by-Inspection Evidence", expanded=False):
            for rec in sorted(assessment.records, key=lambda r: r.date_opened, reverse=True):
                fat = bool(rec.severe_injury_or_fatality) or any(a.fatality for a in rec.accidents)
                n_v = len(rec.violations)
                if fat:
                    badge = f"🔴 {n_v} violations — fatality-linked"
                elif n_v:
                    badge = f"🟡 {n_v} violation{'s' if n_v != 1 else ''}"
                else:
                    badge = "✅ Clean"
                loc_parts = [p for p in (rec.estab_name, rec.site_city, rec.site_state) if p]
                loc_tag = f" @ {', '.join(loc_parts)}" if loc_parts else ""
                label = (
                    f"Inspection {rec.inspection_id}{loc_tag} — "
                    f"{rec.date_opened.strftime('%b %d, %Y')} — {badge}"
                )
                with st.expander(label, expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Violations", n_v)
                    c2.metric("Total Penalty", f"${rec.total_penalties:,.0f}")
                    c3.metric("Accidents", len(rec.accidents))
                    if rec.violations:
                        st.dataframe(
                            pd.DataFrame([{
                                "OSHA Code": v.category,
                                "Theme": map_standard_to_theme(v.category)[0],
                                "Severity": v.severity,
                                "Penalty ($)": f"${v.penalty_amount:,.0f}",
                                "Willful": "✓" if v.is_willful else "",
                                "Repeat": "✓" if v.is_repeat else "",
                            } for v in rec.violations]),
                            width='stretch',
                            hide_index=True,
                        )
