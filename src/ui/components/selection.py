import streamlit as st
import pandas as pd
from src.search.grouped_search import GroupedCompanyResult
from src.ui.styles import confidence_badge_html
from src.data_retrieval.naics_lookup import get_industry_name, load_naics_map


def _get_sector(naics_code: str) -> str:
    """Extract the 2-digit NAICS sector from a raw NAICS code."""
    digits = "".join(c for c in str(naics_code or "") if c.isdigit())
    return digits[:2] if len(digits) >= 2 else "??"


def _render_naics_sector_filter(fac_list: list, parent_key: str) -> list:
    """If facilities span 2+ 2-digit NAICS sectors, render a multiselect filter.

    Returns the (possibly narrowed) facility list.
    """
    if not fac_list:
        return fac_list

    _naics_map = load_naics_map()

    # Group by 2-digit sector
    sectors: dict[str, list] = {}
    for f in fac_list:
        sector = _get_sector(f.naics_code)
        sectors.setdefault(sector, []).append(f)

    if len(sectors) <= 1:
        return fac_list  # single sector — no filter needed

    sector_labels: dict[str, str] = {}
    for sector, facs in sorted(sectors.items()):
        label = get_industry_name(sector, _naics_map)
        if label == "Unknown Industry":
            label = f"NAICS {sector}x"
        sector_labels[sector] = f"{label} ({len(facs)} est.)"

    state_key = f"naics_filter_{parent_key[:30]}"
    all_sector_codes = sorted(sectors.keys())

    st.markdown("**Industry sector filter**")
    selected_sectors = st.multiselect(
        "Include sectors",
        options=all_sector_codes,
        default=st.session_state.get(state_key, all_sector_codes),
        format_func=lambda s: sector_labels.get(s, s),
        key=f"ms_naics_{parent_key[:30]}",
        help="Filter which 2-digit NAICS industry sectors to include in the analysis.",
    )
    st.session_state[state_key] = selected_sectors

    if not selected_sectors:
        st.caption("Select at least one sector, or leave all selected.")
        return fac_list

    selected_set = set(selected_sectors)
    filtered = [f for f in fac_list if _get_sector(f.naics_code) in selected_set]
    if len(filtered) < len(fac_list):
        dropped = len(fac_list) - len(filtered)
        st.caption(
            f"{dropped} establishment{'s' if dropped != 1 else ''} excluded by sector filter."
        )
    return filtered


def _format_facility_option(raw_name: str, group: GroupedCompanyResult) -> str:
    all_fac = group.all_facilities
    match = next((f for f in all_fac if f.raw_name == raw_name), None)
    if match:
        parts = [match.raw_name]
        if match.city and match.state:
            parts.append(f"{match.city}, {match.state}")
        return " — ".join(parts)
    return raw_name


def render_facility_review_table(fac_to_show: list, title_suffix: str = ""):
    if not fac_to_show:
        return
    with st.expander(
        f"📍 Review included locations ({len(fac_to_show)}){title_suffix}",
        expanded=False,
    ):
        st.caption("These are the OSHA establishments that will be included in the analysis.")
        rows = []
        for f in fac_to_show:
            rows.append({
                "Establishment": f.display_name + (f" [{f.facility_code}]" if f.facility_code else ""),
                "Raw OSHA Name": f.raw_name,
                "City": f.city or "—",
                "State": f.state or "—",
                "Address": f.address or "—",
                "Confidence": f.confidence_label,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        low_count = sum(1 for f in fac_to_show if f.confidence_label == "Low")
        if low_count:
            st.warning(
                f"⚠️ {low_count} included establishment(s) have low confidence. "
                "Verify they belong to this company before relying on the score.",
                icon="⚠️",
            )


def render_scope_selector(group: GroupedCompanyResult):
    n = group.total_facilities
    multi = n > 1

    st.markdown(
        '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;'
        'text-transform:uppercase;color:#9BAFC5;margin-bottom:6px">Selected company</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        header_col, change_col = st.columns([5, 2], gap="medium")
        with header_col:
            badge = confidence_badge_html(group.confidence_label)
            st.markdown(
                f'<div style="font-size:1.0rem;font-weight:700;color:#0F2240">'
                f'🏢 {group.parent_name}</div>'
                f'<div style="font-size:0.8rem;color:#6B7E99;margin-top:3px">'
                f'{n} likely related OSHA establishment{"s" if n != 1 else ""}'
                f'&nbsp;{badge}</div>',
                unsafe_allow_html=True,
            )
        with change_col:
            if st.button("Change", key="btn_change_company", width="stretch"):
                st.session_state.selected_groups = []
                st.session_state.selected_group_keys = set()
                st.session_state.analysis_scope = "all"
                st.session_state.selected_facility_names = []
                st.rerun()

    if not multi:
        st.session_state.analysis_scope = "all"
        return "all", None

    st.markdown("**Analysis scope**")
    scope = st.radio(
        "analysis_scope_radio",
        options=["all", "one", "custom"],
        format_func=lambda x: {
            "all":    f"Include all {n} likely related facilities",
            "one":    "Analyze one specific facility only",
            "custom": "Select specific facilities to include",
        }[x],
        index=["all", "one", "custom"].index(st.session_state.analysis_scope),
        label_visibility="collapsed",
        key="radio_scope",
    )
    if scope != st.session_state.analysis_scope:
        st.session_state.analysis_scope = scope
        st.session_state.selected_facility_names = []
        st.rerun()

    if scope == "all":
        st.caption(
            "All establishments sharing this company’s identity in the OSHA database "
            "will be included in the analysis."
        )
        return "all", None

    all_facilities = group.all_facilities

    if scope == "one":
        options = [f.raw_name for f in all_facilities] or [group.parent_name]
        chosen = st.selectbox(
            "Choose a facility",
            options=options,
            format_func=lambda raw: _format_facility_option(raw, group),
            key="sel_single_facility",
        )
        return "one", [chosen] if chosen else None

    options = [f.raw_name for f in all_facilities] or [group.parent_name]
    default = st.session_state.selected_facility_names or []
    chosen = st.multiselect(
        "Select facilities to include",
        options=options,
        default=[d for d in default if d in options],
        format_func=lambda raw: _format_facility_option(raw, group),
        key="ms_custom_facilities",
    )
    if chosen:
        st.session_state.selected_facility_names = chosen
    if not chosen:
        st.caption("Pick at least one facility, or switch scope to 'Include all'.")
    return "custom", chosen if chosen else None


def render_selection_summary() -> tuple:
    selected_groups = st.session_state.selected_groups
    if not selected_groups:
        return None, ""

    st.session_state.naics_check_ok = True
    st.divider()

    # Single group
    if len(selected_groups) == 1:
        group = selected_groups[0]
        scope, chosen_raw_names = render_scope_selector(group)

        all_fac = group.all_facilities
        if scope == "all":
            fac_to_show = list(all_fac)
        elif chosen_raw_names:
            chosen_set = set(chosen_raw_names)
            fac_to_show = [f for f in all_fac if f.raw_name in chosen_set]
        else:
            fac_to_show = []

        fac_to_show = _render_naics_sector_filter(fac_to_show, group.parent_name)
        render_facility_review_table(fac_to_show)

        n_included = len(fac_to_show)
        if n_included > 1:
            st.markdown(
                f'<div class="vc-scope-warning">'
                f'<strong>ℹ️ Aggregated analysis:</strong> This analysis combines '
                f'<strong>{n_included} likely related establishments</strong> and may '
                f'not represent a single physical site.</div>',
                unsafe_allow_html=True,
            )
        # If filter narrowed the selection, use explicit facility names
        filtered_names = [f.raw_name for f in fac_to_show]
        if chosen_raw_names is None and len(fac_to_show) < len(group.all_facilities):
            return filtered_names or None, group.parent_name
        return chosen_raw_names if chosen_raw_names is not None else (filtered_names or None), group.parent_name

    # Multiple groups
    st.markdown(
        '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;'
        'text-transform:uppercase;color:#9BAFC5;margin-bottom:8px">'
        'Selected for combined analysis</div>',
        unsafe_allow_html=True,
    )

    all_fac_objects = []
    for group in selected_groups:
        all_fac_objects.extend(group.all_facilities)

    for group in selected_groups:
        fac = group.all_facilities
        n = len(fac)
        badge = confidence_badge_html(group.confidence_label)
        c_label, c_remove = st.columns([8, 1], gap="small")
        with c_label:
            st.markdown(
                f'<div style="border:1px solid #C8D9EE;border-radius:9px;'
                f'background:#F4F8FD;padding:10px 14px;margin-bottom:5px">'
                f'<span style="font-size:0.9rem;font-weight:600;color:#0F2240">'
                f'✅ 🏢 {group.parent_name}</span>&nbsp;'
                f'<span style="font-size:0.77rem;color:#9BAFC5">'
                f'{n} establishment{"s" if n != 1 else ""}</span>'
                f'&nbsp;{badge}</div>',
                unsafe_allow_html=True,
            )
        with c_remove:
            if st.button(
                "✕",
                key=f"btn_remove_{group.parent_name[:20]}",
                width="stretch",
                help=f"Remove {group.parent_name} from selection",
            ):
                st.session_state.selected_group_keys.discard(group.parent_name)
                st.session_state.selected_groups = [
                    g for g in st.session_state.selected_groups
                    if g.parent_name != group.parent_name
                ]
                if len(st.session_state.selected_groups) <= 1:
                    st.session_state.multi_group_scope = "all"
                    st.session_state.selected_facility_names = []
                st.rerun()

    display_name = " + ".join(g.parent_name for g in selected_groups)
    total_fac = len(all_fac_objects)

    st.session_state.naics_check_ok = True

    # Facility scope selector
    st.markdown("**Facility scope**")
    multi_scope = st.radio(
        "multi_group_scope_radio",
        options=["all", "custom"],
        format_func=lambda x: {
            "all":    f"Include all {total_fac} facilities from selected groups",
            "custom": "Select specific facilities to include",
        }[x],
        index=0 if st.session_state.multi_group_scope == "all" else 1,
        label_visibility="collapsed",
        key="radio_multi_scope",
    )
    if multi_scope != st.session_state.multi_group_scope:
        st.session_state.multi_group_scope = multi_scope
        st.session_state.selected_facility_names = []
        st.rerun()

    if multi_scope == "all":
        st.caption(
            "All high- and medium-confidence facilities from each selected group "
            "will be merged into a single analysis."
        )
        final_raw_names = [f.raw_name for f in all_fac_objects]
        fac_to_show = all_fac_objects
    else:
        option_raw_names = [f.raw_name for f in all_fac_objects]

        def _multi_format(raw: str) -> str:
            for g in selected_groups:
                match = next((f for f in g.all_facilities if f.raw_name == raw), None)
                if match:
                    label = match.display_name
                    if match.facility_code:
                        label += f" [{match.facility_code}]"
                    if match.city and match.state:
                        label += f" — {match.city}, {match.state}"
                    return f"{g.parent_name}  /  {label}"
            return raw

        default_sel = (
            st.session_state.selected_facility_names
            if st.session_state.selected_facility_names
            else option_raw_names
        )
        chosen = st.multiselect(
            "Select facilities to include",
            options=option_raw_names,
            default=[d for d in default_sel if d in option_raw_names],
            format_func=_multi_format,
            key="ms_multi_facilities",
        )
        if chosen:
            st.session_state.selected_facility_names = chosen
        else:
            st.caption("Pick at least one facility, or switch to 'Include all'.")
        chosen_set = set(chosen)
        final_raw_names = chosen if chosen else []
        fac_to_show = [f for f in all_fac_objects if f.raw_name in chosen_set]

    filter_key = "+".join(g.parent_name[:12] for g in selected_groups)
    fac_to_show = _render_naics_sector_filter(fac_to_show, filter_key)
    final_raw_names = [f.raw_name for f in fac_to_show]

    render_facility_review_table(
        fac_to_show,
        title_suffix=f" across {len(selected_groups)} groups",
    )

    n_final = len(final_raw_names)
    st.markdown(
        f'<div class="vc-scope-warning">'
        f'<strong>ℹ️ Combined analysis:</strong> This merges '
        f'<strong>{n_final} establishment{"s" if n_final != 1 else ""}</strong> across '
        f'<strong>{len(selected_groups)} selected groups</strong>. '
        f'Results reflect the aggregated OSHA footprint and may span '
        f'multiple distinct company entities.</div>',
        unsafe_allow_html=True,
    )

    return final_raw_names if final_raw_names else None, display_name
