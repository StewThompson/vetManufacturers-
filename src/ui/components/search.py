import streamlit as st
from src.search.grouped_search import (
    group_establishments, SearchResultSet, GroupedCompanyResult, FacilityCandidate,
)
from src.ui.styles import confidence_badge_html
from src.ui.components.selection import render_selection_summary


def _render_group_row(group, key_suffix, is_top, selected_keys):
    is_selected = group.parent_name in selected_keys
    any_selected = bool(selected_keys)
    n = group.total_facilities
    badge = confidence_badge_html(group.confidence_label)

    box_bg = "#F4F8FD" if is_selected else ("#FFFFFF" if is_top else "#FAFBFD")
    box_border = "#1D5A8E" if is_selected else ("#DDE4EF" if is_top else "#EEF2F8")
    pad = "16px 18px" if is_top else "10px 14px"
    radius = "13px" if is_top else "10px"
    name_size = "1.0rem" if is_top else "0.9rem"
    name_wt = "700" if is_top else "600"
    name_col = "#0F2240" if is_top else "#1A3558"
    check_html = (
        '<span style="color:#1D5A8E;font-size:0.9rem;margin-right:4px;font-weight:700">✓</span>'
        if is_selected else ""
    )
    multi_note = (
        f'&nbsp;<span class="vc-badge vc-badge-multi">'
        f'{n} related establishment{"s" if n != 1 else ""}</span>'
    ) if n > 1 and is_top else ""

    c_name, c_btn = st.columns([5, 2], gap="medium")
    with c_name:
        st.markdown(
            f"""<div style="border:1px solid {box_border};border-radius:{radius};
                background:{box_bg};padding:{pad};margin-bottom:4px">
                {check_html}<span style="font-size:{name_size};font-weight:{name_wt};
                color:{name_col}">🏢 {group.parent_name}</span>
                &nbsp;&nbsp;
                <span style="font-size:0.78rem;color:#9BAFC5">
                    {n} establishment{"s" if n != 1 else ""}
                </span>
                &nbsp;{badge}{multi_note}
            </div>""",
            unsafe_allow_html=True,
        )
    with c_btn:
        if is_selected:
            btn_label, btn_type = "✓ Added", "primary"
        elif any_selected:
            btn_label, btn_type = "＋ Add", "secondary"
        else:
            btn_label = "Select →" if is_top else "Select"
            btn_type = "primary" if is_top else "secondary"

        if st.button(btn_label, key=f"btn_toggle_{key_suffix}", type=btn_type, width="stretch"):
            if is_selected:
                st.session_state.selected_group_keys.discard(group.parent_name)
                st.session_state.selected_groups = [
                    g for g in st.session_state.selected_groups if g.parent_name != group.parent_name
                ]
                if not st.session_state.selected_groups:
                    st.session_state.analysis_scope = "all"
                    st.session_state.selected_facility_names = []
            else:
                st.session_state.selected_group_keys.add(group.parent_name)
                st.session_state.selected_groups.append(group)
                st.session_state.analysis_scope = "all"
                st.session_state.selected_facility_names = []
            st.rerun()


def _render_grouped_search_results(results: SearchResultSet):
    selected_keys = st.session_state.selected_group_keys

    if results.top_group is None and not results.other_groups and not results.unmatched:
        st.info(
            "No matches found in the OSHA cache for that query. "
            "Assessment will query the DOL API directly.",
            icon="ℹ️",
        )
        if st.button("Search live OSHA API →", key="btn_api_fallback"):
            new_g = GroupedCompanyResult(
                parent_name=results.query.title(),
                query=results.query,
                total_facilities=0,
                confidence=0.0,
                confidence_label="Low",
            )
            st.session_state.selected_group_keys.add(new_g.parent_name)
            st.session_state.selected_groups.append(new_g)
            st.rerun()
        return

    n_sel = len(selected_keys)
    if n_sel:
        names_str = " + ".join(
            f"**{g.parent_name}**" for g in st.session_state.selected_groups
        )
        st.success(
            f"{n_sel} group{'s' if n_sel > 1 else ''} selected: {names_str}  \n"
            "Add more below, or scroll down to configure and run the analysis.",
            icon="✅",
        )

    if results.top_group:
        st.markdown(
            '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;'
            'text-transform:uppercase;color:#9BAFC5;margin-bottom:6px">Best match</div>',
            unsafe_allow_html=True,
        )
        _render_group_row(results.top_group, key_suffix="top", is_top=True, selected_keys=selected_keys)

    if results.other_groups:
        st.markdown(
            '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;'
            'text-transform:uppercase;color:#9BAFC5;margin:14px 0 6px">Other possible matches</div>',
            unsafe_allow_html=True,
        )
        for i, group in enumerate(results.other_groups):
            _render_group_row(group, key_suffix=f"other_{i}", is_top=False, selected_keys=selected_keys)

    if results.unmatched:
        with st.expander(
            f"🔍 {len(results.unmatched)} possible variant(s) — lower confidence",
            expanded=False,
        ):
            st.caption(
                "These OSHA names partially matched your search but couldn't be "
                "confidently grouped with a parent company."
            )
            for raw_name in results.unmatched:
                c_n, c_b = st.columns([5, 2], gap="small")
                with c_n:
                    st.markdown(
                        f'<div style="font-size:0.83rem;color:#4A5568;padding:4px 0">{raw_name}</div>',
                        unsafe_allow_html=True,
                    )
                with c_b:
                    if st.button("Use this name", key=f"btn_unmatched_{raw_name[:30]}", width="stretch"):
                        fc = FacilityCandidate(
                            raw_name=raw_name,
                            display_name=raw_name,
                            facility_code=None,
                            city="", state="", address="",
                            confidence=0.3,
                            confidence_label="Low",
                        )
                        new_g = GroupedCompanyResult(
                            parent_name=raw_name,
                            query=raw_name,
                            total_facilities=1,
                            confidence=0.3,
                            confidence_label="Low",
                            high_confidence=[],
                            medium_confidence=[],
                            low_confidence=[fc],
                        )
                        st.session_state.selected_group_keys.add(raw_name)
                        st.session_state.selected_groups.append(new_g)
                        st.rerun()


def render_search_card(vetting_agent, all_companies, name_index=None):
    with st.container(border=True):
        st.markdown(
            '<div style="margin-bottom:14px">'
            '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;'
            'text-transform:uppercase;color:#9BAFC5;margin-bottom:5px">Manufacturer Lookup</div>'
            '<div style="font-size:1.05rem;font-weight:700;color:#0F2240;margin-bottom:3px">'
            'Vet a Manufacturer</div>'
            '<div style="font-size:0.83rem;color:#7B8FA8;line-height:1.5">'
            'Search by company name — select one or more groups to analyse together.</div></div>',
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns([7, 3], gap="large")
        raw_names_to_use = None
        display_name = ""

        with left_col:
            search_term = st.text_input(
                "Company name",
                value=st.session_state.search_term,
                placeholder="Enter company name — e.g. Amazon, Fastenal, Parker Hannifin",
                label_visibility="collapsed",
                key="txt_search_term",
            )
            if search_term != st.session_state.search_term:
                st.session_state.search_term = search_term
                st.session_state.search_results = None

            if search_term and len(search_term.strip()) >= 2:
                if st.session_state.search_results is None:
                    osha_client = vetting_agent.get_osha_client()
                    st.session_state.search_results = group_establishments(
                        query=search_term,
                        all_company_names=name_index or all_companies,
                        osha_client=osha_client,
                    )

            results: SearchResultSet | None = st.session_state.search_results
            if results is not None:
                _render_grouped_search_results(results)

            if st.session_state.selected_groups:
                raw_names_to_use, display_name = render_selection_summary()

        with right_col:
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
            has_selection = bool(st.session_state.selected_groups)
            n_groups = len(st.session_state.selected_groups)
            btn_label = (
                f"Run Assessment ({n_groups} groups) →"
                if n_groups > 1 else "Run Assessment →"
            )
            run_clicked = st.button(
                btn_label,
                type="primary",
                width="stretch",
                key="btn_run_assessment",
                disabled=not has_selection or not st.session_state.get("naics_check_ok", True),
            )
            if has_selection:
                total_estabs = sum(
                    max(len([f for f in g.all_facilities if f.confidence >= 0.55]), 1)
                    for g in st.session_state.selected_groups
                )
                st.markdown(
                    f'<div style="font-size:0.73rem;color:#1D5A8E;margin-top:9px;'
                    f'line-height:1.6;text-align:center;font-weight:600">'
                    f'{n_groups} group{"s" if n_groups > 1 else ""}'
                    f'&nbsp;·&nbsp;~{total_estabs} establishment{"s" if total_estabs != 1 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:0.73rem;color:#9BAFC5;margin-top:9px;'
                    'line-height:1.5;text-align:center">'
                    'Returns risk score, inspection signals,<br>'
                    'and explainable recommendation.</div>',
                    unsafe_allow_html=True,
                )

        if run_clicked and st.session_state.selected_groups:
            display_name = display_name or st.session_state.selected_groups[0].parent_name
            n_g = len(st.session_state.selected_groups)
            years_back = st.session_state.get("sb_year_range", 10)
            min_insp = st.session_state.get("sb_min_insp", 1)

            if n_g == 1 and raw_names_to_use is None:
                label = st.session_state.selected_groups[0].parent_name
            elif n_g == 1 and raw_names_to_use:
                label = f"{st.session_state.selected_groups[0].parent_name} ({len(raw_names_to_use)} facilities)"
            else:
                label = f"{display_name} ({n_g} groups combined)"

            with st.spinner(f"Retrieving OSHA data and scoring {label}…"):
                try:
                    if raw_names_to_use is None and n_g == 1:
                        assessment = vetting_agent.vet_manufacturer(
                            st.session_state.selected_groups[0].parent_name,
                            locations=None,
                            years_back=years_back,
                        )
                    else:
                        names = raw_names_to_use or [
                            f.raw_name
                            for g in st.session_state.selected_groups
                            for f in g.all_facilities
                            if f.confidence >= 0.40
                        ]
                        assessment = vetting_agent.vet_by_raw_estab_names(
                            raw_names=names,
                            display_name=display_name,
                            years_back=years_back,
                        )
                    if len(assessment.records) < min_insp:
                        st.warning(
                            f"Only {len(assessment.records)} inspection(s) found in the last "
                            f"{years_back} year(s) — below your minimum threshold of {min_insp}. "
                            f"Results may be unreliable."
                        )
                    st.session_state.assessment = assessment
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Assessment failed: {e}")
