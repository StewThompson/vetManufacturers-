import time
import threading
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
                            naics_code="",
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
    # ------------------------------------------------------------------ #
    #  Background thread polling — thread never touches st.session_state.
    #  It only writes to a plain Python dict (_shared) captured by closure.
    # ------------------------------------------------------------------ #
    _shared = st.session_state.get("_assess_shared")

    if _shared is not None:
        if _shared["error"]:
            st.error(f"Assessment failed: {_shared['error']}")
            st.session_state.pop("_assess_shared", None)
            _shared = None

        elif _shared["done"] and _shared["result"] is not None:
            _assessment = _shared["result"]
            st.session_state.pop("_assess_shared", None)
            _min_insp = st.session_state.get("sb_min_insp", 1)
            if len(_assessment.records) < _min_insp:
                st.warning(
                    f"Only {len(_assessment.records)} inspection(s) found in the last "
                    f"{st.session_state.get('sb_year_range', 10)} year(s) — below your "
                    f"minimum threshold of {_min_insp}. Results may be unreliable."
                )
            for _k in [_k for _k in st.session_state if _k.startswith("_violations_df_")]:
                del st.session_state[_k]
            st.session_state.assessment = _assessment
            st.session_state.llm_pending = bool(vetting_agent.client)
            st.session_state.messages = []
            st.rerun()

        else:
            # Still running — show progress and poll again
            _label = _shared.get("label", "company")
            with st.status(f"Running assessment for {_label}…", expanded=True) as _s:
                for _msg in list(_shared["progress"]):
                    _s.write(_msg)
            time.sleep(0.3)
            st.rerun()

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
                        company_key_index=name_index,
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
                    max(g.total_facilities, 1)
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

            if n_g == 1 and raw_names_to_use is None:
                label = st.session_state.selected_groups[0].parent_name
            elif n_g == 1 and raw_names_to_use:
                label = f"{st.session_state.selected_groups[0].parent_name} ({len(raw_names_to_use)} facilities)"
            else:
                label = f"{display_name} ({n_g} groups combined)"

            # Plain Python dict — the thread only ever touches this object,
            # never st.session_state (which is not thread-safe / lacks context).
            _shared = {
                "label": label,
                "progress": [],
                "result": None,
                "error": None,
                "done": False,
            }
            st.session_state["_assess_shared"] = _shared

            # Snapshot closure variables before the thread captures them
            _raw_names = raw_names_to_use
            _display_name = display_name
            _n_g = n_g
            _years_back = years_back
            _groups = list(st.session_state.selected_groups)

            def _run_assessment(_s=_shared):
                def _cb(msg: str):
                    _s["progress"].append(msg)

                try:
                    if _raw_names is None and _n_g == 1:
                        result = vetting_agent.vet_manufacturer(
                            _groups[0].parent_name,
                            locations=None,
                            years_back=_years_back,
                            progress_cb=_cb,
                        )
                    else:
                        names = _raw_names or [
                            f.raw_name for g in _groups for f in g.all_facilities
                        ]
                        result = vetting_agent.vet_by_raw_estab_names(
                            raw_names=names,
                            display_name=_display_name,
                            years_back=_years_back,
                            progress_cb=_cb,
                        )
                    _s["result"] = result
                except Exception as exc:
                    _s["error"] = str(exc)
                finally:
                    _s["done"] = True

            threading.Thread(target=_run_assessment, daemon=True).start()
            st.rerun()
