import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agent.vetting_agent import VettingAgent
from src.search.grouped_search import get_or_build_name_index
from src.ui.styles import inject_global_css
from src.ui.components.hero import render_sidebar, render_hero, render_feature_cards, render_how_it_works, render_empty_state
from src.ui.components.search import render_search_card
from src.ui.components.results import render_results

st.set_page_config(
    page_title="Manufacturer Compliance Intelligence",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def _get_vetting_agent():
    return VettingAgent()


vetting_agent = _get_vetting_agent()

for key, default in {
    "messages": [],
    "assessment": None,
    "search_term": "",
    "search_results": None,
    "selected_groups": [],
    "selected_group_keys": set(),
    "analysis_scope": "all",
    "selected_facility_names": [],
    "multi_group_scope": "all",
    "naics_check_ok": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data
def _cached_raw_estab_names():
    return vetting_agent.get_osha_client().get_all_raw_estab_names()


@st.cache_resource
def _cached_name_index(_names):
    return get_or_build_name_index(_names)


all_companies = _cached_raw_estab_names()
name_index = _cached_name_index(tuple(all_companies))

inject_global_css()
render_sidebar()
render_hero()
render_search_card(vetting_agent, all_companies, name_index)

if st.session_state.assessment:
    render_results(st.session_state.assessment, vetting_agent)
else:
    render_feature_cards()
    render_how_it_works()
    render_empty_state()
