import streamlit as st
from src.ui.styles import LOGO_SVG


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">'
            '<span style="font-size:1.3rem">🛡️</span>'
            '<span style="font-size:0.96rem;font-weight:700;color:#0F2240">Compliance Intel</span></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Filters**")
        st.caption("Applied when running an assessment.")
            
        st.number_input("Min. inspections to score", min_value=1, max_value=20, value=1, key="sb_min_insp")
        st.divider()
        st.markdown("**Display**")
        st.checkbox("Show raw feature vector", value=False, key="sb_show_raw")


def render_hero():
    st.markdown(f"""
    <div style="
        background:#FFFFFF;border:1px solid #DDE4EF;border-radius:16px;
        box-shadow:0 2px 10px rgba(15,40,90,0.08);padding:26px 30px;
        display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:18px;
    ">
        <div style="display:flex;align-items:flex-start;gap:18px;flex:1;min-width:0">
            <div style="flex-shrink:0;margin-top:3px">{LOGO_SVG}</div>
            <div>
                <div style="margin-bottom:8px">
                    <span style="display:inline-block;background:#EBF3FB;color:#1D5A8E;
                        font-size:0.67rem;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;
                        padding:3px 10px;border-radius:20px;border:1px solid #BED5EE;">
                        OSHA-Powered Risk Intelligence</span>
                </div>
                <h1 style="margin:0 0 7px 0;font-size:1.95rem;font-weight:800;color:#0F2240;
                    letter-spacing:-0.5px;line-height:1.15;">Manufacturer Compliance Intelligence</h1>
                <p style="margin:0;color:#6B7E99;font-size:0.88rem;line-height:1.55;">
                    Evaluate OSHA inspection history, violation patterns, and injury risk for any US manufacturer<br>
                    — powered by DOL public data and machine learning.</p>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;min-width:195px">
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                background:#F4F7FB;border:1px solid #DDE4EF;border-radius:10px">
                <span style="font-size:1rem">🔎</span>
                <div>
                    <div style="font-size:0.79rem;font-weight:700;color:#1A3558">Entity Resolution</div>
                    <div style="font-size:0.71rem;color:#8FA4BC">Smart name + location matching</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                background:#F4F7FB;border:1px solid #DDE4EF;border-radius:10px">
                <span style="font-size:1rem">📋</span>
                <div>
                    <div style="font-size:0.79rem;font-weight:700;color:#1A3558">Inspection History</div>
                    <div style="font-size:0.71rem;color:#8FA4BC">Full OSHA violation timeline</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                background:#F4F7FB;border:1px solid #DDE4EF;border-radius:10px">
                <span style="font-size:1rem">🤖</span>
                <div>
                    <div style="font-size:0.79rem;font-weight:700;color:#1A3558">Explainable Scoring</div>
                    <div style="font-size:0.71rem;color:#8FA4BC">ML risk score with rationale</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_cards():
    st.markdown("""
    <div style="margin:24px 0 10px">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;color:#9BAFC5;margin-bottom:10px">What you'll get</div>
    </div>
    """, unsafe_allow_html=True)

    _card = ("background:#FFFFFF;border:1px solid #DDE4EF;border-radius:12px;"
             "padding:22px 20px;box-shadow:0 1px 4px rgba(20,50,100,0.06);height:100%")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(f'<div style="{_card}"><div style="font-size:1.8rem;margin-bottom:10px">📊</div>'
                     '<div style="font-size:0.92rem;font-weight:700;color:#0F2240;margin-bottom:6px">Risk Score</div>'
                     '<div style="font-size:0.81rem;color:#7B8FA8;line-height:1.55">'
                     'ML-calibrated 0–100 risk rating benchmarked against the full DOL inspection population.</div></div>',
                     unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="{_card}"><div style="font-size:1.8rem;margin-bottom:10px">📋</div>'
                     '<div style="font-size:0.92rem;font-weight:700;color:#0F2240;margin-bottom:6px">OSHA Record Summary</div>'
                     '<div style="font-size:0.81rem;color:#7B8FA8;line-height:1.55">'
                     'Full inspection history — violations, penalties, repeat offenses, injury reports, and fatalities.</div></div>',
                     unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="{_card}"><div style="font-size:1.8rem;margin-bottom:10px">🔍</div>'
                     '<div style="font-size:0.92rem;font-weight:700;color:#0F2240;margin-bottom:6px">Explainable Recommendation</div>'
                     '<div style="font-size:0.81rem;color:#7B8FA8;line-height:1.55">'
                     'AI-generated compliance narrative with key risk drivers ranked by learned feature importance.</div></div>',
                     unsafe_allow_html=True)


def render_how_it_works():
    _step_num = ("width:28px;height:28px;min-width:28px;border-radius:50%;"
                 "background:#1D5A8E;color:white;font-size:0.78rem;font-weight:700;"
                 "display:flex;align-items:center;justify-content:center;margin-top:1px")
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #DDE4EF;border-radius:14px;
        padding:22px 26px;margin-top:16px;box-shadow:0 1px 5px rgba(20,50,100,0.06);">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;color:#9BAFC5;margin-bottom:14px">How it works</div>
        <div style="display:flex;gap:0;align-items:flex-start">
            <div style="flex:1;display:flex;align-items:flex-start;gap:12px;
                        padding-right:20px;border-right:1px solid #EEF2F8">
                <div style="{_step_num}">1</div>
                <div>
                    <div style="font-size:0.87rem;font-weight:700;color:#0F2240;margin-bottom:3px">Resolve Entity</div>
                    <div style="font-size:0.78rem;color:#7B8FA8;line-height:1.45">Match the company name to OSHA establishment records using fuzzy name and location resolution.</div>
                </div>
            </div>
            <div style="flex:1;display:flex;align-items:flex-start;gap:12px;
                        padding:0 20px;border-right:1px solid #EEF2F8">
                <div style="{_step_num}">2</div>
                <div>
                    <div style="font-size:0.87rem;font-weight:700;color:#0F2240;margin-bottom:3px">Retrieve OSHA History</div>
                    <div style="font-size:0.78rem;color:#7B8FA8;line-height:1.45">Pull all inspections, violations, accident reports, and injury data from the DOL API or bulk cache.</div>
                </div>
            </div>
            <div style="flex:1;display:flex;align-items:flex-start;gap:12px;padding-left:20px">
                <div style="{_step_num}">3</div>
                <div>
                    <div style="font-size:0.87rem;font-weight:700;color:#0F2240;margin-bottom:3px">Score &amp; Explain Risk</div>
                    <div style="font-size:0.78rem;color:#7B8FA8;line-height:1.45">ML model scores the manufacturer relative to the population and generates an explainable risk narrative.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div style="text-align:center;padding:44px 24px;color:#9BAFC5;margin-top:8px">
        <div style="font-size:2.6rem;margin-bottom:12px">🛡️</div>
        <div style="font-size:0.9rem;font-weight:600;color:#6B7E99;margin-bottom:6px">No assessment loaded</div>
        <div style="font-size:0.82rem;color:#9BAFC5;line-height:1.6">
            Run an assessment above to view compliance risk,<br>
            full inspection history, and an explainable recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)
