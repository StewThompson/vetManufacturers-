import streamlit as st

# SVG shield logo used in the hero banner
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 58" width="54" height="65">
  <path d="M24 2 L46 10 L46 30 C46 44 36 53 24 57 C12 53 2 44 2 30 L2 10 Z"
        fill="#C5DDF4" stroke="#5A9FD0" stroke-width="2" stroke-linejoin="round"/>
  <path d="M24 7 L41 14 L41 29 C41 40 33 48 24 52 C15 48 7 40 7 29 L7 14 Z"
        fill="#EAF3FB" stroke="none"/>
  <polyline points="15,28 22,36 33,20"
            fill="none" stroke="#1D5A8E" stroke-width="3.4"
            stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

# OSHA hazard theme definitions: (code_prefixes, theme_name, description, icon)
THEME_DEFS = [
    (["19100147", "29100147"],
     "Lockout / Tagout",
     "Failure to de-energize equipment during maintenance — risk of fatal crush or amputation.", "⚡"),
    (["19100212", "19100213", "19100214", "19100215", "19100216", "19100217",
      "19100218", "19100219"],
     "Machine Guarding",
     "Exposed moving parts, unguarded blades or gears — amputation and entanglement risk.", "⚙️"),
    (["19100303", "19100304", "19100305", "19100334", "19100330", "19100331"],
     "Electrical Safety",
     "Unguarded live conductors, improper wiring, or lack of arc-flash protection.", "🔌"),
    (["19100022", "19100023", "19100024", "19100025", "19100028", "19100029",
      "19260501", "19260502", "19260503", "19260451"],
     "Fall Protection / Walking Surfaces",
     "Unguarded floor openings, improper scaffolding, or absence of fall-arrest systems.", "🧗"),
    (["19100106", "19100157", "19100101", "19100110"],
     "Fire / Flammables / Compressed Gas",
     "Improper storage or handling of flammable liquids, gases, or compressed cylinders.", "🔥"),
    (["19101200"],
     "Hazard Communication / Chemical Safety",
     "Missing SDS, unlabeled containers, or failure to train workers on chemical hazards.", "☣️"),
    (["19100132", "19100133", "19100134", "19100135", "19100136", "19100138", "19100140"],
     "Personal Protective Equipment",
     "Failure to provide or require appropriate PPE for identified hazards.", "🦺"),
    (["19100095", "19100096"],
     "Noise / Hearing Conservation",
     "Inadequate hearing protection or noise-level monitoring in high-decibel environments.", "👂"),
    (["19100139", "19260352"],
     "Welding / Hot Work",
     "Fire ignition hazards from uncontrolled hot-work operations.", "🔩"),
    (["19101050", "19101001", "19260062", "19101025", "19260055"],
     "Toxic Substances (Lead / Asbestos / Silica)",
     "Exposure to carcinogenic or acutely toxic industrial materials.", "☢️"),
    (["19100020"],
     "Recordkeeping",
     "Failure to maintain OSHA-required injury and illness logs.", "📋"),
    (["5A1", "5(A)(1)", "OSHACT", "SECTION5", "GENERALDUTY"],
     "General Duty Clause",
     "Broad safety management failure — employer did not address recognized serious hazards.", "⚠️"),
    (["19260050", "19260051", "19260100", "19260150", "19260200", "19260250",
      "19260300", "19260350", "19260400", "19260450", "19260600"],
     "Construction Safety",
     "Violations of construction-specific OSHA standards (subparts C–R).", "🏗️"),
]


def inject_global_css():
    """Inject all custom CSS into the Streamlit page."""
    st.markdown("""
<style>
html, body { background: #EFF2F7 !important; }
[data-testid="stAppViewContainer"] > .main { background: #EFF2F7 !important; }
.block-container {
    padding: 1.6rem 2.4rem 4rem 2.4rem !important;
    max-width: 1320px !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DDE4EF !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.6rem 1.2rem !important;
    max-width: none !important;
}
h2 { color: #0F2240 !important; font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { color: #2A4A6E !important; font-size: 0.96rem !important; font-weight: 600 !important; }
.stTextInput > div > div > input {
    border-radius: 9px !important;
    border-color: #C8D5E4 !important;
    padding: 0.58rem 0.9rem !important;
    font-size: 0.9rem !important;
    background: #FAFCFF !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1D5A8E !important;
    box-shadow: 0 0 0 3px rgba(29,90,142,0.11) !important;
}
.stSelectbox > div > div { border-radius: 9px !important; }
.stMultiSelect > div > div { border-radius: 9px !important; }
.stButton > button[kind="primary"] {
    background: #1D5A8E !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    transition: background 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover { background: #144572 !important; }
.stButton > button:not([kind="primary"]) {
    border-radius: 9px !important;
    border-color: #C5D3E3 !important;
    color: #2D4A6E !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE4EF !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: 0 1px 5px rgba(20,50,100,0.07) !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7E99 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricValue"] {
    color: #0F2240 !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}
[data-testid="stContainer"] { border-radius: 14px !important; }
[data-testid="stExpander"] {
    border: 1px solid #DDE4EF !important;
    border-radius: 10px !important;
    background: #FAFBFD !important;
}
hr { border-color: #E0E8F0 !important; margin: 1.4rem 0 !important; }
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.87rem !important; }
.vc-group-card {
    background:#FFFFFF;border:1px solid #DDE4EF;border-radius:13px;
    padding:16px 18px;margin-bottom:8px;
    transition:border-color 0.15s ease,box-shadow 0.15s ease;cursor:pointer;
}
.vc-group-card:hover {
    border-color:#1D5A8E;box-shadow:0 2px 8px rgba(29,90,142,0.13);
}
.vc-group-card.selected {
    border-color:#1D5A8E;box-shadow:0 0 0 3px rgba(29,90,142,0.12);background:#F4F8FD;
}
.vc-badge {
    display:inline-block;padding:2px 9px;border-radius:20px;
    font-size:0.68rem;font-weight:700;letter-spacing:0.4px;
}
.vc-badge-high   { background:#D1F0DC; color:#1A6B3A; }
.vc-badge-medium { background:#FEF3CD; color:#7A5500; }
.vc-badge-low    { background:#EEF3FA; color:#2A4A6E; }
.vc-badge-multi  { background:#EBF3FB; color:#1D5A8E; }
.vc-facility-row {
    display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;
    margin-bottom:5px;background:#FAFBFD;border:1px solid #EEF2F8;font-size:0.82rem;
}
.vc-scope-warning {
    background:#FFF8E1;border:1px solid #F3D77A;border-radius:9px;
    padding:10px 14px;font-size:0.82rem;color:#5C4A00;margin-top:10px;
}
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    background: #F8FAFC !important;
    border: 1px solid #E8EEF6 !important;
}
</style>
""", unsafe_allow_html=True)


def confidence_badge_html(label: str) -> str:
    css_class = {
        "High": "vc-badge-high",
        "Medium": "vc-badge-medium",
        "Low": "vc-badge-low",
    }.get(label, "vc-badge-low")
    return f'<span class="vc-badge {css_class}">{label} confidence</span>'


def map_standard_to_theme(code: str):
    """Return (theme_name, icon) for an OSHA standard code string."""
    import re
    code_up = re.sub(r'[\s.\-/()\']', '', (code or "").upper())
    for prefixes, theme, _, icon in THEME_DEFS:
        for p in prefixes:
            p_clean = re.sub(r'[\s.\-/()\']', '', p.upper())
            if code_up.startswith(p_clean):
                return theme, icon
    return "Other Safety Issues", "📌"


def theme_description(theme: str) -> str:
    for _, t, desc, _ in THEME_DEFS:
        if t == theme:
            return desc
    return "Various OSHA regulatory violations."
