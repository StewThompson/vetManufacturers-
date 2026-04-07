import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

features = [
    (0,  'log_inspections',         'Inspection Count',              'log1p(# inspections)',                          'Absolute', 'float'),
    (1,  'log_violations',          'Violation Count',               'log1p(# violations)',                           'Absolute', 'float'),
    (2,  'serious_violations',      'Serious Rate',                  '# serious viols / # insp',                      'Absolute', 'rate'),
    (3,  'willful_violations',      'Willful Rate',                  '# willful viols / # insp',                      'Absolute', 'rate'),
    (4,  'repeat_violations',       'Repeat Rate',                   '# repeat viols / # insp',                       'Absolute', 'rate'),
    (5,  'log_penalties',           'Total Penalty ($)',              'sum of all violation penalties (raw $)',         'Absolute', '$'),
    (6,  'avg_penalty',             'Avg Penalty ($)',                'sum(penalties) / # violations',                 'Absolute', '$'),
    (7,  'max_penalty',             'Max Single Penalty ($)',         'max(per-violation penalty amount)',              'Absolute', '$'),
    (8,  'recent_ratio',            'Recent Activity (3 yr)',         '# insp in last 3 yrs / n_insp',                 'Absolute', 'rate'),
    (9,  'severe_incidents',        'Fatal / Catastrophe Rate',      '# insp with linked accident / n_insp',          'Absolute', 'rate'),
    (10, 'violations_per_insp',     'Violations / Inspection',       '# violations / # inspections',                  'Absolute', 'count'),
    (11, 'accident_count',          'Linked Accident Rate',          'log1p(# accidents / n_insp)',                   'Absolute', 'float'),
    (12, 'fatality_count',          'Fatality Rate',                 'log1p(# fatalities / n_insp)',                  'Absolute', 'float'),
    (13, 'injury_count',            'Injury Rate',                   'log1p(# injuries / n_insp)',                    'Absolute', 'float'),
    (14, 'avg_gravity',             'Avg Violation Gravity',         'mean(gravity 1-10 across all violations)',       'Absolute', 'score'),
    (15, 'penalties_per_insp',      'Penalty / Inspection ($)',      'total_penalty / n_insp',                        'Absolute', '$'),
    (16, 'clean_ratio',             'Clean Inspection Rate',         '# insp with 0 violations / n_insp',             'Absolute', 'rate'),
    (17, 'time_adjusted_penalty',   'Time-Adjusted Penalty ($)',     'sum( pen * exp( -age_yrs / 3 ) )',              'Absolute', '$'),
    (18, 'recent_wr_rate',          'Recent W/R Rate',               'recent W/R viols / total W/R viols',            'Absolute', 'rate'),
    (19, 'trend_delta',             'Violation Trend',               'vpi_all_time - vpi_recent  (pos = improving)',  'Absolute', 'delta'),
    (20, 'log_willful_raw',         'Willful Count (log)',           'log1p(raw # willful violations)',               'Disc.',    'float'),
    (21, 'log_repeat_raw',          'Repeat Count (log)',            'log1p(raw # repeat violations)',                'Disc.',    'float'),
    (22, 'has_any_fatality',        'Fatality Flag',                 '1 if any fatality ever, else 0',               'Disc.',    'binary'),
    (23, 'log_max_insp_penalty',    'Max-Insp Penalty (log)',        'log1p(max total penalty in single inspection)', 'Disc.',    'float'),
    (24, 'log_estab_size',          'Establishment Size (log)',      'log1p(median employee count)',                  'Disc.',    'float'),
    (25, 'relative_violation_rate', 'Viol Rate vs Industry',         'z-score vs 4-digit NAICS peer group',          'Z-Score',  'z'),
    (26, 'relative_penalty',        'Penalty vs Industry',           'z-score vs 4-digit NAICS peer group',          'Z-Score',  'z'),
    (27, 'relative_serious_ratio',  'Serious Ratio vs Industry',     'z-score vs 4-digit NAICS peer group',          'Z-Score',  'z'),
    (28, 'relative_willful_repeat', 'W/R Rate vs Industry',          'z-score vs 4-digit NAICS peer group',          'Z-Score',  'z'),
    (29, 'naics_11', 'Agriculture / Forestry / Fishing',         '1 if 2-digit NAICS prefix = 11',  'NAICS', 'binary'),
    (30, 'naics_21', 'Mining / Oil & Gas Extraction',            '1 if 2-digit NAICS prefix = 21',  'NAICS', 'binary'),
    (31, 'naics_22', 'Utilities',                                '1 if 2-digit NAICS prefix = 22',  'NAICS', 'binary'),
    (32, 'naics_23', 'Construction',                             '1 if 2-digit NAICS prefix = 23',  'NAICS', 'binary'),
    (33, 'naics_31', 'Manufacturing (Food / Textile / Apparel)', '1 if 2-digit NAICS prefix = 31',  'NAICS', 'binary'),
    (34, 'naics_32', 'Manufacturing (Wood / Paper / Chemical)',  '1 if 2-digit NAICS prefix = 32',  'NAICS', 'binary'),
    (35, 'naics_33', 'Manufacturing (Metal / Machinery / Elec)','1 if 2-digit NAICS prefix = 33',  'NAICS', 'binary'),
    (36, 'naics_42', 'Wholesale Trade',                          '1 if 2-digit NAICS prefix = 42',  'NAICS', 'binary'),
    (37, 'naics_44', 'Retail Trade',                             '1 if 2-digit NAICS prefix = 44',  'NAICS', 'binary'),
    (38, 'naics_45', 'Retail Trade (Misc. / Nonstore)',          '1 if 2-digit NAICS prefix = 45',  'NAICS', 'binary'),
    (39, 'naics_48', 'Transportation',                           '1 if 2-digit NAICS prefix = 48',  'NAICS', 'binary'),
    (40, 'naics_49', 'Warehousing & Storage',                    '1 if 2-digit NAICS prefix = 49',  'NAICS', 'binary'),
    (41, 'naics_51', 'Information',                              '1 if 2-digit NAICS prefix = 51',  'NAICS', 'binary'),
    (42, 'naics_52', 'Finance & Insurance',                      '1 if 2-digit NAICS prefix = 52',  'NAICS', 'binary'),
    (43, 'naics_53', 'Real Estate',                              '1 if 2-digit NAICS prefix = 53',  'NAICS', 'binary'),
    (44, 'naics_54', 'Professional / Scientific / Technical',    '1 if 2-digit NAICS prefix = 54',  'NAICS', 'binary'),
    (45, 'naics_55', 'Management of Companies',                  '1 if 2-digit NAICS prefix = 55',  'NAICS', 'binary'),
    (46, 'naics_56', 'Admin / Support / Waste Management',       '1 if 2-digit NAICS prefix = 56',  'NAICS', 'binary'),
    (47, 'naics_61', 'Educational Services',                     '1 if 2-digit NAICS prefix = 61',  'NAICS', 'binary'),
    (48, 'naics_62', 'Health Care & Social Assistance',          '1 if 2-digit NAICS prefix = 62',  'NAICS', 'binary'),
    (49, 'naics_71', 'Arts / Entertainment / Recreation',        '1 if 2-digit NAICS prefix = 71',  'NAICS', 'binary'),
    (50, 'naics_72', 'Accommodation & Food Services',            '1 if 2-digit NAICS prefix = 72',  'NAICS', 'binary'),
    (51, 'naics_81', 'Other Services',                           '1 if 2-digit NAICS prefix = 81',  'NAICS', 'binary'),
    (52, 'naics_92', 'Public Administration',                    '1 if 2-digit NAICS prefix = 92',  'NAICS', 'binary'),
    (53, 'naics_unknown', 'Industry Unknown',                    '1 if no NAICS code available',    'NAICS', 'binary'),
]

GROUP_STYLE = {
    'Absolute': {'bg': '#131D33', 'bg_alt': '#0F1626', 'pill': '#4F8EF7', 'text': '#7BB3FF', 'brk': '#4F8EF7'},
    'Disc.':    {'bg': '#1A1530', 'bg_alt': '#130E27', 'pill': '#A78BFA', 'text': '#C4AAFF', 'brk': '#A78BFA'},
    'Z-Score':  {'bg': '#0F2018', 'bg_alt': '#0B1912', 'pill': '#34D399', 'text': '#6DEBBA', 'brk': '#34D399'},
    'NAICS':    {'bg': '#201508', 'bg_alt': '#180F05', 'pill': '#F5A623', 'text': '#FAC26A', 'brk': '#F5A623'},
}

BG    = '#0D1117'
PANEL = '#161B22'
BORDER= '#2D3748'
WHITE = '#F0F6FC'
DIM   = '#8B949E'
GOLD  = '#FCD34D'

N  = len(features)
RH = 0.90   # row height in data units

# Canvas coordinate system: 0-32 wide
XL, XR = 0.35, 31.65   # left/right table edges
CW = XR - XL           # 31.30 usable width

TOTAL_Y = N * RH + 6.0

fig = plt.figure(figsize=(32, TOTAL_Y * 0.52))
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.01, 0.005, 0.98, 0.99])
ax.set_xlim(0, 32)
ax.set_ylim(-0.6, TOTAL_Y)
ax.axis('off')

XMID = (XL + XR) / 2   # 16.0 — natural centre

def txt(x, y, s, size=9, color=WHITE, ha='left', va='center', bold=False, mono=False):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontfamily='monospace' if mono else 'sans-serif', zorder=5,
            clip_on=False)

def hline(y, color=BORDER, lw=0.7):
    ax.plot([XL, XR], [y, y], color=color, lw=lw, zorder=3)

# Column layout — fields fill XL..XR exactly, centred under title
# (x_start, width, header, align)
COLS = [
    (XL,        1.50, '#',                   'center'),
    (XL+1.50,   5.80, 'Key',                 'left'),
    (XL+7.30,   7.40, 'Description',         'left'),
    (XL+14.70, 16.60, 'Formula / Derivation','left'),
]

# TITLE
ty = TOTAL_Y - 0.60
txt(XMID, ty, '54-Feature Input Vector  —  Field Reference', size=22, bold=True, color=WHITE, ha='center')
txt(XMID, ty - 0.85,
    'Features are computed per-establishment from historical OSHA records, concatenated in index order, and passed to all model heads',
    size=12, color=DIM, ha='center')

# LEGEND
legend = [
    ('Absolute Signals  (idx 0 – 19)', 'Absolute'),
    ('High-Signal Discriminators  (20 – 24)', 'Disc.'),
    ('Industry Z-Scores  (25 – 28)', 'Z-Score'),
    ('NAICS 2-digit One-Hot  (29 – 53)', 'NAICS'),
]
lx = XL + 0.2
for lbl, grp in legend:
    st = GROUP_STYLE[grp]
    pill = mpatches.FancyBboxPatch((lx, ty - 2.0), 0.50, 0.40,
        boxstyle='round,pad=0.05', lw=0, facecolor=st['pill'], zorder=4)
    ax.add_patch(pill)
    txt(lx + 0.68, ty - 1.80, lbl, size=11, color=st['text'])
    lx += 7.6

# COLUMN HEADER
hy = TOTAL_Y - 3.10
hdr = mpatches.FancyBboxPatch((XL, hy - 0.38), CW, 0.76,
    boxstyle='square,pad=0', facecolor='#1C2333', lw=0, zorder=2)
ax.add_patch(hdr)
hline(hy - 0.38, color='#4F8EF7', lw=2.2)
hline(hy + 0.38, color='#4F8EF7', lw=2.2)

for (cx, cw, ch, ca) in COLS:
    hx = cx + (cw/2 if ca == 'center' else 0.18)
    txt(hx, hy, ch, size=13, bold=True, color=GOLD, ha=ca)

# DATA ROWS
section_breaks = {20, 25, 29}

C_IDX   = COLS[0][0]          # 0.35
C_KEY   = COLS[1][0]          # 1.85
C_DESC  = COLS[2][0]          # 7.65
C_FORM  = COLS[3][0]          # 15.05

for i, (idx, key, desc, formula, group, dtype) in enumerate(features):
    y = TOTAL_Y - 3.90 - i * RH
    st = GROUP_STYLE[group]
    fc = st['bg'] if i % 2 == 0 else st['bg_alt']

    bg = mpatches.FancyBboxPatch((XL, y - RH * 0.46), CW, RH * 0.92,
        boxstyle='square,pad=0', lw=0, facecolor=fc, zorder=2)
    ax.add_patch(bg)

    if idx in section_breaks:
        ax.plot([XL, XR], [y + RH * 0.46, y + RH * 0.46],
                color=st['brk'], lw=2.5, zorder=4)

    # index pill
    ip = mpatches.FancyBboxPatch((C_IDX + 0.05, y - 0.22), 1.30, 0.44,
        boxstyle='round,pad=0.05', lw=1.2,
        facecolor=BG, edgecolor=st['pill'], zorder=4)
    ax.add_patch(ip)
    txt(C_IDX + 0.70, y, str(idx), size=12, color=st['pill'], ha='center', mono=True)

    txt(C_KEY  + 0.18, y, key,     size=11,  color=st['text'],  mono=True)
    txt(C_DESC + 0.18, y, desc,    size=12,  color=WHITE)
    txt(C_FORM + 0.18, y, formula, size=11,  color='#A8C4CC',   mono=True)

# FOOTER
fy = TOTAL_Y - 3.90 - N * RH - 0.35
hline(fy, color='#4F8EF7', lw=1.5)
txt(XMID, fy - 0.38,
    'Log-transform applied at inference to indices 0, 1, 11, 12, 13   |   '
    'Z-scores set to 0.0 when NAICS peer group < 10 establishments   |   '
    'Exactly one NAICS bit is set per establishment',
    size=11, color=DIM, ha='center')

plt.savefig('plots/feature_vector_table.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved: plots/feature_vector_table.png')
