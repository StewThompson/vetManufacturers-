import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG    = '#0D1117'
WHITE = '#F0F6FC'
DIM   = '#8B949E'
GOLD  = '#FCD34D'
BLUE  = '#4F8EF7'
RED   = '#E05C5C'
GREEN = '#34D399'

wr_rows = [
    ('WR Event AUROC',           '0.751',   0.751, BLUE),
    ('WR Brier Skill Score',     '0.1885',  0.628, BLUE),
    ('WR PR-AUC / AP',           '0.721',   0.721, BLUE),
    ('WR Top-Decile Lift',       '2.145x',  0.715, BLUE),
]
inj_rows = [
    ('Injury AUROC',             '0.797',   0.797, RED),
    ('Injury Brier Skill Score', '0.0858',  0.429, RED),
    ('Injury Top-10% Capture',   '12.7 %',  0.635, RED),
]
pen_rows = [
    ('Penalty Tier AUROC (P95)', '0.812',   0.812, GREEN),
    ('Penalty PR-AUC / AP',      '0.634',   0.634, GREEN),
    ('Penalty Top-Decile Lift',  '3.2x',    0.800, GREEN),
]

RH    = 0.56
GAP   = 0.30
SEC_H = 0.40
HDR_H = 0.48
TITLE_H  = 1.10
FOOTER_H = 0.55

n_total  = len(wr_rows) + len(inj_rows) + len(pen_rows)
inside_h = HDR_H + 3 * SEC_H + n_total * RH + 2 * GAP
FH = TITLE_H + inside_h + FOOTER_H + 0.4
FW = 16

fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor(BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

XL, XR  = 0.5, 15.5
BX      = XL + 0.70
RX      = 9.5
BAR_X   = 11.2
BAR_W   = 3.8


def txt(x, y, s, size=11, color=WHITE, ha='left', va='center', bold=False, mono=False):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontfamily='monospace' if mono else 'sans-serif', zorder=5, clip_on=False)


def hline(y, color, lw=1.4):
    ax.plot([XL, XR], [y, y], color=color, lw=lw, zorder=4)


# ── TITLE ────────────────────────────────────────────────────────────────────
txt(FW / 2, FH - 0.45, 'Model Validation Results',
    size=22, bold=True, color=WHITE, ha='center')
txt(FW / 2, FH - 0.88,
    'Multi-target composite risk model  |  Hold-out test set  |  Temporal cutoff: 2022-01-01',
    size=10, color=DIM, ha='center')

cy = FH - TITLE_H

# ── COLUMN HEADER ─────────────────────────────────────────────────────────────
hdr_bg = mpatches.FancyBboxPatch((XL, cy - HDR_H), XR - XL, HDR_H,
    boxstyle='square,pad=0', facecolor='#1C2333', lw=0, zorder=2)
ax.add_patch(hdr_bg)
hline(cy,        GOLD, lw=2.0)
hline(cy - HDR_H, GOLD, lw=2.0)
txt(BX,              cy - HDR_H / 2, 'Metric', size=13, bold=True, color=GOLD)
txt(RX,              cy - HDR_H / 2, 'Result', size=13, bold=True, color=GOLD, ha='center')
txt(BAR_X + BAR_W / 2, cy - HDR_H / 2, 'Visual', size=13, bold=True, color=GOLD, ha='center')
cy -= HDR_H


def draw_section(cy, rows, label, head_color):
    _bgs = {
        BLUE:  ('#131D33', '#0F1626'),
        RED:   ('#1A0D0D', '#120808'),
        GREEN: ('#0D1A12', '#09130D'),
    }
    bg_dark, bg_light = _bgs.get(head_color, ('#1A1A1A', '#111111'))

    slb = mpatches.FancyBboxPatch((XL, cy - SEC_H), XR - XL, SEC_H,
        boxstyle='square,pad=0', facecolor='#0D1117', lw=0, zorder=2)
    ax.add_patch(slb)
    ax.plot([XL, XL + 0.18], [cy - SEC_H / 2, cy - SEC_H / 2],
            color=head_color, lw=4, solid_capstyle='round', zorder=4)
    txt(XL + 0.38, cy - SEC_H / 2, label, size=11, bold=True, color=head_color)
    cy -= SEC_H

    for i, (metric, result_str, bar_frac, col) in enumerate(rows):
        y_mid = cy - RH / 2
        fc = bg_dark if i % 2 == 0 else bg_light
        bg = mpatches.FancyBboxPatch((XL, cy - RH), XR - XL, RH,
            boxstyle='square,pad=0', lw=0, facecolor=fc, zorder=2)
        ax.add_patch(bg)

        txt(BX, y_mid, metric, size=12, color=WHITE)

        rb = mpatches.FancyBboxPatch((RX - 1.0, y_mid - 0.18), 2.0, 0.36,
            boxstyle='round,pad=0.04', lw=1.4, facecolor=BG, edgecolor=col, zorder=4)
        ax.add_patch(rb)
        txt(RX, y_mid, result_str, size=12, bold=True, color=col, ha='center', mono=True)

        track = mpatches.FancyBboxPatch((BAR_X, y_mid - 0.10), BAR_W, 0.20,
            boxstyle='round,pad=0.02', lw=0, facecolor='#1E2A35', zorder=4)
        ax.add_patch(track)
        fill_w = max(0.06, bar_frac * BAR_W)
        fill = mpatches.FancyBboxPatch((BAR_X, y_mid - 0.10), fill_w, 0.20,
            boxstyle='round,pad=0.02', lw=0, facecolor=col, alpha=0.88, zorder=5)
        ax.add_patch(fill)
        txt(BAR_X + BAR_W + 0.15, y_mid, f'{bar_frac:.0%}',
            size=9, color=DIM, ha='left', mono=True)
        cy -= RH
    return cy


cy = draw_section(cy, wr_rows,  'WR / Serious Event Head  (composite weight = 60%)',  BLUE)
cy -= GAP
cy = draw_section(cy, inj_rows, 'Injury / Fatality Head  (composite weight = 30%)',   RED)
cy -= GAP
cy = draw_section(cy, pen_rows, 'Large Penalty Head  (composite weight = 10%)',        GREEN)

hline(cy - 0.12, BLUE, lw=1.2)
txt(FW / 2, cy - 0.38,
    'Bar scale: AUROC and PR-AUC plotted as raw value  |  '
    'BSS and capture rate normalised to a 0-1 visual scale  |  '
    'Penalty Tier target: future penalty >= industry P95',
    size=9, color=DIM, ha='center')

plt.savefig('plots/metrics_table.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved: plots/metrics_table.png')
