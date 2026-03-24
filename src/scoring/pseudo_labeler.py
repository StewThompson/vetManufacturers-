import numpy as np


def pseudo_label(row: np.ndarray) -> float:
    """Generate a training label (0-100 risk) from raw features using domain heuristics.

    Uses only absolute OSHA signals and NAICS sector identity.
    Industry-relative z-scores (indices 17-20) are intentionally excluded
    so the GB model learns those relationships from data, not labels.
    """
    # Absolute signals (indices 0-16)
    (n_insp, n_viols, serious, willful, repeat,
     total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
     accident_count, fatality_count, injury_count, avg_gravity,
     pen_per_insp, clean_ratio) = row[:17]

    # NAICS sector identity (last 25 elements: 24 sectors + unknown flag)
    naics_unknown = float(row[-1])  # 1.0 when NAICS is missing

    score = 0.0

    # Direct harm signals (up to 34)
    if fatality_count > 0:
        eff_fatalities = min(fatality_count * max(n_insp, 1), 5)
        score += min(22.0 + (eff_fatalities - 1) * 6.0, 34)
    score += min(severe * 25.0, 6)
    score += min(injury_count * 6.0, 4)
    score += min(accident_count * 10.0, 4)
    score = min(score, 34)

    # Violation type (up to 24)
    score += min(willful * 14.0, 14)
    score += min(repeat * 10.0, 10)

    # Gravity / severity (up to 14)
    score += min(serious * 8.0, 8)
    if avg_gravity > 0:
        score += min(avg_gravity * 0.6, 6)

    # Recency (up to 12)
    score += recent_ratio * 12.0

    # Violation rate per inspection (up to 10)
    score += min(vpi * 2.5, 10)

    # Penalties — corroboration only (up to 6)
    if total_pen > 0:
        score += min(np.log1p(total_pen) * 0.4, 4)
    if pen_per_insp > 0:
        score += min(np.log1p(pen_per_insp) * 0.3, 2)

    # Missing-NAICS uncertainty penalty
    if naics_unknown > 0.5:
        score += 4.0

    # Clean inspection credit (up to -10)
    if (clean_ratio > 0
            and fatality_count == 0
            and accident_count == 0
            and severe == 0):
        if n_insp >= 3:
            score -= clean_ratio * 10.0
        elif n_insp == 2:
            score -= clean_ratio * 4.0

    # Uncertainty / sparse data (up to 5)
    if n_insp <= 1:
        score += 5.0
    elif n_insp <= 3:
        score += 2.5
    elif n_insp <= 5:
        score += 1.0

    # Interaction effects
    if fatality_count > 0 and (willful + repeat) > 0:
        score += 8.0
    if willful > 0 and repeat > 0:
        score += 5.0
    if recent_ratio > 0.5 and serious >= 0.25:
        score += 4.0

    # Conservative floors
    if fatality_count > 0 and recent_ratio >= 0.25:
        score = max(score, 65.0)
    if n_insp <= 1 and score < 18:
        score = 18.0

    return float(np.clip(score, 0, 100))
