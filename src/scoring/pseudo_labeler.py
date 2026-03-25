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
    # Fatality base reduced from 22 → 12 (per-extra from 6 → 3, cap unchanged at 34).
    # Rationale: empirical real-world validation showed that the original 22-pt base
    # pushed historical-fatality companies past 60 even when their recent violation
    # pattern was modest.  Those companies then had better-than-average future S/W/R
    # outcomes (the compliance crackdown following a fatality often improves behaviour),
    # creating a non-monotone tier ordering.  Reducing the single-fatality contribution
    # to 12 pts (roughly one willful violation) preserves the signal while preventing
    # harm events alone from dominating the score at the expense of violation patterns.
    if fatality_count > 0:
        eff_fatalities = min(fatality_count * max(n_insp, 1), 5)
        score += min(12.0 + (eff_fatalities - 1) * 3.0, 24)
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
    # Scale by inspection-count confidence: compound risk patterns require
    # multiple data points to be reliable.  A single inspection cannot confirm
    # that fatality+willful or willful+repeat is a systematic pattern rather
    # than a one-time event.  Full weight is applied at 5+ inspections.
    insp_confidence = min(n_insp / 5.0, 1.0)
    if fatality_count > 0 and (willful + repeat) > 0:
        score += 8.0 * insp_confidence
    if willful > 0 and repeat > 0:
        score += 5.0 * insp_confidence
    if recent_ratio > 0.5 and serious >= 0.25:
        score += 4.0

    # Sparse-data floor — minimum uncertainty premium for single-inspection records.
    if n_insp <= 1 and score < 18:
        score = 18.0
    # NOTE: The previous hard floor (score = max(score, 65) when fatality_count > 0
    # and recent_ratio >= 0.25) has been removed.
    # That floor forced every establishment with any historical fatality and recent
    # OSHA activity to score ≥ 65, regardless of how clean its recent inspections
    # were.  Real-world validation showed this caused the High risk tier to have
    # *lower* future S/W/R rates than the Low tier: the floor was inflating scores
    # for "reformed" companies that had one historical incident but have since
    # improved.  Fatalities are now scored entirely through the organic
    # harm-signal weighting above, producing accurate tier ordering.

    return float(np.clip(score, 0, 100))
