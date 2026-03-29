import numpy as np


def pseudo_label(row: np.ndarray) -> float:
    """Generate a training label (0-100 risk) from raw features using domain heuristics.

    Uses only absolute OSHA signals and NAICS sector identity.
    Industry-relative z-scores (indices 17-20) are intentionally excluded
    so the GB model learns those relationships from data, not labels.

    Design principles:
    - Recency × severity multiplier: identical past violations score lower than
      recent ones, addressing the "reformed past offender" problem.
    - Trajectory signal: rising violation rates get an additional premium.
    - Confirmed-recidivism multiplier: willful+repeat across ≥5 inspections is
      a strong predictor of ongoing non-compliance.
    - Reduced sparse-data floor: single-inspection companies get a small
      uncertainty premium but are no longer homogenised into a degenerate bin
      (the n_insp≤1 hard floor at 18 was creating a phantom D4 gap in deciles
      and teaching the model that 18 is the "neutral" mean, causing MSE-driven
      regression to that value for all high-scored establishments).
    """
    # Absolute signals (indices 0-16)
    (n_insp, n_viols, serious, willful, repeat,
     total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
     accident_count, fatality_count, injury_count, avg_gravity,
     pen_per_insp, clean_ratio,
     time_adj_pen) = row[:18]

    # NAICS sector identity (last 25 elements: 24 sectors + unknown flag)
    naics_unknown = float(row[-1])  # 1.0 when NAICS is missing

    score = 0.0

    # ── Dormant activity scaling ────────────────────────────────────────
    # When a company has had NO inspections in the recent window (recent_ratio
    # near zero), their historical violation record is partially discounted.
    # These "dormant" establishments often reformed after OSHA enforcement or
    # scaled down / ceased operations — their historical extremes over-predict
    # current risk.  Active companies (recent_ratio > 0) receive full credit.
    dormant_scale = 0.5 if recent_ratio < 0.01 else 1.0
    fat_dormant_scale = 0.5 if recent_ratio < 0.01 else 1.0

    # ── Inspection-count confidence ───────────────────────────────────
    # Rate-based features (fat_rate, willful_rate, repeat_rate) computed from
    # a handful of inspections are less reliable than those from a long record.
    # Confidence scales from 0.6 (n=1) to 1.0 (n≥5) — a softer floor than the
    # old 0.2 minimum, which was suppressing genuine 2-3 inspection signals.
    insp_confidence = min(0.5 + n_insp / 10.0, 1.0)

    # ── Direct harm signals (up to 34) ───────────────────────────────
    # fatality_count = fat_rate (fatalities per inspection, already normalised).
    # We scale purely from the rate — NOT from fat_rate * n_insp (absolute count)
    # — so that large companies with many total fatalities but low per-inspection
    # rates do not outscore smaller companies with genuinely higher per-inspection
    # fatality rates.  Any fatality record (fat_rate > 0) earns the 12-pt base;
    # the bonus scales with rate: fat_rate = 0.2 → eff_fat = 1 (base),
    # fat_rate = 0.6 → eff_fat = 3 (+6 pts),  fat_rate = 1.0 → eff_fat = 5 (+24).
    # The contribution is further weighted by insp_confidence: a fat_rate of 1.0
    # computed from a single inspection (insp_confidence=0.6) earns 14.4 pts,
    # preventing single-event catastrophes from dominating the score.
    #
    # Sustained-pattern boost: the feature vector only carries fat_rate, so a
    # company with 1 fatality in 1 inspection looks identical to one with 20
    # fatalities in 20 inspections at the same rate.  n_insp * fat_rate is a
    # proxy for absolute fatality count — log-scaled so it can't overwhelm the
    # rate signal, capped at +8 pts, and also gated by insp_confidence so
    # sparse records don't receive an outsized pattern premium.
    if fatality_count > 0:
        eff_fatalities = min(fatality_count * 5.0, 5.0)
        rate_contrib = min(12.0 + max(eff_fatalities - 1.0, 0.0) * 3.0, 24.0)
        abs_proxy = n_insp * fatality_count          # ≈ estimated total fatalities
        pattern_boost = min(np.log1p(abs_proxy) * 2.0, 8.0)
        score += (rate_contrib + pattern_boost) * fat_dormant_scale * insp_confidence
    score += min(severe * 25.0, 6)
    score += min(injury_count * 6.0, 4)
    score += min(accident_count * 10.0, 4)
    score = min(score, 34)

    # ── Violation type (up to 35) ─────────────────────────────────────
    # Coefficients raised so that moderate violation RATES generate meaningful
    # scores: willful_rate=0.1 → 6 pts, 0.33 → 20 pts (cap).  The old cap of
    # 14 pts total meant even a company with every inspection flagged as willful
    # only matched a company with a single historical fatality.
    score += min(willful * 60.0, 20) * dormant_scale
    score += min(repeat  * 50.0, 15) * dormant_scale

    # ── Gravity / severity (up to 21) ────────────────────────────────
    # Serious violations raised: rate=0.2 → 5 pts, 0.6 → 15 pts.
    score += min(serious * 25.0, 15)
    if avg_gravity > 0:
        score += min(avg_gravity * 0.6, 6)

    # ── Active non-compliance interaction (up to 20) ──────────────────
    # The single strongest forward-looking predictor missing from the old
    # formula: a company that is CURRENTLY being inspected AND continuing to
    # receive willful/repeat citations.  Scales with both recency_ratio (how
    # recently active) and the W/R rate.  Dormant companies earn 0; active
    # recidivists with willful_rate=0.2 and recent_ratio=0.5 earn ~10 pts.
    if recent_ratio > 0 and (willful + repeat) > 0:
        active_wr = recent_ratio * (willful * 3.0 + repeat * 2.0) * 25.0
        score += min(active_wr, 20.0)

    # ── Recency component (up to 12) ──────────────────────────────────
    # Added directly after gravity so it contributes even when absolute
    # violation counts are low.  Kept additive (not multiplicative) so that
    # the temporal train/test split does not introduce a distribution shift:
    # pre-2024 training establishments all have recent_ratio = 0 (all their
    # inspections are now >2 years old), so a multiplicative recency factor
    # would be 1.0 during training but >1.0 for 2025 test establishments —
    # creating a systematic label-scale mismatch the model cannot learn.
    score += recent_ratio * 12.0

    # ── Violation rate per inspection (up to 10) ──────────────────────
    score += min(vpi * 2.5, 10)

    # ── Trajectory signal: rising vs. falling violation trend ─────────
    # Proxy: establish whether recent inspection density is above or below
    # the long-run average using recent_ratio and vpi.
    # - recent_ratio > 0.5 means more than half of all inspections were recent.
    # - vpi is violations/inspection (the intensity).
    # When violations are BOTH recently concentrated AND intense (vpi ≥ 1.5,
    # roughly "more violations than inspections"), add a trajectory premium.
    # When violations are old (recent_ratio < 0.2) and modest (vpi < 0.5),
    # apply a small deduction, rewarding genuinely reformed establishments.
    if recent_ratio > 0.5 and vpi >= 1.5:
        trajectory_premium = min((recent_ratio - 0.5) * 2.0 * (vpi - 1.0), 8.0)
        score += trajectory_premium
    elif recent_ratio < 0.2 and vpi < 0.5 and n_insp >= 3:
        score -= min((0.2 - recent_ratio) * 10.0, 5.0)

    # ── Penalties — linear scaling so large amounts generate proportionally
    # higher risk labels (up to 16) ────────────────────────────────────────
    # time_adj_pen is the primary signal: recent penalties are nearly full-weight,
    # older ones are decayed exponentially (τ=3y).  Linear scaling with a cap
    # gives $10K → 0.25 pts, $200K → 5 pts, $400K → 10 pts — vs log1p which
    # compressed $10K and $1M to within 30% of each other.
    if time_adj_pen > 0:
        score += min(time_adj_pen / 40_000, 10.0)
    # Per-inspection average: flags single catastrophic inspections.
    # Raised cap to 6 pts ($120K+ per inspection = severe signal).
    if pen_per_insp > 0:
        score += min(pen_per_insp / 20_000, 6.0)
    # total_pen (flat historical accumulation) removed: it rewarded companies
    # with large OLD penalty records without requiring current activity, which
    # was driving band inversion.  time_adj_pen already captures the meaningful
    # penalty signal with exponential recency decay.

    # ── Missing-NAICS uncertainty penalty ────────────────────────────
    if naics_unknown > 0.5:
        score += 4.0

    # ── Clean inspection credit (up to -10) ──────────────────────────
    # Dormant companies (recent_ratio < 0.01) that have accumulated clean
    # inspections after an old fatality event should earn this credit — the
    # inspections indicate that the fatality was not systemic.  Active companies
    # still carrying current accident/fatality records remain blocked.
    is_dormant = recent_ratio < 0.01
    fat_blocks_clean = (
        (fatality_count > 0 or accident_count > 0 or severe > 0)
        and not is_dormant
    )
    if clean_ratio > 0 and not fat_blocks_clean:
        if n_insp >= 3:
            score -= clean_ratio * 10.0
        elif n_insp == 2:
            score -= clean_ratio * 4.0

    # ── Uncertainty / sparse data premium (reduced) ───────────────────
    # Reduced from 5.0/2.5/1.0 to 2.5/1.5/0.5 to stop over-inflating sparse
    # records into a degenerate spike.  A small premium is still applied to
    # signal evidence uncertainty without homogenising all single-inspection
    # companies into one score cluster.
    if n_insp <= 1:
        score += 2.5
    elif n_insp <= 3:
        score += 1.5
    elif n_insp <= 5:
        score += 0.5

    # ── Interaction effects ───────────────────────────────────────────
    # Scale by inspection-count confidence: compound risk patterns require
    # multiple data points to be reliable.  A single inspection cannot confirm
    # that fatality+willful or willful+repeat is a systematic pattern rather
    # than a one-time event.  Full weight is applied at 5+ inspections.
    # insp_confidence computed earlier (before fatality scoring).
    if fatality_count > 0 and (willful + repeat) > 0:
        score += 8.0 * insp_confidence * dormant_scale
    if willful > 0 and repeat > 0:
        score += 5.0 * insp_confidence * dormant_scale
    if recent_ratio > 0.5 and serious >= 0.25:
        score += 4.0

    # NOTE: A multiplicative recidivism factor (×1.15 for willful+repeat+5insp)
    # was considered but removed.  It created hard threshold crossings just above
    # the ≥60 "Do Not Recommend" boundary, making many borderline establishments
    # jump into the highest tier without a proportionate change in features.
    # The interaction terms above (willful+repeat interaction +5, fatality+WR +8)
    # already provide meaningful incremental scoring for confirmed recidivists.
    # The production model's Huber loss + tail sample weights further amplify
    # separation at the high end without creating discontinuities in the label.

    # NOTE: The previous hard floor (n_insp <= 1 → score = 18.0) has been
    # removed.  It homogenised thousands of single-inspection establishments
    # into a degenerate spike at score=18, created a phantom D4 gap in decile
    # analysis, and taught the GBR that 18 is the "neutral" prediction target,
    # causing MSE-driven mean-reversion for all high-scored establishments.
    # NOTE: The earlier hard floor (score = max(score, 65) when fatality_count > 0
    # and recent_ratio >= 0.25) was removed in a prior revision for similar reasons.

    return float(np.clip(score, 0, 100))
