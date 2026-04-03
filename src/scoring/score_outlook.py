"""score_outlook.py — Back-project a risk score into a 12-month compliance forecast.

Design
------
The ML model's per-inspection-rate features (violations_per_inspection,
penalties_per_inspection, etc.) are already available in the log-transformed
feature dict returned by ``score()``.  Because only the raw count features
(log_inspections, log_violations, accident/fatality/injury rates) are
log-compressed, the rate features are still in their natural units and can
be multiplied directly by the projected annual inspection frequency to yield
forward-looking estimates.

Annual inspection rate
~~~~~~~~~~~~~~~~~~~~~~
    n_insp       = expm1(log_inspections)   ← undo log1p transform
    recent_insp  = n_insp × recent_ratio    ← inspections in past 3 years
    annual_rate  = recent_insp / 3          ← annualised

A score-based floor prevents the projection from collapsing to zero when a
company has had no recent activity (may have reformed or reduced operations).
High-risk companies face a higher floor because OSHA's Programmed Inspection
targeting increases likelihood of a visit even without a complaint trigger.

Caveats shown in the narrative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The projection is a statistical estimate, not a guarantee.
* Irregular inspection patterns (single-complaint-driven history) may over-
  or under-estimate frequency.
* Companies with very few historical inspections receive wider uncertainty.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


# Score-based annual inspection frequency floors (inspections per year).
# These reflect OSHA's targeting priorities: high-risk establishments appear
# more often in Programmed Inspection lists; low-risk companies may not be
# visited for several years.
_SCORE_FLOOR_HIGH     = 1.5   # score ≥ 60
_SCORE_FLOOR_MODERATE = 0.75  # score 40-59
_SCORE_FLOOR_LOW      = 0.30  # score 20-39
_SCORE_FLOOR_MINIMAL  = 0.10  # score < 20


def compute_12m_outlook(
    risk_score: float,
    features: Dict[str, float],
    n_sites: int = 1,
    risk_targets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Derive a 12-month compliance forecast from risk score and feature values.

    Parameters
    ----------
    risk_score : float
        Calibrated 0-100 risk score for the company.
    features : dict
        Log-transformed feature dict returned by ``MLRiskScorer.score()``
        (keys are ``FEATURE_NAMES``, values are post-log-transform floats).
    n_sites : int
        Number of distinct establishments (used in basis narrative only;
        features are already aggregated across all sites).
    risk_targets : dict or None
        When provided (output of ``MultiTargetRiskScorer.predict()``), the
        model-predicted ``expected_penalty_usd`` and ``expected_citations``
        values replace the heuristic rate-based projections for violations
        and penalties.  The inspection frequency (``expected_inspections_12m``)
        continues to use the feature-rate heuristic because it is not
        predicted by the multi-target heads.

    Returns
    -------
    dict with keys matching ``ComplianceOutlook12M``:
        expected_inspections_12m     – projected OSHA inspection visits
        expected_violations_12m      – projected total violations
        expected_penalties_usd_12m   – projected $ in OSHA penalties
        expected_serious_12m         – projected Serious-type violations
        expected_willful_repeat_12m  – projected Willful + Repeat violations
        risk_band                    – "low" | "moderate" | "high"
        has_history                  – whether any prior OSHA record exists
        basis                        – short narrative explaining the estimate source
        summary_narrative            – human-readable 2-3 sentence forecast
    """
    # ── Recover raw inspection count from log-transformed feature ─────────
    log_insp = features.get("log_inspections", 0.0)
    n_insp = math.expm1(max(log_insp, 0.0))  # total historical inspections (est.)

    # ── Per-inspection rates (not log-transformed) ────────────────────────
    recent_ratio     = features.get("recent_ratio", 0.0)      # fraction in last 3 yrs
    vpi              = features.get("violations_per_inspection", 0.0)  # viols/insp
    pen_per_insp     = features.get("penalties_per_inspection", 0.0)   # $/insp
    serious_per_insp = features.get("serious_violations", 0.0)         # serious/insp
    willful_per_insp = features.get("willful_violations", 0.0)         # willful/insp
    repeat_per_insp  = features.get("repeat_violations", 0.0)          # repeat/insp

    has_history = n_insp >= 0.5  # at least one real inspection in history

    # ── Annual inspection rate estimate ──────────────────────────────────
    recent_inspections = n_insp * recent_ratio  # inspections in past 3 years
    if recent_inspections > 0.01:
        annual_insp_rate = recent_inspections / 3.0
        basis = (
            f"Based on {recent_inspections:.1f} inspections "
            f"in the past 3 years ({n_sites} site{'s' if n_sites != 1 else ''})"
        )
    elif n_insp > 0.5:
        # History exists but all inspections are older than 3 years.
        # Discount heavily — company may have reformed or reduced operations.
        annual_insp_rate = (n_insp / 10.0) * 0.2
        basis = (
            f"Low recent activity (all {n_insp:.0f} historical inspections are "
            f"older than 3 years); forward rate heavily discounted"
        )
    else:
        # No OSHA history at all.
        annual_insp_rate = 0.0
        basis = "No prior OSHA inspection history; estimate is score-based only"

    # ── Apply score-based floor ───────────────────────────────────────────
    if risk_score >= 60:
        floor = _SCORE_FLOOR_HIGH
    elif risk_score >= 40:
        floor = _SCORE_FLOOR_MODERATE
    elif risk_score >= 20:
        floor = _SCORE_FLOOR_LOW
    else:
        floor = _SCORE_FLOOR_MINIMAL

    annual_insp_rate = max(annual_insp_rate, floor)

    # ── 12-month projections ──────────────────────────────────────────────
    expected_inspections    = annual_insp_rate

    # When multi-target model predictions are available, prefer them for
    # violations and penalties — they are trained on real future outcomes
    # rather than extrapolated from historical rates.
    if risk_targets is not None:
        expected_penalties      = float(risk_targets.get("expected_penalty_usd", 0.0))
        expected_violations     = float(risk_targets.get("expected_citations", 0.0))
        # Serious/WR counts remain rate-based (not direct targets)
        expected_serious        = annual_insp_rate * serious_per_insp
        expected_willful_repeat = annual_insp_rate * (willful_per_insp + repeat_per_insp)
        basis += " (violations + penalties from probabilistic model)"
    else:
        expected_violations     = annual_insp_rate * vpi
        expected_penalties      = annual_insp_rate * pen_per_insp
        expected_serious        = annual_insp_rate * serious_per_insp
        expected_willful_repeat = annual_insp_rate * (willful_per_insp + repeat_per_insp)

    # ── Risk band ─────────────────────────────────────────────────────────
    if risk_score >= 60:
        risk_band = "high"
    elif risk_score >= 30:
        risk_band = "moderate"
    else:
        risk_band = "low"

    # ── Human-readable narrative ──────────────────────────────────────────
    summary_narrative = _build_narrative(
        risk_score=risk_score,
        risk_band=risk_band,
        has_history=has_history,
        expected_inspections=expected_inspections,
        expected_violations=expected_violations,
        expected_penalties=expected_penalties,
        expected_serious=expected_serious,
        expected_willful_repeat=expected_willful_repeat,
        n_sites=n_sites,
    )

    return {
        "expected_inspections_12m": round(expected_inspections, 1),
        "expected_violations_12m": round(expected_violations, 1),
        "expected_penalties_usd_12m": int(round(expected_penalties)),
        "expected_serious_12m": round(expected_serious, 1),
        "expected_willful_repeat_12m": round(expected_willful_repeat, 1),
        "risk_band": risk_band,
        "has_history": has_history,
        "basis": basis,
        "summary_narrative": summary_narrative,
    }


def _fmt_viols(v: float) -> str:
    if v < 0.5:
        return "fewer than 1 violation"
    return f"approximately {v:.1f} violation{'s' if v != 1.0 else ''}"


def _fmt_pen(p: float) -> str:
    if p < 500:
        return "minimal penalties"
    if p < 1_000:
        return f"~${p:,.0f} in penalties"
    return f"~${p:,.0f} in OSHA penalties"


def _build_opening(
    risk_band: str,
    risk_score: float,
    site_str: str,
    insp_desc: str,
    viols_sent: str,
    pen_sent: str,
) -> str:
    if risk_band == "low":
        return (
            f"This supplier presents a low compliance risk (score {risk_score:.0f}). "
            f"Based on recent inspection trends across {site_str}, we project "
            f"{insp_desc} in the next 12 months, resulting in {viols_sent} "
            f"and {pen_sent}."
        )
    if risk_band == "moderate":
        return (
            f"This supplier carries a moderate compliance risk (score {risk_score:.0f}). "
            f"Projecting from recent inspection patterns across {site_str}, expect "
            f"{insp_desc}, {viols_sent}, and {pen_sent} over the next 12 months."
        )
    return (
        f"This supplier is flagged as high-risk (score {risk_score:.0f}). "
        f"Based on observed enforcement patterns across {site_str}, the next "
        f"12 months may see {insp_desc}, {viols_sent}, "
        f"and {pen_sent}."
    )


def _build_severity_sentence(
    expected_serious: float,
    expected_willful_repeat: float,
) -> str:
    parts = []
    if expected_serious >= 0.3:
        parts.append(f"~{expected_serious:.1f} Serious")
    if expected_willful_repeat >= 0.2:
        parts.append(f"~{expected_willful_repeat:.1f} Willful/Repeat")
    if not parts:
        return ""
    plural = sum(1 for _ in parts) > 1 or expected_serious + expected_willful_repeat != 1.0
    return (
        f" Of the projected violations, {' and '.join(parts)} "
        f"citation{'s are' if plural else ' is'} estimated."
    )


def _build_narrative(
    risk_score: float,
    risk_band: str,
    has_history: bool,
    expected_inspections: float,
    expected_violations: float,
    expected_penalties: float,
    expected_serious: float,
    expected_willful_repeat: float,
    n_sites: int,
) -> str:
    """Compose a 2-3 sentence plain-English forecast."""
    site_str = f"{n_sites} site{'s' if n_sites != 1 else ''}"

    if not has_history:
        return (
            f"This supplier has no prior OSHA inspection record. "
            f"The risk score of {risk_score:.0f} is derived from industry benchmarks "
            f"and population patterns rather than direct enforcement history. "
            f"Absence of a record does not guarantee compliance — an initial inspection "
            f"could result in violations if underlying safety practices are not established."
        )

    insp_desc = (
        f"roughly {expected_inspections:.1f} inspection visit{'s' if expected_inspections != 1.0 else ''}"
        if expected_inspections >= 0.5
        else "a low likelihood of an inspection visit"
    )

    opening = _build_opening(
        risk_band, risk_score, site_str, insp_desc,
        _fmt_viols(expected_violations), _fmt_pen(expected_penalties),
    )
    severity_sent = _build_severity_sentence(expected_serious, expected_willful_repeat)
    caveat = (
        " These are statistical projections based on historical rates; "
        "actual outcomes depend on OSHA targeting priorities, operational changes, "
        "and corrective actions taken."
    )
    return opening + severity_sent + caveat
