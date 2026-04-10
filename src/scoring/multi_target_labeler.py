"""multi_target_labeler.py — Build multi-target training labels for the
probabilistic risk model.

Architecture: Sequential Conditional Multi-Stage Pipeline
---------------------------------------------------------
Zero future outcome does not always mean low compliance risk; it may mean
no inspection exposure. This model separates inspection exposure from
violation severity so risk estimates are more interpretable and less
distorted by structural zeros.

The labeler produces per-establishment labels for a 3-stage pipeline:

  **Stage 1 — Inspection Exposure** (ALL establishments):
    future_has_inspection   — 0/1, establishment inspected in the future window
    future_inspection_count — int, count of future inspections

  **Stage 2 — Violation | Inspection** (inspection rows only):
    future_has_violation    — 0/1, any violation in the future window
    future_violation_count  — int, total violation count
    future_has_serious      — 0/1, any S/W/R violation
    any_injury_fatal        — 0/1, any hospitalization or fatality

  **Stage 3 — Magnitude | Violation** (violation rows only):
    future_total_penalty    — float ($), total penalties
    log_penalty             — log1p(penalty)
    gravity_weighted_score  — float, Σ(gravity × viol_weight)
    future_citation_count   — int, total citation count

  **Backward compat**:
    any_wr_serious          — alias for future_has_serious
    real_label              — 0-100 composite adverse outcome score

Design notes
------------
* All feature computation is delegated to helpers imported from
  ``temporal_labeler``, keeping code deduplication high and leakage logic
  centrally maintained.
* Penalty tier thresholds come from ``penalty_percentiles.py`` and MUST be
  pre-computed on training-fold data before calling this module.
* The stratified sampling strategy is NAICS sector × staged-outcome quartile
  — ensuring broad coverage across industries and risk levels.
* Returns dict rows that include both ``features_46`` (pre-log-transform) and
  all target scalars, plus the ``real_label`` composite score for backward
  compatibility with diagnostics.
* **Stage filtering**: callers (the scorer's fit method) must filter rows to
  the proper training subset for each stage:
    - Stage 1: all rows
    - Stage 2: rows where ``future_has_inspection == 1``
    - Stage 3: rows where ``future_has_violation == 1``
"""
from __future__ import annotations

import csv
import logging
import math
import os
import pickle
from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.scoring.ml_risk_scorer import MLRiskScorer

from src.scoring.labeling.helpers import (
    ADV_MAX,
    _aggregate_hist_features,
    _build_violation_index,
    _compute_adverse,
    _normalize_adverse,
    _parse_date,
    _stratified_sample,
    _stream_inspections,
)
from src.scoring.labeling.inspection_propensity import InspectionPropensityModel
from src.scoring.penalty_percentiles import lookup_threshold

logger = logging.getLogger(__name__)


CACHE_FILENAME = "multi_target_labels.pkl"


# ── Per-establishment multi-target computation ──────────────────────────────


def _build_hosp_fatal_stats(
    accidents_path: str,
    injuries_path: str,
    activity_nrs: set,
) -> Dict[str, bool]:
    """Return {activity_nr: True} if any hospitalization or fatality was
    recorded in the injuries table for the given OSHA inspection activity numbers.

    OSHA degree_of_inj codes used:
        '1.0' or '1' → Fatality
        '2.0' or '2' → Hospitalized injury
    Both are treated as positive for the p_injury target.
    """
    # Build summary_nr sets per inspection
    summaries_by_act: Dict[str, set] = defaultdict(set)
    serious_inj_by_key: Dict[str, bool] = {}

    csv.field_size_limit(10 * 1024 * 1024)
    try:
        with open(
            injuries_path, "r", newline="", encoding="utf-8", errors="replace"
        ) as f:
            for row in csv.DictReader(f):
                act = str(row.get("rel_insp_nr", ""))
                if act not in activity_nrs:
                    continue
                snr = str(row.get("summary_nr", ""))
                if not snr:
                    continue
                summaries_by_act[act].add(snr)
                doi_raw = str(row.get("degree_of_inj", "")).strip()
                try:
                    doi = float(doi_raw)
                except ValueError:
                    doi = -1.0
                if doi in (1.0, 2.0):  # fatality or hospitalized
                    serious_inj_by_key[f"{act}|{snr}"] = True
    except FileNotFoundError:
        pass

    result: Dict[str, bool] = {}
    for act in activity_nrs:
        hit = False
        for snr in summaries_by_act.get(act, set()):
            if serious_inj_by_key.get(f"{act}|{snr}", False):
                hit = True
                break
        result[act] = hit
    return result


def _compute_multi_targets(
    future_viols: List[dict],
    any_injury_fatal: bool,
    n_future_insp: int,
    penalty_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    naics_2digit: Optional[str] = None,
) -> Dict[str, float]:
    """Compute all target variables from post-cutoff violations and injuries.

    Returns labels for ALL three pipeline stages:
      Stage 1: future_has_inspection, future_inspection_count
      Stage 2: future_has_violation, future_violation_count, future_has_serious
      Stage 3: future_total_penalty, log_penalty, gravity_weighted_score,
               future_citation_count

    Parameters
    ----------
    future_viols : list of dicts with keys ``viol_type``, ``current_penalty``,
        ``initial_penalty``, ``gravity`` from the violations CSV.
    any_injury_fatal : bool
        True if any future inspection linked to this establishment had a
        hospitalized or fatal injury (from ``_build_hosp_fatal_stats``).
    n_future_insp : int
        Number of post-cutoff inspections.
    penalty_thresholds : dict, optional
        Output of ``penalty_percentiles.compute_penalty_percentiles()``.
        When provided, penalty tier binary targets are computed.
    naics_2digit : str or None
        2-digit NAICS sector for per-industry threshold lookup.

    Returns
    -------
    dict with keys:
        # Stage 1 — Inspection exposure
        future_has_inspection, future_inspection_count,
        # Stage 2 — Violation given inspection
        future_has_violation, future_violation_count,
        future_has_serious, any_wr_serious (alias), any_injury_fatal,
        # Stage 3 — Magnitude given violation
        future_total_penalty, log_penalty, gravity_weighted_score,
        future_citation_count,
        # Penalty tiers
        is_moderate_penalty, is_large_penalty, is_extreme_penalty,
        # Backward compat
        real_label_raw, real_label
    """
    n_viols = len(future_viols)
    total_penalty = sum(
        float(v.get("current_penalty") or v.get("initial_penalty") or 0)
        for v in future_viols
    )
    wr_serious = sum(
        1 for v in future_viols
        if v.get("viol_type") in ("W", "R", "S")
    )

    # Gravity-weighted severity: Σ(gravity × viol_weight)
    # viol_weight: W/R→3, S→2, all others→1
    grav_total = 0.0
    for v in future_viols:
        raw_g = v.get("gravity", "")
        try:
            g = float(str(raw_g).strip()) if raw_g else 0.0
        except ValueError:
            g = 0.0
        vt = v.get("viol_type", "")
        weight = 3.0 if vt in ("W", "R") else (2.0 if vt == "S" else 1.0)
        grav_total += g * weight

    # Composite adverse (for calibration / backward compat)
    # Pass 0 fatalities for the composite calc — injury data now drives p_injury
    raw_adv   = _compute_adverse(future_viols, 0, n_future_insp)
    real_label = _normalize_adverse(raw_adv)

    # Penalty tier binary targets (NAICS-normalized thresholds)
    is_moderate = 0
    is_large = 0
    is_extreme = 0
    if penalty_thresholds is not None:
        p75 = lookup_threshold(penalty_thresholds, naics_2digit, "p75")
        p90 = lookup_threshold(penalty_thresholds, naics_2digit, "p90")
        p95 = lookup_threshold(penalty_thresholds, naics_2digit, "p95")
        if total_penalty >= p75:
            is_moderate = 1
        if total_penalty >= p90:
            is_large = 1
        if total_penalty >= p95:
            is_extreme = 1

    return {
        # ── Stage 1: Inspection exposure ──────────────────────────────────
        "future_has_inspection":  int(n_future_insp >= 1),
        "future_inspection_count": n_future_insp,
        # ── Stage 2: Violation conditional on inspection ──────────────────
        "future_has_violation":   int(n_viols > 0),
        "future_violation_count": n_viols,
        "future_has_serious":     int(wr_serious > 0),
        "any_wr_serious":         int(wr_serious > 0),   # backward compat alias
        "any_injury_fatal":       int(any_injury_fatal),
        # ── Stage 3: Magnitude conditional on violation ───────────────────
        "future_total_penalty":   total_penalty,
        "log_penalty":            math.log1p(total_penalty),
        "gravity_weighted_score": grav_total,
        "future_citation_count":  n_viols,  # citation count = violation count
        # ── Penalty tiers ─────────────────────────────────────────────────
        "is_moderate_penalty":    is_moderate,
        "is_large_penalty":       is_large,
        "is_extreme_penalty":     is_extreme,
        # ── Backward compat ───────────────────────────────────────────────
        "real_label_raw":  raw_adv,
        "real_label":      real_label,
    }


# ── Cache fingerprint ────────────────────────────────────────────────────────

def _build_fingerprint(
    cutoff_date: date,
    outcome_end_date: date,
    sample_size: int,
    inspections_path: str,
    violations_path: str,
) -> dict:
    def _mtime(p: str) -> float:
        try:
            return round(os.path.getmtime(p), 1)
        except OSError:
            return 0.0

    return {
        "cutoff":      cutoff_date.isoformat(),
        "outcome_end": outcome_end_date.isoformat(),
        "sample_size": sample_size,
        "insp_mtime":  _mtime(inspections_path),
        "viol_mtime":  _mtime(violations_path),
        "schema":      "multi_target_v11",  # bumped: +staged pipeline labels (future_has_inspection, future_has_violation, etc.)
    }


# ── Public API ───────────────────────────────────────────────────────────────

def build_multi_target_sample(
    scorer: "MLRiskScorer",
    cutoff_date: date,
    outcome_end_date: date,
    inspections_path: str,
    violations_path: str,
    accidents_path: str,
    injuries_path: str,
    naics_map: dict,
    penalty_thresholds: Dict[str, Dict[str, float]],
    sample_size: int = 50_000,
    rng_seed: int = 42,
    min_hist_insp: int = 2,
    include_uninspected: bool = False,
) -> List[Dict]:
    """Build a stratified sample of (features, multi-targets) training rows.

    This function is the multi-target analogue of
    ``temporal_labeler.build_temporal_training_labels()``.  It uses the same
    streaming helpers to avoid memory pressure and the same stratification
    strategy (NAICS × pseudo-label quartile) to ensure broad coverage.

    Parameters
    ----------
    scorer : MLRiskScorer
        Required for ``_encode_naics()``, ``_industry_stats``,
        `_naics_map``.
    cutoff_date : date
        History / future split.  All inspections before this date feed into
        features; all inspections after (up to ``outcome_end_date``) define
        the outcome label.
    outcome_end_date : date
        Latest date to include in the outcome window.
    inspections_path, violations_path, accidents_path, injuries_path : str
        Absolute paths to the corresponding bulk CSVs in ml_cache/.
    naics_map : dict
        NAICS reference table from ``load_naics_map()``.
    penalty_thresholds : dict
        Output of ``penalty_percentiles.compute_penalty_percentiles()``.
        MUST be pre-computed on training-only data.
    sample_size : int
        Maximum number of rows to return (stratified down-sample).
    rng_seed : int
        Random seed for reproducible stratified sampling.
    min_hist_insp : int
        Minimum pre-cutoff inspections to include an establishment.

    Returns
    -------
    List of row dicts:
        name               : str
        features_46        : list[float]  — pre-log-transform, len 47
        # Stage 1 — Inspection exposure
        future_has_inspection   : int (0/1)
        future_inspection_count : int
        # Stage 2 — Violation given inspection
        future_has_violation    : int (0/1)
        future_violation_count  : int
        future_has_serious      : int (0/1)
        any_wr_serious          : int (0/1, backward compat alias)
        any_injury_fatal        : int (0/1)
        # Stage 3 — Magnitude given violation
        future_total_penalty : float
        log_penalty        : float
        gravity_weighted_score : float
        future_citation_count : int
        # Penalty tiers
        is_moderate_penalty : int (0/1)
        is_large_penalty   : int (0/1)
        is_extreme_penalty : int (0/1)
        # Meta
        real_label_raw     : float
        real_label         : float
        naics_2digit       : str
        cutoff_date        : str (ISO)
    """
    from src.scoring.industry_stats import (
        compute_industry_stats,
        compute_relative_features,
    )

    rng = np.random.default_rng(rng_seed)
    one_year_ago = cutoff_date - timedelta(days=1095)

    # ── PASS 1: split inspections into hist / future per establishment ─────
    logger.info("  [MultiTargetLabeler] Scanning inspections (cutoff=%s) …", cutoff_date)
    estab_hist:   Dict[str, list] = defaultdict(list)
    estab_future: Dict[str, list] = defaultdict(list)

    csv.field_size_limit(10 * 1024 * 1024)
    with open(inspections_path, "r", newline="", encoding="utf-8",
              errors="replace") as f:
        for row in csv.DictReader(f):
            d = _parse_date(row.get("open_date", ""))
            if d is None:
                continue
            name = (row.get("estab_name") or "UNKNOWN").upper()
            rec = {
                "activity_nr": str(row.get("activity_nr", "")),
                "open_date":   row.get("open_date", ""),
                "naics_code":  row.get("naics_code", ""),
                "estab_name":  name,
            }
            if d < cutoff_date:
                estab_hist[name].append(rec)
            elif d <= outcome_end_date:
                estab_future[name].append(rec)

    paired_names = [
        n for n in estab_hist
        if len(estab_hist[n]) >= min_hist_insp and len(estab_future.get(n, [])) >= 1
    ]
    logger.info("  [MultiTargetLabeler] %s paired establishments found.", f"{len(paired_names):,}")

    # ── Include UNPAIRED establishments for Stage 1 (inspection exposure) ──
    # These have historical data but NO future inspections — they are the
    # negative class for the inspection-exposure model.  Including them
    # enables the sequential pipeline to separate "not inspected" from
    # "inspected but clean" — critical for handling zero-inflation.
    if include_uninspected:
        unpaired_names = [
            n for n in estab_hist
            if len(estab_hist[n]) >= min_hist_insp and len(estab_future.get(n, [])) == 0
        ]
    else:
        unpaired_names = []
    all_names = paired_names + unpaired_names
    logger.info(
        "  [MultiTargetLabeler] %s total establishments (%s paired + %s unpaired for Stage 1).",
        f"{len(all_names):,}", f"{len(paired_names):,}", f"{len(unpaired_names):,}",
    )

    if not all_names:
        return []

    # ── IPW: fit industry-stratified inspection propensity model ──────────
    # IPW weights are only meaningful for paired (inspected) establishments.
    # Unpaired establishments get weight=1.0 by default.
    ipw_model = InspectionPropensityModel(naics_sectors=scorer.NAICS_SECTORS)
    ipw_model.fit(
        paired_hist=[estab_hist[n] for n in paired_names],
        unpaired_hist=[estab_hist[n] for n in unpaired_names],
    )
    ipw_weights: np.ndarray = ipw_model.ipw_weights(
        [estab_hist[n] for n in paired_names]
    )
    # Map name → weight for O(1) lookup in PASS 3
    ipw_by_name: dict = dict(zip(paired_names, ipw_weights.tolist()))

    # ── PASS 2: build violation/accident indices ───────────────────────────
    all_hist_acts: set = {
        r["activity_nr"] for n in all_names for r in estab_hist[n]
    }
    all_future_acts: set = {
        r["activity_nr"] for n in all_names for r in estab_future.get(n, [])
    }

    logger.info(
        "  [MultiTargetLabeler] Building violation indices (%s hist + %s future acts)…",
        f"{len(all_hist_acts):,}", f"{len(all_future_acts):,}",
    )
    hist_viol_index   = _build_violation_index(violations_path, all_hist_acts)
    future_viol_index = _build_violation_index(violations_path, all_future_acts)

    logger.info("  [MultiTargetLabeler] Building injury/fatality stats …")
    hist_fatals   = _build_hosp_fatal_stats(accidents_path, injuries_path, all_hist_acts)
    future_hosp_fatal = _build_hosp_fatal_stats(accidents_path, injuries_path, all_future_acts)

    # ── PASS 3: aggregate features and targets ─────────────────────────────
    logger.info("  [MultiTargetLabeler] Computing features + targets for %s establishments …", f"{len(all_names):,}")
    rows_raw: list = []
    scratch_industry: list = []

    for name in all_names:
        hist_insp   = estab_hist[name]
        future_insp = estab_future.get(name, [])

        # LEAKAGE GUARD: features from hist_insp only
        agg = _aggregate_hist_features(
            hist_insp, hist_viol_index, hist_fatals, one_year_ago,
        )
        if len(agg) != 6 or not agg[0]:
            continue
        features_17, naics_group, raw_vpi, raw_avg_pen, raw_sr, raw_wr = agg

        # LEAKAGE GUARD: outcomes from future_insp only
        # For unpaired establishments (no future inspections), all future
        # targets will be zero — this is the negative class for Stage 1.
        fut_acts       = [r["activity_nr"] for r in future_insp]
        fut_viols      = [v for act in fut_acts for v in future_viol_index.get(act, [])]
        fut_any_injury = any(future_hosp_fatal.get(act, False) for act in fut_acts)

        naics_2d = naics_group[:2] if naics_group else None
        targets = _compute_multi_targets(
            fut_viols, fut_any_injury, len(future_insp),
            penalty_thresholds=penalty_thresholds,
            naics_2digit=naics_2d,
        )

        rows_raw.append({
            "name":         name,
            "features_17":  features_17,
            "naics_group":  naics_group,
            "naics_2digit": naics_2d or "__unknown__",
            "_raw_vpi":     raw_vpi,
            "_raw_avg_pen": raw_avg_pen,
            "_raw_sr":      raw_sr,
            "_raw_wr":      raw_wr,
            "ipw_weight":   ipw_by_name.get(name, 1.0),
            **targets,
        })
        scratch_industry.append({
            "industry_group":   naics_group,
            "raw_vpi":          raw_vpi,
            "raw_avg_pen":      raw_avg_pen,
            "raw_serious_rate": raw_sr,
            "raw_wr_rate":      raw_wr,
        })

    if not rows_raw:
        return []

    # ── Build industry stats from paired sample (leakage guard) ───────────
    pop_df = pd.DataFrame(scratch_industry)
    industry_stats = compute_industry_stats(
        pop_df, min_sample=10, naics_map=naics_map,
    )

    # ── Complete features → 46 dims + pseudo label ─────────────────────────
    _NAICS_SECTORS = scorer.NAICS_SECTORS

    for row in rows_raw:
        ig  = row["naics_group"]
        rel = compute_relative_features(
            {
                "industry_group":    ig,
                "raw_vpi":           row.pop("_raw_vpi"),
                "raw_avg_pen":       row.pop("_raw_avg_pen"),
                "raw_serious_rate":  row.pop("_raw_sr"),
                "raw_wr_rate":       row.pop("_raw_wr"),
            },
            industry_stats,
            naics_map,
            min_sample=10,
        )

        def _safe(v: float) -> float:
            return 0.0 if (v != v) else v

        naics_2digit = ig[:2] if ig else None
        naics_vec    = scorer._encode_naics(naics_2digit)

        features_46 = row["features_17"] + [
            _safe(rel["relative_violation_rate"]),
            _safe(rel["relative_penalty"]),
            _safe(rel["relative_serious_ratio"]),
            _safe(rel["relative_willful_repeat"]),
        ] + naics_vec

        f46_arr = np.array(features_46)

        row["features_46"] = features_46
        row.pop("features_17")
        row.pop("naics_group")

    # ── Stratified sampling ────────────────────────────────────────────────
    # Stratify by NAICS × inspection/no-inspection × WR outcome to ensure
    # balanced coverage across all pipeline stages.
    strata_keys = []
    for row in rows_raw:
        f46 = row["features_46"]
        naics_prefix = "??"
        for j, sector in enumerate(_NAICS_SECTORS):
            if f46[22 + j] == 1:
                naics_prefix = sector
                break
        has_insp = row.get("future_has_inspection", 0)
        q_label = int(row.get("any_wr_serious", 0)) if has_insp else 0
        strata_keys.append(f"{naics_prefix}_I{has_insp}_Q{q_label}")

    indices   = list(range(len(rows_raw)))
    selected  = _stratified_sample(indices, strata_keys, sample_size, rng)
    sampled   = [rows_raw[i] for i in selected]

    # Attach cutoff metadata
    for row in sampled:
        row["cutoff_date"] = cutoff_date.isoformat()

    pos   = sum(1 for r in sampled if r["any_wr_serious"])
    neg   = len(sampled) - pos
    inj   = sum(1 for r in sampled if r["any_injury_fatal"])
    n_insp = sum(1 for r in sampled if r["future_has_inspection"])
    n_viol = sum(1 for r in sampled if r["future_has_violation"])
    logger.info(
        "  [MultiTargetLabeler] Sample: %s rows — "
        "Inspected: %s (%.1f%%), Has Violation: %s (%.1f%%), "
        "WR/Serious: %s pos / %s neg (%.1f%%), "
        "Injury/Fatal: %s (%.1f%%)",
        f"{len(sampled):,}",
        f"{n_insp:,}", n_insp / max(1, len(sampled)) * 100,
        f"{n_viol:,}", n_viol / max(1, len(sampled)) * 100,
        f"{pos:,}", f"{neg:,}",
        pos / max(1, len(sampled)) * 100,
        f"{inj:,}", inj / max(1, len(sampled)) * 100,
    )
    return sampled


def load_or_build(
    scorer: "MLRiskScorer",
    cutoff_date: date,
    outcome_end_date: date,
    cache_dir: str,
    inspections_path: str,
    violations_path: str,
    accidents_path: str,
    injuries_path: str,
    naics_map: dict,
    penalty_thresholds: Dict[str, Dict[str, float]],
    sample_size: int = 50_000,
    min_hist_insp: int = 2,
) -> List[Dict]:
    """Load from cache or build anew, with fingerprint-based invalidation."""
    cache_path  = os.path.join(cache_dir, CACHE_FILENAME)
    fp_new      = _build_fingerprint(
        cutoff_date, outcome_end_date, sample_size,
        inspections_path, violations_path,
    )

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("fingerprint") == fp_new:
                rows = blob["rows"]
                logger.info("  [MultiTargetLabeler] Loaded %s rows from cache.", f"{len(rows):,}")
                return rows
        except Exception as e:
            logger.warning("  [MultiTargetLabeler] Cache load failed (%s); rebuilding.", e)

    rows = build_multi_target_sample(
        scorer=scorer,
        cutoff_date=cutoff_date,
        outcome_end_date=outcome_end_date,
        inspections_path=inspections_path,
        violations_path=violations_path,
        accidents_path=accidents_path,
        injuries_path=injuries_path,
        naics_map=naics_map,
        penalty_thresholds=penalty_thresholds,
        sample_size=sample_size,
        min_hist_insp=min_hist_insp,
        include_uninspected=True,   # Stage 1 needs uninspected rows
    )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fp_new, "rows": rows}, f)
    except Exception as e:
        logger.warning("  [MultiTargetLabeler] Could not write cache: %s", e)

    return rows
