"""multi_target_labeler.py — Build multi-target training labels for the
probabilistic risk model.

Unlike the single-target ``temporal_labeler`` (which returns one composite
adverse-outcome score), this module produces **7 target columns** per
establishment that directly correspond to the 4 prediction heads:

    1. any_wr_serious       — 0/1, any Willful/Repeat/Serious violation post-cutoff
    2. future_total_penalty — float ($), total OSHA penalties in the outcome window
    3. log_penalty          — log1p(future_total_penalty)
    4. future_citation_count — int, total violations (citations) post-cutoff
    5. is_moderate_penalty  — 0/1, total penalty ≥ NAICS P75 threshold
    6. is_large_penalty     — 0/1, total penalty ≥ NAICS P90 threshold
    7. is_extreme_penalty   — 0/1, total penalty ≥ NAICS P95 threshold

Design notes
------------
* All feature computation is delegated to helpers imported from
  ``temporal_labeler``, keeping code deduplication high and leakage logic
  centrally maintained.
* Penalty tier thresholds come from ``penalty_percentiles.py`` and MUST be
  pre-computed on training-fold data before calling this module.
* The stratified sampling strategy mirrors ``temporal_labeler`` — NAICS sector
  × pseudo-label quartile — so the multi-target sample is drawn from the same
  distribution as the real-label sample used by the existing GBR head.
* Returns dict rows that include both ``features_46`` (pre-log-transform) and
  all 7 target scalars, plus the ``real_label`` composite score for backward
  compatibility with diagnostics.
"""
from __future__ import annotations

import csv
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

from src.scoring.labeling.temporal_labeler import (
    _parse_date,
    _stream_inspections,
    _build_violation_index,
    _build_accident_stats,
    _aggregate_hist_features,
    _compute_adverse,
    _normalize_adverse,
    _stratified_sample,
    ADV_MAX,
)
from src.scoring.penalty_percentiles import lookup_threshold


CACHE_FILENAME = "multi_target_labels.pkl"


# ── Per-establishment multi-target computation ──────────────────────────────

def _compute_multi_targets(
    future_viols: List[dict],
    future_fatalities: int,
    n_future_insp: int,
    penalty_thresholds: Dict[str, Dict[str, float]],
    naics_2digit: Optional[str],
) -> Dict[str, float]:
    """Compute all 7 target variables from post-cutoff violations.

    Parameters
    ----------
    future_viols : list of dicts with keys ``viol_type``, ``current_penalty``,
        ``initial_penalty`` from the violations CSV.
    future_fatalities : int
        Number of post-cutoff fatality events linked to this establishment.
    n_future_insp : int
        Number of post-cutoff inspections.
    penalty_thresholds : dict
        Output of ``penalty_percentiles.compute_penalty_percentiles()``.
    naics_2digit : str or None
        2-digit NAICS sector for per-industry threshold lookup.

    Returns
    -------
    dict with keys:
        any_wr_serious, future_total_penalty, log_penalty,
        future_citation_count, is_moderate_penalty, is_large_penalty,
        is_extreme_penalty, real_label_raw, real_label
    """
    total_penalty = sum(
        float(v.get("current_penalty") or v.get("initial_penalty") or 0)
        for v in future_viols
    )
    n_citations  = len(future_viols)
    wr_serious   = sum(
        1 for v in future_viols
        if v.get("viol_type") in ("W", "R", "S")
    )

    p75 = lookup_threshold(penalty_thresholds, naics_2digit, "p75")
    p90 = lookup_threshold(penalty_thresholds, naics_2digit, "p90")
    p95 = lookup_threshold(penalty_thresholds, naics_2digit, "p95")

    # Composite adverse (for calibration / backward compat)
    raw_adv   = _compute_adverse(future_viols, future_fatalities, n_future_insp)
    real_label = _normalize_adverse(raw_adv)

    return {
        "any_wr_serious":       int(wr_serious > 0),
        "future_total_penalty": total_penalty,
        "log_penalty":          math.log1p(total_penalty),
        "future_citation_count": n_citations,
        "is_moderate_penalty":  int(total_penalty >= p75),
        "is_large_penalty":     int(total_penalty >= p90),
        "is_extreme_penalty":   int(total_penalty >= p95),
        # Kept for diagnostics / backward compat with existing calibrator
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
        "schema":      "multi_target_v1",
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
    min_hist_insp: int = 1,
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
        ``_naics_map``, and ``_pseudo_label()``.
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
        pseudo_label       : float
        any_wr_serious     : int (0/1)
        future_total_penalty : float
        log_penalty        : float
        future_citation_count : int
        is_moderate_penalty : int (0/1)
        is_large_penalty   : int (0/1)
        is_extreme_penalty : int (0/1)
        real_label_raw     : float
        real_label         : float
        naics_2digit       : str
        cutoff_date        : str (ISO)
    """
    from src.scoring.pseudo_labeler import pseudo_label as _pseudo_label
    from src.scoring.industry_stats import (
        compute_industry_stats,
        compute_relative_features,
    )

    rng = np.random.default_rng(rng_seed)
    one_year_ago = cutoff_date - timedelta(days=1095)

    # ── PASS 1: split inspections into hist / future per establishment ─────
    print(f"  [MultiTargetLabeler] Scanning inspections (cutoff={cutoff_date}) …")
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
    print(f"  [MultiTargetLabeler] {len(paired_names):,} paired establishments found.")

    if not paired_names:
        return []

    # ── PASS 2: build violation/accident indices ───────────────────────────
    all_hist_acts: set = {
        r["activity_nr"] for n in paired_names for r in estab_hist[n]
    }
    all_future_acts: set = {
        r["activity_nr"] for n in paired_names for r in estab_future.get(n, [])
    }

    print(
        f"  [MultiTargetLabeler] Building violation indices "
        f"({len(all_hist_acts):,} hist + {len(all_future_acts):,} future acts) …"
    )
    hist_viol_index   = _build_violation_index(violations_path, all_hist_acts)
    future_viol_index = _build_violation_index(violations_path, all_future_acts)

    print("  [MultiTargetLabeler] Building accident/fatality stats …")
    hist_fatals   = _build_accident_stats(accidents_path, injuries_path, all_hist_acts)
    future_fatals = _build_accident_stats(accidents_path, injuries_path, all_future_acts)

    # ── PASS 3: aggregate features and targets ─────────────────────────────
    print(f"  [MultiTargetLabeler] Computing features + targets for {len(paired_names):,} establishments …")
    rows_raw: list = []
    scratch_industry: list = []

    for name in paired_names:
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
        fut_acts  = [r["activity_nr"] for r in future_insp]
        fut_viols = [v for act in fut_acts for v in future_viol_index.get(act, [])]
        fut_fats  = sum(future_fatals.get(act, 0) for act in fut_acts)

        naics_2d = naics_group[:2] if naics_group else None
        targets = _compute_multi_targets(
            fut_viols, fut_fats, len(future_insp),
            penalty_thresholds, naics_2d,
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

        f46_arr       = np.array(features_46)
        pseudo        = float(_pseudo_label(f46_arr))

        row["features_46"] = features_46
        row["pseudo_label"] = pseudo
        row.pop("features_17")
        row.pop("naics_group")

    # ── Stratified sampling ────────────────────────────────────────────────
    pseudo_arr = np.array([r["pseudo_label"] for r in rows_raw])
    quartiles  = pd.qcut(
        pd.Series(pseudo_arr).clip(0, 100),
        q=4, labels=[0, 1, 2, 3], duplicates="drop",
    )

    strata_keys = []
    for i, row in enumerate(rows_raw):
        f46 = row["features_46"]
        naics_prefix = "??"
        for j, sector in enumerate(_NAICS_SECTORS):
            if f46[22 + j] == 1:
                naics_prefix = sector
                break
        q_val   = quartiles.iloc[i]
        q_label = int(q_val) if not pd.isna(q_val) else 0
        strata_keys.append(f"{naics_prefix}_Q{q_label}")

    indices   = list(range(len(rows_raw)))
    selected  = _stratified_sample(indices, strata_keys, sample_size, rng)
    sampled   = [rows_raw[i] for i in selected]

    # Attach cutoff metadata
    for row in sampled:
        row["cutoff_date"] = cutoff_date.isoformat()

    pos   = sum(1 for r in sampled if r["any_wr_serious"])
    neg   = len(sampled) - pos
    print(
        f"  [MultiTargetLabeler] Sample: {len(sampled):,} rows, "
        f"WR/Serious: {pos:,} pos / {neg:,} neg ({pos/max(1,len(sampled)):.1%} rate), "
        f"Large-penalty: {sum(1 for r in sampled if r['is_large_penalty']):,}"
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
                print(
                    f"  [MultiTargetLabeler] Loaded {len(rows):,} rows from cache."
                )
                return rows
        except Exception as e:
            print(f"  [MultiTargetLabeler] Cache load failed ({e}); rebuilding.")

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
    )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fp_new, "rows": rows}, f)
    except Exception as e:
        print(f"  [MultiTargetLabeler] Could not write cache ({e}).")

    return rows
