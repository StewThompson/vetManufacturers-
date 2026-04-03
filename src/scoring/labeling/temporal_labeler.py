"""
temporal_labeler.py — Build a stratified (features, real_outcome) training sample
from historical OSHA data.

For each establishment that has inspections both before and after the cutoff date,
this module pairs:

  * Historical features — aggregated from all inspections BEFORE the cutoff.
  * Real outcome labels — the future_adverse_outcome_score computed from
    inspections AFTER the cutoff (up to outcome_end_date).

Sampling strategy
-----------------
The paired pool is stratified by 2-digit NAICS sector, ensuring all industries
are represented.  The sample is drawn WITHOUT replacement; each stratum
contributes proportionally, floored at min_per_stratum samples.

Caching
-------
A build fingerprint (cutoff_date, outcome_end_date, sample_size, CSV paths)
is stored alongside the pkl file.  If the fingerprint matches on load, the
cached file is returned immediately without re-scanning the CSVs.  Delete
ml_cache/temporal_labels.pkl to force a rebuild.

Leakage safeguards
------------------
* Historical features are built exclusively from pre-cutoff inspections.
  [LEAKAGE GUARD] labels throughout the code mark enforcement points.
* Industry stats (z-scores) are computed from the sampled pre-cutoff
  population — never from the full population or future data.
* Future violation/accident indices are looked up ONLY for post-cutoff
  activity_nrs.
"""

from __future__ import annotations

import csv
import logging
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

logger = logging.getLogger(__name__)

CACHE_FILENAME = "temporal_labels.pkl"


def _build_accident_stats(
    accidents_path: str,
    injuries_path: str,
    activity_nrs: set,
) -> Dict[str, int]:
    """Return {activity_nr: fatality_count} for the given activity_nrs.

    Uses the same injury-table join logic as build_cache.py and the test suite.
    Only fatality counts are needed here (violations and penalties come from
    the violation index directly).
    """
    # injuries table: rel_insp_nr → activity_nr, summary_nr
    summaries_by_insp: Dict[str, set] = defaultdict(set)
    inj_by_snr: Dict[str, list] = defaultdict(list)

    csv.field_size_limit(10 * 1024 * 1024)
    try:
        with open(injuries_path, "r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                act = str(row.get("rel_insp_nr", ""))
                snr = str(row.get("summary_nr", ""))
                if act in activity_nrs and snr:
                    summaries_by_insp[act].add(snr)
                    inj_by_snr[f"{act}|{snr}"].append(
                        str(row.get("degree_of_inj", ""))
                    )
    except FileNotFoundError:
        pass

    acc_by_summary: Dict[str, dict] = {}
    try:
        with open(accidents_path, "r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                snr = str(row.get("summary_nr", ""))
                if snr:
                    acc_by_summary[snr] = row
    except FileNotFoundError:
        pass

    result: Dict[str, int] = {}
    for act in activity_nrs:
        fatalities = 0
        for snr in summaries_by_insp.get(act, set()):
            acc = acc_by_summary.get(snr, {})
            is_fatal = str(acc.get("fatality", "")).strip() in ("1", "Y", "True")
            if not is_fatal:
                is_fatal = any(
                    d.startswith("1")
                    for d in inj_by_snr.get(f"{act}|{snr}", [])
                )
            if is_fatal:
                fatalities += 1
        result[act] = fatalities
    return result

# ====================================================================== #
#  Build fingerprint (for cache invalidation)
# ====================================================================== #

def _build_fingerprint(
    cutoff_date: date,
    outcome_end_date: date,
    sample_size: int,
    inspections_path: str,
    violations_path: str,
) -> dict:
    """Lightweight cache fingerprint based on parameters and file mtimes."""

    def _mtime(p: str) -> float:
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    return {
        "cutoff":       cutoff_date.isoformat(),
        "outcome_end":  outcome_end_date.isoformat(),
        "sample_size":  sample_size,
        "insp_mtime":   round(_mtime(inspections_path), 1),
        "viol_mtime":   round(_mtime(violations_path), 1),
    }


# ====================================================================== #
#  Public API
# ====================================================================== #

def build_temporal_training_labels(
    scorer: "MLRiskScorer",
    cutoff_date: date,
    outcome_end_date: date,
    inspections_path: str,
    violations_path: str,
    accidents_path: str,
    injuries_path: str,
    naics_map: dict,
    sample_size: int = 20_000,
    rng_seed: int    = 42,
    min_hist_insp: int = 2,
) -> List[Dict]:
    """Build a stratified sample of (pre-cutoff features, real-outcome label) pairs.

    Data flow
    ---------
    1. Stream inspections_path → split per establishment into hist / future.
    2. Keep only establishments with >= min_hist_insp historical inspections
       AND >= 1 future inspection within [cutoff_date, outcome_end_date].
    3. For each eligible establishment, build the 47-feature vector from hist
       inspections and compute the real adverse outcome from future inspections.
    4. Assign a pseudo-label from the 17 absolute features (for diagnostics).
    5. Normalise the real adverse label to [0, 100].
    6. Stratify and downsample to sample_size.

    Parameters
    ----------
    scorer          : MLRiskScorer stub (needs _encode_naics, _industry_stats,
                      _naics_map).
    cutoff_date     : History/future split boundary.  All inspections before this
                      date contribute to features; all after contribute to labels.
    outcome_end_date: Latest date to include in the future outcome window.
                      Inspections after this date are ignored.
    inspections_path: Absolute path to inspections_bulk.csv (or equivalent).
    violations_path : Absolute path to violations_bulk.csv.
    accidents_path  : Absolute path to accidents_bulk.csv.
    injuries_path   : Absolute path to accident_injuries_bulk.csv.
    naics_map       : NAICS reference table (loaded via load_naics_map()).
    sample_size     : Maximum number of real-label rows to return.
    rng_seed        : Random seed for reproducible stratified sampling.
    min_hist_insp   : Minimum pre-cutoff inspections required to include an
                      establishment.

    Returns
    -------
    List of dicts, one per sampled establishment:
        name            : str   — establishment name (uppercase)
        features_46     : list  — 47-element pre-cutoff feature vector (pre-log)
        real_label_raw  : float — unnormalised future adverse score (0–ADV_MAX)
        real_label      : float — normalised to [0, 100]
        cutoff_date     : str   — ISO cutoff used to build this row
    """
    from src.scoring.industry_stats import (
        compute_industry_stats,
        compute_relative_features,
    )

    rng        = np.random.default_rng(rng_seed)
    one_year_ago = cutoff_date - timedelta(days=1095)  # matches production scorer

    # ── PASS 1: stream inspections, split hist/future per establishment ────
    logger.info("  [TemporalLabeler] Scanning inspections for cutoff=%s …", cutoff_date)
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
            rec  = {
                "activity_nr": str(row.get("activity_nr", "")),
                "open_date":   row.get("open_date", ""),
                "naics_code":  row.get("naics_code", ""),
                "estab_name":  name,
            }
            if d < cutoff_date:
                estab_hist[name].append(rec)
            elif d <= outcome_end_date:
                estab_future[name].append(rec)

    # Paired: establishments with both hist and future data.
    paired_names = [
        n for n in estab_hist
        if len(estab_hist[n]) >= min_hist_insp and len(estab_future.get(n, [])) >= 1
    ]
    logger.info("  [TemporalLabeler] %s paired establishments found.", f"{len(paired_names):,}")

    if not paired_names:
        return []

    # ── PASS 2: build activity_nr sets for violation/accident index ────────
    all_hist_acts:   set = {
        r["activity_nr"]
        for n in paired_names
        for r in estab_hist[n]
    }
    all_future_acts: set = {
        r["activity_nr"]
        for n in paired_names
        for r in estab_future.get(n, [])
    }

    logger.info(
        "  [TemporalLabeler] Building violation indices (%s hist + %s future acts)…",
        f"{len(all_hist_acts):,}", f"{len(all_future_acts):,}",
    )
    hist_viol_index   = _build_violation_index(violations_path, all_hist_acts)
    future_viol_index = _build_violation_index(violations_path, all_future_acts)

    logger.info("  [TemporalLabeler] Building accident/fatality stats…")
    hist_fatals   = _build_accident_stats(accidents_path, injuries_path, all_hist_acts)
    future_fatals = _build_accident_stats(accidents_path, injuries_path, all_future_acts)

    # ── PASS 3: compute features and real labels ───────────────────────────
    logger.info("  [TemporalLabeler] Aggregating features for %s establishments…", f"{len(paired_names):,}")

    rows_raw: list = []   # unsampled, all paired
    scratch_industry: list = []

    for name in paired_names:
        hist_insp   = estab_hist[name]
        future_insp = estab_future.get(name, [])

        # LEAKAGE GUARD: aggregate features from hist_insp only.
        agg = _aggregate_hist_features(
            hist_insp, hist_viol_index, hist_fatals, one_year_ago,
        )
        if len(agg) != 6 or not agg[0]:
            continue
        features_17, naics_group, raw_vpi, raw_avg_pen, raw_sr, raw_wr = agg

        # LEAKAGE GUARD: compute adverse outcome from future_insp only.
        fut_acts  = [r["activity_nr"] for r in future_insp]
        fut_viols = [v for act in fut_acts for v in future_viol_index.get(act, [])]
        fut_fats  = sum(future_fatals.get(act, 0) for act in fut_acts)

        raw_adv   = _compute_adverse(fut_viols, fut_fats, len(future_insp))
        real_label = _normalize_adverse(raw_adv)

        rows_raw.append({
            "name":          name,
            "features_17":   features_17,
            "naics_group":   naics_group,
            "_raw_vpi":      raw_vpi,
            "_raw_avg_pen":  raw_avg_pen,
            "_raw_sr":       raw_sr,
            "_raw_wr":       raw_wr,
            "real_label_raw": raw_adv,
            "real_label":    real_label,
        })
        scratch_industry.append({
            "industry_group": naics_group,
            "raw_vpi":        raw_vpi,
            "raw_avg_pen":    raw_avg_pen,
            "raw_serious_rate": raw_sr,
            "raw_wr_rate":    raw_wr,
        })

    if not rows_raw:
        return []

    # ── Compute industry stats from THIS sampled population (no leakage) ──
    pop_df = pd.DataFrame(scratch_industry)
    industry_stats = compute_industry_stats(
        pop_df,
        min_sample=10,
        naics_map=naics_map,
    )

    # ── Append industry z-scores + NAICS one-hot → 46 features ───────────
    # Also compute pseudo-labels BEFORE sampling (for stratification).
    for row in rows_raw:
        ig  = row["naics_group"]
        rel = compute_relative_features(
            {
                "industry_group":   ig,
                "raw_vpi":          row.pop("_raw_vpi"),
                "raw_avg_pen":      row.pop("_raw_avg_pen"),
                "raw_serious_rate": row.pop("_raw_sr"),
                "raw_wr_rate":      row.pop("_raw_wr"),
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

        # Compute 46-dim feature vector (no pseudo-label)
        f46_arr = np.array(features_46)

        row["features_46"] = features_46
        row.pop("features_17")
        row.pop("naics_group")

    # ── Stratified sampling ────────────────────────────────────────────────
    real_arr  = np.array([r["real_label"] for r in rows_raw])
    # Wrap in Series so pd.qcut returns a Series (supports .iloc).
    # labels=False returns integer bin indices (0..N-1) regardless of how many
    # bins survive after duplicates="drop" — avoids "labels must be one fewer
    # than bin edges" when the 2022 cutoff yields a zero-heavy distribution.
    quartiles = pd.qcut(
        pd.Series(real_arr).clip(0, 100),
        q=4, labels=False, duplicates="drop",
    )

    _NAICS_SECTORS = [
        "11", "21", "22", "23", "31", "32", "33",
        "42", "44", "45", "48", "49", "51", "52",
        "53", "54", "55", "56", "61", "62", "71",
        "72", "81", "92",
    ]
    strata_keys = []
    for i, row in enumerate(rows_raw):
        f46 = row["features_46"]
        # Features index 21–44 are NAICS one-hot (one per sector in _NAICS_SECTORS)
        naics_prefix = "??"
        for j, sector in enumerate(_NAICS_SECTORS):
            if f46[22 + j] == 1:
                naics_prefix = sector
                break
        q_val   = quartiles.iloc[i]
        q_label = int(q_val) if not pd.isna(q_val) else 0
        strata_keys.append(f"{naics_prefix}_Q{q_label}")

    all_indices  = list(range(len(rows_raw)))
    sample_idxs  = _stratified_sample(
        all_indices, strata_keys, sample_size, rng, min_per_stratum=5,
    )

    sampled = [
        {
            "name":           rows_raw[i]["name"],
            "features_46":    rows_raw[i]["features_46"],
            "real_label_raw": rows_raw[i]["real_label_raw"],
            "real_label":     rows_raw[i]["real_label"],
            "cutoff_date":    cutoff_date.isoformat(),
        }
        for i in sample_idxs
    ]

    n_nonzero = sum(1 for r in sampled if r["real_label_raw"] > 0)
    logger.info(
        "  [TemporalLabeler] Sample: %s rows (%s non-zero real label, %s clean).",
        f"{len(sampled):,}", f"{n_nonzero:,}", f"{len(sampled) - n_nonzero:,}",
    )
    return sampled


def load_or_build_temporal_labels(
    scorer: "MLRiskScorer",
    cutoff_date: date,
    outcome_end_date: date,
    cache_dir: str,
    inspections_path: str,
    violations_path: str,
    accidents_path: str,
    injuries_path: str,
    naics_map: dict,
    sample_size: int = 50_000,
    rng_seed: int    = 42,
    min_hist_insp: int = 2,
) -> List[Dict]:
    """Load cached temporal labels from pkl, or build and cache them.

    The cache is invalidated whenever the parameters change or the source
    CSV files are modified.  Delete ml_cache/temporal_labels.pkl to force
    a full rebuild at any time.

    Returns an empty list if any required CSV path does not exist.
    """
    for req in (inspections_path, violations_path):
        if not os.path.exists(req):
            logger.warning("  [TemporalLabeler] Required CSV not found: %s — skipping.", req)
            return []

    fp = _build_fingerprint(
        cutoff_date, outcome_end_date, sample_size,
        inspections_path, violations_path,
    )

    cache_path = os.path.join(cache_dir, CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fp:
                rows = cached["rows"]
                logger.info(
                    "  [TemporalLabeler] Loaded %s cached rows (cutoff=%s).",
                    f"{len(rows):,}", cutoff_date,
                )
                return rows
            else:
                logger.info("  [TemporalLabeler] Cache fingerprint mismatch — rebuilding.")
        except Exception as e:
            logger.warning("  [TemporalLabeler] Cache load error (%s) — rebuilding.", e)

    rows = build_temporal_training_labels(
        scorer=scorer,
        cutoff_date=cutoff_date,
        outcome_end_date=outcome_end_date,
        inspections_path=inspections_path,
        violations_path=violations_path,
        accidents_path=accidents_path,
        injuries_path=injuries_path,
        naics_map=naics_map,
        sample_size=sample_size,
        rng_seed=rng_seed,
        min_hist_insp=min_hist_insp,
    )

    if rows:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"fingerprint": fp, "rows": rows}, f)
            logger.info(
                "  [TemporalLabeler] Saved %s rows to %s.",
                f"{len(rows):,}", cache_path,
            )
        except Exception as e:
            logger.warning("  [TemporalLabeler] Could not save cache: %s", e)

    return rows


def summarise_temporal_labels(rows: List[Dict]) -> None:
    """Print a diagnostics summary of a built label set."""
    if not rows:
        logger.info("  [TemporalLabeler] No rows to summarise.")
        return
    real_labels = np.array([r["real_label"] for r in rows])
    n_nonzero   = int((real_labels > 0).sum())
    logger.info(
        "TEMPORAL LABEL SUMMARY  (%s rows)  "
        "Real label range: [%.1f, %.1f]  mean=%.1f  std=%.1f  "
        "Non-zero: %s (%.1f%%)",
        f"{len(rows):,}",
        real_labels.min(), real_labels.max(),
        real_labels.mean(), real_labels.std(),
        f"{n_nonzero:,}", n_nonzero / len(rows) * 100,
    )
