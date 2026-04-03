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
import json
import math
import os
import pickle
from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.scoring.ml_risk_scorer import MLRiskScorer


# ── Adverse-outcome formula weights (mirror test_real_world_validation.py) ──
ADV_FATALITY_FLAG     = 20.0
ADV_FATALITY_PER      = 5.0
ADV_FATALITY_MAX_XTRA = 15.0
ADV_WR_FLAG           = 8.0
ADV_WR_PER            = 3.0
ADV_WR_MAX            = 15.0
ADV_SERIOUS_PER       = 1.0
ADV_SERIOUS_MAX       = 10.0
ADV_PENALTY_SCALE     = 0.8
ADV_PENALTY_MAX       = 10.0
ADV_RATE_SCALE        = 2.0
ADV_RATE_MAX          = 10.0
# Theoretical maximum: 20 + 15 + 8 + 15 + 10 + 10 + 10 = 88
ADV_MAX = 88.0

CACHE_FILENAME = "temporal_labels.pkl"


# ====================================================================== #
#  Low-level adverse-outcome computation (no external dependencies)
# ====================================================================== #

def _compute_adverse(
    future_viols: list,
    future_fatalities: int,
    n_future_insp: int,
) -> float:
    """Compute the composite future_adverse_outcome_score.

    Mirrors the formula in test_real_world_validation._compute_future_outcomes.
    Returns 0.0 when there are no future inspections.
    """
    if n_future_insp == 0:
        return 0.0

    n_viols        = len(future_viols)
    willful_repeat = sum(1 for v in future_viols if v.get("viol_type") in ("W", "R"))
    serious        = sum(1 for v in future_viols if v.get("viol_type") == "S")
    penalties      = [
        float(v.get("current_penalty") or v.get("initial_penalty") or 0)
        for v in future_viols
    ]
    total_penalty  = sum(penalties)
    violation_rate = n_viols / n_future_insp
    any_fatal      = int(future_fatalities > 0)

    adv = 0.0
    adv += ADV_FATALITY_FLAG * any_fatal
    adv += min((future_fatalities - 1) * ADV_FATALITY_PER, ADV_FATALITY_MAX_XTRA) \
           if future_fatalities > 1 else 0.0
    adv += ADV_WR_FLAG  * int(willful_repeat > 0)
    adv += min(willful_repeat * ADV_WR_PER, ADV_WR_MAX)
    adv += min(serious * ADV_SERIOUS_PER, ADV_SERIOUS_MAX)
    adv += min(math.log1p(total_penalty) * ADV_PENALTY_SCALE, ADV_PENALTY_MAX)
    adv += min(violation_rate * ADV_RATE_SCALE, ADV_RATE_MAX)
    return adv


def _normalize_adverse(raw: float) -> float:
    """Rescale raw adverse score [0, ADV_MAX] → [0, 100]."""
    return float(np.clip(raw / ADV_MAX * 100.0, 0.0, 100.0))


# ====================================================================== #
#  CSV streaming helpers
# ====================================================================== #

def _parse_date(date_str: str) -> Optional[date]:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return None


def _stream_inspections(
    path: str,
    earliest: Optional[date] = None,
    latest: Optional[date] = None,
) -> List[dict]:
    """Stream inspection rows from path, optionally filtering by date range.

    To keep memory bounded, only the required fields are retained.
    """
    required_fields = {
        "activity_nr", "estab_name", "open_date", "naics_code", "nr_in_estab",
    }
    result: list = []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            d = _parse_date(row.get("open_date", ""))
            if d is None:
                continue
            if earliest and d < earliest:
                continue
            if latest and d > latest:
                continue
            result.append({k: row.get(k, "") for k in required_fields})
    return result


def _build_violation_index(path: str, activity_nrs: set) -> Dict[str, list]:
    """Stream violations CSV, keeping only rows whose activity_nr is requested.

    Only the fields needed for adverse-outcome computation are kept in memory.
    """
    needed_fields = {"activity_nr", "viol_type", "current_penalty",
                     "initial_penalty", "gravity", "delete_flag"}
    index: Dict[str, list] = defaultdict(list)
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("delete_flag") == "X":
                continue
            act = str(row.get("activity_nr", ""))
            if act in activity_nrs:
                index[act].append({k: row.get(k, "") for k in needed_fields})
    return dict(index)


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
#  Per-establishment feature aggregation
# ====================================================================== #

def _aggregate_hist_features(
    hist_inspections: list,
    viol_index: Dict[str, list],
    acc_fatals: Dict[str, int],
    one_year_ago: date,
) -> Tuple[list, Optional[str]]:
    """Compute the 18 absolute features for a set of historical inspections.

    Returns (features_17, naics_group) where naics_group is the majority-vote
    4-digit NAICS code (or None when no NAICS data is present).

    LEAKAGE GUARD: hist_inspections must only contain pre-cutoff records.
    """
    n_insp = len(hist_inspections)
    if n_insp == 0:
        return ([], None)

    # Anchor for exponential decay: the cutoff date (equivalent to
    # one_year_ago + 3 years), matching the anchor used in production scoring.
    decay_anchor = one_year_ago + timedelta(days=1095)

    recent             = 0
    severe             = 0
    clean              = 0
    viols:    list     = []
    acc_count          = 0
    fat_count          = 0
    inj_count          = 0
    time_adj_pen       = 0.0
    max_insp_pen       = 0.0
    recent_viol_count  = 0
    recent_wr_raw      = 0
    estab_sizes: list  = []
    naics_votes: Dict[str, int] = defaultdict(int)

    for insp in hist_inspections:
        act = str(insp.get("activity_nr", ""))
        d   = _parse_date(insp.get("open_date", ""))

        insp_viols = viol_index.get(act, [])
        viols.extend(insp_viols)
        if not insp_viols:
            clean += 1

        if d and d >= one_year_ago:
            recent += 1
            recent_viol_count += len(insp_viols)
            recent_wr_raw += sum(
                1 for v in insp_viols if v.get("viol_type") in {"W", "R"}
            )

        # Time-adjusted penalty: sum per-inspection penalties weighted by
        # exponential decay from cutoff date (τ = 3 years).
        if insp_viols:
            insp_pen_sum = sum(
                float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                for v in insp_viols
            )
            if insp_pen_sum > max_insp_pen:
                max_insp_pen = insp_pen_sum
            if insp_pen_sum > 0 and d:
                age_years = max(0.0, (decay_anchor - d).days / 365.25)
                time_adj_pen += insp_pen_sum * math.exp(-age_years / 3.0)

        fat = acc_fatals.get(act, 0)
        fat_count += fat
        if fat > 0:
            severe += 1
            acc_count += 1
            inj_count += fat  # approximate; injuries not available in slim index

        nc = str(insp.get("naics_code") or "").strip()
        if nc and nc.isdigit() and len(nc) >= 4:
            naics_votes[nc[:4]] += 1

        nr_raw = str(insp.get("nr_in_estab") or "").strip()
        if nr_raw:
            try:
                estab_sizes.append(float(nr_raw))
            except ValueError:
                pass

    naics_group = max(naics_votes, key=naics_votes.get) if naics_votes else None

    n_viols      = len(viols)
    serious_raw  = sum(1 for v in viols if v.get("viol_type") == "S")
    willful_raw  = sum(1 for v in viols if v.get("viol_type") == "W")
    repeat_raw   = sum(1 for v in viols if v.get("viol_type") == "R")
    penalties    = [
        float(v.get("current_penalty") or v.get("initial_penalty") or 0)
        for v in viols
    ]
    total_pen    = sum(penalties)
    avg_pen      = float(np.mean(penalties)) if penalties else 0.0
    max_pen      = max(penalties)             if penalties else 0.0
    gravities    = []
    for v in viols:
        g = v.get("gravity", "")
        if g:
            try:
                gravities.append(float(g))
            except (ValueError, TypeError):
                pass
    avg_gravity  = float(np.mean(gravities)) if gravities else 0.0

    recent_ratio = recent      / n_insp
    vpi          = n_viols     / n_insp
    pen_per_insp = total_pen   / n_insp
    clean_ratio  = clean       / n_insp
    serious_rate = serious_raw / n_insp
    willful_rate = willful_raw / n_insp
    repeat_rate  = repeat_raw  / n_insp
    severe_rate  = severe      / n_insp
    acc_rate     = acc_count   / n_insp
    fat_rate     = fat_count   / n_insp
    inj_rate     = inj_count   / n_insp

    total_wr       = willful_raw + repeat_raw
    recent_wr_rate = recent_wr_raw / max(total_wr, 1)
    vpi_recent     = recent_viol_count / max(recent, 1)
    trend_delta    = vpi - vpi_recent

    # Median establishment size — proxy for OSHA penalty multiplier scale.
    # Use median (not mean) to be robust against one-off large-site inspections.
    estab_sizes_clean = [s for s in estab_sizes if s > 0]
    median_estab_size = float(np.median(estab_sizes_clean)) if estab_sizes_clean else 0.0

    features_17 = [
        n_insp, n_viols,
        serious_rate, willful_rate, repeat_rate,
        total_pen, avg_pen, max_pen,
        recent_ratio, severe_rate, vpi,
        acc_rate, fat_rate, inj_rate, avg_gravity,
        pen_per_insp, clean_ratio,
        time_adj_pen,
        recent_wr_rate,
        trend_delta,
        # High-signal penalty discriminators (Option B expansion)
        math.log1p(willful_raw),
        math.log1p(repeat_raw),
        1.0 if fat_count > 0 else 0.0,
        math.log1p(max_insp_pen),
        math.log1p(median_estab_size),
    ]

    # Also return scratch fields for industry relative features
    raw_serious_rate = serious_raw / max(n_viols, 1)
    raw_wr_rate      = (willful_raw + repeat_raw) / max(n_viols, 1)

    return features_17, naics_group, vpi, avg_pen, raw_serious_rate, raw_wr_rate


# ====================================================================== #
#  Stratified sampling
# ====================================================================== #

def _stratified_sample(
    indices: List[int],
    strata_keys: List[str],
    sample_size: int,
    rng: np.random.Generator,
    min_per_stratum: int = 5,
) -> List[int]:
    """Stratified sampling by strata_keys.

    Allocates sample_size proportionally across strata, floored at
    min_per_stratum per stratum when the stratum is large enough.
    Returns a flat list of row indices to include in the sample.
    """
    assert len(indices) == len(strata_keys)
    n_total = len(indices)
    if n_total <= sample_size:
        return list(indices)

    # Group indices by stratum
    groups: Dict[str, list] = defaultdict(list)
    for idx, key in zip(indices, strata_keys):
        groups[key].append(idx)

    # Proportional allocation
    result: List[int] = []
    for key, grp in groups.items():
        n_grp = len(grp)
        alloc = max(min_per_stratum, int(round(n_grp / n_total * sample_size)))
        alloc = min(alloc, n_grp)
        chosen = rng.choice(grp, size=alloc, replace=False).tolist()
        result.extend(chosen)

    # Trim or top-up to exactly sample_size
    if len(result) > sample_size:
        result = rng.choice(result, size=sample_size, replace=False).tolist()
    elif len(result) < sample_size:
        remaining = list(set(indices) - set(result))
        extra = min(sample_size - len(result), len(remaining))
        if extra > 0:
            result.extend(
                rng.choice(remaining, size=extra, replace=False).tolist()
            )
    return result


def _make_stratum_key(naics_group: Optional[str], real_label_quartile: int) -> str:
    sector = (naics_group or "??")[:2]
    return f"{sector}_Q{real_label_quartile}"


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
    print(f"  [TemporalLabeler] Scanning inspections for cutoff={cutoff_date} …")
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
    print(f"  [TemporalLabeler] {len(paired_names):,} paired establishments found.")

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

    print(f"  [TemporalLabeler] Building violation indices "
          f"({len(all_hist_acts):,} hist + {len(all_future_acts):,} future acts)…")
    hist_viol_index   = _build_violation_index(violations_path, all_hist_acts)
    future_viol_index = _build_violation_index(violations_path, all_future_acts)

    print("  [TemporalLabeler] Building accident/fatality stats…")
    hist_fatals   = _build_accident_stats(accidents_path, injuries_path, all_hist_acts)
    future_fatals = _build_accident_stats(accidents_path, injuries_path, all_future_acts)

    # ── PASS 3: compute features and real labels ───────────────────────────
    print(f"  [TemporalLabeler] Aggregating features for {len(paired_names):,} establishments…")

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
    print(
        f"  [TemporalLabeler] Sample: {len(sampled):,} rows  "
        f"({n_nonzero:,} with non-zero real label, "
        f"{len(sampled) - n_nonzero:,} with clean future record)."
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
            print(f"  [TemporalLabeler] Required CSV not found: {req} — skipping.")
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
                print(
                    f"  [TemporalLabeler] Loaded {len(rows):,} cached rows "
                    f"(cutoff={cutoff_date})."
                )
                return rows
            else:
                print("  [TemporalLabeler] Cache fingerprint mismatch — rebuilding.")
        except Exception as e:
            print(f"  [TemporalLabeler] Cache load error ({e}) — rebuilding.")

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
            print(f"  [TemporalLabeler] Saved {len(rows):,} rows to {cache_path}.")
        except Exception as e:
            print(f"  [TemporalLabeler] Could not save cache: {e}")

    return rows


def summarise_temporal_labels(rows: List[Dict]) -> None:
    """Print a diagnostics summary of a built label set."""
    if not rows:
        print("  [TemporalLabeler] No rows to summarise.")
        return
    real_labels = np.array([r["real_label"] for r in rows])
    n_nonzero   = int((real_labels > 0).sum())
    print("\n" + "=" * 60)
    print(f"TEMPORAL LABEL SUMMARY  ({len(rows):,} rows)")
    print(f"  Real label  range: [{real_labels.min():.1f}, {real_labels.max():.1f}]  "
          f"mean={real_labels.mean():.1f}  std={real_labels.std():.1f}")
    print(f"  Non-zero real label: {n_nonzero:,} ({n_nonzero/len(rows):.1%})")
    print("=" * 60 + "\n")
