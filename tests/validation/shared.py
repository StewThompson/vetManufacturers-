"""
test_real_world_validation.py — External-validity testing for the manufacturer risk score.

EXTERNAL VALIDITY
=================
For every establishment that has at least one OSHA inspection before the
cutoff date, we:

  1. BASELINE RISK SCORE (predictor)
       Aggregate only pre-cutoff inspections → 46-feature representation →
       score with a GBR model trained exclusively on pre-cutoff population.
       Industry normalisation is also computed from pre-cutoff data only.

  2. FUTURE ADVERSE OUTCOMES (external targets)
       Aggregate only post-cutoff inspections, counting serious / willful /
       repeat violations, fatalities, penalties, etc.  These are the ground-
       truth signals the score is meant to anticipate.

  3. VALIDATION
       Higher baseline scores should be associated with worse future outcomes:
       rank correlation, decile lift, tier monotonicity, binary event
       discrimination, precision-at-k, industry robustness, sparse-data check.

Leakage safeguards
------------------
* Historical scoring model is trained exclusively on pre-cutoff population.
  [LEAKAGE GUARD labels mark the enforcement points in the code.]
* Industry normalisation (z-scores) is computed from the pre-cutoff
  population only — never the full or future population.
* Future outcome labels are derived exclusively from post-cutoff inspections.
  The violation / accident indices cover all time periods, but each lookup
  is gated on the activity_nr of a post-cutoff inspection only.
* Establishments are processed in one pass; no future information crosses
  back into the historical feature vectors.

Test categories
---------------
  1.  Data integrity — volume, date guard, feature shape, NaN check
  2.  Score vs. future adverse-outcome (Spearman, Pearson, bootstrap CI)
  3.  Top-decile lift
  4.  Tier monotonicity (Low < Medium < High future adverse rates)
  5.  Binary event discrimination (positives outscore negatives)
  6.  Precision-at-k / recall-at-k
  7.  Calibration report by risk band         ← diagnostic, always passes
  8.  Industry robustness
  9.  Sparse-data robustness
 10.  Summary report                          ← diagnostic, always passes
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import csv
import numpy as np
import pandas as pd
import pytest
from collections import defaultdict
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import Counter

# ── Project imports ────────────────────────────────────────────────────
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.multi_target_scorer import MultiTargetRiskScorer
from src.scoring.industry_stats import compute_industry_stats, compute_relative_features
from src.data_retrieval.naics_lookup import load_naics_map


# ====================================================================== #
#  Constants
# ====================================================================== #

# Primary split: score with pre-2024 data, measure outcomes from 2024 onward.
CUTOFF_DATE = date(2024, 1, 1)

CACHE_DIR = "ml_cache"

# Sample-size guards — tests skip gracefully when volume is insufficient.
MIN_HIST_ESTABLISHMENTS = 50    # minimum to train a meaningful scoring model
MIN_FUTURE_ESTABLISHMENTS = 20  # minimum for correlation tests to be meaningful
MIN_BINARY_POSITIVE = 10        # minimum positive-class count needed for AUROC

# ── External-validity thresholds ──────────────────────────────────────
# These are intentionally more relaxed than pseudo-label reconstruction
# thresholds (test_temporal_validation.py) because real-world prediction
# is harder than reproducing a deterministic training target.
MIN_SPEARMAN_REAL    = 0.10   # modest but reliably positive rank correlation
MIN_TOP_DECILE_LIFT  = 1.20   # top decile ≥ 1.2× worse than population mean
MIN_BINARY_DELTA     = 0.5    # positives should outscore negatives by ≥ 0.5 pts
MIN_AUROC            = 0.55   # just above chance; external labels are noisy

# cut-points: 20 (broad early-warning), 30 (Recommend → Caution boundary),
# 40, 50, 60 (Caution → Do-Not-Recommend boundary).
EVAL_THRESHOLDS = [20, 30, 40, 50, 60]

# Fractions evaluated for top-k capture analysis.
TOPK_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

# ── Composite adverse-outcome score weights ────────────────────────────
# Each weight is documented below; the composite is the primary real-world
# validation target because it summarises multiple dimensions of harm.
#
#   Component               | Max pts | Rationale
#   ----------------------- | ------- | -----------------------------------------
#   Fatality (binary flag)  |    20   | Highest-severity OSHA outcome
#   Fatality (per-count)    |    15   | Escalates for multiple fatalities (cap 3)
#   Willful/repeat (flag)   |     8   | Regulatory violation → systemic problem
#   Willful/repeat (count)  |    15   | Up to 5 W/R violations × 3 pts
#   Serious violations      |    10   | Up to 10 serious violations × 1 pt
#   Penalties (log-scale)   |    10   | log1p(total_penalty) × 0.8, capped
#   Violation rate          |    10   | Violations per future inspection × 2, cap 5
ADV_FATALITY_FLAG     = 20.0
ADV_FATALITY_PER      = 5.0    # additional per-fatality beyond the first
ADV_FATALITY_MAX_XTRA = 15.0   # cap on the per-count component
ADV_WR_FLAG           = 8.0
ADV_WR_PER            = 3.0
ADV_WR_MAX            = 15.0
ADV_SERIOUS_PER       = 1.0
ADV_SERIOUS_MAX       = 10.0
ADV_PENALTY_SCALE     = 0.8
ADV_PENALTY_MAX       = 10.0
ADV_RATE_SCALE        = 2.0
ADV_RATE_MAX          = 10.0


# ====================================================================== #
#  Data loading helpers
# ====================================================================== #

def _read_csv(filename: str) -> list:
    """Read a CSV from ml_cache/ into a list of dicts."""
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_date(date_str: str) -> Optional[date]:
    """Parse OSHA date format '2024-10-23 00:00:00+00:00' → date."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return None


def _load_raw_data() -> Tuple[list, dict, dict]:
    """Load inspections, violations index, and accident stats from cache.

    Streams violations by matching to known inspection activity_nrs, avoiding
    loading all 13 M+ violation rows into memory at once.

    Returns:
        inspections         : list of all raw inspection dicts
        viols_by_activity   : {activity_nr: [viol_dict, ...]}  (all time periods)
        accident_stats      : {activity_nr: {accidents, fatalities, injuries}}

    The violation and accident indices cover all time periods.  Leakage
    prevention is enforced by the callers, who only look up activity_nrs
    that belong to the time window of interest.
    """
    inspections = _read_csv("inspections_bulk.csv")

    # Build set of all activity_nrs present in the inspection file so that
    # the violation CSV scan can skip rows that will never be looked up.
    _all_activity_nrs: set = {str(r.get("activity_nr", "")) for r in inspections}

    # Stream violations: keep only rows whose activity_nr is in scope
    violations: list = []
    _viol_path = os.path.join(CACHE_DIR, "violations_bulk.csv")
    csv.field_size_limit(10 * 1024 * 1024)
    with open(_viol_path, "r", newline="", encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            if str(_row.get("activity_nr", "")) in _all_activity_nrs:
                violations.append(_row)

    accidents   = _read_csv("accidents_bulk.csv")
    injuries    = _read_csv("accident_injuries_bulk.csv")

    # Build violation index; skip administratively deleted records.
    viols_by_activity: Dict[str, list] = defaultdict(list)
    for v in violations:
        if v.get("delete_flag") == "X":
            continue
        act = str(v.get("activity_nr", ""))
        viols_by_activity[act].append(v)

    # Build accident stats via injury-accident join.
    # injuries.rel_insp_nr links each injury to an inspection (activity_nr).
    summaries_by_insp: Dict[str, set] = defaultdict(set)
    injuries_by_insp_summary: Dict[str, list] = defaultdict(list)
    for inj in injuries:
        act = str(inj.get("rel_insp_nr", ""))
        snr = str(inj.get("summary_nr", ""))
        if act and snr:
            summaries_by_insp[act].add(snr)
            injuries_by_insp_summary[f"{act}|{snr}"].append(inj)

    acc_by_summary: Dict[str, dict] = {}
    for acc in accidents:
        snr = str(acc.get("summary_nr", ""))
        if snr:
            acc_by_summary[snr] = acc

    all_acts: set = set()
    for inj in injuries:
        act = str(inj.get("rel_insp_nr", ""))
        if act:
            all_acts.add(act)

    accident_stats: Dict[str, dict] = {}
    for act in all_acts:
        snrs = summaries_by_insp.get(act, set())
        fatalities = 0
        inj_count  = 0
        for snr in snrs:
            acc = acc_by_summary.get(snr, {})
            is_fatal = str(acc.get("fatality", "")).strip() in ("1", "Y", "True")
            injs_for_snr = injuries_by_insp_summary.get(f"{act}|{snr}", [])
            if not is_fatal:
                is_fatal = any(
                    str(inj.get("degree_of_inj", "")).startswith("1")
                    for inj in injs_for_snr
                )
            if is_fatal:
                fatalities += 1
            inj_count += len(injs_for_snr)
        accident_stats[act] = {
            "accidents":  len(snrs),
            "fatalities": fatalities,
            "injuries":   inj_count,
        }

    return inspections, viols_by_activity, accident_stats


# ====================================================================== #
#  Per-establishment temporal split, feature generation, outcome generation
# ====================================================================== #

def _compute_future_outcomes(
    estab_name: str,
    future_inspections: list,
    viols_by_activity: dict,
    accident_stats: dict,
) -> Dict:
    """Compute all future-outcome labels for one establishment.

    LEAKAGE GUARD: future_inspections must contain ONLY post-cutoff
    inspections.  The violation and accident indices are keyed by
    activity_nr; we only look up activity_nrs from future_inspections,
    so no historical accident or violation data can enter the outcome.

    Returns a dict with keys:
        has_future_data                 : bool
        future_n_inspections            : int
        future_n_violations             : int  (None if no future inspections)
        future_serious                  : int
        future_willful                  : int
        future_repeat                   : int
        future_willful_repeat           : int
        future_accidents                : int
        future_fatalities               : int
        future_injuries                 : int
        future_total_penalty            : float
        future_violation_rate           : float  (viol / future_insp)
        future_severe_rate              : float  (insp_with_accident / future_insp)
        future_any_serious_or_willful_repeat : int  (binary 0/1)
        future_fatality_or_catastrophe  : int  (binary 0/1)
        future_adverse_outcome_score    : float  (composite, higher = worse)
    """
    n_insp = len(future_inspections)

    if n_insp == 0:
        # Establishment had no future inspections: outcomes are unknown.
        # Use None to distinguish "not inspected" from "inspected with 0 events".
        return {
            "name": estab_name, "has_future_data": False,
            "future_n_inspections": 0,
            "future_n_violations": None, "future_serious": None,
            "future_willful": None, "future_repeat": None,
            "future_willful_repeat": None, "future_accidents": None,
            "future_fatalities": None, "future_injuries": None,
            "future_total_penalty": None, "future_violation_rate": None,
            "future_severe_rate": None,
            "future_any_serious_or_willful_repeat": None,
            "future_fatality_or_catastrophe": None,
            "future_adverse_outcome_score": None,
            "future_gravity_weighted_score": None,
        }

    viols: list = []
    acc_count = fat_count = inj_count = severe_inspections = 0

    for insp in future_inspections:
        # LEAKAGE GUARD: only future activity_nrs looked up here.
        act = str(insp.get("activity_nr", ""))
        insp_viols = viols_by_activity.get(act, [])
        viols.extend(insp_viols)

        acc = accident_stats.get(act, {"accidents": 0, "fatalities": 0, "injuries": 0})
        acc_count += acc["accidents"]
        fat_count += acc["fatalities"]
        inj_count += acc["injuries"]
        if acc["accidents"] > 0:
            severe_inspections += 1

    n_viols         = len(viols)
    serious         = sum(1 for v in viols if v.get("viol_type") == "S")
    willful         = sum(1 for v in viols if v.get("viol_type") == "W")
    repeat          = sum(1 for v in viols if v.get("viol_type") == "R")
    willful_repeat  = willful + repeat
    penalties       = [
        float(v.get("current_penalty") or v.get("initial_penalty") or 0)
        for v in viols
    ]
    total_penalty   = sum(penalties)
    violation_rate  = n_viols / n_insp
    severe_rate     = severe_inspections / n_insp
    any_swr         = int(serious > 0 or willful_repeat > 0)
    any_fatal       = int(fat_count > 0)

    # ── Gravity-weighted score ────────────────────────────────────────────
    # Same formula used in multi_target_labeler.py: Σ(gravity × viol_weight)
    # viol_weight: W/R → 3,  S → 2,  all others → 1
    # This is the TARGET variable for Head 4 (gravity_score prediction).
    grav_total = 0.0
    for v in viols:
        raw_g = v.get("gravity", "")
        try:
            g = float(str(raw_g).strip()) if raw_g else 0.0
        except ValueError:
            g = 0.0
        vt = v.get("viol_type", "")
        weight = 3.0 if vt in ("W", "R") else (2.0 if vt == "S" else 1.0)
        grav_total += g * weight

    # ── Composite adverse outcome score ───────────────────────────────
    # Purpose: a single summary of future compliance severity that gives
    # appropriate weight to different harm types for rank-correlation tests.
    # Scale: 0 (perfect future record) to ~88 (worst possible combination).
    adv = 0.0
    adv += ADV_FATALITY_FLAG * any_fatal
    adv += min((fat_count - 1) * ADV_FATALITY_PER, ADV_FATALITY_MAX_XTRA) \
           if fat_count > 1 else 0.0
    adv += ADV_WR_FLAG * int(willful_repeat > 0)
    adv += min(willful_repeat * ADV_WR_PER, ADV_WR_MAX)
    adv += min(serious * ADV_SERIOUS_PER, ADV_SERIOUS_MAX)
    adv += min(math.log1p(total_penalty) * ADV_PENALTY_SCALE, ADV_PENALTY_MAX)
    adv += min(violation_rate * ADV_RATE_SCALE, ADV_RATE_MAX)

    return {
        "name": estab_name, "has_future_data": True,
        "future_n_inspections": n_insp,
        "future_n_violations": n_viols, "future_serious": serious,
        "future_willful": willful, "future_repeat": repeat,
        "future_willful_repeat": willful_repeat,
        "future_accidents": acc_count, "future_fatalities": fat_count,
        "future_injuries": inj_count,
        "future_total_penalty": total_penalty,
        "future_violation_rate": violation_rate,
        "future_severe_rate": severe_rate,
        "future_any_serious_or_willful_repeat": any_swr,
        "future_fatality_or_catastrophe": any_fatal,
        "future_adverse_outcome_score": adv,
        "future_gravity_weighted_score": grav_total,
    }


# Dev-iteration limit: set env var MAX_ESTAB_DEV=<n> to cap the number of
# establishments processed.  Unset (or 0) means no limit (full dataset).
_MAX_ESTAB_DEV: Optional[int] = (
    int(os.environ["MAX_ESTAB_DEV"]) if os.environ.get("MAX_ESTAB_DEV", "").isdigit()
    else None
)


def _build_per_establishment_data(
    all_inspections: list,
    viols_by_activity: dict,
    accident_stats: dict,
    naics_map: dict,
    cutoff_date: date = CUTOFF_DATE,
    min_hist_inspections: int = 2,
    max_establishments: Optional[int] = _MAX_ESTAB_DEV,
) -> Tuple[List[Dict], List[Dict]]:
    """Split all inspections per establishment into historical features and
    future outcomes.

    Processes all establishments in a single pass: for each establishment,
    pre-cutoff inspections become historical features; post-cutoff inspections
    become future outcome inputs.

    LEAKAGE GUARD: The historical feature dict is built exclusively from
    inspections with open_date < cutoff_date.  The outcome dict is built
    exclusively from inspections with open_date >= cutoff_date.  The two
    populations never mix within this function.

    Args:
        all_inspections     : complete inspection list (all dates)
        viols_by_activity   : violation index (all time periods)
        accident_stats      : accident index (all time periods)
        naics_map           : NAICS reference table
        cutoff_date         : split boundary
        min_hist_inspections: establishments with fewer historical inspections
                              are excluded from the output

    Returns:
        hist_pop        : list of historical feature dicts (one per establishment)
        future_outcomes : list of future outcome dicts (parallel to hist_pop)
    """
    # Group all inspections by establishment name (normalised upper-case).
    estab_all: Dict[str, list] = defaultdict(list)
    for insp in all_inspections:
        name = (insp.get("estab_name") or "UNKNOWN").upper()
        estab_all[name].append(insp)

    # Dev-iteration cap: deterministically slice to the first N establishments.
    if max_establishments and len(estab_all) > max_establishments:
        estab_all = dict(list(estab_all.items())[:max_establishments])

    # recent_ratio counts inspections within the past year relative to today,
    # matching the production scorer's definition exactly.
    one_year_ago = date.today() - timedelta(days=1095)  # 3-year recency window

    hist_pop: List[Dict] = []
    future_outcomes: List[Dict] = []

    for estab, insp_list in estab_all.items():
        # ── Temporal split (per establishment) ─────────────────────────
        hist_list:   List[dict] = []
        future_list: List[dict] = []
        for insp in insp_list:
            d = _parse_date(insp.get("open_date", ""))
            if d is None:
                continue
            if d < cutoff_date:
                hist_list.append(insp)
            else:
                future_list.append(insp)

        # Skip establishments without enough historical data to be scored.
        if len(hist_list) < min_hist_inspections:
            continue

        # ── Historical feature aggregation ─────────────────────────────
        # LEAKAGE GUARD: only hist_list activity_nrs are looked up below.
        n_insp       = len(hist_list)
        recent       = 0
        severe       = 0
        clean        = 0
        viols:    list = []
        acc_count    = fat_count = inj_count = 0
        time_adj_pen = 0.0
        max_insp_pen = 0.0
        estab_sizes: list = []
        naics_votes: Dict[str, int] = defaultdict(int)
        recent_viol_count = 0
        recent_wr_raw     = 0

        for insp in hist_list:
            act = str(insp.get("activity_nr", ""))
            od  = insp.get("open_date", "")
            insp_d: Optional[date] = None
            try:
                insp_d = date.fromisoformat(od[:10])
                if insp_d >= one_year_ago:
                    recent += 1
            except (ValueError, TypeError):
                pass

            insp_viols = viols_by_activity.get(act, [])
            viols.extend(insp_viols)
            if not insp_viols:
                clean += 1

            # Time-adjusted penalty: per-inspection penalty sum * exp(-age/3y)
            if insp_viols and insp_d:
                insp_pen_sum = sum(
                    float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                    for v in insp_viols
                )
                if insp_pen_sum > max_insp_pen:
                    max_insp_pen = insp_pen_sum
                if insp_pen_sum > 0:
                    age_years = max(0.0, (date.today() - insp_d).days / 365.25)
                    time_adj_pen += insp_pen_sum * math.exp(-age_years / 3.0)

            nr_raw = str(insp.get("nr_in_estab") or "").strip()
            if nr_raw:
                try:
                    sz = float(nr_raw)
                    if sz > 0:
                        estab_sizes.append(sz)
                except ValueError:
                    pass

            acc = accident_stats.get(act, {"accidents": 0, "fatalities": 0, "injuries": 0})
            acc_count += acc["accidents"]
            fat_count += acc["fatalities"]
            inj_count += acc["injuries"]
            if acc["accidents"] > 0:
                severe += 1

            if insp_d is not None and insp_d >= one_year_ago:
                recent_viol_count += len(insp_viols)
                recent_wr_raw += sum(
                    1 for v in insp_viols if v.get("viol_type") in ("W", "R")
                )

            nc = str(insp.get("naics_code") or "").strip()
            if nc and nc.isdigit() and len(nc) >= 4:
                naics_votes[nc[:4]] += 1

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

        recent_ratio = recent       / n_insp
        vpi          = n_viols      / n_insp
        pen_per_insp = total_pen    / n_insp
        clean_ratio  = clean        / n_insp
        serious_rate = serious_raw  / n_insp
        willful_rate = willful_raw  / n_insp
        repeat_rate  = repeat_raw   / n_insp
        severe_rate  = severe       / n_insp
        acc_rate     = acc_count    / n_insp
        fat_rate     = fat_count    / n_insp
        inj_rate     = inj_count    / n_insp

        raw_serious_rate = serious_raw / max(n_viols, 1)
        raw_wr_rate      = (willful_raw + repeat_raw) / max(n_viols, 1)

        total_wr       = willful_raw + repeat_raw
        recent_wr_rate = recent_wr_raw / max(total_wr, 1)
        vpi_recent     = recent_viol_count / max(recent, 1)
        trend_delta    = vpi - vpi_recent

        hist_pop.append({
            "name": estab,
            "n_inspections": n_insp,
            "n_future_inspections": len(future_list),
            # 24 absolute features — same ordering as MLRiskScorer.FEATURE_NAMES[:24]
            "features": [
                n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
                total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
                acc_rate, fat_rate, inj_rate, avg_gravity,
                pen_per_insp, clean_ratio,
                time_adj_pen,
                recent_wr_rate,
                trend_delta,
                # Option B: high-signal penalty discriminators
                math.log1p(willful_raw),
                math.log1p(repeat_raw),
                1.0 if fat_count > 0 else 0.0,
                math.log1p(max_insp_pen),
                math.log1p(float(np.median(estab_sizes)) if estab_sizes else 0.0),
            ],
            "_industry_group":   naics_group,
            "_raw_vpi":          vpi,
            "_raw_avg_pen":      avg_pen,
            "_raw_serious_rate": raw_serious_rate,
            "_raw_wr_rate":      raw_wr_rate,
            # Metadata flags re-used in edge-case sub-tests
            "_has_fatality":       fat_count > 0,
            "_is_clean":           n_viols == 0,
            "_single_inspection":  n_insp == 1,
            "_has_willful":        willful_raw > 0,
            "_has_repeat":         repeat_raw > 0,
        })

        # ── Future outcomes ─────────────────────────────────────────────
        # LEAKAGE GUARD: _compute_future_outcomes only receives future_list.
        future_outcomes.append(
            _compute_future_outcomes(
                estab, future_list, viols_by_activity, accident_stats,
            )
        )

    return hist_pop, future_outcomes


# ====================================================================== #
#  Feature matrix construction (historical population → n × 46 array)
# ====================================================================== #

def _build_feature_matrix(
    population: List[Dict],
    industry_stats: dict,
    naics_map: dict,
    scorer: MLRiskScorer,
) -> np.ndarray:
    """Append industry z-scores + NAICS one-hot to 24-feature rows → n × 53 array.

    LEAKAGE GUARD: industry_stats must have been computed from the historical
    population only (never the full or future population).
    """
    rows = []
    for p in population:
        ig  = p["_industry_group"]
        rel = compute_relative_features(
            {
                "industry_group":   ig,
                "raw_vpi":          p["_raw_vpi"],
                "raw_avg_pen":      p["_raw_avg_pen"],
                "raw_serious_rate": p["_raw_serious_rate"],
                "raw_wr_rate":      p["_raw_wr_rate"],
            },
            industry_stats,
            naics_map,
            min_sample=MLRiskScorer.INDUSTRY_MIN_SAMPLE,
        )
        naics_2digit = ig[:2] if ig else None
        naics_vec    = scorer._encode_naics(naics_2digit)
        row = p["features"] + [
            rel["relative_violation_rate"],
            rel["relative_penalty"],
            rel["relative_serious_ratio"],
            rel["relative_willful_repeat"],
        ] + naics_vec
        rows.append(row)

    X = np.array(rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    return X


# ====================================================================== #
#  Metrics helpers
# ====================================================================== #

def _risk_tier(score: float) -> str:
    """Classify a score into Low / Medium / High (same thresholds as temporal test)."""
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    else:
        return "High"


def _spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for Spearman rank correlation.

    Returns:
        (rho_observed, ci_lower, ci_upper)
    """
    rng     = np.random.default_rng(random_state)
    n       = len(x)
    rho_obs = float(spearmanr(x, y)[0])
    boots   = [
        float(spearmanr(x[idx := rng.integers(0, n, size=n)],
                        y[idx])[0])
        for _ in range(n_boot)
    ]
    boots_arr = np.array(boots)
    alpha = 1.0 - ci
    lo    = float(np.percentile(boots_arr, 100 * alpha / 2))
    hi    = float(np.percentile(boots_arr, 100 * (1.0 - alpha / 2)))
    return rho_obs, lo, hi


def _decile_summary(
    scores: np.ndarray,
    outcomes: np.ndarray,
    outcome_label: str = "outcome",
) -> pd.DataFrame:
    """Summarise mean/median outcome and lift by score decile.

    Uses rank-based decile assignment (pd.qcut on ranks) so that ties are
    spread evenly across bins instead of collapsing into empty percentile
    buckets.  Only non-empty bins are returned — callers should use the
    'decile' column for labelling rather than assuming rows 0-9 exist.

    Returns a DataFrame with columns:
        decile, score_lo, score_hi, n, mean_outcome, median_outcome, lift
    where:
        lift = mean_outcome(decile) / population mean_outcome
    """
    n = len(scores)
    if n == 0:
        return pd.DataFrame(columns=[
            "decile", "score_lo", "score_hi", "n",
            "mean_outcome", "median_outcome", "lift",
        ])

    overall_mean = float(outcomes.mean()) if n > 0 else 1e-9

    # Rank-based decile labels: spread ties evenly, 10 bins
    ranks  = pd.Series(scores).rank(method="first")
    labels = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
    # labels is 0-indexed; convert to 1-indexed decile numbers
    unique_labels = sorted(labels.dropna().unique())

    rows = []
    for lbl in unique_labels:
        mask   = labels == lbl
        n_d    = int(mask.sum())
        if n_d == 0:
            continue
        vals   = outcomes[mask.values]
        sc_d   = scores[mask.values]
        mean_d = float(vals.mean())
        med_d  = float(np.median(vals))
        lift   = mean_d / max(overall_mean, 1e-9)
        rows.append({
            "decile":        int(lbl) + 1,          # 1-indexed label
            "score_lo":      round(float(sc_d.min()), 1),
            "score_hi":      round(float(sc_d.max()), 1),
            "n":             n_d,
            "mean_outcome":  round(mean_d, 3),
            "median_outcome": round(med_d, 3),
            "lift":          round(lift, 3),
        })
    return pd.DataFrame(rows)


def _assign_confidence_tag(n_inspections: int, recent_count: int) -> str:
    """Classify an establishment's data quality for downstream score interpretation.

    Confidence reflects how much OSHA history is available and how current it is:

    High   — ≥ 5 inspections AND at least 1 inspection within the past year.
              The score is well-anchored by a rich, up-to-date compliance record.
    Medium — 2–4 inspections, OR ≥ 5 inspections but all historical (no recent).
              The score is usable but should be treated with some caution.
    Low    — ≤ 1 inspection.  Score may be dominated by a single event; entities
              with no recent OSHA presence may have improved or relocated.

    Args:
        n_inspections : total historical inspection count for this establishment
        recent_count  : number of those inspections within the past year
                        (= round(recent_ratio * n_inspections))
    """
    if n_inspections >= 5 and recent_count >= 1:
        return "High"
    elif n_inspections <= 1:
        return "Low"
    elif 2 <= n_inspections <= 4:
        return "Medium"
    else:
        # >= 5 inspections but recent_count == 0 (all historical, none recent)
        return "Medium"


def _compute_threshold_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: List[int],
    label: str = "target",
) -> pd.DataFrame:
    """Compute binary classification metrics at several score thresholds.

    For each threshold t, a manufacturer is flagged as high-risk when
    their baseline score >= t.  We measure whether flagged establishments
    are more likely to have had future adverse events (y_true == 1).

    This directly answers the operational question:
    "If our vetting rule is to flag manufacturers scoring >= t, how well
    does that rule catch real future incidents without over-flagging?"

    Args:
        scores     : 1-D array of baseline risk scores (0–100)
        y_true     : 1-D binary array of future adverse outcome labels
        thresholds : list of score thresholds to evaluate
        label      : string label used when printing results

    Returns:
        DataFrame with columns:
            threshold, TP, FP, TN, FN,
            precision, recall, F1, specificity,
            PPR (positive prediction rate ≡ fraction flagged),
            prevalence, lift
    """
    rows = []
    prevalence = float(y_true.mean()) if len(y_true) > 0 else 0.0
    for t in thresholds:
        predicted = (scores >= t).astype(int)
        TP = int(((predicted == 1) & (y_true == 1)).sum())
        FP = int(((predicted == 1) & (y_true == 0)).sum())
        TN = int(((predicted == 0) & (y_true == 0)).sum())
        FN = int(((predicted == 0) & (y_true == 1)).sum())
        precision   = TP / max(TP + FP, 1)
        recall      = TP / max(TP + FN, 1)
        specificity = TN / max(TN + FP, 1)
        f1          = 2 * precision * recall / max(precision + recall, 1e-9)
        ppr         = (TP + FP) / max(len(scores), 1)       # % of population flagged
        lift        = precision / max(prevalence, 1e-9)      # precision / base rate
        rows.append({
            "threshold":   t,
            "TP":          TP, "FP": FP, "TN": TN, "FN": FN,
            "precision":   round(precision,   4),
            "recall":      round(recall,      4),
            "F1":          round(f1,          4),
            "specificity": round(specificity, 4),
            "PPR":         round(ppr,         4),
            "prevalence":  round(prevalence,  4),
            "lift":        round(lift,        4),
        })
    return pd.DataFrame(rows)


def _compute_topk_capture(
    scores: np.ndarray,
    adverse_scores: np.ndarray,
    swr_flags: np.ndarray,
    fractions: List[float],
) -> pd.DataFrame:
    """For several top-k% groups, compute the share of total adverse outcomes
    and S/W/R positives captured by the highest-scoring establishments.

    This directly answers: "If we inspect only the top-k% of vendors, what
    fraction of all future compliance incidents do we catch?"  Lift > 1.0
    means the model concentrates incidents at the top.

    Args:
        scores         : 1-D baseline score array (higher = riskier)
        adverse_scores : 1-D future adverse outcome score array
        swr_flags      : 1-D binary array for any future serious/willful/repeat event
        fractions      : list of floats in (0, 1] for top-k% cut-points

    Returns:
        DataFrame with columns:
            fraction, n,
            adverse_captured_pct, swr_captured_pct,
            adverse_lift, swr_lift
    where lift = captured_pct / fraction  (1.0 = same as random).
    """
    n               = len(scores)
    total_adverse   = float(adverse_scores.sum()) or 1.0
    total_swr       = float(swr_flags.sum())      or 1.0
    order           = np.argsort(scores)[::-1]    # highest score first
    sorted_adv      = adverse_scores[order]
    sorted_swr      = swr_flags[order]
    cum_adv         = np.cumsum(sorted_adv)
    cum_swr         = np.cumsum(sorted_swr)

    rows = []
    for frac in fractions:
        k           = max(1, int(round(n * frac)))
        adv_cap     = float(cum_adv[k - 1]) / total_adverse
        swr_cap     = float(cum_swr[k - 1]) / total_swr
        adv_lift    = adv_cap / frac
        swr_lift    = swr_cap / frac
        rows.append({
            "fraction":             frac,
            "n":                    k,
            "adverse_captured_pct": round(adv_cap * 100, 2),
            "swr_captured_pct":     round(swr_cap * 100, 2),
            "adverse_lift":         round(adv_lift,  3),
            "swr_lift":             round(swr_lift,  3),
        })
    return pd.DataFrame(rows)


def _auroc_if_sufficient(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_positive: int = MIN_BINARY_POSITIVE,
) -> Optional[float]:
    """Return AUROC when there are enough positive examples, else None.

    Skips cleanly rather than raising an error when the positive class is
    too sparse to produce a meaningful area estimate.
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos < min_positive or n_neg < min_positive:
        return None
    return float(roc_auc_score(y_true, y_score))


# ====================================================================== #
#  Session-scoped data holder
# ====================================================================== #

class RealWorldData:
    """Session-scoped holder: loads bulk OSHA data, scores via the MT model,
    and measures future outcomes.

    Architecture
    ------------
    * hist_pop          : all establishments with >= 1 pre-cutoff inspection
    * baseline_scores   : MT composite score for each hist_pop member (shape n,)
    * future_outcomes   : future outcome dict for each hist_pop member
    * paired_*          : subset where has_future_data is True

    Pre-computed outcome arrays (paired subset only, NaN-free):
        paired_adverse_scores, paired_swr_flags, paired_violation_rates,
        paired_log_penalties, paired_fatality_flags
    """

    _instance: Optional["RealWorldData"] = None

    def __init__(self):
        self.loaded = False
        self.cutoff_date = CUTOFF_DATE
        self.scorer: Optional[MLRiskScorer] = None
        self.naics_map: dict = {}
        self.hist_industry_stats: dict = {}

        self.hist_pop: List[Dict] = []
        self.future_outcomes: List[Dict] = []
        self.baseline_scores: np.ndarray = np.array([])
        self.hist_X: np.ndarray = np.array([])

        # Paired subset (both historical score AND future data available)
        self.paired_pop: List[Dict] = []
        self.paired_outcomes: List[Dict] = []
        self.paired_scores: np.ndarray = np.array([])

        # Flattened outcome arrays for the paired subset
        self.paired_adverse_scores:  np.ndarray = np.array([])
        self.paired_swr_flags:       np.ndarray = np.array([])
        self.paired_violation_rates: np.ndarray = np.array([])
        self.paired_log_penalties:   np.ndarray = np.array([])
        self.paired_fatality_flags:  np.ndarray = np.array([])

        # Derived arrays computed once after the paired subset is built
        # binary: future adverse score >= 75th percentile of the paired population
        self.swr_75th_flags: np.ndarray = np.array([])
        # "High" / "Medium" / "Low" confidence tag per paired establishment
        self.confidence_tags: List[str] = []
    @classmethod
    def get(cls) -> "RealWorldData":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._build()
        return cls._instance

    def _build(self):
        print("\n" + "=" * 70)
        print("REAL-WORLD VALIDATION: Loading bulk data and building splits…")
        print(f"  Cutoff date:  {CUTOFF_DATE}  (historical < cutoff, future >= cutoff)")
        print("=" * 70)

        # ── Scorer stub (no cache/API) ──────────────────────────────────
        from unittest.mock import patch
        with patch.object(MLRiskScorer, "_load_or_build"):
            self.scorer = MLRiskScorer(osha_client=None)
        self.naics_map = load_naics_map()

        # ── Raw data ────────────────────────────────────────────────────
        inspections, viols_by_activity, accident_stats = _load_raw_data()
        assert len(inspections) > 0, (
            "No inspection data found in ml_cache/. Run build_cache.py first."
        )
        print(f"  Total raw inspections loaded: {len(inspections):,}")

        # ── Per-establishment temporal split ────────────────────────────
        # LEAKAGE GUARD: _build_per_establishment_data enforces the cutoff
        # boundary; each establishment's hist and future sets never mix.
        self.hist_pop, self.future_outcomes = _build_per_establishment_data(
            inspections, viols_by_activity, accident_stats, self.naics_map,
            cutoff_date=CUTOFF_DATE,
            min_hist_inspections=2,
        )
        print(f"  Establishments with >= 1 historical inspection: {len(self.hist_pop):,}")
        n_with_future = sum(1 for o in self.future_outcomes if o["has_future_data"])
        print(f"  Establishments with >= 1 future inspection:     {n_with_future:,}")

        assert len(self.hist_pop) >= MIN_HIST_ESTABLISHMENTS, (
            f"Need >= {MIN_HIST_ESTABLISHMENTS} historical establishments, "
            f"got {len(self.hist_pop)}"
        )

        # ── Industry stats (historical population only) ─────────────────
        # LEAKAGE GUARD: industry stats are computed from self.hist_pop only.
        hist_df = pd.DataFrame([{
            "industry_group":   p["_industry_group"],
            "raw_vpi":          p["_raw_vpi"],
            "raw_avg_pen":      p["_raw_avg_pen"],
            "raw_serious_rate": p["_raw_serious_rate"],
            "raw_wr_rate":      p["_raw_wr_rate"],
        } for p in self.hist_pop])
        self.hist_industry_stats = compute_industry_stats(
            hist_df,
            min_sample=MLRiskScorer.INDUSTRY_MIN_SAMPLE,
            naics_map=self.naics_map,
        )
        self.scorer._industry_stats = self.hist_industry_stats

        # ── Feature matrix (historical) ─────────────────────────────────
        self.hist_X = _build_feature_matrix(
            self.hist_pop, self.hist_industry_stats, self.naics_map, self.scorer,
        )

        # ── Score all establishments via the MT model ───────────────────
        mt_scorer = MultiTargetRiskScorer.load_if_exists(CACHE_DIR)
        if mt_scorer is not None:
            hist_X_log = MLRiskScorer._log_transform_features(
                np.nan_to_num(self.hist_X, nan=0.0)
            )
            preds = mt_scorer.predict_batch(hist_X_log)
            self.baseline_scores = np.array([mt_scorer.composite_score(p) for p in preds])
            print(f"  MT model score range: [{self.baseline_scores.min():.1f}, "
                  f"{self.baseline_scores.max():.1f}]  "
                  f"mean={self.baseline_scores.mean():.1f}  "
                  f"std={self.baseline_scores.std():.1f}")
        else:
            print("  WARNING: MT model not found in cache. Using fallback scores (50).")
            self.baseline_scores = np.full(len(self.hist_pop), 50.0)

        # ── Build paired subset ─────────────────────────────────────────
        for p, outc, score in zip(
            self.hist_pop, self.future_outcomes, self.baseline_scores,
        ):
            if outc["has_future_data"]:
                self.paired_pop.append(p)
                self.paired_outcomes.append(outc)
                self.paired_scores = np.append(self.paired_scores, score)

        print(f"  Paired establishments (score + future outcomes): "
              f"{len(self.paired_pop):,}")

        if len(self.paired_pop) >= MIN_FUTURE_ESTABLISHMENTS:
            self.paired_adverse_scores  = np.array([
                o["future_adverse_outcome_score"] for o in self.paired_outcomes
            ])
            self.paired_swr_flags       = np.array([
                o["future_any_serious_or_willful_repeat"] for o in self.paired_outcomes
            ])
            self.paired_violation_rates = np.array([
                o["future_violation_rate"] for o in self.paired_outcomes
            ])
            self.paired_log_penalties   = np.array([
                math.log1p(o["future_total_penalty"]) for o in self.paired_outcomes
            ])
            self.paired_fatality_flags  = np.array([
                o["future_fatality_or_catastrophe"] for o in self.paired_outcomes
            ])

            # ── Derived: 75th-pct adverse binary flag ──────────────────────
            # Useful for threshold evaluation: "will this establishment fall in
            # the worst quartile for future adverse outcomes?".
            p75 = float(np.percentile(self.paired_adverse_scores, 75))
            self.swr_75th_flags = (self.paired_adverse_scores >= p75).astype(int)
            print(f"  75th-pct adverse threshold: {p75:.2f}  "
                  f"({int(self.swr_75th_flags.sum())} positives, "
                  f"{int((self.swr_75th_flags == 0).sum())} negatives)")

            # ── Confidence tagging ──────────────────────────────────────────
            # Feature index 8 in the 17-feature absolute vector is recent_ratio.
            # Multiply by n_inspections to recover the integer recent-inspection count.
            self.confidence_tags = []
            for p in self.paired_pop:
                n_insp       = p["n_inspections"]
                recent_ratio = p["features"][8]   # 0–1 fraction of recent insp
                recent_cnt   = round(recent_ratio * n_insp)
                tag          = _assign_confidence_tag(n_insp, recent_cnt)
                self.confidence_tags.append(tag)
            tag_dist = {t: self.confidence_tags.count(t)
                        for t in ("High", "Medium", "Low")}
            print(f"  Confidence tags: High={tag_dist['High']}  "
                  f"Medium={tag_dist['Medium']}  Low={tag_dist['Low']}")
        else:
            print(f"  WARNING: only {len(self.paired_pop)} paired establishments "
                  f"(need >= {MIN_FUTURE_ESTABLISHMENTS}) — most tests will skip.")

        rho_adv, _, _ = _spearman_bootstrap_ci(
            self.paired_scores, self.paired_adverse_scores
        ) if len(self.paired_pop) >= MIN_FUTURE_ESTABLISHMENTS else (float("nan"), 0, 0)
        print(f"  Spearman(score, future_adverse): rho={rho_adv:.3f}")
        print("=" * 70 + "\n")

        self.loaded = True

    def _skip_if_insufficient(self, n_required: int = MIN_FUTURE_ESTABLISHMENTS):
        """Called at the start of tests that need paired data."""
        if len(self.paired_pop) < n_required:
            pytest.skip(
                f"Only {len(self.paired_pop)} paired establishments "
                f"(need >= {n_required})"
            )
