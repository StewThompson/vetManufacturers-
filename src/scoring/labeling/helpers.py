"""helpers.py — Shared CSV-streaming and feature-aggregation helpers.

Used by both temporal_labeler and multi_target_labeler to avoid duplication.
All functions are pure (no global state) and operate on dicts from bulk CSVs.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Adverse-outcome formula weights ──────────────────────────────────────────
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

    Only the required fields are retained to keep memory bounded.
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
    """Stream violations CSV, keeping only rows whose activity_nr is in the set."""
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


def _aggregate_hist_features(
    hist_inspections: list,
    viol_index: Dict[str, list],
    acc_fatals: Dict[str, int],
    one_year_ago: date,
) -> Tuple[list, Optional[str], float, float, float, float]:
    """Compute the 25 absolute features for a set of historical inspections.

    Returns (features_25, naics_group, vpi, avg_pen, raw_serious_rate, raw_wr_rate)
    where naics_group is the majority-vote 4-digit NAICS code (or None).

    LEAKAGE GUARD: hist_inspections must only contain pre-cutoff records.
    """
    n_insp = len(hist_inspections)
    if n_insp == 0:
        return ([], None, 0.0, 0.0, 0.0, 0.0)

    # Anchor for exponential decay: cutoff date (one_year_ago + 3 years)
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
            inj_count += fat

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

    estab_sizes_clean = [s for s in estab_sizes if s > 0]
    median_estab_size = float(np.median(estab_sizes_clean)) if estab_sizes_clean else 0.0

    features_25 = [
        n_insp, n_viols,
        serious_rate, willful_rate, repeat_rate,
        total_pen, avg_pen, max_pen,
        recent_ratio, severe_rate, vpi,
        acc_rate, fat_rate, inj_rate, avg_gravity,
        pen_per_insp, clean_ratio,
        time_adj_pen,
        recent_wr_rate,
        trend_delta,
        math.log1p(willful_raw),
        math.log1p(repeat_raw),
        1.0 if fat_count > 0 else 0.0,
        math.log1p(max_insp_pen),
        math.log1p(median_estab_size),
    ]

    raw_serious_rate = serious_raw / max(n_viols, 1)
    raw_wr_rate      = (willful_raw + repeat_raw) / max(n_viols, 1)

    return features_25, naics_group, vpi, avg_pen, raw_serious_rate, raw_wr_rate


def _stratified_sample(
    indices: List[int],
    strata_keys: List[str],
    sample_size: int,
    rng: np.random.Generator,
    min_per_stratum: int = 5,
) -> List[int]:
    """Stratified proportional sampling.

    Allocates sample_size proportionally across strata, floored at
    min_per_stratum per stratum when the stratum is large enough.
    Returns a flat list of row indices.
    """
    assert len(indices) == len(strata_keys)
    n_total = len(indices)
    if n_total <= sample_size:
        return list(indices)

    groups: Dict[str, list] = defaultdict(list)
    for idx, key in zip(indices, strata_keys):
        groups[key].append(idx)

    result: List[int] = []
    for key, grp in groups.items():
        n_grp = len(grp)
        alloc = max(min_per_stratum, int(round(n_grp / n_total * sample_size)))
        alloc = min(alloc, n_grp)
        result.extend(rng.choice(grp, size=alloc, replace=False).tolist())

    if len(result) > sample_size:
        result = rng.choice(result, size=sample_size, replace=False).tolist()
    elif len(result) < sample_size:
        remaining = list(set(indices) - set(result))
        extra = min(sample_size - len(result), len(remaining))
        if extra > 0:
            result.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return result
