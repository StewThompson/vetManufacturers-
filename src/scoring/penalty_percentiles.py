"""penalty_percentiles.py — NAICS-stratified penalty tier thresholds.

Computes P75 (moderate), P90 (large), and P95 (extreme) penalty thresholds
per 2-digit NAICS sector from historical training data.  These thresholds are
stored in ml_cache/penalty_percentiles.json and consumed by the multi-target
labeler to assign binary ``is_large_penalty`` labels without data leakage.

Leakage rule
------------
Percentiles MUST be computed using only training-fold (pre-cutoff) data and
then applied to test-fold (post-cutoff) penalty values.  The callers enforce
this; this module contains no split logic.

Fallback
--------
If a NAICS-2-digit group has fewer than ``min_group_n`` samples, the function
falls back to the global (all-NAICS) percentiles.  This prevents threshold
instability for rare industry codes.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


PERCENTILES = [75, 90, 95]
PERCENTILE_KEYS = ["p75", "p90", "p95"]
DEFAULT_MIN_GROUP_N = 50
CACHE_FILENAME = "penalty_percentiles.json"


# ── Public API ──────────────────────────────────────────────────────────────

def compute_penalty_percentiles(
    violations_df: pd.DataFrame,
    naics_col: str = "naics_2digit",
    penalty_col: str = "penalty_amount",
    min_group_n: int = DEFAULT_MIN_GROUP_N,
) -> Dict[str, Dict[str, float]]:
    """Compute P75/P90/P95 penalty thresholds stratified by 2-digit NAICS.

    Parameters
    ----------
    violations_df : pd.DataFrame
        Must contain at least ``naics_col`` (string, 2-digit sector code like
        "33") and ``penalty_col`` (numeric penalty amount ≥ 0).
    naics_col : str
        Column name for 2-digit NAICS code.
    penalty_col : str
        Column name for penalty amount.
    min_group_n : int
        Groups with fewer than this many samples fall back to global
        percentiles.

    Returns
    -------
    dict mapping naics_2digit (str) or ``"__global__"`` → ``{p75, p90, p95}``.
    All establishments — including those with missing NAICS — can look up
    their threshold via ``"__global__"``.
    """
    df = violations_df[[naics_col, penalty_col]].copy()
    df[penalty_col] = pd.to_numeric(df[penalty_col], errors="coerce").fillna(0.0)
    # Only use rows with a positive penalty to avoid zero-inflation skewing
    # the threshold down.  Zero-penalty rows represent clean inspections or
    # other-than-serious violations; they should not define "large penalty".
    positive = df[df[penalty_col] > 0]

    # Global fallback thresholds (computed first, used for small groups)
    global_thresh = _compute_thresholds(positive[penalty_col].values)

    result: Dict[str, Dict[str, float]] = {"__global__": global_thresh}

    if positive.empty:
        return result

    # Per-NAICS thresholds — vectorized groupby
    group_counts = positive.groupby(naics_col)[penalty_col].count()
    large_groups = group_counts[group_counts >= min_group_n].index.tolist()

    for naics in large_groups:
        vals = positive.loc[positive[naics_col] == naics, penalty_col].values
        result[str(naics)] = _compute_thresholds(vals)

    return result


def label_penalty_tiers(
    df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    naics_col: str = "naics_2digit",
    penalty_col: str = "penalty_amount",
) -> pd.DataFrame:
    """Assign is_moderate/large/extreme_penalty columns (no data loops).

    Computes ``log_penalty`` as well.  The function is vectorized — no
    per-row iteration.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``naics_col`` and ``penalty_col``.
    thresholds : dict
        Output of ``compute_penalty_percentiles()``.
    naics_col, penalty_col : str
        Column names (same as used when computing thresholds).

    Returns
    -------
    DataFrame with new columns:
        is_moderate_penalty (int 0/1 ≥ P75),
        is_large_penalty    (int 0/1 ≥ P90),
        is_extreme_penalty  (int 0/1 ≥ P95),
        log_penalty         (float, log1p-transformed penalty).
    """
    out = df.copy()
    out[penalty_col] = pd.to_numeric(out[penalty_col], errors="coerce").fillna(0.0)
    out["log_penalty"] = np.log1p(out[penalty_col])

    global_thresh = thresholds["__global__"]

    # Map each row's NAICS to its thresholds; fall back to global when missing
    naics_series = out[naics_col].astype(str)
    p75_thresh = naics_series.map(
        lambda n: thresholds.get(n, global_thresh)["p75"]
    )
    p90_thresh = naics_series.map(
        lambda n: thresholds.get(n, global_thresh)["p90"]
    )
    p95_thresh = naics_series.map(
        lambda n: thresholds.get(n, global_thresh)["p95"]
    )

    out["is_moderate_penalty"] = (out[penalty_col] >= p75_thresh).astype(int)
    out["is_large_penalty"]    = (out[penalty_col] >= p90_thresh).astype(int)
    out["is_extreme_penalty"]  = (out[penalty_col] >= p95_thresh).astype(int)

    return out


def save_percentiles(thresholds: Dict[str, Dict[str, float]], path: str) -> None:
    """Serialise penalty thresholds to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def load_percentiles(path: str) -> Dict[str, Dict[str, float]]:
    """Load penalty thresholds from JSON.  Returns empty global fallback if missing."""
    if not os.path.exists(path):
        return {"__global__": {"p75": 5000.0, "p90": 15000.0, "p95": 40000.0}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup_threshold(
    thresholds: Dict[str, Dict[str, float]],
    naics_2digit: Optional[str],
    percentile: str = "p90",
) -> float:
    """Return the penalty threshold for a given NAICS sector and percentile.

    Falls back to ``"__global__"`` when the sector is missing or small.
    ``percentile`` must be one of ``"p75"``, ``"p90"``, ``"p95"``.
    """
    key = str(naics_2digit) if naics_2digit else "__global__"
    entry = thresholds.get(key, thresholds.get("__global__", {}))
    if not entry:
        # Ultimate fallback: hardcoded OSHA-typical thresholds
        fallback = {"p75": 5_000.0, "p90": 15_000.0, "p95": 40_000.0}
        return fallback.get(percentile, 15_000.0)
    return entry.get(percentile, entry.get("p90", 15_000.0))


# ── Internal helpers ────────────────────────────────────────────────────────

def _compute_thresholds(values: np.ndarray) -> Dict[str, float]:
    """Return {p75, p90, p95} for a 1-D array of positive penalty values."""
    if len(values) == 0:
        return {"p75": 5_000.0, "p90": 15_000.0, "p95": 40_000.0}
    arr = np.asarray(values, dtype=float)
    pcts = np.percentile(arr, PERCENTILES)
    return dict(zip(PERCENTILE_KEYS, [float(v) for v in pcts]))
