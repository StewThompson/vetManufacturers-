"""ranking_metrics.py — Ranking-focused evaluation metrics.

Provides decile lift, top-k precision, KS statistic, and gain chart
computations for evaluating risk stratification models.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def decile_lift(
    scores: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute lift per score decile.

    Sorts by ``scores`` descending, splits into ``n_bins`` equal-size groups,
    and returns ``mean_outcome_in_bin / population_mean_outcome`` for each.

    Returns
    -------
    lifts : ndarray of shape (n_bins,)
        Lift values ordered from highest-score decile to lowest.
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    pop_mean = outcomes.mean()
    if pop_mean < 1e-12:
        return np.ones(n_bins)

    order = np.argsort(-scores)  # descending
    sorted_outcomes = outcomes[order]

    bin_size = len(scores) // n_bins
    lifts = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(scores)
        bin_mean = sorted_outcomes[start:end].mean()
        lifts.append(bin_mean / pop_mean)

    return np.array(lifts)


def decile_means(
    scores: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean score and mean outcome per decile (highest-score first).

    Returns
    -------
    mean_scores : ndarray of shape (n_bins,)
    mean_outcomes : ndarray of shape (n_bins,)
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_outcomes = outcomes[order]

    bin_size = len(scores) // n_bins
    mean_s, mean_o = [], []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(scores)
        mean_s.append(sorted_scores[start:end].mean())
        mean_o.append(sorted_outcomes[start:end].mean())

    return np.array(mean_s), np.array(mean_o)


def top_k_precision(
    scores: np.ndarray,
    binary_outcomes: np.ndarray,
    k_fracs: List[float] = None,
) -> Dict[str, float]:
    """Precision in the top-k% of scored entities.

    Parameters
    ----------
    scores : array of risk scores (higher = riskier)
    binary_outcomes : array of 0/1 event labels
    k_fracs : fractions to evaluate (default: [0.05, 0.10, 0.20])

    Returns
    -------
    dict mapping 'top_5%', 'top_10%', etc. to precision values.
    """
    if k_fracs is None:
        k_fracs = [0.05, 0.10, 0.20]

    scores = np.asarray(scores, dtype=float)
    binary_outcomes = np.asarray(binary_outcomes, dtype=float)
    order = np.argsort(-scores)

    result = {}
    for frac in k_fracs:
        k = max(1, int(len(scores) * frac))
        top_idx = order[:k]
        precision = float(binary_outcomes[top_idx].mean())
        label = f"top_{int(frac * 100)}%"
        result[label] = precision

    return result


def ks_statistic(
    scores: np.ndarray,
    binary_outcomes: np.ndarray,
) -> float:
    """Kolmogorov-Smirnov statistic for binary classification.

    Measures the maximum separation between the cumulative distributions
    of scores for positives vs negatives.  Range [0, 1]; higher is better.
    """
    scores = np.asarray(scores, dtype=float)
    binary_outcomes = np.asarray(binary_outcomes, dtype=int)

    pos_scores = scores[binary_outcomes == 1]
    neg_scores = scores[binary_outcomes == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    # Build combined sorted thresholds
    all_vals = np.sort(np.unique(np.concatenate([pos_scores, neg_scores])))

    pos_cdf = np.searchsorted(np.sort(pos_scores), all_vals, side="right") / len(pos_scores)
    neg_cdf = np.searchsorted(np.sort(neg_scores), all_vals, side="right") / len(neg_scores)

    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def gain_chart(
    scores: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative gains chart data.

    Parameters
    ----------
    scores : array of risk scores
    binary_outcomes : array of 0/1 event labels
    n_bins : int

    Returns
    -------
    cumulative_gain : ndarray of shape (n_bins+1,)
        Fraction of all positives captured at each decile (starts at 0).
    random_baseline : ndarray of shape (n_bins+1,)
        Expected gain under random scoring.
    """
    scores = np.asarray(scores, dtype=float)
    binary_outcomes = np.asarray(binary_outcomes, dtype=float)
    total_pos = binary_outcomes.sum()

    if total_pos < 1:
        fracs = np.linspace(0, 1, n_bins + 1)
        return fracs, fracs

    order = np.argsort(-scores)
    sorted_outcomes = binary_outcomes[order]

    bin_size = len(scores) // n_bins
    cum_gain = [0.0]
    for i in range(n_bins):
        start = 0
        end = (i + 1) * bin_size if i < n_bins - 1 else len(scores)
        cum_gain.append(sorted_outcomes[:end].sum() / total_pos)

    random_baseline = np.linspace(0, 1, n_bins + 1)
    return np.array(cum_gain), random_baseline
