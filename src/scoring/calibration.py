"""calibration.py — Probability calibration evaluation utilities.

Provides functions for assessing the quality of calibrated probability
outputs from binary classifiers, including reliability curves, Brier
scores, and expected calibration error (ECE).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability (calibration) curve.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth labels (0 or 1).
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    mean_predicted : ndarray of shape (n_nonempty_bins,)
        Mean predicted probability in each non-empty bin.
    mean_observed : ndarray of shape (n_nonempty_bins,)
        Observed fraction of positives in each non-empty bin.
    counts : ndarray of shape (n_nonempty_bins,)
        Number of samples in each non-empty bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted, mean_observed, counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi < 1.0:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n = mask.sum()
        if n == 0:
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        mean_observed.append(float(y_true[mask].mean()))
        counts.append(int(n))

    return (
        np.array(mean_predicted),
        np.array(mean_observed),
        np.array(counts),
    )


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and binary labels.

    Lower is better.  Range [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Skill Score (BSS): improvement over a climatological baseline.

    BSS = 1 - Brier(model) / Brier(climatology)

    where climatology predicts the prevalence for every sample.
    Range (-inf, 1]; 0 = no skill, 1 = perfect.
    """
    bs_model = brier_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    bs_ref = prevalence * (1.0 - prevalence)
    if bs_ref < 1e-12:
        return 0.0
    return 1.0 - bs_model / bs_ref


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of |mean_predicted - mean_observed| across bins,
    weighted by the fraction of samples in each bin.
    """
    mean_pred, mean_obs, counts = reliability_curve(y_true, y_prob, n_bins)
    if len(counts) == 0:
        return 0.0
    total = counts.sum()
    return float(np.sum(counts / total * np.abs(mean_pred - mean_obs)))
