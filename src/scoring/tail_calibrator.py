"""Isotonic regression calibrator for the tail of the risk score distribution.

The raw GBR output correctly ranks establishments but its absolute values
compress above ~40: companies scoring 40-90 all have similar predicted
adverse outcomes.  An isotonic (monotone) calibration layer:

1. Learns the mapping raw_score → mean_future_adverse from held-out
   temporal validation data (MUST be post-cutoff to avoid leakage).
2. Re-normalises the calibrated values back to the original 0-100 range so
   the user-visible score scale and API contract are unchanged.
3. Is fully monotone — it cannot reorder any two establishments; it only
   stretches or compresses the scale to better reflect actual future risk.

Usage
-----
Calibrator is fitted inside ``MLRiskScorer._train()`` using a temporal
held-out split, saved to ``ml_cache/tail_calibrator.pkl``, and loaded
automatically when the model is loaded.  ``score_establishments()`` calls
``calibrate()`` on every raw prediction before returning it.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional

from sklearn.isotonic import IsotonicRegression


class TailCalibrator:
    """Isotonic regression wrapper with 0-100 re-normalisation.

    Parameters
    ----------
    out_of_bounds : str
        How to handle scores outside the training range.
        ``"clip"`` (default) clamps to the nearest trained endpoint, which
        is safe for rare out-of-distribution scores at inference time.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self._iso: Optional[IsotonicRegression] = None
        self._out_min: float = 0.0
        self._out_max: float = 100.0
        self._out_of_bounds = out_of_bounds

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        raw_scores: np.ndarray,
        future_outcomes: np.ndarray,
        bin_width: float = 5.0,
    ) -> "TailCalibrator":
        """Fit the calibrator on (score, future_outcome) pairs.

        To avoid noise from individual data points the calibration is done on
        *binned* means: scores are bucketed into ``bin_width``-wide bins, and
        each bin's mean future outcome is used as the regression target.  This
        gives the isotonic fit a smoother signal and is robust to outliers.

        Parameters
        ----------
        raw_scores : array-like of float
            Model raw scores (0-100) from temporal validation data.
        future_outcomes : array-like of float
            Observed future adverse-outcome scores for the same establishments.
        bin_width : float
            Width of score bins used to compute mean outcomes before fitting.
        """
        raw_scores = np.asarray(raw_scores, dtype=float)
        future_outcomes = np.asarray(future_outcomes, dtype=float)

        if len(raw_scores) < 10:
            # Not enough data to calibrate — leave unfitted
            return self

        # Bin scores and compute mean outcome per bin
        bins = np.arange(0, 100 + bin_width, bin_width)
        bin_idx = np.digitize(raw_scores, bins) - 1  # 0-indexed
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

        bin_scores: list[float] = []
        bin_means: list[float] = []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() >= 3:  # require at least 3 observations per bin
                bin_scores.append(bins[b] + bin_width / 2)  # bin mid-point
                bin_means.append(float(np.mean(future_outcomes[mask])))

        if len(bin_scores) < 3:
            return self

        x = np.array(bin_scores)
        y = np.array(bin_means)

        self._iso = IsotonicRegression(increasing=True, out_of_bounds=self._out_of_bounds)
        self._iso.fit(x, y)

        # Store output range for re-normalisation.
        # Use the actual min/max of the binned-mean predictions rather than
        # percentile clipping: the inputs are already bin averages (smoothed),
        # so there are no outlier bins to guard against, and clipping the 1st/99th
        # percentile was actively compressing the extreme tail into a wall.
        calibrated = self._iso.predict(x)
        self._out_min = float(calibrated.min())
        self._out_max = float(calibrated.max())

        return self

    # ------------------------------------------------------------------ #
    #  Calibration at inference
    # ------------------------------------------------------------------ #
    def calibrate(self, raw_score: float) -> float:
        """Map a single raw score to a calibrated 0-100 value.

        If the calibrator has not been fitted (e.g., insufficient training
        data) it returns *raw_score* unchanged so inference always succeeds.
        """
        if self._iso is None:
            return raw_score

        cal = float(self._iso.predict([raw_score])[0])

        # Re-normalise to 0-100 while preserving monotonicity
        span = self._out_max - self._out_min
        if span < 1e-6:
            return raw_score
        normalised = (cal - self._out_min) / span * 100.0
        return float(np.clip(normalised, 0.0, 100.0))

    def calibrate_array(self, raw_scores: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`calibrate`."""
        if self._iso is None:
            return np.asarray(raw_scores, dtype=float)

        cal = self._iso.predict(np.asarray(raw_scores, dtype=float))
        span = self._out_max - self._out_min
        if span < 1e-6:
            return np.asarray(raw_scores, dtype=float)
        normalised = (cal - self._out_min) / span * 100.0
        return np.clip(normalised, 0.0, 100.0)

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "TailCalibrator":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected TailCalibrator, got {type(obj)}")
        return obj
