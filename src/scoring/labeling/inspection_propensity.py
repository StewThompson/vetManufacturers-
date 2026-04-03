"""inspection_propensity.py — Inverse Probability Weighting (IPW) model for
inspection-selection bias correction.

OSHA establishments are not inspected at random: industry (NAICS), size, and
prior inspection history all strongly predict whether an establishment receives
a follow-up inspection.  Because ``build_multi_target_sample`` retains only
*paired* establishments (≥ ``min_hist_insp`` historical AND ≥ 1 future
inspection), the training set over-represents high-activity, large employers —
particularly in NAICS 33 (manufacturing) which is inspected ~8× more often
than NAICS 54 (professional services).

This module fits a logistic propensity model

    P(future_inspection = 1 | log_hist_count, NAICS)

over the combined paired (y=1) + unpaired (y=0) population and returns
clip(1 / P, 1, max_weight) IPW weights for each paired row, so that the
GBM heads learn from a sample that better approximates the full employer
population rather than the inspection-conditional subset.

Features (industry-stratified)
-------------------------------
  * log1p(hist_insp_count)         — prior inspection volume proxy
  * NAICS 2-digit one-hot (N + 1)  — industry × inspection rate confound;
                                     the +1 dim captures unknown / unmapped codes

The model is intentionally lightweight (LogisticRegression, C=1.0) because
its purpose is bias correction, not accurate classification.  The propensity
scores only need to rank establishments coarsely by their inspection
probability within each industry stratum.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class InspectionPropensityModel:
    """Logistic propensity model P(reinspected | hist_insp_count, NAICS).

    Parameters
    ----------
    naics_sectors : list[str]
        Ordered list of 2-digit NAICS sector codes (e.g. MLRiskScorer.NAICS_SECTORS).
        Defines the one-hot encoding dimension; must be identical between
        fit() and ipw_weights() calls.
    max_weight : float
        Upper clip for computed IPW weights (default 5.0).  Prevents extreme
        upweighting of near-zero-probability establishments.
    """

    def __init__(self, naics_sectors: List[str], max_weight: float = 5.0) -> None:
        self._naics_sectors = list(naics_sectors)
        self._max_weight = float(max_weight)
        self._n_features: int = 1 + len(naics_sectors) + 1  # log1p + one-hot + unknown
        self._model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _encode(self, hist_count: int, naics_2d: Optional[str]) -> List[float]:
        """Build a single-row feature vector [log1p_count, *one_hot, unknown]."""
        vec = [float(np.log1p(hist_count))]
        for sector in self._naics_sectors:
            vec.append(1.0 if naics_2d == sector else 0.0)
        # Unknown dimension: 1 when naics_2d is None or not in NAICS_SECTORS
        vec.append(0.0 if (naics_2d and naics_2d in self._naics_sectors) else 1.0)
        return vec

    @staticmethod
    def _most_common_naics(history_rows: list) -> Optional[str]:
        """Return the 2-digit NAICS prefix most frequent in history_rows."""
        codes = [
            r.get("naics_code", "")[:2]
            for r in history_rows
            if r.get("naics_code", "")[:2]
        ]
        if not codes:
            return None
        return Counter(codes).most_common(1)[0][0]

    # ------------------------------------------------------------------ #
    #  Fit / predict
    # ------------------------------------------------------------------ #

    def fit(
        self,
        paired_hist: List[list],
        unpaired_hist: List[list],
    ) -> "InspectionPropensityModel":
        """Fit P(reinspected | features) on paired (y=1) + unpaired (y=0) pop.

        Parameters
        ----------
        paired_hist : list of lists-of-dicts
            Each inner list is the pre-cutoff inspection history rows for one
            *paired* establishment (≥ min_hist_insp AND ≥ 1 future inspection).
        unpaired_hist : list of lists-of-dicts
            Each inner list is the history for an *unpaired* establishment
            (≥ min_hist_insp pre-cutoff inspections, 0 future).
        """
        X_rows: list = []
        y: list = []

        for hist in paired_hist:
            naics = self._most_common_naics(hist)
            X_rows.append(self._encode(len(hist), naics))
            y.append(1)

        for hist in unpaired_hist:
            naics = self._most_common_naics(hist)
            X_rows.append(self._encode(len(hist), naics))
            y.append(0)

        X = np.array(X_rows, dtype=float)
        y_arr = np.array(y, dtype=int)

        n_pos = int(y_arr.sum())
        n_neg = len(y_arr) - n_pos

        if n_pos < 5 or n_neg < 5:
            # Not enough data on one side — fall back to uniform weights
            print(
                f"  [IPW] Insufficient paired/unpaired counts "
                f"(pos={n_pos}, neg={n_neg}); using uniform weights."
            )
            self._model = None
            return self

        self._model.fit(X, y_arr)

        # Diagnostic: print propensity P95 and per-industry mean
        probs = self._model.predict_proba(X[y_arr == 1])[:, 1]
        weights = np.clip(1.0 / np.clip(probs, 0.05, 1.0), 1.0, self._max_weight)
        print(
            f"  [IPW] Propensity model fitted  "
            f"paired={n_pos:,}  unpaired={n_neg:,}  "
            f"weight P50={np.percentile(weights, 50):.2f}  "
            f"P95={np.percentile(weights, 95):.2f}  "
            f"max={weights.max():.2f}"
        )
        return self

    def ipw_weights(self, paired_hist: List[list]) -> np.ndarray:
        """Return IPW weights = clip(1 / P(reinspected | features), 1, max_weight).

        Parameters
        ----------
        paired_hist : list of lists-of-dicts
            Same ordering as ``paired_names`` in build_multi_target_sample.
        """
        if self._model is None:
            return np.ones(len(paired_hist), dtype=float)

        X_rows = [
            self._encode(len(hist), self._most_common_naics(hist))
            for hist in paired_hist
        ]
        X = np.array(X_rows, dtype=float)
        probs = self._model.predict_proba(X)[:, 1]
        # Floor at 0.05 to prevent division by near-zero probability
        probs = np.clip(probs, 0.05, 1.0)
        return np.clip(1.0 / probs, 1.0, self._max_weight)
