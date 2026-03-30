"""multi_target_scorer.py — Probabilistic multi-target risk model.

Replaces the single-head GBR pseudo-label model with four prediction heads
that each target a distinct, directly-observable future OSHA outcome.

Architecture
------------
All four heads share the same 47-feature input vector (identical to the
existing ``MLRiskScorer`` feature space, log-transformed).

Head 1 — Binary: serious/willful/repeat event
    GradientBoostingClassifier (balanced) → CalibratedClassifierCV (cv=5)
    Output: ``p_serious_wr_event`` ∈ [0, 1]
    Why: Best single-variable predictor of future OSHA enforcement action.

Head 2 — Regression: log-penalty
    GradientBoostingRegressor (Huber) → predict log1p(total_penalty) → expm1
    Output: ``expected_penalty_usd_12m`` (dollars, ≥ 0)
    Why: Directly predicts the dollar magnitude of future non-compliance costs.

Head 3 — Regression: citation count
    GradientBoostingRegressor (Huber)
    Output: ``expected_citations_12m`` (count, ≥ 0)
    Why: Total violations expose the breadth of non-compliance risk.

Head 4 — Binary: large-penalty event (three thresholds)
    3 × GradientBoostingClassifier (balanced) → CalibratedClassifierCV (cv=5)
    Outputs: ``p_moderate_penalty`` (P75), ``p_large_penalty`` (P90),
             ``p_extreme_penalty`` (P95)
    Why: Separate probability predictions for tiered severity, with industry-
         normalized thresholds so "large" means the same thing across sectors.

Composite risk score formula
----------------------------

    composite = w1 * p_wr * 100
              + w2 * min(log1p(η̂$) / log1p(P_ref), 1) * 100
              + w3 * min(Ĉ / C_ref, 1) * 100
              + w4 * p_large * 100

where weights (w1=0.40, w2=0.25, w3=0.20, w4=0.15) and reference values
(P_ref=$200k, C_ref=20 citations) are configurable.  Weights are chosen so
that the binary WR head (empirically the strongest ranker) dominates the
composite.  Optional weight optimization is available via ``fit_weights()``.

Evidence ceiling
----------------
Consistent with ``MLRiskScorer``, the composite score is capped at 50 when
n_inspections ≤ 2 and at 58 when n_inspections ≤ 4 (except when confirmed
fatality + willful, which caps at 70).  This prevents single-event noise
from dominating sparse-data establishments.
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FILE = "multi_target_model.pkl"

# ── Default composite weights ──────────────────────────────────────────────
_DEFAULT_W1 = 0.35   # p_wr_serious
_DEFAULT_W2 = 0.30   # normalized expected penalty
_DEFAULT_W3 = 0.20   # p_injury_event
_DEFAULT_W4 = 0.15   # normalized gravity score

# Reference values for normalizing regression outputs into 0-100 components
_PENALTY_REF_USD = 200_000.0   # typical large-employer penalties in DON'T-REC range
_GRAVITY_REF     = 100.0        # benchmark gravity total (see composite_score)

# Population-mean prior for evidence shrinkage (replaces hard ceilings)
_SCORE_PRIOR = 15.0   # approximate unconditional mean composite score

# ── GBC / GBR hyper-parameters ─────────────────────────────────────────────
# Binary heads use HistGradientBoostingClassifier with early stopping.
# Early stopping finds the optimal number of iterations automatically,
# preventing the over-confidence (compressed predictions) that harms
# calibration slope when a fixed n_estimators is used.

# Head 1 (p_event / WR-serious): early stopping with moderate patience.
# lr=0.02 + n_iter_no_change=80 gives ~400-800 effective iterations before
# stopping, significantly more than the previous lr=0.04/n=40 (~200-350 iter)
# but while keeping the model from over-fitting (which causes covariate-shift
# amplification that breaks post-hoc Platt calibration on 2021 holdout).
_HGBC_PARAMS = dict(
    max_iter=5000,
    max_depth=5,
    learning_rate=0.02,
    min_samples_leaf=15,
    l2_regularization=0.8,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=80,
    random_state=42,
)
# Head 3 (p_injury / hospitalization-fatality): no early stopping, moderate
# regularisation.  This combination (from Round 7) gave BSS=0.103, AUROC=0.797.
_HGBC_INJURY_PARAMS = dict(
    max_iter=700,
    max_depth=4,
    learning_rate=0.04,
    min_samples_leaf=20,
    l2_regularization=1.5,
    early_stopping=False,
    random_state=42,
)
_GBR_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.75,
    loss="huber",
    alpha=0.9,
    min_samples_leaf=2,
    random_state=42,
)


# ── Platt calibration helpers ─────────────────────────────────────────────

def _fit_platt(y_val: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
    """Fit 2-parameter Platt (logistic) calibration: p_cal = sigmoid(a*logit(p)+b).

    Minimises binary cross-entropy on the holdout.  Returns [a, b].
    a scales the logit (corrects discriminative slope);
    b shifts the logit  (corrects the mean / intercept).
    """
    from scipy.optimize import minimize

    p_clipped = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
    logits    = np.log(p_clipped / (1.0 - p_clipped))

    def nll(params: np.ndarray) -> float:
        a, b = params
        p_cal = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        return float(-np.mean(
            y_val * np.log(np.clip(p_cal, 1e-9, 1.0)) +
            (1.0 - y_val) * np.log(np.clip(1.0 - p_cal, 1e-9, 1.0))
        ))

    result = minimize(
        nll, [1.0, 0.0], method="L-BFGS-B",
        bounds=[(0.05, 10.0), (-10.0, 10.0)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return result.x   # [a, b]


def _apply_platt_batch(p_arr: np.ndarray, platt: np.ndarray) -> np.ndarray:
    """Vectorised Platt transform.  platt = [a, b]."""
    a, b = float(platt[0]), float(platt[1])
    if abs(a - 1.0) < 1e-6 and abs(b) < 1e-6:
        return p_arr
    p_clipped = np.clip(p_arr, 1e-7, 1.0 - 1e-7)
    logits    = np.log(p_clipped / (1.0 - p_clipped))
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


def _apply_platt_scalar(p: float, platt: np.ndarray) -> float:
    """Scalar Platt transform.  platt = [a, b]."""
    a, b = float(platt[0]), float(platt[1])
    if abs(a - 1.0) < 1e-6 and abs(b) < 1e-6:
        return p
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


class MultiTargetRiskScorer:
    """4-head probabilistic OSHA risk model.

    Usage
    -----
    1.  ``fit(X, multi_target_rows)``  ← from ``multi_target_labeler``
    2.  ``predict(X)``                 ← returns dict with all outputs
    3.  ``composite_score(pred_dict)`` ← converts to 0-100 user-visible score
    4.  ``save(path)`` / ``load(path)`` via pickle
    """

    def __init__(self) -> None:
        # Binary head 1: WR/serious event
        self._head_wr: Optional[Pipeline] = None
        # Regression head 2: log penalty
        self._head_log_pen: Optional[Pipeline] = None
        # Binary head 3: hospitalization/fatality event  [NEW]
        self._head_injury: Optional[Pipeline] = None
        # Regression head 4: gravity-weighted severity  [NEW]
        self._head_gravity: Optional[Pipeline] = None

        # Composite weight vector [w1, w2, w3, w4]
        self._weights: List[float] = [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4]

        # Post-training temperature slots are kept for backward-compatibility
        # with old pickled models; Platt calibration replaces them.
        self._temp_wr: float = 1.0
        self._temp_inj: float = 1.0
        # Platt calibration parameters [[a_wr, b_wr], [a_inj, b_inj]].
        # Defaults = identity (no calibration).  Set by fit().
        _identity = np.array([1.0, 0.0])
        self._platt_wr:  np.ndarray = _identity.copy()
        self._platt_inj: np.ndarray = _identity.copy()

        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        rows: List[Dict],
        optimize_weights: bool = True,
        val_fraction: float = 0.20,
        rng_seed: int = 42,
    ) -> "MultiTargetRiskScorer":
        """Train all 4 prediction heads.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Log-transformed feature matrix.
        rows : list of dicts from ``multi_target_labeler.build_multi_target_sample``
        optimize_weights : bool
            When True, empirically optimize composite score weights on a held-
            out validation fold to maximize Spearman ρ with the real adverse
            outcome.  Requires ≥50 samples.
        val_fraction : float
            Fraction of X held out for weight optimization (default 20%).
        rng_seed : int
            Random seed for train/val split.
        """
        assert len(rows) == len(X), "X and rows must have the same length"
        n = len(rows)
        if n < 20:
            raise ValueError(f"MultiTargetRiskScorer.fit requires ≥20 rows; got {n}")

        # Extract target arrays
        y_wr    = np.array([r["any_wr_serious"]         for r in rows], dtype=int)
        y_logp  = np.array([r["log_penalty"]             for r in rows], dtype=float)
        y_inj   = np.array([r["any_injury_fatal"]        for r in rows], dtype=int)
        y_grav  = np.array([r["gravity_weighted_score"]  for r in rows], dtype=float)

        rng = np.random.default_rng(rng_seed)
        # Train/val split for weight optimization
        idx = rng.permutation(n)
        split = max(int(n * val_fraction), min(50, n))
        if not optimize_weights or n < 100:
            # Use all data for fitting; skip weight optimization on small sets
            train_idx, val_idx = idx, np.array([], dtype=int)
            optimize_weights = False
        else:
            val_idx   = idx[:split]
            train_idx = idx[split:]

        X_tr = X[train_idx];  X_val = X[val_idx]

        def _train_hgbc_raw(X_fit, y_fit, hgbc_params=None):
            """Train raw (un-calibrated) HistGBT; calibration is added later."""
            n_pos = y_fit.sum()
            n_neg = len(y_fit) - n_pos
            if n_pos < 5 or n_neg < 5:
                return _ConstantClassifier(float(n_pos / max(1, len(y_fit))))
            if hgbc_params is None:
                hgbc_params = _HGBC_PARAMS
            hgbc = HistGradientBoostingClassifier(**hgbc_params)
            hgbc.fit(X_fit, y_fit)
            return hgbc

        def _train_gbr_pipe(X_fit, y_fit):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(**_GBR_PARAMS)),
            ])
            pipe.fit(X_fit, y_fit)
            return pipe

        print("  [MultiTargetScorer] Training Head 1: WR/Serious event (binary) …")
        self._head_wr = _train_hgbc_raw(X_tr, y_wr[train_idx], hgbc_params=_HGBC_PARAMS)

        print("  [MultiTargetScorer] Training Head 2: log-penalty regression \u2026")
        self._head_log_pen = _train_gbr_pipe(X_tr, y_logp[train_idx])

        print("  [MultiTargetScorer] Training Head 3: hospitalization/fatality (binary) \u2026")
        # Separate lighter params; no class imbalance correction so that
        # calibration tracks the natural training prevalence.
        self._head_injury = _train_hgbc_raw(
            X_tr, y_inj[train_idx], hgbc_params=_HGBC_INJURY_PARAMS
        )

        print("  [MultiTargetScorer] Training Head 4: gravity-weighted severity (regression) \u2026")
        self._head_gravity = _train_gbr_pipe(X_tr, y_grav[train_idx])

        self._is_fitted = True

        # \u2500\u2500 Post-hoc Platt calibration on training holdout \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Calibrate the *actual* trained models (100% of X_tr) using a sigmoid
        # Platt layer fitted on the held-out 20% training fold.  Unlike 5-fold
        # CalibratedClassifierCV, this avoids the 80%-vs-100%-training mismatch
        # that causes calibration slope to remain above 1.0 on held-out data.
        # For p_event (prevalences 48.6% train vs 47.3% val): calibration
        # transfers near-perfectly.  For p_injury (12.8% train vs 9.5% val):
        # the 'a' (slope) parameter transfers; 'b' only partially compensates
        # the 3.3pp prevalence shift.
        if len(val_idx) >= 200:
            print("  [MultiTargetScorer] Post-hoc Platt calibration on training holdout ...")
            p_wr_val  = _clf_proba_batch(self._head_wr,     X_val)
            p_inj_val = _clf_proba_batch(self._head_injury, X_val)
            self._platt_wr  = _fit_platt(y_wr[val_idx].astype(float),  p_wr_val)
            self._platt_inj = _fit_platt(y_inj[val_idx].astype(float), p_inj_val)
            print(
                f"  [MultiTargetScorer] Platt WR:  a={self._platt_wr[0]:.3f}  "
                f"b={self._platt_wr[1]:.3f}"
            )
            print(
                f"  [MultiTargetScorer] Platt Inj: a={self._platt_inj[0]:.3f}  "
                f"b={self._platt_inj[1]:.3f}"
            )

        # Temperature slots kept for backward-compat with old pickled models.
        self._temp_wr  = 1.0
        self._temp_inj = 1.0

        # ── Post-training temperature scaling ─────────────────────────
        # Fit temperature parameters on the 20% training-holdout validation
        # fold to bring the OLS calibration slope near 1.0.  Temperature is
        # applied at inference time (predict / predict_batch) so AUROC is
        # preserved (monotone transform) and the slope meets the [0.9, 1.1]
        # target on held-out data with similar prevalence.
        # Temperature scaling: no longer applied (Platt calibration on holdout
        # subsumes temperature scaling with better statistical efficiency).
        self._temp_wr  = 1.0
        self._temp_inj = 1.0

        # ── Composite weight optimization ──────────────────────────────
        if optimize_weights and len(val_idx) >= 50:
            y_adv_val = np.array([rows[i]["real_label"] for i in val_idx])
            pred_val  = self.predict_batch(X_val)
            self._weights = _optimize_weights(pred_val, y_adv_val)
            print(
                f"  [MultiTargetScorer] Optimized weights: "
                f"w1={self._weights[0]:.3f} w2={self._weights[1]:.3f} "
                f"w3={self._weights[2]:.3f} w4={self._weights[3]:.3f}"
            )
        else:
            print(
                f"  [MultiTargetScorer] Using default weights: "
                f"w1={_DEFAULT_W1} w2={_DEFAULT_W2} w3={_DEFAULT_W3} w4={_DEFAULT_W4}"
            )

        return self

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> Dict[str, float]:
        """Run all 4 prediction heads on a single feature row or batch.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,) or (1, n_features)

        Returns
        -------
        dict:
            p_serious_wr_event  (float 0-1)
            expected_penalty_usd (float >= 0)
            p_injury_event      (float 0-1)
            gravity_score       (float >= 0)
        """
        if not self._is_fitted:
            return self._empty_prediction()
        x2d = np.atleast_2d(x)
        platt_wr  = getattr(self, "_platt_wr",  np.array([1.0, 0.0]))
        platt_inj = getattr(self, "_platt_inj", np.array([1.0, 0.0]))
        return {
            "p_serious_wr_event":  _apply_platt_scalar(float(_clf_proba(self._head_wr,     x2d)), platt_wr),
            "expected_penalty_usd": float(max(0.0, math.expm1(
                float(_reg_predict(self._head_log_pen, x2d))
            ))),
            "p_injury_event":      _apply_platt_scalar(float(_clf_proba(self._head_injury, x2d)), platt_inj),
            "gravity_score":       float(max(0.0, _reg_predict(self._head_gravity, x2d))),
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict for a batch of feature rows (efficient vectorized path)."""
        if not self._is_fitted:
            return [self._empty_prediction() for _ in range(len(X))]
        platt_wr  = getattr(self, "_platt_wr",  np.array([1.0, 0.0]))
        platt_inj = getattr(self, "_platt_inj", np.array([1.0, 0.0]))
        wr   = _apply_platt_batch(_clf_proba_batch(self._head_wr,     X), platt_wr)
        logp = _reg_predict_batch(self._head_log_pen, X)
        inj  = _apply_platt_batch(_clf_proba_batch(self._head_injury, X), platt_inj)
        grav = _reg_predict_batch(self._head_gravity, X)

        return [
            {
                "p_serious_wr_event":  float(wr[i]),
                "expected_penalty_usd": float(max(0.0, math.expm1(float(logp[i])))),
                "p_injury_event":      float(inj[i]),
                "gravity_score":       float(max(0.0, float(grav[i]))),
            }
            for i in range(len(X))
        ]

    def composite_score(
        self,
        pred: Dict[str, float],
        n_inspections: int = 99,
        has_fatality: bool = False,
        has_willful: bool = False,
    ) -> float:
        """Compute 0-100 composite risk score from a prediction dict.

        Applies smooth evidence shrinkage toward the population mean instead
        of hard evidence ceilings, eliminating the score wall at ~55 observed
        in validation plots.
        """
        w1, w2, w3, w4 = self._weights

        pen_component = min(
            math.log1p(max(0.0, pred["expected_penalty_usd"])) /
            math.log1p(_PENALTY_REF_USD) * 100.0,
            100.0,
        )
        grav_component = min(
            math.log1p(max(0.0, pred["gravity_score"])) /
            math.log1p(_GRAVITY_REF) * 100.0,
            100.0,
        )

        raw = (
            w1 * pred["p_serious_wr_event"] * 100.0
            + w2 * pen_component
            + w3 * pred["p_injury_event"]   * 100.0
            + w4 * grav_component
        )

        # Smooth evidence shrinkage: pulls low-inspection scores toward the
        # population mean rather than hard-capping them.
        # confidence approaches 0 for n=1 (~0.28) and 1 for n>=10 (~0.96).
        confidence = 1.0 - math.exp(-n_inspections / 3.0)
        score = raw * confidence + _SCORE_PRIOR * (1.0 - confidence)

        # Hard cap only for the most extreme evidence case (fatality + willful)
        if has_fatality and has_willful:
            score = min(score, 70.0)

        return float(np.clip(score, 0.0, 100.0))

    # ------------------------------------------------------------------ #
    #  Weight optimization (public — callable after fit for diagnostics)
    # ------------------------------------------------------------------ #
    def fit_weights(
        self,
        X_val: np.ndarray,
        y_adv_val: np.ndarray,
    ) -> None:
        """Optimize composite weights on a held-out set (Spearman objective)."""
        if not self._is_fitted:
            return
        pred_val      = self.predict_batch(X_val)
        self._weights = _optimize_weights(pred_val, y_adv_val)

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MultiTargetRiskScorer":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_if_exists(cls, cache_dir: str = "ml_cache") -> Optional["MultiTargetRiskScorer"]:
        path = os.path.join(cache_dir, MODEL_FILE)
        if not os.path.exists(path):
            return None
        try:
            return cls.load(path)
        except Exception as e:
            print(f"  [MultiTargetScorer] Could not load from {path}: {e}")
            return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _empty_prediction(self) -> Dict[str, float]:
        return {
            "p_serious_wr_event":   0.0,
            "expected_penalty_usd": 0.0,
            "p_injury_event":       0.0,
            "gravity_score":        0.0,
        }


# ── Composite weight optimizer ──────────────────────────────────────────────

def _optimize_weights(
    pred_batch: List[Dict[str, float]],
    y_adv: np.ndarray,
) -> List[float]:
    """Find weights that maximize Spearman ρ between composite score and y_adv.

    Strategy: pre-compute the vectorized component arrays once, then use a
    fast grid search over a coarse lattice followed by Nelder-Mead refinement.
    Nelder-Mead tolerates the non-differentiability of Spearman correlation
    much better than gradient-based SLSQP.
    """
    from scipy.stats import spearmanr

    n = len(pred_batch)

    # Pre-compute component arrays (vectorized — avoids per-iteration Python loop)
    p_wr  = np.array([p["p_serious_wr_event"] * 100.0       for p in pred_batch])
    pen_n = np.array([
        min(
            math.log1p(max(0.0, p["expected_penalty_usd"])) /
            math.log1p(_PENALTY_REF_USD) * 100.0, 100.0,
        )
        for p in pred_batch
    ])
    p_inj = np.array([p["p_injury_event"] * 100.0            for p in pred_batch])
    grav_n = np.array([
        min(
            math.log1p(max(0.0, p["gravity_score"])) /
            math.log1p(_GRAVITY_REF) * 100.0, 100.0,
        )
        for p in pred_batch
    ])

    components = np.stack([p_wr, pen_n, p_inj, grav_n], axis=1)  # (n, 4)

    def _neg_spearman_vec(raw_weights: np.ndarray) -> float:
        w = np.maximum(raw_weights, 0.0)
        s = w.sum()
        if s < 1e-9:
            return 0.0
        w = w / s
        scores = components @ w          # fast dot product
        if np.all(scores == scores[0]):
            return 0.0
        rho, _ = spearmanr(scores, y_adv)
        return -float(rho) if not math.isnan(rho) else 0.0

    w0      = np.array([_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4])
    rho_def = -_neg_spearman_vec(w0)

    # ── Stage 1: coarse grid search over 25 candidate starting points ──
    best_w   = w0.copy()
    best_val = _neg_spearman_vec(w0)

    grid_points = [
        # Near-default exploration around [0.35, 0.30, 0.20, 0.15]
        [0.35, 0.30, 0.20, 0.15],
        [0.40, 0.25, 0.20, 0.15],
        [0.40, 0.30, 0.15, 0.15],
        [0.35, 0.30, 0.25, 0.10],
        [0.30, 0.30, 0.25, 0.15],
        # Injury-heavy (test if p_injury head is underweighted)
        [0.30, 0.25, 0.30, 0.15],
        [0.35, 0.25, 0.30, 0.10],
        [0.30, 0.30, 0.30, 0.10],
        # WR-dominant
        [0.50, 0.25, 0.15, 0.10],
        [0.45, 0.25, 0.20, 0.10],
        [0.45, 0.30, 0.15, 0.10],
        [0.50, 0.20, 0.20, 0.10],
        # Penalty-heavy
        [0.30, 0.40, 0.20, 0.10],
        [0.35, 0.35, 0.20, 0.10],
        [0.30, 0.35, 0.25, 0.10],
        # Gravity-heavy
        [0.35, 0.25, 0.15, 0.25],
        [0.35, 0.30, 0.15, 0.20],
        [0.30, 0.25, 0.20, 0.25],
        # Equal-ish
        [0.25, 0.25, 0.25, 0.25],
        [0.30, 0.25, 0.25, 0.20],
    ]
    for gw in grid_points:
        gw_arr = np.array(gw)
        gw_arr /= gw_arr.sum()
        val = _neg_spearman_vec(gw_arr)
        if val < best_val:
            best_val = val
            best_w   = gw_arr.copy()

    # ── Stage 2: Nelder-Mead starting from the best grid point ──────────
    # Parametrise as 3 free weights; 4th = max(0, 1 - sum)
    def _nm_objective(v3: np.ndarray) -> float:
        v = np.concatenate([np.maximum(v3, 0.0), [max(0.0, 1.0 - float(np.sum(np.maximum(v3, 0.0))))]])
        # L2 regularization towards defaults (λ=0.1) — light constraint to prevent
        # degenerate all-zero weights while allowing meaningful exploration
        reg = 0.1 * float(np.sum((v - w0) ** 2))
        return _neg_spearman_vec(v) + reg

    v0_3 = best_w[:3].copy()
    try:
        res = minimize(
            _nm_objective,
            v0_3,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5},
        )
        if res.success or res.fun < best_val:
            v3   = np.maximum(res.x, 0.0)
            w4   = max(0.0, 1.0 - float(v3.sum()))
            w_nm = np.concatenate([v3, [w4]])
            w_nm = np.maximum(w_nm, 0.0)
            w_nm /= w_nm.sum()
            if _neg_spearman_vec(w_nm) <= best_val:
                best_w   = w_nm
                best_val = _neg_spearman_vec(w_nm)
    except Exception:
        pass

    # Enforce minimum w4 >= 0.05: the gravity signal always contributes,
    # preventing degenerate solutions that completely ignore violation severity.
    if best_w[3] < 0.05:
        best_w[3] = 0.05
        best_w = best_w / np.sum(best_w)

    rho_opt = -_neg_spearman_vec(best_w)
    if rho_opt >= rho_def - 0.001:
        return best_w.tolist()

    return [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4]


# ── Internal prediction helpers ─────────────────────────────────────────────

class _ConstantClassifier:
    """Fallback for all-one-class training sets."""

    def __init__(self, p: float) -> None:
        self._p = float(p)

    def predict_proba(self, X):
        n = len(np.atleast_2d(X))
        return np.column_stack([
            np.full(n, 1.0 - self._p),
            np.full(n, self._p),
        ])


def _clf_proba(clf, x2d: np.ndarray) -> float:
    """Probability of positive class (index 1) for a single row (1×n array)."""
    if clf is None:
        return 0.0
    proba = clf.predict_proba(x2d)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return float(proba[0, 1])
    return float(proba[0])


def _clf_proba_batch(clf, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities for a batch, returns 1-D array."""
    if clf is None:
        return np.zeros(len(X))
    proba = clf.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def _reg_predict(model, x2d: np.ndarray) -> float:
    if model is None:
        return 0.0
    return float(model.predict(x2d)[0])


def _reg_predict_batch(model, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X))
    return model.predict(X).ravel()
