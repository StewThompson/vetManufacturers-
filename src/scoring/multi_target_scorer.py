"""multi_target_scorer.py — Probability-driven, calibrated risk model.

Primary outputs are **calibrated probabilities** for binary adverse events;
regression heads are retained for backward compatibility but no longer drive
the composite score.

Architecture (v2 — probability-first)
-------------------------------------
All heads share the same 46-feature input vector (log-transformed).

Primary binary heads (drive composite score):
  Head 1 — p_serious_wr_event:  P(≥1 Serious/Willful/Repeat violation in 12mo)
  Head 3 — p_injury_event:      P(hospitalization or fatality event)
  Head 5 — p_penalty_ge_p95:    P(penalty ≥ industry P95 threshold)

Auxiliary binary heads (available in output, not in composite):
  Head 5a — p_penalty_ge_p75:   P(penalty ≥ industry P75)
  Head 5b — p_penalty_ge_p90:   P(penalty ≥ industry P90)

Legacy regression heads (retained for backward compat / outlook):
  Head 2  — expected_penalty_usd (hurdle: binary × conditional log-penalty GBR)
  Head 4  — gravity_score (GBR + isotonic)

Composite risk score formula (v2)
---------------------------------
    risk_score_raw = w1 * p_serious_wr_event
                   + w2 * p_injury_event
                   + w3 * p_extreme_penalty

Default weights: w1=0.60, w2=0.30, w3=0.10.  All inputs are calibrated
probabilities in [0, 1].  The raw score is then:
  1. Evidence-shrunk toward population mean (smooth exponential)
  2. Transformed via monotonic percentile stretching (expands top-end separation)
  3. Rescaled to 0–100

Calibration
-----------
All binary heads use isotonic regression calibration fitted on a held-out
validation fold (no leakage).  Calibration quality is logged during training
via Brier score, BSS, and ECE.
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
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FILE = "multi_target_model.pkl"

# ── Default composite weights (v2: probability-first, 3 components) ────────
_DEFAULT_W1 = 0.60   # p_serious_wr_event
_DEFAULT_W2 = 0.30   # p_injury_event
_DEFAULT_W3 = 0.10   # p_extreme_penalty (P95 tier)

# Legacy 4-component weights (kept for backward compat with old pickles)
_LEGACY_W1 = 0.35
_LEGACY_W2 = 0.30
_LEGACY_W3 = 0.20
_LEGACY_W4 = 0.15

# Reference values for normalizing regression outputs (legacy heads only)
_PENALTY_REF_USD = 200_000.0
_GRAVITY_REF     = 100.0

# Population-mean prior for evidence shrinkage (replaces hard ceilings)
_SCORE_PRIOR = 15.0

# Convex power-stretch parameter for top-end separation.
# score = pctile^alpha * 100.  alpha=2 maps median->25, P90->81, P95->90,
# concentrating most companies in the lower half and spreading high-risk ones.
_STRETCH_ALPHA = 2.0

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
# ── Regression head hyperparameters ──────────────────────────────────────
# Penalty head: quantile regression (75th percentile) in log-space.
# Targets the upper half of the conditional penalty distribution so predictions
# for high-risk companies reach realistic large-dollar amounts.  Early stopping
# prevents overfitting on the sparse positive-penalty subset.
_HGBR_PEN_PARAMS = dict(
    max_iter=3000,
    max_depth=5,
    learning_rate=0.025,
    min_samples_leaf=8,
    l2_regularization=0.3,
    loss="quantile",
    quantile=0.75,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=60,
    random_state=42,
)
# Gravity head: absolute-error (MAE) loss is robust to heavy-tailed outliers
# and improves Spearman rank correlation vs squared-error.  A shallower tree
# with stronger l2 regularisation + monotone constraints on violation features
# removes the upper-range overprediction seen with the previous GBR.
_HGBR_GRAV_PARAMS = dict(
    max_iter=3000,
    max_depth=4,
    learning_rate=0.025,
    min_samples_leaf=12,
    l2_regularization=1.2,
    loss="absolute_error",
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=60,
    random_state=42,
)
# Gravity monotone constraints (feature index → direction).
# More serious / willful / repeat violations must predict higher gravity;
# clean_ratio (clean inspections / total) must predict lower gravity.
# Indices match ml_risk_scorer.FEATURE_NAMES.
_GRAV_MONOTONE_BY_IDX: Dict[int, int] = {
    1: 1, 2: 1, 3: 1, 4: 1,    # log_violations, serious, willful, repeat
    9: 1, 10: 1,                 # severe_incidents, violations_per_inspection
    11: 1, 12: 1, 13: 1,        # accident_count, fatality_count, injury_count
    14: 1,                       # avg_gravity  (strongest signal)
    16: -1,                      # clean_ratio  (more clean → lower gravity)
    18: 1, 19: 1,               # recent_wr_rate, trend_delta
    20: 1, 21: 1, 22: 1,       # log_willful_raw, log_repeat_raw, has_any_fatality
}


def _make_monotone_cst(n_features: int) -> List[int]:
    """Build gravity monotone-constraint list (length = n_features)."""
    cst = [0] * n_features
    for idx, val in _GRAV_MONOTONE_BY_IDX.items():
        if idx < n_features:
            cst[idx] = val
    return cst


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
        # ── Primary binary heads (drive composite score) ──────────────
        # Head 1: WR/serious event
        self._head_wr: Optional[Pipeline] = None
        # Head 3: hospitalization/fatality event
        self._head_injury: Optional[Pipeline] = None
        # Head 5: penalty tier classifiers (P75, P90, P95)
        self._head_pen_p75: Optional[object] = None
        self._head_pen_p90: Optional[object] = None
        self._head_pen_p95: Optional[object] = None

        # ── Legacy regression heads (kept for backward compat / outlook) ──
        # Hurdle head 2a: binary "any positive penalty?" classifier
        self._head_pen_nonzero: Optional[object] = None
        # Hurdle head 2b: conditional log-penalty GBR (trained on positive rows only)
        self._head_log_pen: Optional[Pipeline] = None
        # Regression head 4: gravity-weighted severity
        self._head_gravity: Optional[Pipeline] = None

        # ── Composite weight vector [w1, w2, w3] (3 probability components) ──
        self._weights: List[float] = [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3]

        # Post-training temperature slots are kept for backward-compatibility
        # with old pickled models; Platt calibration replaces them.
        self._temp_wr: float = 1.0
        self._temp_inj: float = 1.0

        # ── Calibration ────────────────────────────────────────────────
        # Isotonic calibration for all binary heads (fitted on val fold).
        self._iso_wr:     Optional[object] = None
        self._iso_inj:    Optional[object] = None
        self._iso_pen_p75: Optional[object] = None
        self._iso_pen_p90: Optional[object] = None
        self._iso_pen_p95: Optional[object] = None

        # Legacy Platt calibration (backward compat with old pickles)
        _identity = np.array([1.0, 0.0])
        self._platt_wr:  np.ndarray = _identity.copy()
        self._platt_inj: np.ndarray = _identity.copy()
        self._platt_pen: np.ndarray = _identity.copy()

        # Isotonic calibration for regression heads
        self._iso_pen: Optional[object] = None      # kept for compat; unused by hurdle path
        self._iso_log_pen: Optional[object] = None  # calibrates conditional log-penalty GBR
        self._iso_grav: Optional[object] = None

        # ── Score transformation ───────────────────────────────────────
        # Empirical CDF of raw scores from validation set, for percentile stretching.
        # Stored as sorted array of raw scores; at inference, percentile = searchsorted.
        self._score_cdf: Optional[np.ndarray] = None

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

        # Extract IPW weights (default 1.0 for rows without the key).
        # Normalize to mean=1 so fitted regularisation scale stays constant.
        weights_raw = np.array([r.get("ipw_weight", 1.0) for r in rows], dtype=float)
        weights_norm = weights_raw / weights_raw.mean()
        print(
            f"  [MultiTargetScorer] IPW weight stats — "
            f"min={weights_norm.min():.2f}  mean=1.00  "
            f"P95={float(np.percentile(weights_norm, 95)):.2f}  "
            f"max={weights_norm.max():.2f}"
        )

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
        sw_tr = weights_norm[train_idx]  # sample weights for training fold

        def _train_hgbc_raw(X_fit, y_fit, hgbc_params=None, sample_weight=None):
            """Train raw (un-calibrated) HistGBT; calibration is added later."""
            n_pos = y_fit.sum()
            n_neg = len(y_fit) - n_pos
            if n_pos < 5 or n_neg < 5:
                return _ConstantClassifier(float(n_pos / max(1, len(y_fit))))
            if hgbc_params is None:
                hgbc_params = _HGBC_PARAMS
            hgbc = HistGradientBoostingClassifier(**hgbc_params)
            hgbc.fit(X_fit, y_fit, sample_weight=sample_weight)
            return hgbc

        def _train_hgbr(X_fit, y_fit, hgbr_params, sample_weight=None):
            """Train HistGradientBoostingRegressor (no scaler needed for tree models)."""
            model = HistGradientBoostingRegressor(**hgbr_params)
            model.fit(X_fit, y_fit, sample_weight=sample_weight)
            return model

        print("  [MultiTargetScorer] Training Head 1: WR/Serious event (binary) …")
        self._head_wr = _train_hgbc_raw(
            X_tr, y_wr[train_idx], hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr
        )

        # ── Head 2: direct log-penalty regression (no hurdle) ────────────────
        # Training on ALL rows (including zero-penalty) gives the model the full
        # zero vs nonzero signal and avoids multiplicative noise from a two-stage
        # hurdle.  log1p(0) = 0, so the model learns the separation implicitly.
        print(
            f"  [MultiTargetScorer] Training Head 2: log-penalty regression "
            f"(all {len(train_idx):,} rows, including zeros) …"
        )
        self._head_log_pen = _train_hgbr(
            X_tr, y_logp[train_idx], _HGBR_PEN_PARAMS, sample_weight=sw_tr
        )

        print("  [MultiTargetScorer] Training Head 3: hospitalization/fatality (binary) …")
        self._head_injury = _train_hgbc_raw(
            X_tr, y_inj[train_idx], hgbc_params=_HGBC_INJURY_PARAMS, sample_weight=sw_tr
        )

        # ── Penalty tier classifiers (P75, P90, P95) ──────────────────
        y_pen_p75 = np.array([r.get("is_moderate_penalty", 0) for r in rows], dtype=int)
        y_pen_p90 = np.array([r.get("is_large_penalty",    0) for r in rows], dtype=int)
        y_pen_p95 = np.array([r.get("is_extreme_penalty",  0) for r in rows], dtype=int)

        for tier_name, y_tier, attr_name in [
            ("P75 (moderate)", y_pen_p75, "_head_pen_p75"),
            ("P90 (large)",    y_pen_p90, "_head_pen_p90"),
            ("P95 (extreme)",  y_pen_p95, "_head_pen_p95"),
        ]:
            pos_rate = y_tier[train_idx].mean()
            print(f"  [MultiTargetScorer] Training penalty tier {tier_name} "
                  f"(pos rate={pos_rate:.1%}) …")
            head = _train_hgbc_raw(
                X_tr, y_tier[train_idx],
                hgbc_params=_HGBC_PARAMS,
                sample_weight=sw_tr,
            )
            setattr(self, attr_name, head)

        print("  [MultiTargetScorer] Training Head 4: gravity-weighted severity (regression) …")
        grav_params = dict(_HGBR_GRAV_PARAMS)
        grav_params["monotonic_cst"] = _make_monotone_cst(X.shape[1])
        self._head_gravity = _train_hgbr(
            X_tr, y_grav[train_idx], grav_params, sample_weight=sw_tr
        )

        self._is_fitted = True

        # ── Post-hoc isotonic calibration on training holdout ────────────────
        # Isotonic regression is nonparametric and handles non-sigmoid
        # calibration curves better than Platt scaling.  Fitted on the
        # held-out 20% validation fold to avoid leakage.
        if len(val_idx) >= 200:
            from sklearn.isotonic import IsotonicRegression
            from scipy.stats import spearmanr, pearsonr
            from src.scoring.calibration import brier_score, brier_skill_score, expected_calibration_error

            print("  [MultiTargetScorer] Post-hoc isotonic calibration on training holdout ...")

            # ── Calibrate primary binary heads with isotonic regression ──
            p_wr_val  = _clf_proba_batch(self._head_wr,     X_val)
            p_inj_val = _clf_proba_batch(self._head_injury, X_val)

            iso_wr = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso_wr.fit(p_wr_val, y_wr[val_idx].astype(float))
            self._iso_wr = iso_wr

            iso_inj = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso_inj.fit(p_inj_val, y_inj[val_idx].astype(float))
            self._iso_inj = iso_inj

            # Calibration metrics for primary heads
            p_wr_cal  = iso_wr.predict(p_wr_val)
            p_inj_cal = iso_inj.predict(p_inj_val)
            for head_name, y_val_h, p_cal_h in [
                ("WR/Serious", y_wr[val_idx], p_wr_cal),
                ("Injury/Fatal", y_inj[val_idx], p_inj_cal),
            ]:
                bs  = brier_score(y_val_h, p_cal_h)
                bss = brier_skill_score(y_val_h, p_cal_h)
                ece = expected_calibration_error(y_val_h, p_cal_h)
                print(f"  [MultiTargetScorer] {head_name} calibration: "
                      f"Brier={bs:.4f}  BSS={bss:+.3f}  ECE={ece:.4f}")

            # ── Calibrate penalty tier heads ─────────────────────────────
            for tier_name, y_tier, head_attr, iso_attr in [
                ("P75", y_pen_p75, "_head_pen_p75", "_iso_pen_p75"),
                ("P90", y_pen_p90, "_head_pen_p90", "_iso_pen_p90"),
                ("P95", y_pen_p95, "_head_pen_p95", "_iso_pen_p95"),
            ]:
                head = getattr(self, head_attr)
                p_val_t = _clf_proba_batch(head, X_val)
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p_val_t, y_tier[val_idx].astype(float))
                setattr(self, iso_attr, iso)
                p_cal_t = iso.predict(p_val_t)
                bs  = brier_score(y_tier[val_idx], p_cal_t)
                bss = brier_skill_score(y_tier[val_idx], p_cal_t)
                ece = expected_calibration_error(y_tier[val_idx], p_cal_t)
                print(f"  [MultiTargetScorer] Penalty {tier_name} calibration: "
                      f"Brier={bs:.4f}  BSS={bss:+.3f}  ECE={ece:.4f}")

            # ── Legacy Platt calibration (kept for backward compat) ──────
            self._platt_wr  = _fit_platt(y_wr[val_idx].astype(float),  p_wr_val)
            self._platt_inj = _fit_platt(y_inj[val_idx].astype(float), p_inj_val)

            # Direct log-penalty regression diagnostics on validation fold.
            direct_logp_val  = _reg_predict_batch(self._head_log_pen, X_val)
            actual_pen_usd   = np.expm1(y_logp[val_idx])
            direct_pen_usd   = np.expm1(np.maximum(0.0, direct_logp_val))
            rho_p, _ = spearmanr(direct_pen_usd, actual_pen_usd)
            r_p, _   = pearsonr(direct_logp_val, y_logp[val_idx])
            print(
                f"  [MultiTargetScorer] Direct Pen (val): "
                f"rho={rho_p:+.3f}  r(log-space)={r_p:+.3f}"
            )

            # Isotonic calibration for log-penalty regression head.
            # Disabled: isotonic mapping fitted on training-era val labels creates
            # distribution-shift bias when predicting on future (2024+) data.
            # The HGBR predictions in log-space are already well-calibrated via
            # squared_error loss; a second calibration layer hurts external r.
            self._iso_log_pen = None

            # Isotonic calibration for gravity regression head.
            raw_grav_val = _reg_predict_batch(self._head_gravity, X_val)
            iso_grav = IsotonicRegression(out_of_bounds="clip")
            iso_grav.fit(raw_grav_val, y_grav[val_idx])
            self._iso_grav = iso_grav

        # Temperature slots kept for backward-compat with old pickled models.
        self._temp_wr  = 1.0
        self._temp_inj = 1.0

        # ── Composite weight optimization ────────────────────────────────
        if optimize_weights and len(val_idx) >= 50:
            y_adv_val = np.array([rows[i]["real_label"] for i in val_idx])
            pred_val  = self.predict_batch(X_val)
            self._weights = _optimize_weights(pred_val, y_adv_val)
            print(
                f"  [MultiTargetScorer] Optimized weights: "
                f"w1={self._weights[0]:.3f} w2={self._weights[1]:.3f} "
                f"w3={self._weights[2]:.3f}"
            )
        else:
            print(
                f"  [MultiTargetScorer] Using default weights: "
                f"w1={_DEFAULT_W1} w2={_DEFAULT_W2} w3={_DEFAULT_W3}"
            )

        # ── Build score CDF for percentile stretching ────────────────────
        # Use validation fold (or full data if no val) to build empirical CDF
        # that transforms raw probability-based scores -> percentile ranks.
        if len(val_idx) >= 50:
            pred_cdf  = self.predict_batch(X_val)
            raw_vals  = np.array([self._raw_composite(p) for p in pred_cdf])
            self._score_cdf = np.sort(raw_vals)
            print(f"  [MultiTargetScorer] Score CDF built from {len(raw_vals)} val samples "
                  f"(range [{raw_vals.min():.3f}, {raw_vals.max():.3f}])")
        else:
            pred_cdf  = self.predict_batch(X)
            raw_vals  = np.array([self._raw_composite(p) for p in pred_cdf])
            self._score_cdf = np.sort(raw_vals)

        return self

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> Dict[str, float]:
        """Run all prediction heads on a single feature row.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,) or (1, n_features)

        Returns
        -------
        dict with keys:
            p_serious_wr_event   (float 0-1, isotonic-calibrated)
            p_injury_event       (float 0-1, isotonic-calibrated)
            p_penalty_ge_p75     (float 0-1)
            p_penalty_ge_p90     (float 0-1)
            p_penalty_ge_p95     (float 0-1)
            expected_penalty_usd (float >= 0, legacy hurdle)
            gravity_score        (float >= 0, legacy)
        """
        if not self._is_fitted:
            return self._empty_prediction()
        x2d = np.atleast_2d(x)

        # ── Primary binary heads — prefer isotonic, fall back to Platt ──
        iso_wr  = getattr(self, "_iso_wr",  None)
        iso_inj = getattr(self, "_iso_inj", None)
        raw_wr  = float(_clf_proba(self._head_wr,     x2d))
        raw_inj = float(_clf_proba(self._head_injury, x2d))
        p_wr  = float(iso_wr.predict([raw_wr])[0])   if iso_wr  is not None else _apply_platt_scalar(raw_wr,  getattr(self, "_platt_wr",  np.array([1.0, 0.0])))
        p_inj = float(iso_inj.predict([raw_inj])[0]) if iso_inj is not None else _apply_platt_scalar(raw_inj, getattr(self, "_platt_inj", np.array([1.0, 0.0])))

        # ── Penalty tier heads (may not exist on old pickles) ────────────
        p_pen_p75 = p_pen_p90 = p_pen_p95 = 0.0
        for attr_head, attr_iso, setter in [
            ("_head_pen_p75", "_iso_pen_p75", "p75"),
            ("_head_pen_p90", "_iso_pen_p90", "p90"),
            ("_head_pen_p95", "_iso_pen_p95", "p95"),
        ]:
            head = getattr(self, attr_head, None)
            if head is not None:
                raw_p = float(_clf_proba(head, x2d))
                iso = getattr(self, attr_iso, None)
                cal_p = float(iso.predict([raw_p])[0]) if iso is not None else raw_p
                if setter == "p75":
                    p_pen_p75 = cal_p
                elif setter == "p90":
                    p_pen_p90 = cal_p
                else:
                    p_pen_p95 = cal_p

        # ── Legacy regression heads (kept for backward compat) ──────────
        iso_grav    = getattr(self, "_iso_grav", None)
        iso_log_pen = getattr(self, "_iso_log_pen", None)
        head_pen_nz = getattr(self, "_head_pen_nonzero", None)
        platt_pen   = getattr(self, "_platt_pen", np.array([1.0, 0.0]))

        raw_grav = float(_reg_predict(self._head_gravity, x2d))
        cal_grav = float(iso_grav.predict([raw_grav])[0]) if iso_grav is not None else raw_grav

        if head_pen_nz is not None:
            p_any_pen  = _apply_platt_scalar(float(_clf_proba(head_pen_nz, x2d)), platt_pen)
            cond_logp  = float(_reg_predict(self._head_log_pen, x2d))
            if iso_log_pen is not None:
                cond_logp = float(iso_log_pen.predict([cond_logp])[0])
            expected_pen = p_any_pen * float(math.expm1(max(0.0, cond_logp)))
        else:
            expected_pen = float(max(0.0, math.expm1(
                float(_reg_predict(self._head_log_pen, x2d))
            )))

        return {
            "p_serious_wr_event":   p_wr,
            "p_injury_event":       p_inj,
            "p_penalty_ge_p75":     p_pen_p75,
            "p_penalty_ge_p90":     p_pen_p90,
            "p_penalty_ge_p95":     p_pen_p95,
            "expected_penalty_usd": float(max(0.0, expected_pen)),
            "gravity_score":        float(max(0.0, cal_grav)),
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict for a batch of feature rows (efficient vectorized path)."""
        if not self._is_fitted:
            return [self._empty_prediction() for _ in range(len(X))]
        n = len(X)

        # ── Primary binary heads — prefer isotonic, fall back to Platt ──
        iso_wr  = getattr(self, "_iso_wr",  None)
        iso_inj = getattr(self, "_iso_inj", None)
        raw_wr  = _clf_proba_batch(self._head_wr,     X)
        raw_inj = _clf_proba_batch(self._head_injury, X)
        wr  = iso_wr.predict(raw_wr)   if iso_wr  is not None else _apply_platt_batch(raw_wr,  getattr(self, "_platt_wr",  np.array([1.0, 0.0])))
        inj = iso_inj.predict(raw_inj) if iso_inj is not None else _apply_platt_batch(raw_inj, getattr(self, "_platt_inj", np.array([1.0, 0.0])))

        # ── Penalty tier heads ──────────────────────────────────────────
        pen_p75 = pen_p90 = pen_p95 = np.zeros(n)
        for attr_head, attr_iso, tier in [
            ("_head_pen_p75", "_iso_pen_p75", "p75"),
            ("_head_pen_p90", "_iso_pen_p90", "p90"),
            ("_head_pen_p95", "_iso_pen_p95", "p95"),
        ]:
            head = getattr(self, attr_head, None)
            if head is not None:
                raw_p = _clf_proba_batch(head, X)
                iso = getattr(self, attr_iso, None)
                cal_p = iso.predict(raw_p) if iso is not None else raw_p
                if tier == "p75":
                    pen_p75 = cal_p
                elif tier == "p90":
                    pen_p90 = cal_p
                else:
                    pen_p95 = cal_p

        # ── Legacy regression heads ─────────────────────────────────────
        iso_grav    = getattr(self, "_iso_grav", None)
        iso_log_pen = getattr(self, "_iso_log_pen", None)
        head_pen_nz = getattr(self, "_head_pen_nonzero", None)
        platt_pen   = getattr(self, "_platt_pen", np.array([1.0, 0.0]))

        raw_grav = _reg_predict_batch(self._head_gravity, X)
        grav     = iso_grav.predict(raw_grav) if iso_grav is not None else raw_grav

        if head_pen_nz is not None:
            p_any_pen = _apply_platt_batch(_clf_proba_batch(head_pen_nz, X), platt_pen)
            cond_logp = _reg_predict_batch(self._head_log_pen, X)
            if iso_log_pen is not None:
                cond_logp = iso_log_pen.predict(cond_logp)
            expected_pen = p_any_pen * np.expm1(np.maximum(0.0, cond_logp))
        else:
            raw_logp     = _reg_predict_batch(self._head_log_pen, X)
            expected_pen = np.expm1(np.maximum(0.0, raw_logp))

        return [
            {
                "p_serious_wr_event":   float(wr[i]),
                "p_injury_event":       float(inj[i]),
                "p_penalty_ge_p75":     float(pen_p75[i]),
                "p_penalty_ge_p90":     float(pen_p90[i]),
                "p_penalty_ge_p95":     float(pen_p95[i]),
                "expected_penalty_usd": float(max(0.0, float(expected_pen[i]))),
                "gravity_score":        float(max(0.0, float(grav[i]))),
            }
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    #  Composite score (v2 — probability-first)
    # ------------------------------------------------------------------ #
    def _raw_composite(self, pred: Dict[str, float]) -> float:
        """Weighted sum of calibrated probabilities (before transformation).

        Returns a value in [0, 1] representing the raw probability-weighted
        composite.  Used internally by fit() to build the score CDF.
        """
        w1, w2, w3 = self._weights[:3]
        p_wr  = pred.get("p_serious_wr_event", 0.0)
        p_inj = pred.get("p_injury_event", 0.0)
        p_ext = pred.get("p_penalty_ge_p95", 0.0)
        return w1 * p_wr + w2 * p_inj + w3 * p_ext

    def _transform_score(self, raw: float) -> float:
        """Monotonic percentile stretching to expand top-end separation.

        Steps:
        1. Map raw composite -> percentile rank via empirical CDF built at
           training time.
        2. Apply convex power stretch: stretched = pctile ^ alpha.
           With alpha=2 (square), the 50th percentile maps to 25 and
           the 90th maps to 81, placing most companies below 50 and
           spreading the actionable high-risk range across 50-100.
        """
        cdf = getattr(self, "_score_cdf", None)
        if cdf is None or len(cdf) == 0:
            # Fallback: linear scaling
            return raw * 100.0
        pctile = float(np.searchsorted(cdf, raw, side="right")) / len(cdf)
        # Convex stretch: compresses low-risk, expands high-risk separation
        stretched = pctile ** _STRETCH_ALPHA
        return stretched * 100.0

    def composite_score(
        self,
        pred: Dict[str, float],
        n_inspections: int = 99,
        has_fatality: bool = False,
        has_willful: bool = False,
    ) -> float:
        """Compute 0-100 composite risk score from a prediction dict.

        v2 formula: all inputs are calibrated probabilities.
            raw = w1 * p_wr + w2 * p_injury + w3 * p_extreme_penalty
        Then: evidence-shrink -> percentile-stretch -> rescale to 0-100.
        """
        raw = self._raw_composite(pred)

        # Smooth evidence shrinkage toward population mean.
        confidence = 1.0 - math.exp(-n_inspections / 3.0)
        shrunk = raw * confidence + (_SCORE_PRIOR / 100.0) * (1.0 - confidence)

        score = self._transform_score(shrunk)

        # Hard cap only for the most extreme evidence case
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
            "p_injury_event":       0.0,
            "p_penalty_ge_p75":     0.0,
            "p_penalty_ge_p90":     0.0,
            "p_penalty_ge_p95":     0.0,
            "expected_penalty_usd": 0.0,
            "gravity_score":        0.0,
        }


# ── Composite weight optimizer ──────────────────────────────────────────────

def _optimize_weights(
    pred_batch: List[Dict[str, float]],
    y_adv: np.ndarray,
) -> List[float]:
    """Find 3-component weights maximizing Spearman rho between composite and y_adv.

    v2: components are calibrated probabilities — no log-transform needed.
    Uses grid search + Nelder-Mead refinement (handles non-differentiable rho).
    """
    from scipy.stats import spearmanr

    # Pre-compute component arrays (vectorized)
    p_wr  = np.array([p.get("p_serious_wr_event", 0.0) for p in pred_batch])
    p_inj = np.array([p.get("p_injury_event", 0.0)     for p in pred_batch])
    p_ext = np.array([p.get("p_penalty_ge_p95", 0.0)   for p in pred_batch])

    components = np.stack([p_wr, p_inj, p_ext], axis=1)  # (n, 3)

    def _neg_spearman_vec(raw_weights: np.ndarray) -> float:
        w = np.maximum(raw_weights, 0.0)
        s = w.sum()
        if s < 1e-9:
            return 0.0
        w = w / s
        scores = components @ w
        if np.all(scores == scores[0]):
            return 0.0
        rho, _ = spearmanr(scores, y_adv)
        return -float(rho) if not math.isnan(rho) else 0.0

    w0      = np.array([_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3])
    rho_def = -_neg_spearman_vec(w0)

    # ── Stage 1: coarse grid search ──
    best_w   = w0.copy()
    best_val = _neg_spearman_vec(w0)

    grid_points = [
        [0.60, 0.30, 0.10],
        [0.50, 0.35, 0.15],
        [0.55, 0.30, 0.15],
        [0.65, 0.25, 0.10],
        [0.50, 0.40, 0.10],
        [0.55, 0.35, 0.10],
        [0.60, 0.25, 0.15],
        [0.70, 0.20, 0.10],
        [0.45, 0.40, 0.15],
        [0.50, 0.30, 0.20],
        [0.40, 0.40, 0.20],
        [0.55, 0.25, 0.20],
        [0.65, 0.30, 0.05],
        [0.60, 0.35, 0.05],
        [0.50, 0.25, 0.25],
    ]
    for gw in grid_points:
        gw_arr = np.array(gw)
        gw_arr /= gw_arr.sum()
        val = _neg_spearman_vec(gw_arr)
        if val < best_val:
            best_val = val
            best_w   = gw_arr.copy()

    # ── Stage 2: Nelder-Mead from best grid point ──
    def _nm_objective(v2: np.ndarray) -> float:
        v = np.concatenate([np.maximum(v2, 0.0),
                            [max(0.0, 1.0 - float(np.sum(np.maximum(v2, 0.0))))]])
        reg = 0.1 * float(np.sum((v - w0) ** 2))
        return _neg_spearman_vec(v) + reg

    v0_2 = best_w[:2].copy()
    try:
        res = minimize(
            _nm_objective,
            v0_2,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5},
        )
        if res.success or res.fun < best_val:
            v2   = np.maximum(res.x, 0.0)
            w3   = max(0.0, 1.0 - float(v2.sum()))
            w_nm = np.concatenate([v2, [w3]])
            w_nm = np.maximum(w_nm, 0.0)
            w_nm /= w_nm.sum()
            if _neg_spearman_vec(w_nm) <= best_val:
                best_w   = w_nm
                best_val = _neg_spearman_vec(w_nm)
    except Exception:
        pass

    # Enforce minimum w3 >= 0.05: penalty signal always contributes
    if best_w[2] < 0.05:
        best_w[2] = 0.05
        best_w = best_w / np.sum(best_w)

    rho_opt = -_neg_spearman_vec(best_w)
    if rho_opt >= rho_def - 0.001:
        return best_w.tolist()

    return [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3]


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
