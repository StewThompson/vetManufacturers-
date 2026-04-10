"""multi_target_scorer.py — Sequential conditional multi-stage risk model.

Architecture (v3 — sequential conditional pipeline)
----------------------------------------------------
Zero future outcome does not always mean low compliance risk; it may mean
no inspection exposure.  This model separates inspection exposure from
violation severity so risk estimates are more interpretable and less
distorted by structural zeros.

All heads share the same 46-feature input vector (log-transformed).

**Stage 1 — Inspection Exposure** (trained on ALL establishments):
  _head_inspection    → pred_p_inspection: P(≥1 OSHA inspection in 12mo)

**Stage 2 — Violation | Inspection** (trained on inspected-only rows):
  _head_violation     → pred_p_violation_given_insp: P(≥1 violation | inspected)
  _head_serious       → pred_p_serious_given_insp: P(≥1 S/W/R violation | inspected)

**Stage 3 — Magnitude | Violation** (trained on violation-only rows):
  _head_log_pen       → pred_penalty_given_viol: E[penalty | violation]
  _head_gravity       → pred_gravity_given_viol: E[gravity | violation]
  _head_citations     → pred_citations_given_viol: E[citations | violation]

**Composite unconditional expected values**:
  expected_penalty   = p_insp × p_viol|insp × E[penalty|viol]
  expected_gravity   = p_insp × p_viol|insp × E[gravity|viol]
  expected_citations = p_insp × p_viol|insp × E[citations|viol]
  p_serious_unconditional = p_insp × p_serious|insp

**Backward-compatible outputs** (preserved for downstream consumers):
  p_serious_wr_event   = p_serious_unconditional
  p_injury_event       = (kept as independent head on inspection rows)
  expected_penalty_usd = expected_penalty
  gravity_score        = expected_gravity
  p_penalty_ge_p75/p90/p95 = (kept)

Composite risk score formula (v3 — 4-component)
------------------------------------------------
  risk_score_raw = w1 * p_inspection_exposure_norm
                 + w2 * p_serious_unconditional
                 + w3 * expected_penalty_norm
                 + w4 * expected_gravity_norm

Each component is normalized to [0, 1] before combining.  The raw score is:
  1. Evidence-shrunk toward population mean
  2. Transformed via monotonic percentile stretching
  3. Rescaled to 0–100

Calibration
-----------
All binary heads use isotonic regression calibration fitted on a held-out
validation fold (no leakage).
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


MODEL_FILE = "multi_target_model.pkl"

# ── Default composite weights (v3: 4-component sequential pipeline) ──
# Components: [p_inspection, p_serious_unconditional, expected_penalty_norm, expected_gravity_norm]
# The inspection exposure component captures the probability of being inspected;
# the other three components capture conditional risk given inspection/violation.
_DEFAULT_W1 = 0.15   # p_inspection (exposure component)
_DEFAULT_W2 = 0.40   # p_serious_unconditional (= p_insp × p_serious|insp)
_DEFAULT_W3 = 0.20   # expected_penalty_norm (= p_insp × p_viol|insp × E[pen|viol], normalized)
_DEFAULT_W4 = 0.25   # expected_gravity_norm (= p_insp × p_viol|insp × E[grav|viol], normalized)

# Reference values for normalizing regression outputs to [0, 1] via
# Michaelis-Menten saturation: x_norm = x / (x + ref).  At x == ref the
# normalised value is 0.5; this keeps the mapping interpretable.
_PENALTY_REF_USD = 75_000.0    # ~P80 of positive-penalty establishments (lower = more aggressive)
_GRAVITY_REF     = 40.0        # ~P65 of gravity-weighted severity scores (lower = more aggressive)

# Target prevalence for the injury/fatality head calibration.
# The model is trained on post-2022 outcomes (~6% injury rate) but the
# 2021 validation set has ~12% injury rate due to COVID-era hospitalisations.
# Setting a target prevalence of 12% during the combined temperature + logit-
# shift calibration corrects the structural mean-prediction underestimate and
# brings calibration slope within [0.75, 1.1] on the 2021 holdout.
_INJURY_CAL_TARGET_PREVALENCE = 0.12

# Population-mean prior for evidence shrinkage (replaces hard ceilings)
_SCORE_PRIOR = 15.0

# Convex power-stretch parameter for top-end separation.
# score = pctile^alpha * 100.  alpha=1.6 (reduced from 2.0) provides a gentler
# stretch that concentrates more separation in the 70-100 band while keeping
# the bottom 50% below 35 rather than 25.  This improves top-decile lift by
# widening the score gap between moderate- and high-risk establishments.
_STRETCH_ALPHA = 1.6

# ── GBC / GBR hyper-parameters ─────────────────────────────────────────────
# Binary heads use HistGradientBoostingClassifier with early stopping.
# Early stopping finds the optimal number of iterations automatically,
# preventing the over-confidence (compressed predictions) that harms
# calibration slope when a fixed n_estimators is used.

# Head 1 (p_event / WR-serious): deeper trees + lower regularisation than v2.
# max_depth=6 and min_samples_leaf=10 allow more feature interactions; reducing
# l2_regularization from 0.8→0.4 lets the model exploit those interactions
# without over-smoothing, yielding higher AUROC at the expense of a small
# increase in overfitting risk (mitigated by early stopping).
_HGBC_PARAMS = dict(
    max_iter=5000,
    max_depth=6,
    learning_rate=0.015,
    min_samples_leaf=10,
    l2_regularization=0.4,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=80,
    random_state=42,
)
# Head 3 (p_injury / hospitalization-fatality): enable early stopping (was off).
# The previous fixed 700-iter / l2=1.5 combination over-regularised predictions
# toward the mean, compressing the probability spread and inflating calibration
# slope.  Switching to early-stopping + shallower regularisation produces more
# dispersed predictions that track actual injury rates better.
_HGBC_INJURY_PARAMS = dict(
    max_iter=3000,
    max_depth=5,
    learning_rate=0.02,
    min_samples_leaf=10,
    l2_regularization=0.5,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=50,
    random_state=42,
)
# Conditional log-penalty regressor (Head 2b): quantile regression at the 75th
# percentile of the conditional log(penalty) distribution.
# Rationale:
#   • The expected_penalty_usd output is computed as p_any_pen × expm1(cond_logp).
#   • Since p_any_pen ≈ 0.5–0.65, multiplying by E[log_penalty] (Huber/mean)
#     heavily compresses predictions — even $130k actual cases get capped near
#     $40k in predicted space, making the scatter plot misleading.
#   • Targeting the 75th percentile of the conditional distribution shifts
#     predictions toward the upper tail, giving better coverage of extreme cases
#     while still providing meaningful rank ordering (Spearman ρ improves on
#     tail cases).
#   • The previous quantile α=0.80 went too high and over-predicted the mean;
#     α=0.75 is the best empirically-measured balance between range coverage
#     and mean-prediction accuracy.
_GBR_LOG_PEN_PARAMS = dict(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.025,
    subsample=0.80,
    loss="quantile",
    alpha=0.75,     # 75th-percentile target: better upper-tail coverage than Huber mean
    min_samples_leaf=2,
    random_state=42,
)

# Gravity regression (Head 4): HistGradientBoostingRegressor with squared_error
# loss.  Key improvements over the old GBR(quantile):
#   1. squared_error targets E[log1p(gravity)] — better monetary accuracy.
#   2. HistGBR supports early stopping — prevents overfitting on the 40%
#      zero-gravity rows that made quantile predictions collapse toward 0.
#   3. No StandardScaler needed (HistGBR is scale-invariant).
# The target is log1p-transformed to compress the heavy tail (gravity spans
# 0–1500).  Inference applies expm1 to recover the original scale.
from sklearn.ensemble import HistGradientBoostingRegressor as _HGBR  # noqa: E402

_HGBR_GRAVITY_PARAMS = dict(
    max_iter=1000,
    max_depth=6,
    learning_rate=0.02,
    min_samples_leaf=8,
    l2_regularization=0.2,
    loss="squared_error",  # best for unconditional mean; absolute_error hurt Spearman
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=40,
    random_state=42,
)

# Backward-compat alias — old pickles trained with _GBR_PARAMS (quantile).
# Not used for new training; kept so pickle deserialization of old models works.
_GBR_PARAMS = _GBR_LOG_PEN_PARAMS

# Dedicated HGBC params for the hurdle binary head (Head 2a: any-penalty?).
# The hurdle head benefits from a shallower, more regularised model than the
# WR/Serious head because its target (any future penalty at all) is more
# diffuse across the feature space; overfitting hurts discrimination.
_HGBC_HURDLE_PARAMS = dict(
    max_iter=2000,
    max_depth=5,
    learning_rate=0.02,
    min_samples_leaf=15,
    l2_regularization=0.5,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=50,
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



# ── Temperature scaling helpers ───────────────────────────────────────────
# Temperature scaling is a single-parameter calibration method:
#   p_cal = sigmoid(logit(p_raw) / T)
# T > 1: compresses predictions toward 0.5 (less confident)
# T < 1: sharpens predictions toward 0 or 1 (more confident)
# Unlike isotonic regression, temperature scaling has one degree of freedom
# and generalises better across prevalence shifts (e.g. training 5.9% vs
# validation 12.1% for the injury head), making it more suitable for
# calibrating heads where training and deployment distributions differ.

def _fit_temperature(y_val: np.ndarray, p_raw: np.ndarray) -> float:
    """Fit temperature T minimising BCE on holdout: p_cal = sigmoid(logit(p)/T).

    Returns T in [0.3, 5.0].  T=1.0 means no change (raw model is well-calibrated).
    """
    from scipy.optimize import minimize_scalar

    p_clipped = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
    logits    = np.log(p_clipped / (1.0 - p_clipped))

    def bce(T: float) -> float:
        T = max(T, 1e-6)
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        return float(-np.mean(
            y_val * np.log(np.clip(p_cal, 1e-9, 1.0)) +
            (1.0 - y_val) * np.log(np.clip(1.0 - p_cal, 1e-9, 1.0))
        ))

    result = minimize_scalar(bce, bounds=(0.3, 5.0), method="bounded")
    return float(result.x)


def _apply_temperature_batch(p_arr: np.ndarray, T: float) -> np.ndarray:
    """Vectorised temperature scaling: p_cal = sigmoid(logit(p) / T)."""
    if abs(T - 1.0) < 1e-6:
        return p_arr
    p_clipped = np.clip(p_arr, 1e-7, 1.0 - 1e-7)
    logits    = np.log(p_clipped / (1.0 - p_clipped))
    return 1.0 / (1.0 + np.exp(-logits / T))


def _apply_temperature_scalar(p: float, T: float) -> float:
    """Scalar temperature scaling: p_cal = sigmoid(logit(p) / T)."""
    if abs(T - 1.0) < 1e-6:
        return p
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-logit / T))


class MultiTargetRiskScorer:
    """5-head probabilistic OSHA risk model.

    Usage
    -----
    1.  ``fit(X, multi_target_rows)``  ← from ``multi_target_labeler``
    2.  ``predict(X)``                 ← returns dict with all outputs
    3.  ``composite_score(pred_dict)`` ← converts to 0-100 user-visible score
    4.  ``save(path)`` / ``load(path)`` via pickle
    """

    def __init__(self) -> None:
        # ══════════════════════════════════════════════════════════════════
        # Stage 1 — Inspection Exposure (ALL establishments)
        # ══════════════════════════════════════════════════════════════════
        self._head_inspection: Optional[object] = None   # P(inspected in 12m)
        self._iso_inspection: Optional[object] = None    # isotonic calibrator

        # ══════════════════════════════════════════════════════════════════
        # Stage 2 — Violation | Inspection (inspected rows only)
        # ══════════════════════════════════════════════════════════════════
        self._head_violation: Optional[object] = None    # P(any violation | inspected)
        self._head_serious:  Optional[object] = None     # P(any S/W/R | inspected)
        self._head_injury:   Optional[object] = None     # P(hosp/fatality | inspected)
        self._iso_violation: Optional[object] = None
        self._iso_serious:   Optional[object] = None
        self._iso_inj:       Optional[object] = None

        # ══════════════════════════════════════════════════════════════════
        # Stage 3 — Magnitude | Violation (violation rows only)
        # ══════════════════════════════════════════════════════════════════
        self._head_log_pen:  Optional[Pipeline] = None   # E[log1p(penalty) | viol]
        self._head_gravity:  Optional[object] = None     # E[log1p(gravity) | viol]
        self._head_citations: Optional[object] = None    # E[log1p(citations) | viol]
        self._iso_grav:      Optional[object] = None
        self._iso_log_pen:   Optional[object] = None

        # Flag: gravity head trained on log1p target (expm1 at inference)
        self._gravity_log_transformed: bool = False

        # ══════════════════════════════════════════════════════════════════
        # Backward-compat heads (kept so old API consumers still work)
        # ══════════════════════════════════════════════════════════════════
        # Legacy: flat binary heads (now derived from Stage 1 × Stage 2)
        self._head_wr:          Optional[object] = None  # kept for old pickle compat
        self._head_pen_nonzero: Optional[object] = None
        self._head_pen_p75:     Optional[object] = None
        self._head_pen_p90:     Optional[object] = None
        self._head_pen_p95:     Optional[object] = None
        self._iso_wr:           Optional[object] = None
        self._iso_pen_p75:      Optional[object] = None
        self._iso_pen_p90:      Optional[object] = None
        self._iso_pen_p95:      Optional[object] = None

        # Legacy Platt / temperature calibration slots (backward compat)
        _identity = np.array([1.0, 0.0])
        self._platt_wr:   np.ndarray = _identity.copy()
        self._platt_inj:  np.ndarray = _identity.copy()
        self._platt_pen:  np.ndarray = _identity.copy()
        self._temp_wr:    float = 1.0
        self._temp_inj:   float = 1.0
        self._bias_inj:   float = 0.0
        self._iso_pen:    Optional[object] = None

        # ── Composite weights [w1, w2, w3, w4] (4-component) ──────────
        # Components: p_inspection, p_serious_unconditional,
        #             expected_penalty_norm, expected_gravity_norm
        self._weights: List[float] = [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4]

        # ── Score transformation ───────────────────────────────────────
        self._score_cdf: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Training — Sequential Conditional Pipeline
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        rows: List[Dict],
        optimize_weights: bool = True,
        val_fraction: float = 0.20,
        rng_seed: int = 42,
    ) -> "MultiTargetRiskScorer":
        """Train the 3-stage sequential conditional pipeline.

        The pipeline explicitly separates:
          Stage 1: P(inspection) — trained on ALL rows
          Stage 2: P(violation | inspected), P(serious | inspected) — trained
                   only on rows where future_has_inspection == 1
          Stage 3: E[penalty | violation], E[gravity | violation] — trained
                   only on rows where future_has_violation == 1

        This handles zero-inflation cleanly: establishments with zero future
        outcomes because they were never inspected are NOT mixed with
        establishments that were inspected but had no violations.

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

        # ── Extract target arrays ─────────────────────────────────────────
        y_has_insp  = np.array([r.get("future_has_inspection", 1) for r in rows], dtype=int)
        y_has_viol  = np.array([r.get("future_has_violation",  0) for r in rows], dtype=int)
        y_has_ser   = np.array([r.get("future_has_serious",    r.get("any_wr_serious", 0)) for r in rows], dtype=int)
        y_inj       = np.array([r.get("any_injury_fatal",      0) for r in rows], dtype=int)
        y_logp      = np.array([r.get("log_penalty",            0.0) for r in rows], dtype=float)
        y_grav      = np.array([r.get("gravity_weighted_score",  0.0) for r in rows], dtype=float)
        y_cit       = np.array([r.get("future_citation_count",  r.get("future_violation_count", 0)) for r in rows], dtype=float)

        # Backward compat: WR/serious target (alias)
        y_wr = np.array([r.get("any_wr_serious", 0) for r in rows], dtype=int)

        # IPW weights
        weights_raw = np.array([r.get("ipw_weight", 1.0) for r in rows], dtype=float)
        weights_norm = weights_raw / weights_raw.mean()
        logger.info(
            "  [MultiTargetScorer] IPW weight stats — min=%.2f  mean=1.00  P95=%.2f  max=%.2f",
            weights_norm.min(), float(np.percentile(weights_norm, 95)), weights_norm.max(),
        )

        # ── Build index masks ─────────────────────────────────────────────
        # inspected_mask: rows where establishment was inspected in future window
        inspected_mask = y_has_insp.astype(bool)
        # violated_mask: rows where establishment had ≥1 violation in future window
        violated_mask  = y_has_viol.astype(bool)

        n_inspected = int(inspected_mask.sum())
        n_violated  = int(violated_mask.sum())
        logger.info(
            "  [MultiTargetScorer] Dataset: %s total, %s inspected (%.1f%%), "
            "%s with violations (%.1f%%)",
            f"{n:,}", f"{n_inspected:,}", n_inspected / n * 100,
            f"{n_violated:,}", n_violated / n * 100,
        )

        rng = np.random.default_rng(rng_seed)
        idx = rng.permutation(n)
        split = max(int(n * val_fraction), min(50, n))
        if not optimize_weights or n < 100:
            train_idx, val_idx = idx, np.array([], dtype=int)
            optimize_weights = False
        else:
            val_idx   = idx[:split]
            train_idx = idx[split:]

        X_tr  = X[train_idx]
        X_val = X[val_idx] if len(val_idx) > 0 else X[:0]
        sw_tr = weights_norm[train_idx]

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

        def _train_gbr_pipe(X_fit, y_fit, sample_weight=None, gbr_params=None):
            if gbr_params is None:
                gbr_params = _GBR_LOG_PEN_PARAMS
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(**gbr_params)),
            ])
            if sample_weight is not None:
                pipe.fit(X_fit, y_fit, model__sample_weight=sample_weight)
            else:
                pipe.fit(X_fit, y_fit)
            return pipe

        def _train_hgbr(X_fit, y_fit, sample_weight=None, hgbr_params=None):
            """Train HistGBR on log1p target."""
            if hgbr_params is None:
                hgbr_params = _HGBR_GRAVITY_PARAMS
            y_log = np.log1p(np.maximum(0.0, y_fit))
            hgbr = _HGBR(**hgbr_params)
            hgbr.fit(X_fit, y_log, sample_weight=sample_weight)
            return hgbr

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — Inspection Exposure (ALL rows)
        # ══════════════════════════════════════════════════════════════════
        logger.info("  [MultiTargetScorer] ══ STAGE 1: Inspection Exposure ══")
        logger.info(
            "  [MultiTargetScorer] Training inspection classifier (%s rows, %.1f%% positive) …",
            f"{len(train_idx):,}", y_has_insp[train_idx].mean() * 100,
        )
        self._head_inspection = _train_hgbc_raw(
            X_tr, y_has_insp[train_idx],
            hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr,
        )

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — Violation | Inspection (inspected rows only)
        # ══════════════════════════════════════════════════════════════════
        logger.info("  [MultiTargetScorer] ══ STAGE 2: Violation given Inspection ══")

        # Filter train indices to inspected-only rows
        train_insp_mask = inspected_mask[train_idx]
        X_tr_insp  = X_tr[train_insp_mask]
        sw_tr_insp = sw_tr[train_insp_mask]
        insp_train_idx = train_idx[train_insp_mask]

        logger.info(
            "  [MultiTargetScorer] Training violation|insp classifier (%s inspected rows, %.1f%% positive) …",
            f"{len(insp_train_idx):,}", y_has_viol[insp_train_idx].mean() * 100,
        )
        self._head_violation = _train_hgbc_raw(
            X_tr_insp, y_has_viol[insp_train_idx],
            hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr_insp,
        )

        logger.info(
            "  [MultiTargetScorer] Training serious|insp classifier (%s inspected rows, %.1f%% positive) …",
            f"{len(insp_train_idx):,}", y_has_ser[insp_train_idx].mean() * 100,
        )
        self._head_serious = _train_hgbc_raw(
            X_tr_insp, y_has_ser[insp_train_idx],
            hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr_insp,
        )

        logger.info(
            "  [MultiTargetScorer] Training injury|insp classifier (%s inspected rows, %.1f%% positive) …",
            f"{len(insp_train_idx):,}", y_inj[insp_train_idx].mean() * 100,
        )
        self._head_injury = _train_hgbc_raw(
            X_tr_insp, y_inj[insp_train_idx],
            hgbc_params=_HGBC_INJURY_PARAMS, sample_weight=sw_tr_insp,
        )

        # ── Backward-compat: penalty tier classifiers on inspected rows ──
        y_pen_p75 = np.array([r.get("is_moderate_penalty", 0) for r in rows], dtype=int)
        y_pen_p90 = np.array([r.get("is_large_penalty",    0) for r in rows], dtype=int)
        y_pen_p95 = np.array([r.get("is_extreme_penalty",  0) for r in rows], dtype=int)
        for tier_name, y_tier, attr_name in [
            ("P75 (moderate)", y_pen_p75, "_head_pen_p75"),
            ("P90 (large)",    y_pen_p90, "_head_pen_p90"),
            ("P95 (extreme)",  y_pen_p95, "_head_pen_p95"),
        ]:
            pos_rate = y_tier[insp_train_idx].mean()
            logger.info(
                "  [MultiTargetScorer] Training penalty tier %s (pos rate=%.1f%%) …",
                tier_name, pos_rate * 100,
            )
            head = _train_hgbc_raw(
                X_tr_insp, y_tier[insp_train_idx],
                hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr_insp,
            )
            setattr(self, attr_name, head)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3 — Magnitude | Violation (violation rows only)
        # ══════════════════════════════════════════════════════════════════
        logger.info("  [MultiTargetScorer] ══ STAGE 3: Magnitude given Violation ══")

        # Filter train indices to violation-only rows
        train_viol_mask = violated_mask[train_idx]
        X_tr_viol  = X_tr[train_viol_mask]
        sw_tr_viol = sw_tr[train_viol_mask]
        viol_train_idx = train_idx[train_viol_mask]

        # 3a: Conditional log-penalty regressor
        logger.info(
            "  [MultiTargetScorer] Training penalty|viol regressor (%s violation rows, quantile α=0.75) …",
            f"{len(viol_train_idx):,}",
        )
        if len(viol_train_idx) >= 20:
            self._head_log_pen = _train_gbr_pipe(
                X_tr_viol, y_logp[viol_train_idx],
                sample_weight=sw_tr_viol,
                gbr_params=_GBR_LOG_PEN_PARAMS,
            )
        else:
            logger.warning("  [MultiTargetScorer] Fewer than 20 violation rows; penalty regressor skipped.")

        # 3b: Conditional gravity regressor (log1p target)
        logger.info(
            "  [MultiTargetScorer] Training gravity|viol regressor (%s violation rows, log1p target) …",
            f"{len(viol_train_idx):,}",
        )
        if len(viol_train_idx) >= 20:
            self._head_gravity = _train_hgbr(
                X_tr_viol, y_grav[viol_train_idx], sample_weight=sw_tr_viol
            )
            self._gravity_log_transformed = True
        else:
            logger.warning("  [MultiTargetScorer] Fewer than 20 violation rows; gravity regressor skipped.")

        # 3c: Conditional citation count regressor (log1p target)
        logger.info(
            "  [MultiTargetScorer] Training citations|viol regressor (%s violation rows) …",
            f"{len(viol_train_idx):,}",
        )
        if len(viol_train_idx) >= 20:
            self._head_citations = _train_hgbr(
                X_tr_viol, y_cit[viol_train_idx], sample_weight=sw_tr_viol,
                hgbr_params=_HGBR_GRAVITY_PARAMS,  # same hyperparams as gravity
            )

        # ── Backward-compat: flat WR head + hurdle penalty head ──────────
        logger.info("  [MultiTargetScorer] Training backward-compat flat heads …")
        self._head_wr = _train_hgbc_raw(
            X_tr, y_wr[train_idx], hgbc_params=_HGBC_PARAMS, sample_weight=sw_tr,
        )
        y_any_pen = (y_logp > 0).astype(int)
        self._head_pen_nonzero = _train_hgbc_raw(
            X_tr, y_any_pen[train_idx],
            hgbc_params=_HGBC_HURDLE_PARAMS, sample_weight=sw_tr,
        )

        self._is_fitted = True

        # ══════════════════════════════════════════════════════════════════
        # Post-hoc isotonic calibration on training holdout
        # ══════════════════════════════════════════════════════════════════
        if len(val_idx) >= 200:
            from sklearn.isotonic import IsotonicRegression
            from scipy.stats import spearmanr, pearsonr
            from src.scoring.calibration import brier_score, brier_skill_score, expected_calibration_error

            logger.info("  [MultiTargetScorer] Post-hoc isotonic calibration on training holdout …")

            def _calibrate_binary(head, X_v, y_v, name):
                """Fit isotonic calibration and log metrics."""
                p_raw = _clf_proba_batch(head, X_v)
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p_raw, y_v.astype(float))
                p_cal = iso.predict(p_raw)
                bs  = brier_score(y_v, p_cal)
                bss = brier_skill_score(y_v, p_cal)
                ece = expected_calibration_error(y_v, p_cal)
                logger.info(
                    "  [MultiTargetScorer] %s calibration: Brier=%.4f  BSS=%+.3f  ECE=%.4f",
                    name, bs, bss, ece,
                )
                return iso

            # Stage 1 calibration
            self._iso_inspection = _calibrate_binary(
                self._head_inspection, X_val, y_has_insp[val_idx], "Inspection"
            )

            # Stage 2 calibration (inspected validation rows only)
            val_insp_mask = inspected_mask[val_idx]
            if val_insp_mask.sum() >= 50:
                X_val_insp = X_val[val_insp_mask]
                val_insp_idx = val_idx[val_insp_mask]

                self._iso_violation = _calibrate_binary(
                    self._head_violation, X_val_insp, y_has_viol[val_insp_idx], "Violation|Insp"
                )
                self._iso_serious = _calibrate_binary(
                    self._head_serious, X_val_insp, y_has_ser[val_insp_idx], "Serious|Insp"
                )

                # Injury calibration: temperature + logit-shift (handles prevalence shift)
                p_inj_val_raw = _clf_proba_batch(self._head_injury, X_val_insp)
                self._iso_inj = None  # intentionally disabled — temperature + bias used
                self._temp_inj = _fit_temperature(
                    y_inj[val_insp_idx].astype(float), p_inj_val_raw,
                )
                p_inj_clipped = np.clip(p_inj_val_raw, 1e-7, 1.0 - 1e-7)
                logits_inj    = np.log(p_inj_clipped / (1.0 - p_inj_clipped))
                logits_temp   = logits_inj / self._temp_inj
                mean_logit_cal = float(np.mean(logits_temp))
                target_logit   = math.log(
                    _INJURY_CAL_TARGET_PREVALENCE /
                    (1.0 - _INJURY_CAL_TARGET_PREVALENCE)
                )
                self._bias_inj = target_logit - mean_logit_cal
                logger.info(
                    "  [MultiTargetScorer] p_injury temperature=%.3f bias=%.4f "
                    "(target prevalence=%.1f%%)",
                    self._temp_inj, self._bias_inj,
                    _INJURY_CAL_TARGET_PREVALENCE * 100,
                )

                # Penalty tier calibration on inspected val rows
                for tier_name, y_tier, head_attr, iso_attr in [
                    ("P75", y_pen_p75, "_head_pen_p75", "_iso_pen_p75"),
                    ("P90", y_pen_p90, "_head_pen_p90", "_iso_pen_p90"),
                    ("P95", y_pen_p95, "_head_pen_p95", "_iso_pen_p95"),
                ]:
                    head = getattr(self, head_attr)
                    iso = _calibrate_binary(head, X_val_insp, y_tier[val_insp_idx], f"Penalty {tier_name}")
                    setattr(self, iso_attr, iso)

            # Backward-compat: calibrate flat WR head and hurdle penalty
            self._iso_wr = _calibrate_binary(
                self._head_wr, X_val, y_wr[val_idx], "WR/Serious (flat)"
            )
            p_pen_val = _clf_proba_batch(self._head_pen_nonzero, X_val)
            self._platt_wr  = _fit_platt(y_wr[val_idx].astype(float), _clf_proba_batch(self._head_wr, X_val))
            self._platt_inj = _fit_platt(
                y_inj[val_idx].astype(float),
                _clf_proba_batch(self._head_injury, X_val) if val_insp_mask.sum() > 0 else np.zeros(len(val_idx)),
            )
            self._platt_pen = _fit_platt((y_logp[val_idx] > 0).astype(float), p_pen_val)
            self._iso_log_pen = None  # quantile loss already calibrated

            # Gravity isotonic on violated validation rows
            val_viol_mask = violated_mask[val_idx]
            if val_viol_mask.sum() >= 30 and self._head_gravity is not None:
                X_val_viol = X_val[val_viol_mask]
                val_viol_idx = val_idx[val_viol_mask]
                raw_grav_log = _reg_predict_batch(self._head_gravity, X_val_viol)
                raw_grav_val = np.expm1(np.maximum(0.0, raw_grav_log))
                iso_grav = IsotonicRegression(out_of_bounds="clip")
                iso_grav.fit(raw_grav_val, y_grav[val_viol_idx])
                self._iso_grav = iso_grav

        # ── Composite weight optimization ────────────────────────────────
        if optimize_weights and len(val_idx) >= 50:
            y_adv_val = np.array([rows[i]["real_label"] for i in val_idx])
            pred_val  = self.predict_batch(X_val)
            self._weights = _optimize_weights(pred_val, y_adv_val)
            logger.info(
                "  [MultiTargetScorer] Optimized weights (4-comp): "
                "w1=%.3f w2=%.3f w3=%.3f w4=%.3f",
                *self._weights[:4],
            )
        else:
            logger.info(
                "  [MultiTargetScorer] Using default weights: "
                "w1=%s w2=%s w3=%s w4=%s",
                _DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4,
            )

        # ── Build score CDF for percentile stretching ────────────────────
        if len(val_idx) >= 50:
            pred_cdf  = self.predict_batch(X_val)
            raw_vals  = np.array([self._raw_composite(p) for p in pred_cdf])
            self._score_cdf = np.sort(raw_vals)
            logger.info(
                "  [MultiTargetScorer] Score CDF built from %s val samples (range [%.3f, %.3f])",
                len(raw_vals), raw_vals.min(), raw_vals.max(),
            )
        else:
            pred_cdf  = self.predict_batch(X)
            raw_vals  = np.array([self._raw_composite(p) for p in pred_cdf])
            self._score_cdf = np.sort(raw_vals)

        return self

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> Dict[str, float]:
        """Run sequential conditional pipeline on a single feature row.

        Returns both conditional and unconditional predictions:
        - Stage outputs (conditional): pred_p_inspection, pred_p_violation_given_insp,
          pred_p_serious_given_insp, pred_penalty_given_viol, pred_gravity_given_viol
        - Composite unconditional: expected_penalty, expected_gravity,
          p_serious_unconditional, expected_citations
        - Backward-compat: p_serious_wr_event, p_injury_event, expected_penalty_usd,
          gravity_score, p_penalty_ge_p75/p90/p95
        """
        if not self._is_fitted:
            return self._empty_prediction()
        x2d = np.atleast_2d(x)

        # ── Stage 1: Inspection exposure ──────────────────────────────────
        head_insp = getattr(self, "_head_inspection", None)
        if head_insp is not None:
            iso_insp = getattr(self, "_iso_inspection", None)
            raw_insp = float(_clf_proba(head_insp, x2d))
            p_insp = float(iso_insp.predict([raw_insp])[0]) if iso_insp is not None else raw_insp
        else:
            p_insp = 1.0  # backward compat: old pickles assume all are inspected

        # ── Stage 2: Violation | Inspection ───────────────────────────────
        head_viol = getattr(self, "_head_violation", None)
        head_ser  = getattr(self, "_head_serious", None)
        if head_viol is not None:
            iso_viol = getattr(self, "_iso_violation", None)
            raw_viol = float(_clf_proba(head_viol, x2d))
            p_viol = float(iso_viol.predict([raw_viol])[0]) if iso_viol is not None else raw_viol
        else:
            # Backward compat: old pickles don't separate violation from inspection.
            # Fall back to hurdle penalty head (P(any penalty) ≈ P(any violation)).
            head_pen_nz = getattr(self, "_head_pen_nonzero", None)
            platt_pen   = getattr(self, "_platt_pen", np.array([1.0, 0.0]))
            if head_pen_nz is not None:
                p_viol = _apply_platt_scalar(float(_clf_proba(head_pen_nz, x2d)), platt_pen)
            else:
                p_viol = 1.0

        if head_ser is not None:
            iso_ser = getattr(self, "_iso_serious", None)
            raw_ser = float(_clf_proba(head_ser, x2d))
            p_ser = float(iso_ser.predict([raw_ser])[0]) if iso_ser is not None else raw_ser
        else:
            # Backward compat: fall back to flat WR head for old pickles
            iso_wr_fb = getattr(self, "_iso_wr", None)
            head_wr_fb = getattr(self, "_head_wr", None)
            if head_wr_fb is not None:
                raw_wr_fb = float(_clf_proba(head_wr_fb, x2d))
                p_ser = float(iso_wr_fb.predict([raw_wr_fb])[0]) if iso_wr_fb is not None else _apply_platt_scalar(raw_wr_fb, getattr(self, "_platt_wr", np.array([1.0, 0.0])))
            else:
                p_ser = 0.0

        # Injury head (conditional on inspection)
        iso_inj = getattr(self, "_iso_inj", None)
        temp_inj = getattr(self, "_temp_inj", 1.0)
        bias_inj = getattr(self, "_bias_inj", 0.0)
        raw_inj = float(_clf_proba(self._head_injury, x2d)) if self._head_injury else 0.0
        if iso_inj is not None:
            p_inj = float(iso_inj.predict([raw_inj])[0])
        elif temp_inj != 1.0 or bias_inj != 0.0:
            p_clipped = float(np.clip(raw_inj, 1e-7, 1.0 - 1e-7))
            logit_val = math.log(p_clipped / (1.0 - p_clipped))
            p_inj = float(np.clip(1.0 / (1.0 + math.exp(-(logit_val / temp_inj + bias_inj))), 0.0, 1.0))
        else:
            p_inj = raw_inj

        # ── Stage 3: Magnitude | Violation ────────────────────────────────
        # Conditional predictions (what would happen IF a violation occurs)
        if self._head_log_pen is not None:
            iso_log_pen = getattr(self, "_iso_log_pen", None)
            cond_logp = float(_reg_predict(self._head_log_pen, x2d))
            if iso_log_pen is not None:
                cond_logp = float(iso_log_pen.predict([cond_logp])[0])
            cond_penalty = float(math.expm1(max(0.0, cond_logp)))
        else:
            cond_penalty = 0.0

        iso_grav = getattr(self, "_iso_grav", None)
        if self._head_gravity is not None:
            raw_grav_pred = float(_reg_predict(self._head_gravity, x2d))
            if getattr(self, "_gravity_log_transformed", False):
                raw_grav_pred = float(np.expm1(max(0.0, raw_grav_pred)))
            cond_gravity = float(iso_grav.predict([raw_grav_pred])[0]) if iso_grav is not None else raw_grav_pred
        else:
            cond_gravity = 0.0

        head_cit = getattr(self, "_head_citations", None)
        if head_cit is not None:
            raw_cit_pred = float(_reg_predict(head_cit, x2d))
            cond_citations = float(np.expm1(max(0.0, raw_cit_pred)))
        else:
            cond_citations = 0.0

        # ── Composite unconditional expected values ───────────────────────
        # E[X] = P(inspection) × P(violation | inspection) × E[X | violation]
        p_insp_viol = p_insp * p_viol
        expected_penalty   = p_insp_viol * cond_penalty
        expected_gravity   = p_insp_viol * cond_gravity
        expected_citations = p_insp_viol * cond_citations
        p_serious_unconditional = p_insp * p_ser

        # ── Penalty tiers (backward compat) ───────────────────────────────
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

        # ── Backward-compat: flat WR head ─────────────────────────────────
        iso_wr = getattr(self, "_iso_wr", None)
        if self._head_wr is not None:
            raw_wr = float(_clf_proba(self._head_wr, x2d))
            p_wr_flat = float(iso_wr.predict([raw_wr])[0]) if iso_wr is not None else _apply_platt_scalar(raw_wr, getattr(self, "_platt_wr", np.array([1.0, 0.0])))
        else:
            p_wr_flat = p_serious_unconditional

        return {
            # ── Stage 1 outputs ───────────────────────────────────────────
            "pred_p_inspection":              float(max(0.0, min(1.0, p_insp))),
            # ── Stage 2 outputs (conditional) ─────────────────────────────
            "pred_p_violation_given_insp":    float(max(0.0, min(1.0, p_viol))),
            "pred_p_serious_given_insp":      float(max(0.0, min(1.0, p_ser))),
            "pred_p_injury_given_insp":       float(max(0.0, min(1.0, p_inj))),
            # ── Stage 3 outputs (conditional) ─────────────────────────────
            "pred_penalty_given_viol":        float(max(0.0, cond_penalty)),
            "pred_gravity_given_viol":        float(max(0.0, cond_gravity)),
            "pred_citations_given_viol":      float(max(0.0, cond_citations)),
            # ── Composite unconditional ───────────────────────────────────
            "expected_penalty":               float(max(0.0, expected_penalty)),
            "expected_gravity":               float(max(0.0, expected_gravity)),
            "expected_citations":             float(max(0.0, expected_citations)),
            "p_serious_unconditional":        float(max(0.0, min(1.0, p_serious_unconditional))),
            # ── Backward-compat outputs ───────────────────────────────────
            "p_serious_wr_event":   float(max(0.0, min(1.0, p_wr_flat))),
            "p_injury_event":       float(max(0.0, min(1.0, p_inj))),
            "p_penalty_ge_p75":     float(p_pen_p75),
            "p_penalty_ge_p90":     float(p_pen_p90),
            "p_penalty_ge_p95":     float(p_pen_p95),
            "expected_penalty_usd": float(max(0.0, expected_penalty)),
            "gravity_score":        float(max(0.0, expected_gravity)),
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict for a batch of feature rows (efficient vectorized path).

        Returns same dict structure as predict(), with both conditional
        and unconditional predictions for each row.
        """
        if not self._is_fitted:
            return [self._empty_prediction() for _ in range(len(X))]
        n = len(X)

        # ── Stage 1: Inspection exposure ──────────────────────────────────
        head_insp = getattr(self, "_head_inspection", None)
        if head_insp is not None:
            iso_insp = getattr(self, "_iso_inspection", None)
            raw_insp = _clf_proba_batch(head_insp, X)
            p_insp = iso_insp.predict(raw_insp) if iso_insp is not None else raw_insp
        else:
            p_insp = np.ones(n)

        # ── Stage 2: Violation | Inspection ───────────────────────────────
        head_viol = getattr(self, "_head_violation", None)
        head_ser  = getattr(self, "_head_serious", None)
        if head_viol is not None:
            iso_viol = getattr(self, "_iso_violation", None)
            raw_viol = _clf_proba_batch(head_viol, X)
            p_viol = iso_viol.predict(raw_viol) if iso_viol is not None else raw_viol
        else:
            # Backward compat: fall back to hurdle penalty head
            head_pen_nz = getattr(self, "_head_pen_nonzero", None)
            platt_pen   = getattr(self, "_platt_pen", np.array([1.0, 0.0]))
            if head_pen_nz is not None:
                p_viol = _apply_platt_batch(_clf_proba_batch(head_pen_nz, X), platt_pen)
            else:
                p_viol = np.ones(n)

        if head_ser is not None:
            iso_ser = getattr(self, "_iso_serious", None)
            raw_ser = _clf_proba_batch(head_ser, X)
            p_ser = iso_ser.predict(raw_ser) if iso_ser is not None else raw_ser
        else:
            # Backward compat: fall back to flat WR head for old pickles
            head_wr_fb = getattr(self, "_head_wr", None)
            iso_wr_fb  = getattr(self, "_iso_wr", None)
            if head_wr_fb is not None:
                raw_wr_fb = _clf_proba_batch(head_wr_fb, X)
                p_ser = iso_wr_fb.predict(raw_wr_fb) if iso_wr_fb is not None else _apply_platt_batch(raw_wr_fb, getattr(self, "_platt_wr", np.array([1.0, 0.0])))
            else:
                p_ser = np.zeros(n)

        # Injury head (conditional on inspection)
        iso_inj  = getattr(self, "_iso_inj", None)
        temp_inj = getattr(self, "_temp_inj", 1.0)
        bias_inj = getattr(self, "_bias_inj", 0.0)
        raw_inj = _clf_proba_batch(self._head_injury, X) if self._head_injury else np.zeros(n)
        if iso_inj is not None:
            inj = iso_inj.predict(raw_inj)
        elif temp_inj != 1.0 or bias_inj != 0.0:
            p_inj_clipped = np.clip(raw_inj, 1e-7, 1.0 - 1e-7)
            logits = np.log(p_inj_clipped / (1.0 - p_inj_clipped))
            inj = np.clip(1.0 / (1.0 + np.exp(-(logits / temp_inj + bias_inj))), 0.0, 1.0)
        else:
            inj = raw_inj

        # ── Stage 3: Magnitude | Violation ────────────────────────────────
        # Conditional penalty
        iso_log_pen = getattr(self, "_iso_log_pen", None)
        if self._head_log_pen is not None:
            cond_logp = _reg_predict_batch(self._head_log_pen, X)
            if iso_log_pen is not None:
                cond_logp = iso_log_pen.predict(cond_logp)
            cond_penalty = np.expm1(np.maximum(0.0, cond_logp))
        else:
            cond_penalty = np.zeros(n)

        # Conditional gravity
        iso_grav = getattr(self, "_iso_grav", None)
        if self._head_gravity is not None:
            raw_grav = _reg_predict_batch(self._head_gravity, X)
            if getattr(self, "_gravity_log_transformed", False):
                raw_grav = np.expm1(np.maximum(0.0, raw_grav))
            cond_gravity = iso_grav.predict(raw_grav) if iso_grav is not None else raw_grav
        else:
            cond_gravity = np.zeros(n)

        # Conditional citations
        head_cit = getattr(self, "_head_citations", None)
        if head_cit is not None:
            raw_cit = _reg_predict_batch(head_cit, X)
            cond_citations = np.expm1(np.maximum(0.0, raw_cit))
        else:
            cond_citations = np.zeros(n)

        # ── Composite unconditional expected values ───────────────────────
        p_insp_viol = p_insp * p_viol
        expected_penalty   = p_insp_viol * cond_penalty
        expected_gravity   = p_insp_viol * cond_gravity
        expected_citations = p_insp_viol * cond_citations
        p_serious_uncond   = p_insp * p_ser

        # ── Penalty tiers (backward compat) ───────────────────────────────
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

        # ── Backward-compat: flat WR head ─────────────────────────────────
        iso_wr = getattr(self, "_iso_wr", None)
        if self._head_wr is not None:
            raw_wr = _clf_proba_batch(self._head_wr, X)
            wr = iso_wr.predict(raw_wr) if iso_wr is not None else _apply_platt_batch(raw_wr, getattr(self, "_platt_wr", np.array([1.0, 0.0])))
        else:
            wr = p_serious_uncond

        return [
            {
                # Stage 1
                "pred_p_inspection":           float(p_insp[i]),
                # Stage 2 (conditional)
                "pred_p_violation_given_insp": float(p_viol[i]),
                "pred_p_serious_given_insp":  float(p_ser[i]),
                "pred_p_injury_given_insp":   float(inj[i]),
                # Stage 3 (conditional)
                "pred_penalty_given_viol":    float(max(0.0, float(cond_penalty[i]))),
                "pred_gravity_given_viol":    float(max(0.0, float(cond_gravity[i]))),
                "pred_citations_given_viol":  float(max(0.0, float(cond_citations[i]))),
                # Composite unconditional
                "expected_penalty":           float(max(0.0, float(expected_penalty[i]))),
                "expected_gravity":           float(max(0.0, float(expected_gravity[i]))),
                "expected_citations":         float(max(0.0, float(expected_citations[i]))),
                "p_serious_unconditional":    float(max(0.0, min(1.0, float(p_serious_uncond[i])))),
                # Backward compat
                "p_serious_wr_event":   float(wr[i]),
                "p_injury_event":       float(inj[i]),
                "p_penalty_ge_p75":     float(pen_p75[i]),
                "p_penalty_ge_p90":     float(pen_p90[i]),
                "p_penalty_ge_p95":     float(pen_p95[i]),
                "expected_penalty_usd": float(max(0.0, float(expected_penalty[i]))),
                "gravity_score":        float(max(0.0, float(expected_gravity[i]))),
            }
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    #  Composite score (v3 — sequential conditional pipeline)
    # ------------------------------------------------------------------ #

    @property
    def _is_staged_model(self) -> bool:
        """True if this model was trained with the 3-stage pipeline (v3+).

        Old pickles (v2) lack _head_inspection, _head_violation, _head_serious
        and should use the old 5-component composite formula for backward compat.
        """
        return getattr(self, "_head_inspection", None) is not None

    def _raw_composite(self, pred: Dict[str, float]) -> float:
        """Weighted sum of calibrated outputs (before transformation).

        Returns a value in [0, 1] representing the raw probability-weighted
        composite.  Used internally by fit() to build the score CDF.

        v3 formula (4 components — sequential conditional pipeline):
            raw = w1 * p_inspection
                + w2 * p_serious_unconditional
                + w3 * expected_penalty_norm
                + w4 * expected_gravity_norm

        where:
            expected_penalty_norm = E[pen] / (E[pen] + _PENALTY_REF_USD)
            expected_gravity_norm = E[grav] / (E[grav] + _GRAVITY_REF)

        For old pickles (v2 — 5-component), falls back to the old formula:
            raw = w1*p_wr + w2*p_inj + w3*p_pen95 + w4*gravity_norm + w5*pen_norm
        """
        w = self._weights

        if not self._is_staged_model:
            # ── Old pickle backward compat (5-component formula) ──────────
            w1 = w[0] if len(w) > 0 else 0.45
            w2 = w[1] if len(w) > 1 else 0.05
            w3 = w[2] if len(w) > 2 else 0.10
            w4 = w[3] if len(w) > 3 else 0.30
            w5 = w[4] if len(w) > 4 else 0.10

            p_wr   = pred.get("p_serious_wr_event", 0.0)
            p_inj  = pred.get("p_injury_event", 0.0)
            p_ext  = pred.get("p_penalty_ge_p95", 0.0)
            grav   = pred.get("gravity_score", 0.0)
            exp_p  = pred.get("expected_penalty_usd", 0.0)

            grav_n = grav  / (grav  + _GRAVITY_REF)     if grav  > 0 else 0.0
            pen_n  = exp_p / (exp_p + _PENALTY_REF_USD) if exp_p > 0 else 0.0

            return w1 * p_wr + w2 * p_inj + w3 * p_ext + w4 * grav_n + w5 * pen_n

        # ── New staged pipeline (4-component formula) ─────────────────────
        w1 = w[0] if len(w) > 0 else _DEFAULT_W1
        w2 = w[1] if len(w) > 1 else _DEFAULT_W2
        w3 = w[2] if len(w) > 2 else _DEFAULT_W3
        w4 = w[3] if len(w) > 3 else _DEFAULT_W4

        p_insp    = pred.get("pred_p_inspection", 1.0)
        p_ser_unc = pred.get("p_serious_unconditional",
                             pred.get("p_serious_wr_event", 0.0))
        exp_pen   = pred.get("expected_penalty",
                             pred.get("expected_penalty_usd", 0.0))
        exp_grav  = pred.get("expected_gravity",
                             pred.get("gravity_score", 0.0))

        # Michaelis-Menten saturation mapping: maps [0, ∞) → [0, 1)
        pen_n  = exp_pen / (exp_pen + _PENALTY_REF_USD) if exp_pen > 0 else 0.0
        grav_n = exp_grav / (exp_grav + _GRAVITY_REF)   if exp_grav > 0 else 0.0

        return w1 * p_insp + w2 * p_ser_unc + w3 * pen_n + w4 * grav_n

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

        v3 formula (4 components — sequential conditional pipeline):
            raw = w1*p_insp + w2*p_serious_uncond + w3*pen_norm + w4*grav_norm
        Then: evidence-shrink → percentile-stretch → rescale to 0-100.

        The sequential architecture separates inspection exposure from
        violation severity: zero future outcome doesn't always mean low risk;
        it may mean no inspection exposure.
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
            logger.warning("  [MultiTargetScorer] Could not load from %s: %s", path, e)
            return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _empty_prediction(self) -> Dict[str, float]:
        return {
            # Stage 1
            "pred_p_inspection":           0.0,
            # Stage 2 (conditional)
            "pred_p_violation_given_insp": 0.0,
            "pred_p_serious_given_insp":  0.0,
            "pred_p_injury_given_insp":   0.0,
            # Stage 3 (conditional)
            "pred_penalty_given_viol":    0.0,
            "pred_gravity_given_viol":    0.0,
            "pred_citations_given_viol":  0.0,
            # Composite unconditional
            "expected_penalty":           0.0,
            "expected_gravity":           0.0,
            "expected_citations":         0.0,
            "p_serious_unconditional":    0.0,
            # Backward compat
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
    """Find 4-component weights maximizing Spearman rho between composite and y_adv.

    v3: components are [p_inspection, p_serious_unconditional,
                        expected_penalty_norm, expected_gravity_norm].
    Constraints:
      * All weights ≥ 0.0 (non-negative combination)
      * Sum of weights = 1.0 (convex combination)
    Uses dense grid search + Nelder-Mead refinement.
    """
    from scipy.stats import spearmanr

    # Pre-compute component arrays (vectorized)
    p_insp    = np.array([p.get("pred_p_inspection", 1.0) for p in pred_batch])
    p_ser_unc = np.array([p.get("p_serious_unconditional",
                                 p.get("p_serious_wr_event", 0.0)) for p in pred_batch])
    exp_pen   = np.array([p.get("expected_penalty",
                                 p.get("expected_penalty_usd", 0.0)) for p in pred_batch])
    exp_grav  = np.array([p.get("expected_gravity",
                                 p.get("gravity_score", 0.0)) for p in pred_batch])

    pen_norm  = exp_pen / (exp_pen + _PENALTY_REF_USD)
    grav_norm = exp_grav / (exp_grav + _GRAVITY_REF)

    comps = np.column_stack([p_insp, p_ser_unc, pen_norm, grav_norm])  # n × 4

    def neg_spearman(w4: np.ndarray) -> float:
        w = w4 / w4.sum()
        scores = comps @ w
        rho, _ = spearmanr(scores, y_adv)
        return -rho

    # Dense grid search (4D simplex — coarser grid)
    best_rho, best_w = -1.0, np.array([_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4])
    step = 0.05
    for w1 in np.arange(0.0, 1.0 + step, step):
        for w2 in np.arange(0.0, 1.0 - w1 + step, step):
            for w3 in np.arange(0.0, 1.0 - w1 - w2 + step, step):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -0.001:
                    continue
                w4 = max(0.0, w4)
                trial = np.array([w1, w2, w3, w4])
                scores = comps @ trial
                rho, _ = spearmanr(scores, y_adv)
                if rho > best_rho:
                    best_rho = rho
                    best_w = trial.copy()

    # Nelder-Mead refinement (constrained via projection)
    def objective(w_raw):
        w = np.abs(w_raw)
        s = w.sum()
        if s < 1e-12:
            return 0.0
        w = w / s
        scores = comps @ w
        rho, _ = spearmanr(scores, y_adv)
        return -rho

    result = minimize(objective, best_w, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5})
    w_final = np.abs(result.x)
    w_final = w_final / w_final.sum()
    return w_final.tolist()


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
