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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FILE = "multi_target_model.pkl"

# ── Default composite weights ──────────────────────────────────────────────
_DEFAULT_W1 = 0.40   # p_wr_serious
_DEFAULT_W2 = 0.25   # normalized expected penalty
_DEFAULT_W3 = 0.20   # normalized expected citations
_DEFAULT_W4 = 0.15   # p_large_penalty

# Reference values for normalizing regression outputs into 0-100 components
_PENALTY_REF_USD = 200_000.0   # typical large-employer penalties in DON'T-REC range
_CITATION_REF    = 20.0         # ~20 citations/year → max citation component

# ── GBC / GBR hyper-parameters ─────────────────────────────────────────────
_GBC_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42,
)
_GBR_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    loss="huber",
    alpha=0.9,
    min_samples_leaf=3,
    random_state=42,
)


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
        # Regression head 3: citation count
        self._head_cit: Optional[Pipeline] = None
        # Binary head 4a/b/c: moderate / large / extreme penalty
        self._head_mod_pen: Optional[Pipeline]  = None
        self._head_lrg_pen: Optional[Pipeline]  = None
        self._head_ext_pen: Optional[Pipeline]  = None

        # Composite weight vector [w1, w2, w3, w4]
        self._weights: List[float] = [_DEFAULT_W1, _DEFAULT_W2, _DEFAULT_W3, _DEFAULT_W4]

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
        y_wr    = np.array([r["any_wr_serious"]      for r in rows], dtype=int)
        y_logp  = np.array([r["log_penalty"]          for r in rows], dtype=float)
        y_cit   = np.array([r["future_citation_count"] for r in rows], dtype=float)
        y_mod   = np.array([r["is_moderate_penalty"]  for r in rows], dtype=int)
        y_lrg   = np.array([r["is_large_penalty"]     for r in rows], dtype=int)
        y_ext   = np.array([r["is_extreme_penalty"]   for r in rows], dtype=int)

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

        def _train_gbc_calibrated(X_fit, y_fit):
            n_pos = y_fit.sum()
            n_neg = len(y_fit) - n_pos
            if n_pos < 5 or n_neg < 5:
                # Not enough of one class — return a constant dummy
                return _ConstantClassifier(float(n_pos / max(1, len(y_fit))))
            gbc = GradientBoostingClassifier(**_GBC_PARAMS)
            # Use isotonic calibration with cv=3 to preserve data on small sets
            cv = 3 if len(X_fit) < 500 else 5
            cal = CalibratedClassifierCV(gbc, method="isotonic", cv=cv)
            cal.fit(X_fit, y_fit)
            return cal

        def _train_gbr_pipe(X_fit, y_fit):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(**_GBR_PARAMS)),
            ])
            pipe.fit(X_fit, y_fit)
            return pipe

        print("  [MultiTargetScorer] Training Head 1: WR/Serious event (binary) …")
        self._head_wr = _train_gbc_calibrated(X_tr, y_wr[train_idx])

        print("  [MultiTargetScorer] Training Head 2: log-penalty regression …")
        self._head_log_pen = _train_gbr_pipe(X_tr, y_logp[train_idx])

        print("  [MultiTargetScorer] Training Head 3: citation count regression …")
        self._head_cit = _train_gbr_pipe(X_tr, y_cit[train_idx])

        print("  [MultiTargetScorer] Training Head 4a: moderate penalty (P75) …")
        self._head_mod_pen = _train_gbc_calibrated(X_tr, y_mod[train_idx])

        print("  [MultiTargetScorer] Training Head 4b: large penalty (P90) …")
        self._head_lrg_pen = _train_gbc_calibrated(X_tr, y_lrg[train_idx])

        print("  [MultiTargetScorer] Training Head 4c: extreme penalty (P95) …")
        self._head_ext_pen = _train_gbc_calibrated(X_tr, y_ext[train_idx])

        self._is_fitted = True

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
            p_serious_wr_event     (float 0-1)
            expected_penalty_usd   (float ≥ 0)
            expected_citations     (float ≥ 0)
            p_moderate_penalty     (float 0-1)
            p_large_penalty        (float 0-1)
            p_extreme_penalty      (float 0-1)
        """
        if not self._is_fitted:
            return self._empty_prediction()
        x2d = np.atleast_2d(x)
        return {
            "p_serious_wr_event":  float(_clf_proba(self._head_wr,      x2d)),
            "expected_penalty_usd": float(max(0.0, math.expm1(
                float(_reg_predict(self._head_log_pen, x2d))
            ))),
            "expected_citations":   float(max(0.0, _reg_predict(self._head_cit, x2d))),
            "p_moderate_penalty":   float(_clf_proba(self._head_mod_pen, x2d)),
            "p_large_penalty":      float(_clf_proba(self._head_lrg_pen, x2d)),
            "p_extreme_penalty":    float(_clf_proba(self._head_ext_pen, x2d)),
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict for a batch of feature rows (efficient vectorized path)."""
        if not self._is_fitted:
            return [self._empty_prediction() for _ in range(len(X))]
        wr   = _clf_proba_batch(self._head_wr,      X)
        logp = _reg_predict_batch(self._head_log_pen, X)
        cit  = _reg_predict_batch(self._head_cit,    X)
        mod  = _clf_proba_batch(self._head_mod_pen,  X)
        lrg  = _clf_proba_batch(self._head_lrg_pen,  X)
        ext  = _clf_proba_batch(self._head_ext_pen,  X)

        return [
            {
                "p_serious_wr_event":  float(wr[i]),
                "expected_penalty_usd": float(max(0.0, math.expm1(float(logp[i])))),
                "expected_citations":   float(max(0.0, float(cit[i]))),
                "p_moderate_penalty":   float(mod[i]),
                "p_large_penalty":      float(lrg[i]),
                "p_extreme_penalty":    float(ext[i]),
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

        Applies the same evidence-ceiling logic as ``MLRiskScorer`` so sparse-
        data establishments are subject to the same caps.
        """
        w1, w2, w3, w4 = self._weights

        pen_component = min(
            math.log1p(max(0.0, pred["expected_penalty_usd"])) /
            math.log1p(_PENALTY_REF_USD) * 100.0,
            100.0,
        )
        cit_component = min(
            max(0.0, pred["expected_citations"]) / _CITATION_REF * 100.0,
            100.0,
        )

        raw = (
            w1 * pred["p_serious_wr_event"]  * 100.0
            + w2 * pen_component
            + w3 * cit_component
            # Blend large (P90) and extreme (P95) penalty tiers for w4;
            # extreme penalty is a rarer, higher-signal event
            + w4 * (0.5 * pred["p_large_penalty"] + 0.5 * pred["p_extreme_penalty"]) * 100.0
        )

        # Evidence ceiling (mirrors MLRiskScorer.score_establishments)
        if has_fatality and has_willful:
            ceiling = 70.0
        elif n_inspections <= 2:
            ceiling = 50.0
        elif n_inspections <= 4:
            ceiling = 58.0
        else:
            ceiling = 100.0

        return float(np.clip(raw, 0.0, ceiling))

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
            "expected_citations":   0.0,
            "p_moderate_penalty":   0.0,
            "p_large_penalty":      0.0,
            "p_extreme_penalty":    0.0,
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
    cit_n = np.array([
        min(max(0.0, p["expected_citations"]) / _CITATION_REF * 100.0, 100.0)
        for p in pred_batch
    ])
    p_lrg = np.array([
        # Blend large+extreme for the w4 component (matches composite_score blending)
        (p["p_large_penalty"] + p["p_extreme_penalty"]) * 0.5 * 100.0
        for p in pred_batch
    ])

    components = np.stack([p_wr, pen_n, cit_n, p_lrg], axis=1)  # (n, 4)

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
        [0.50, 0.20, 0.15, 0.15],
        [0.40, 0.30, 0.20, 0.10],
        [0.35, 0.30, 0.25, 0.10],
        [0.45, 0.25, 0.20, 0.10],
        [0.40, 0.35, 0.15, 0.10],
        [0.30, 0.35, 0.25, 0.10],
        [0.50, 0.25, 0.15, 0.10],
        [0.40, 0.20, 0.30, 0.10],
        [0.40, 0.20, 0.20, 0.20],
        [0.45, 0.20, 0.20, 0.15],
        [0.35, 0.25, 0.25, 0.15],
        [0.30, 0.30, 0.25, 0.15],
        [0.50, 0.15, 0.20, 0.15],
        [0.55, 0.20, 0.15, 0.10],
        [0.60, 0.15, 0.15, 0.10],
        # Citation-heavy candidates (analytically found to be near-optimal)
        [0.33, 0.28, 0.39, 0.00],
        [0.35, 0.25, 0.40, 0.00],
        [0.30, 0.30, 0.40, 0.00],
        [0.40, 0.25, 0.35, 0.00],
        [0.35, 0.30, 0.35, 0.00],
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

    # Enforce minimum w4 ≥ 0.05: the penalty-tier signal always contributes,
    # ensuring the composite score retains sensitivity to extreme-penalty events
    # even when pure Spearman optimization would set w4=0.
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
