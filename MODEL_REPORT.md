# Manufacturer Compliance Intelligence — Model Technical Report

**Project:** Manufacturer Compliance Intelligence & Vetting Agent  
**Report Date:** April 6, 2026  
**Model Version:** Multi-Target v2 (probability-first composite)  
**Status:** All 113 tests passing, 1 skipped ✅

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Source & Ingestion](#2-data-source--ingestion)
3. [Entity Resolution & Company Grouping](#3-entity-resolution--company-grouping)
4. [Feature Engineering — 49-Feature Vector](#4-feature-engineering--49-feature-vector)
5. [Multi-Target Model Architecture](#5-multi-target-model-architecture)
6. [Training Data Construction](#6-training-data-construction)
7. [Target Label Construction (7 Targets)](#7-target-label-construction-7-targets)
8. [Penalty Percentile Thresholds](#8-penalty-percentile-thresholds)
9. [Inspection-Selection Bias Correction (IPW)](#9-inspection-selection-bias-correction-ipw)
10. [Model Training — Head-by-Head Details](#10-model-training--head-by-head-details)
11. [Post-Hoc Calibration](#11-post-hoc-calibration)
12. [Composite Score Formula (v2)](#12-composite-score-formula-v2)
13. [Evidence Shrinkage & Score Transformation](#13-evidence-shrinkage--score-transformation)
14. [Weight Optimization](#14-weight-optimization)
15. [Legacy GBR Scorer (MLRiskScorer)](#15-legacy-gbr-scorer-mlriskscorer)
16. [Per-Establishment Scoring & Multi-Site Aggregation](#16-per-establishment-scoring--multi-site-aggregation)
17. [Bühlmann Credibility Aggregation](#17-bühlmann-credibility-aggregation)
18. [Risk Assessment & Recommendation Logic](#18-risk-assessment--recommendation-logic)
19. [12-Month Compliance Outlook](#19-12-month-compliance-outlook)
20. [Confidence Signal](#20-confidence-signal)
21. [Adverse Outcome Formula](#21-adverse-outcome-formula)
22. [Validation Results & Empirical Thresholds](#22-validation-results--empirical-thresholds)
23. [Cache & Invalidation Strategy](#23-cache--invalidation-strategy)
24. [API Layer](#24-api-layer)
25. [LLM Enhancement Layer (Gemini)](#25-llm-enhancement-layer-gemini)
26. [System Architecture Diagram (Text)](#26-system-architecture-diagram-text)

---

## 1. System Overview

The system is an AI-assisted manufacturer compliance risk prediction platform.  
Given a manufacturer name, it:

1. Resolves the company to one or more physical OSHA-registered establishments
2. Retrieves all publicly available OSHA enforcement history from the Department of Labor API / bulk CSV cache
3. Constructs a 49-element feature vector per establishment
4. Runs a probabilistic multi-target ML model to predict 3 primary adverse outcomes
5. Produces a single 0–100 composite risk score with a 3-tier recommendation
6. Generates a 12-month forward compliance outlook
7. Optionally passes the assessment through a Gemini LLM for plain-English explanation

**Primary goal:** Predict FUTURE compliance risk, not just summarize past violations.

---

## 2. Data Source & Ingestion

### Source
- **DOL API v4** (`https://apiprod.dol.gov/v4/get/OSHA`)
- Endpoints used: `OSHA/inspection`, `OSHA/violation`

### Bulk Cache Strategy
Heavy data builds are done offline by `scripts/build_cache.py`; the live API makes at most 2 lightweight delta calls for recent records.

### Cached Files (in `ml_cache/`)

| File | Contents |
|---|---|
| `inspections_bulk.csv` | All OSHA inspection case records |
| `violations_bulk.csv` | All violation + penalty details per inspection |
| `accidents_bulk.csv` | Accident events linked to inspections |
| `accident_injuries_bulk.csv` | Individual injury records (degree, nature, body part) |
| `accident_abstracts_bulk.csv` | Pre-joined narrative text for accidents |
| `gen_duty_narratives_bulk.csv` | Inspector plain-language notes for high-priority Gen Duty violations |
| `bulk_meta.json` | Cache build metadata (date, record counts) |
| `company_key_index.json` | Pre-built normalized name → raw OSHA name index |

### SQLite Fast Path
When `osha_cache.db` is present, queries use SQLite (`SELECT * FROM inspections WHERE estab_name = ?`) instead of scanning CSV files.

### Injury Severity Codes Decoded

| Code | Label |
|---|---|
| 1 / 1.0 | Fatality |
| 2 / 2.0 | Hospitalized injury |
| 3 / 3.0 | Non-hospitalized injury |

---

## 3. Entity Resolution & Company Grouping

**Module:** `src/search/grouped_search.py`  
**Purpose:** Map a user's search query ("Amazon") to the correct set of physical OSHA establishments, handling the wide variety of noisy naming variants in OSHA records.

### Name Normalization Pipeline
1. Pre-clean (strip punctuation, lowercase)
2. Strip noise words (Inc, LLC, Corp, Co., Ltd., etc.)
3. Canonicalize tokens (phonetic/common abbreviation mapping)
4. Compute a `company_match_key` for deterministic prefix matching

### Fuzzy Matching
- Uses `rapidfuzz` for fast token-set ratio matching
- Score cutoff: **55** (0–100 scale)
- Result limit: **500** raw candidates per search

### Confidence Tiers

| Tier | Description |
|---|---|
| High | Strong token overlap, shared prefix match key |
| Medium | Partial overlap, likely same parent |
| Low | Weak match, flagged for user review |

### Output: `GroupedCompanyResult`
- `parent_name`: Clean display label
- `total_facilities`: Count of likely-related establishments
- `high_confidence` / `medium_confidence` / `low_confidence`: Lists of `FacilityCandidate`
- Each `FacilityCandidate` carries: raw OSHA name, display name, facility code (e.g. "CMH1"), city, state, address, NAICS code, confidence score

---

## 4. Feature Engineering — 49-Feature Vector

**Module:** `src/scoring/features.py` (online path), `src/scoring/ml_risk_scorer.py::_fetch_population` (population build path)

All features are computed identically in training and inference via shared logic.  
Per-establishment rates (not raw counts) are used as the base so training on single-site records and inference on multi-site companies operate on the same numeric scale.

### 4.1 Absolute Signals (25 features)

These 25 features are extracted from raw OSHA records for each establishment:

| # | Feature Name | How Computed |
|---|---|---|
| 0 | `log_inspections` | `n_insp` (raw count; log1p applied later) |
| 1 | `log_violations` | `n_viols` (raw count; log1p applied later) |
| 2 | `serious_violations` | serious violations / n_insp (per-inspection rate) |
| 3 | `willful_violations` | willful violations / n_insp |
| 4 | `repeat_violations` | repeat violations / n_insp |
| 5 | `log_penalties` | total penalties ($) (log1p applied later) |
| 6 | `avg_penalty` | mean violation penalty ($) |
| 7 | `max_penalty` | max single-violation penalty ($) |
| 8 | `recent_ratio` | inspections in last 3-year window / n_insp |
| 9 | `severe_incidents` | inspections with linked accidents / n_insp |
| 10 | `violations_per_inspection` (vpi) | n_viols / n_insp |
| 11 | `accident_count` | accidents / n_insp (log1p applied later) |
| 12 | `fatality_count` | fatalities / n_insp (log1p applied later) |
| 13 | `injury_count` | injuries / n_insp (log1p applied later) |
| 14 | `avg_gravity` | mean violation gravity score (0–10 OSHA scale) |
| 15 | `penalties_per_inspection` | total penalty / n_insp ($) |
| 16 | `clean_ratio` | inspections with 0 violations / n_insp |
| 17 | `time_adjusted_penalty` | Σ(insp_penalty × exp(−age_years / 3)) — exponential decay weighting |
| 18 | `recent_wr_rate` | W/R violations in last 3yr / total W/R (0 if no W/R) |
| 19 | `trend_delta` | vpi_all_time − vpi_recent (vpi from recent inspections) |
| 20 | `log_willful_raw` | log1p(total willful count) |
| 21 | `log_repeat_raw` | log1p(total repeat count) |
| 22 | `has_any_fatality` | 1.0 if any fatality ever recorded, else 0.0 |
| 23 | `log_max_insp_penalty` | log1p(max penalty from a single inspection) |
| 24 | `log_estab_size` | log1p(median nr_in_estab across inspections) |

**Recency window:** 3 years (1,095 days)  
**Time-decay half-life:** 3 years (`exp(-age/3)`)

### 4.2 Industry Z-Score Features (4 features)

Computed in `src/scoring/industry_stats.py`.  
Industry groups resolved at 4-digit NAICS → fallback 3-digit → fallback 2-digit.  
Minimum sample to use a group: **10 establishments** (global fallback below that).  
All z-scores clipped to **±3σ** to prevent tail outlier inflation.

| # | Feature | Formula |
|---|---|---|
| 25 | `relative_violation_rate` | (company_vpi − industry_mean_vpi) / industry_std_vpi |
| 26 | `relative_penalty` | (company_avg_pen − industry_mean_pen) / industry_std_pen |
| 27 | `relative_serious_ratio` | (company_serious_rate − industry_mean_serious) / industry_std_serious |
| 28 | `relative_willful_repeat` | (company_wr_rate − industry_mean_wr) / industry_std_wr |

NaN (when NAICS unavailable) → replaced with 0.0 sentinel.

### 4.3 NAICS Sector One-Hot Encoding (25 features)

One-hot encoded at the 2-digit NAICS prefix level (24 sectors + 1 unknown bucket).

| Index | Sector |
|---|---|
| 29 | naics_11: Agriculture/Forestry/Fishing |
| 30 | naics_21: Mining/Oil & Gas |
| 31 | naics_22: Utilities |
| 32 | naics_23: Construction |
| 33 | naics_31: Mfg (Food/Textile/Apparel) |
| 34 | naics_32: Mfg (Wood/Paper/Chemical/Plastics) |
| 35 | naics_33: Mfg (Metal/Machinery/Electronics) |
| 36 | naics_42: Wholesale Trade |
| 37 | naics_44: Retail Trade |
| 38 | naics_45: Retail Trade (Misc.) |
| 39 | naics_48: Transportation/Warehousing |
| 40 | naics_49: Warehousing/Storage |
| 41 | naics_51: Information |
| 42 | naics_52: Finance/Insurance |
| 43 | naics_53: Real Estate |
| 44 | naics_54: Professional/Scientific/Technical |
| 45 | naics_55: Management of Companies |
| 46 | naics_56: Admin/Support/Waste Mgmt |
| 47 | naics_61: Educational Services |
| 48 | naics_62: Health Care/Social Assistance |
| 49 | naics_71: Arts/Entertainment/Recreation |
| 50 | naics_72: Accommodation/Food Services |
| 51 | naics_81: Other Services |
| 52 | naics_92: Public Administration |
| 53 | naics_unknown |

### 4.4 Log Transform

Applied **after** all features are assembled, to indices `[0, 1, 11, 12, 13]` only:

```
X[..., i] = log1p(max(X[..., i], 0))
```

Indices 5 (`log_penalties`), 6 (`avg_penalty`), 7 (`max_penalty`), 17 (`time_adjusted_penalty`) are intentionally **NOT** log-transformed because large penalty magnitudes are meaningful discrimination signals ($10K vs $1M matters).

---

## 5. Multi-Target Model Architecture

**Module:** `src/scoring/multi_target_scorer.py`  
**Class:** `MultiTargetRiskScorer`  
**Saved artifact:** `ml_cache/multi_target_model.pkl`

The model contains **6 prediction heads** total (3 primary + 3 auxiliary + 2 legacy):

### Primary Heads (drive composite score)

| Head | Name | Type | Output |
|---|---|---|---|
| 1 | WR/Serious event | HistGradientBoostingClassifier + Isotonic | `p_serious_wr_event` ∈ [0, 1] |
| 3 | Hospitalization/Fatality event | HistGradientBoostingClassifier + Isotonic | `p_injury_event` ∈ [0, 1] |
| 5 (P95) | Extreme penalty tier | HistGradientBoostingClassifier + Isotonic | `p_penalty_ge_p95` ∈ [0, 1] |

### Auxiliary Penalty Tier Heads

| Head | Name | Output |
|---|---|---|
| 5a | P75 moderate penalty | `p_penalty_ge_p75` ∈ [0, 1] |
| 5b | P90 large penalty | `p_penalty_ge_p90` ∈ [0, 1] |

### Legacy Regression Heads (backward compatibility / outlook)

| Head | Name | Type | Output |
|---|---|---|---|
| 2 (hurdle) | Expected penalty USD | Binary HGBC × conditional GBR | `expected_penalty_usd` ≥ 0 |
| 4 | Gravity-weighted severity | GBR + Isotonic | `gravity_score` ≥ 0 |

### Hurdle Model (Head 2) — Detail

- **Stage 2a:** Binary HGBC — "Does any future penalty exist?" → P(any_penalty)
- **Stage 2b:** Conditional GBR — Trained ONLY on positive-penalty rows (zero rows excluded to prevent zero-inflation anchor). Targets log1p(penalty); prediction after training: `P(any_pen) × expm1(conditional_log_pen)`
- **GBR loss for Head 2:** Quantile loss at α=0.80 (80th percentile targeting for aggressive upper-tail prediction)

---

## 6. Training Data Construction

**Module:** `src/scoring/labeling/multi_target_labeler.py`  
**Script:** `scripts/train_multi_target.py`

### Temporal Split

| Parameter | Value | Rationale |
|---|---|---|
| Cutoff date | 2022-01-01 | Avoids COVID-era inspection suppression (2020–2021) which creates false-negative labels for genuinely risky companies |
| Outcome window | cutoff → today (2026-04-06) | All available post-cutoff data used |
| Default sample size | 50,000 rows | Exceeds the ~30k available paired pool, so effectively "use all" |

### Paired Pool Construction

1. Stream `inspections_bulk.csv`
2. For each establishment, split into `hist_inspections` (before 2022-01-01) and `future_inspections` (after 2022-01-01)
3. **Eligibility:** ≥ 2 historical inspections AND ≥ 1 future inspection in the outcome window
4. Build 49-feature vector from historical inspections only (`[LEAKAGE GUARD]` enforced throughout)
5. Compute multi-target labels from future inspections only

### Stratified Sampling

- **Stratification axes:** 2-digit NAICS sector × binary WR/Serious outcome quartile
- Ensures all industries and risk levels are represented in the training set
- Drawn without replacement; each stratum contributes proportionally with `min_per_stratum` floor

---

## 7. Target Label Construction (7 Targets)

Each training row carries 7 target variables:

| # | Name | Type | Description |
|---|---|---|---|
| 1 | `any_wr_serious` | int 0/1 | Any Willful/Repeat/Serious violation in post-cutoff window |
| 2 | `future_total_penalty` | float ($) | Sum of all post-cutoff OSHA penalties |
| 3 | `log_penalty` | float | log1p(future_total_penalty) |
| 4 | `any_injury_fatal` | int 0/1 | Any hospitalization (degree_of_inj=2) OR fatality (degree_of_inj=1) |
| 5 | `gravity_weighted_score` | float | Σ(gravity × viol_weight) where W/R→3, S→2, others→1 |
| 6 | `real_label` | float 0–100 | Composite adverse outcome score (for calibration/weight optimization) |
| 7 | `is_moderate_penalty` | int 0/1 | future_total_penalty ≥ NAICS-P75 threshold |
| 8 | `is_large_penalty` | int 0/1 | future_total_penalty ≥ NAICS-P90 threshold |
| 9 | `is_extreme_penalty` | int 0/1 | future_total_penalty ≥ NAICS-P95 threshold |

---

## 8. Penalty Percentile Thresholds

**Module:** `src/scoring/penalty_percentiles.py`  
**Artifact:** `ml_cache/penalty_percentiles.json`

### Computation Rules

- Computed **once** on **pre-cutoff data only** (pre-2022 violations) to prevent leakage
- Thresholds are per 2-digit NAICS sector
- Minimum group size to use a sector-specific threshold: **50 samples**
- Sectors with < 50 samples fall back to global thresholds
- Only **positive-penalty** rows used (zero-penalty rows excluded so the threshold isn't deflated by clean inspections)

### Global Fallback Thresholds (default)

| Percentile | Threshold |
|---|---|
| P75 (moderate) | $5,000 |
| P90 (large) | $15,000 |
| P95 (extreme) | $40,000 |

---

## 9. Inspection-Selection Bias Correction (IPW)

**Module:** `src/scoring/labeling/inspection_propensity.py`  
**Class:** `InspectionPropensityModel`

**Problem:** OSHA does not inspect randomly. Industry, size, and prior inspection history all predict whether an establishment gets a follow-up. The training sample of "paired" establishments (those with pre- AND post-cutoff inspections) is therefore biased toward establishments that OSHA specifically targeted.

**Solution:** Inverse Probability Weighting (IPW)

1. Fit a Logistic Regression classifier: `P(reinspected | log1p(hist_count), NAICS_one_hot)`
2. Compute IPW weight per row: `clip(1 / P(reinspected), 1.0, 5.0)` (max weight 5.0 to prevent extreme upweighting)
3. Normalize weights to `mean = 1.0` so the effective regularization scale stays constant
4. Apply as `sample_weight` when fitting each GBM head

**Features for propensity model:**
- `log1p(historical_inspection_count)` (1 feature)
- 2-digit NAICS one-hot (24 features + unknown = 25 features)
- Total: 26 features

---

## 10. Model Training — Head-by-Head Details

### Train/Validation Split

- `val_fraction = 0.20` (default)
- Training fold: 80% of paired rows → fit each head
- Validation fold: 20% → isotonic calibration, weight optimization, CDF building
- Minimum 50 rows to enable weight optimization; below that, default weights used

### Head 1: WR/Serious Event Classifier

**Algorithm:** `HistGradientBoostingClassifier`  
**Loss:** Binary cross-entropy (log-loss)  
**Hyperparameters:**
```
max_iter        = 5,000 (with early stopping)
max_depth       = 5
learning_rate   = 0.02
min_samples_leaf = 15
l2_regularization = 0.8
early_stopping  = True
validation_fraction = 0.12
n_iter_no_change = 80   (patience)
random_state    = 42
```
Early stopping finds the optimal iteration count automatically (~400–800 effective iterations), preventing over-confidence that harms calibration slope.

### Head 3: Hospitalization/Fatality Classifier

**Algorithm:** `HistGradientBoostingClassifier`  
**Hyperparameters:**
```
max_iter        = 700 (no early stopping)
max_depth       = 4
learning_rate   = 0.04
min_samples_leaf = 20
l2_regularization = 1.5
early_stopping  = False
random_state    = 42
```
Heavier regularization (l2=1.5, larger min_samples_leaf=20) to control the rarer injury/fatality signal.

### Heads 5a/5b/5c: Penalty Tier Classifiers (P75 / P90 / P95)

Same `HistGradientBoostingClassifier` config as Head 1.

### Head 2b: Conditional Log-Penalty Regressor

**Algorithm:** `GradientBoostingRegressor` wrapped in `StandardScaler → Pipeline`  
**Hyperparameters:**
```
n_estimators    = 600
max_depth       = 5
learning_rate   = 0.04
subsample       = 0.75
loss            = "quantile"   (80th-percentile quantile regression)
alpha           = 0.80
min_samples_leaf = 2
random_state    = 42
```
Quantile loss at α=0.80 targets the upper tail of the penalty distribution (aggressive prediction for high-risk establishments).

### Head 4: Gravity-Weighted Severity Regressor

Same `GradientBoostingRegressor` config as Head 2b.

---

## 11. Post-Hoc Calibration

### Isotonic Regression (primary — fitted on 20% validation fold)

All binary heads use **Isotonic Regression** for post-hoc calibration:
```python
IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
```

- Fitted on the 20% held-out validation fold (no leakage)
- Nonparametric — handles non-sigmoid calibration curves better than Platt scaling
- Stored as `_iso_wr`, `_iso_inj`, `_iso_pen_p75`, `_iso_pen_p90`, `_iso_pen_p95`

### Legacy Platt Scaling (kept for backward compatibility with old pickles)

- 2-parameter: `p_cal = sigmoid(a × logit(p) + b)`
- Minimizes binary cross-entropy on holdout via L-BFGS-B
- Parameters stored as `_platt_wr`, `_platt_inj`, `_platt_pen`
- At inference, isotonic is preferred; Platt used as fallback on old pickles

### Metrics Logged During Training

For each head: **Brier Score**, **Brier Skill Score (BSS)**, **Expected Calibration Error (ECE)**

| Metric | Formula | Interpretation |
|---|---|---|
| Brier Score | mean((p − y)²) | Lower = better; 0 = perfect |
| BSS | 1 − BS_model / BS_climatology | Positive = better than base rate |
| ECE | Weighted mean |p̄_bin − ȳ_bin| | Calibration gap; 0 = perfect |

---

## 12. Composite Score Formula (v2)

**Formula:**
```
risk_score_raw = w1 × p_serious_wr_event
               + w2 × p_injury_event
               + w3 × p_penalty_ge_p95
```

**Default weights:**

| Weight | Component | Default Value |
|---|---|---|
| w1 | p_serious_wr_event | 0.60 |
| w2 | p_injury_event | 0.30 |
| w3 | p_penalty_ge_p95 (extreme tier) | 0.10 |

All inputs are calibrated probabilities in [0, 1]. The raw output is therefore in [0, 1].

---

## 13. Evidence Shrinkage & Score Transformation

### Step 1: Evidence Shrinkage

Shrinks the raw composite toward the population mean when inspection evidence is sparse:

```
confidence   = 1 - exp(-n_inspections / 3.0)
shrunk_score = raw × confidence + (15.0 / 100.0) × (1 - confidence)
```

- **Population prior:** 15.0 (on 0–100 scale)
- At 3 inspections: confidence ≈ 0.632 (∼63% weight on model prediction)
- At 10 inspections: confidence ≈ 0.964 (∼96% weight on model)
- At 1 inspection: confidence ≈ 0.283 (∼28% weight on model; heavily shrunk)

### Step 2: Percentile Stretching via Empirical CDF

```
pctile     = searchsorted(score_cdf, shrunk_score) / len(score_cdf)
stretched  = pctile ^ alpha   (with alpha = 2.0)
final_score = stretched × 100
```

- **CDF built from:** validation fold scores computed during training
- **Alpha = 2.0 (square):** Convex stretch that maps:
  - 50th percentile → 25 (on 0–100 scale)
  - 90th percentile → 81
  - 95th percentile → 90
  - Places most companies below 50; spreads actionable high-risk range across 50–100

### Step 3: Hard Cap

Fatality + willful combination: `score = min(score, 70.0)`

---

## 14. Weight Optimization

**Module:** `src/scoring/multi_target_scorer.py::_optimize_weights`

**Objective:** Maximize **Spearman ρ** between composite score and real adverse outcome (`y_adv`).  
Spearman rho is used (instead of RMSE) because the model goal is correct risk **ranking**, not accurate absolute magnitude.

**Method:**
1. Grid search over 15 points on the 3-simplex (w1 + w2 + w3 = 1.0)
2. Best grid point fed to **Nelder-Mead refinement** (`scipy.optimize.minimize`)
3. Constraint: `w3 ≥ 0.05` enforced (penalty signal always contributes at least 5%)

**Runs on:** 20% validation fold held out from training  
**Minimum rows required:** 50 (otherwise defaults used)

---

## 15. Legacy GBR Scorer (MLRiskScorer)

**Module:** `src/scoring/ml_risk_scorer.py`  
**Class:** `MLRiskScorer`  
**Artifacts:** `ml_cache/risk_model.pkl`, `ml_cache/population_data.json`, `ml_cache/tail_calibrator.pkl`

This is the **original scoring engine**, still active. It provides:

1. Population-relative percentile rankings (~370,000 establishments baseline)
2. Feature importance weights (shown in explanation/UI)
3. Industry comparison context
4. Per-establishment site scores (used as input to MT model)

### GBR Configuration
```
n_estimators    = 500
max_depth       = 5
learning_rate   = 0.05
subsample       = 0.8
loss            = "huber"
alpha           = 0.95
min_samples_leaf = 3
random_state    = 42
```

### Tail Sample Weighting
High-risk real-label rows receive up to 8× weight:
```
sample_weight = clip(1.0 + max(0, real_label - 30.0) / 10.0, 1.0, 8.0)
```

### TailCalibrator

`src/scoring/tail_calibrator.py`

- Uses **Isotonic Regression** (`IsotonicRegression(increasing=True)`)
- Fitted on **binned** (score → mean future adverse) pairs (bin width = 5.0 score points, minimum 3 observations per bin)
- Re-normalizes calibrated output back to 0–100 range
- Fully monotone: cannot reorder any two establishments, only stretches/compresses scale

### Evidence-Gated Score Ceilings (per establishment)

| Condition | Ceiling |
|---|---|
| Fatality AND willful violations | 70.0 |
| ≤ 2 inspections | 50.0 |
| 3–4 inspections | 58.0 |
| ≥ 5 inspections | 100.0 (uncapped) |

### Cache Validity
- Cache refreshed if older than **7 days**
- **Feature shape guard:** if cached feature dim ≠ `len(FEATURE_NAMES)`, model deleted and retrained
- **Feature name guard:** if cached feature names list ≠ current `FEATURE_NAMES`, retrained

---

## 16. Per-Establishment Scoring & Multi-Site Aggregation

`MLRiskScorer.score_establishments()` groups `OSHARecord` objects by `estab_name`, scores each independently, then returns aggregate metrics.

### Site Score Fields

Each site score dict contains:
- `name`: establishment name (uppercase)
- `score`: calibrated risk score (0–100)
- `n_inspections`: count of inspections at this site
- `naics_code`: most common NAICS code at this site
- `city` / `state`: from most-recent inspection record
- `_log_feats`: saved 49-dim log-transformed feature array (used by MT model per-site scoring)

### Aggregate Metrics

| Field | Definition |
|---|---|
| `weighted_avg_score` | Inspection-count-weighted mean of site scores |
| `max_score` | Highest single-site score |
| `median_score` | Median of all site scores |
| `risk_concentration` | Fraction of sites scoring ≥ 60 |
| `systemic_risk_flag` | True when >50% high-risk OR willful/repeat across ≥ 2 sites |

---

## 17. Bühlmann Credibility Aggregation

**Module:** `src/scoring/risk_assessor.py`  
**Applicable for:** Companies with ≥ 2 distinct establishments

**Setting:** `_CREDIBILITY_K = 5` (matching NCCI workers'-comp experience rating: sites with < 5 inspections regress toward portfolio mean)

### Formula

**Step 1:** Compute portfolio prior (simple mean of all site composites)

**Step 2:** Credibility-adjust each site score:
```
Z_i        = n_i / (n_i + K)           K = 5
credible_i = Z_i × comp_i + (1 - Z_i) × portfolio_prior
```

**Step 3:** Tail-exposure blend:
```
alpha    = min(risk_concentration × 0.30, 0.30)
composite = (1 - alpha) × credibility_mean + alpha × max_credible
```

- `_TAIL_BLEND_MAX_ALPHA = 0.30`: worst site contributes up to 30% weight when risk is maximally concentrated
- At `risk_concentration = 0`: pure credibility mean (no tail weighting)
- At `risk_concentration = 1.0`: 30% weighted toward worst-site score

### Single-Establishment Fallback

When there is only 1 establishment, the credibility formula simplifies to the direct MT composite score.

---

## 18. Risk Assessment & Recommendation Logic

**Module:** `src/scoring/risk_assessor.py`  
**Output class:** `RiskAssessment` (Pydantic model)

### Recommendation Thresholds

| Score Range | Recommendation |
|---|---|
| 0–39 | Recommend |
| 40–59 | Proceed with Caution |
| 60–100 | Do Not Recommend |

*(Note: exact thresholds are derived from the `risk_score` value after MT composite replaces legacy GBR score when the MT model is available.)*

### Confidence Score Logic

`confidence_score` (0–1) reflects data availability and ambiguity:
- Driven by inspection count, recency, and model agreement between legacy GBR and MT composite
- Returns `risk_confidence`: "high" / "medium" / "low"
- Returns `confidence_detail` dict: `{n_inspections, recency_years, model_agreement, ...}`

### Industry Percentile

- Computed from `population_data.json` (all ~370k establishments)
- Filtered to the same 2–4 digit NAICS group
- `industry_percentile`: percentile rank within industry peers (0–100)

---

## 19. 12-Month Compliance Outlook

**Module:** `src/scoring/score_outlook.py`  
**Output:** `ComplianceOutlook12M`

### Inspection Frequency Estimation

```
recent_inspections  = n_insp × recent_ratio
annual_insp_rate    = recent_inspections / 3.0        (3-year window)
```

#### Score-Based Floor (minimum annual inspection probability)

| Risk Score | Floor (inspections/year) |
|---|---|
| ≥ 60 (high) | 1.5 |
| 40–59 (moderate) | 0.75 |
| 20–39 (low) | 0.30 |
| < 20 (minimal) | 0.10 |

### 12-Month Projections

When MT model predictions are available:
- `expected_violations_12m`: from `expected_citations` (MT regression head)
- `expected_penalties_usd_12m`: from `expected_penalty_usd` (MT hurdle head)
- Serious/WR counts: remain rate-based (historical rates × annual inspection rate)

When only legacy model available:
- All projections = `annual_insp_rate × per-inspection rate` from historical features

### Narrative Construction

Templated narratives at 3 risk bands (low / moderate / high), parameterized with:
- Projected inspection count
- Projected violation count (formatted: "fewer than 1" or "approximately N")
- Projected penalty amount (formatted: "minimal" / "~$X,XXX" / "~$X,XXX")
- Site count

---

## 20. Confidence Signal

`RiskAssessment` exposes a structured confidence signal:

| Field | Type | Description |
|---|---|---|
| `confidence_score` | float 0–1 | Overall model confidence |
| `risk_confidence` | str | "high" / "medium" / "low" |
| `confidence_detail` | dict | `n_inspections`, `recency_years`, `model_agreement`, etc. |

---

## 21. Adverse Outcome Formula

**Module:** `src/scoring/labeling/helpers.py`  
**Function:** `_compute_adverse(future_viols, future_fatalities, n_future_insp)`

Used to compute the `real_label` composite for:
1. Training label (post-cutoff outcomes)
2. TailCalibrator fitting
3. Weight optimization objective

### Formula

```
adv = ADV_FATALITY_FLAG × any_fatal                            (20 pts if any fatality)
    + min((future_fatalities − 1) × 5, 15)                    (5 pts per extra fatality, max 15)
    + ADV_WR_FLAG × any_wr                                     (8 pts if any W/R)
    + min(willful_repeat_count × 3, 15)                        (3 pts per W/R, max 15)
    + min(serious_count × 1, 10)                               (1 pt per Serious, max 10)
    + min(log1p(total_penalty) × 0.8, 10)                      (penalty signal, max 10)
    + min(violation_rate × 2, 10)                              (density signal, max 10)
```

| Component | Max Contribution |
|---|---|
| First fatality flag | 20 |
| Additional fatalities | 15 |
| W/R flag | 8 |
| W/R count (×3) | 15 |
| Serious count (×1) | 10 |
| Log-penalty (×0.8) | 10 |
| Violation rate (×2) | 10 |
| **Theoretical maximum** | **88** |

**Normalization:** `_normalize_adverse(raw) = clip(raw / 88 × 100, 0, 100)`

---

## 22. Validation Results & Empirical Thresholds

**Suite:** `tests/validation/`  
**Run command:** `python -m pytest tests/test_establishment_scoring.py tests/validation/ -v --tb=short`  
**Last run result:** 113 passed, 1 skipped ✅

### Empirical Performance on 2021 Holdout (training cutoff 2022-01-01)

| Metric | Threshold Required | Empirical Result |
|---|---|---|
| WR Event AUROC | ≥ 0.73 | **0.751** ✅ |
| WR Event AUROC "Strong" | ≥ 0.75 | **0.751** ✅ |
| Brier Skill Score (WR head) | ≥ 0.16 | **0.1885** ✅ |
| PR-AUC / AP (WR event, @ 48.5% prevalence) | ≥ 0.70 | **0.721** ✅ |
| Top-decile lift (WR) | ≥ 1.80 | **2.145×** ✅ |
| Top-decile lift "Strong" | ≥ 2.10 | **2.145×** ✅ |
| Top-10% capture rate (injury) | ≥ 11% | **12.7%** ✅ |
| Top-10% capture "Strong" | ≥ 12% | **12.7%** ✅ |
| Injury BSS | ≥ 0.07 | **0.0858** ✅ |
| Injury PR-AUC ratio (vs. random) | ≥ 2.70× | **2.95×** ✅ |
| Injury calibration slope | ≤ 1.40 | **1.297** ✅ |
| Injury AUROC | ≥ some threshold | **0.797** ✅ |

**Note on Injury Calibration Slope (1.297):** Elevated above 1.0 because the training set has 5.8% injury prevalence while the 2021 validation holdout has 11.9% prevalence (COVID-era reduced inspections created a prevalence shift). This is expected and structurally bounded by `CALIB_SLOPE_MAX_P_INJURY = 1.40`.

**Note on WR structural ceiling:** AUROC ~0.751 is near-structural for the 2021 holdout because COVID inspection suppression creates "false negatives" in the outcome data (risky companies that weren't reinspected during the pandemic). The ceiling cannot be meaningfully exceeded without different training data.

### Threshold Constants (in `tests/validation/mt_shared.py`)

| Constant | Value |
|---|---|
| AUROC_P_EVENT_TARGET | 0.73 |
| AUROC_STRONG | 0.75 |
| BRIER_SS_MIN | 0.16 |
| PR_AUC_AP_FLOOR_EVENT | 0.70 |
| PR_AUC_RATIO_P_INJURY | 2.70 |
| LIFT_MINIMUM | 1.80 |
| LIFT_STRONG | 2.10 |
| CAPTURE_MINIMUM | 0.11 |
| CAPTURE_STRONG | 0.12 |
| BRIER_SS_MIN_P_INJURY | 0.07 |
| CALIB_SLOPE_MAX_P_INJURY | 1.40 |

---

## 23. Cache & Invalidation Strategy

| Artifact | Location | Refresh Trigger |
|---|---|---|
| `risk_model.pkl` | `ml_cache/` | Feature shape/name mismatch or cache > 7 days |
| `population_data.json` | `ml_cache/` | Rebuilt with `risk_model.pkl` |
| `tail_calibrator.pkl` | `ml_cache/` | Rebuilt with `risk_model.pkl` |
| `multi_target_model.pkl` | `ml_cache/` | Manual re-run of `train_multi_target.py` |
| `penalty_percentiles.json` | `ml_cache/` | Manual re-run or `--force` flag |
| `multi_target_labels.pkl` | `ml_cache/` | Fingerprint mismatch (cutoff date, CSV mtimes, sample size) |
| `temporal_labels.pkl` | `ml_cache/` | Same fingerprint logic |

**Cache fingerprint** checks: cutoff date, outcome end date, sample size, inspection CSV mtime, violation CSV mtime.

**Force retrain:** Delete `ml_cache/risk_model.pkl` or `ml_cache/multi_target_model.pkl` and restart.  
**Env var shortcut:** `MAX_ESTAB_DEV=<n>` caps the establishment count in real-world validation tests.

---

## 24. API Layer

**Module:** `api/main.py`  
**Framework:** FastAPI  
**Start command:** `uvicorn api.main:app --reload --port 8000`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/companies` | Sorted deduplicated company name list |
| GET | `/api/search?q=...` | Grouped entity search results (SSE stream or JSON) |
| GET | `/api/locations?company=` | Address list for a specific company |
| POST | `/api/assess` | Full risk assessment (SSE stream: progress events → result) |

### SSE Streaming (POST /api/assess)
Progress events are emitted during assessment so the frontend can show live status:
- `"🔍 Searching OSHA records…"`
- `"🏗 Building inspection records…"`
- `"🤖 Scoring risk…"`

Assessment result is the final SSE event.

### CORS Configuration
Allowed origins: `http://localhost:5173` (Vite dev), `http://localhost:3000`

### Response Schema (`AssessmentResponse`)

Full schema includes:
- `manufacturer_name`, `risk_score` (0–100), `recommendation`, `explanation`
- `confidence_score`, `risk_confidence`, `confidence_detail`
- `feature_weights` (ML feature importances for UI display)
- `percentile_rank`, `industry_label`, `industry_group`, `industry_percentile`
- `industry_comparison` (human-readable comparison strings)
- `establishment_count`, `site_scores` (per-site breakdown), `risk_concentration`
- `systemic_risk_flag`, `aggregation_warning`, `concentration_warning`
- `records` (full OSHA inspection + violation + accident data)
- `outlook` (`ComplianceOutlook12M`)
- `risk_targets` (`ProbabilisticRiskTargetsOut` — all 7 probabilistic outputs)

---

## 25. LLM Enhancement Layer (Gemini)

**Module:** `src/agent/vetting_agent.py::enhance_explanation`  
**Model used:** `gemini-2.5-flash`  
**API:** Google GenAI SDK (`google.genai`)  
**Key:** `GOOGLE_API_KEY` env var (optional; gracefully disabled if absent)

### Prompt Design

The LLM receives:
- Manufacturer name, risk score, recommendation tier
- All 7 probabilistic targets (p_serious_wr, p_injury, penalties, etc.)
- Raw technical findings from the deterministic explanation

**Persona:** Manufacturing compliance expert  
**Audience:** Procurement officer with no legal background  
**Task:** 2–3 sentence executive summary with section headers

### Integration Point
- `enhance_explanation()` is called **after** the deterministic assessment is complete
- It **replaces** `assessment.explanation` with the LLM-enriched text
- Failure is graceful: warning logged, original deterministic explanation kept

---

## 26. System Architecture Diagram (Text)

```
User Input: Manufacturer Name
         │
         ▼
┌──────────────────────────────────────┐
│         VettingAgent                 │
│  (src/agent/vetting_agent.py)        │
└───────────────┬──────────────────────┘
                │
    ┌───────────▼───────────┐
    │     OSHAClient        │
    │  (data_retrieval/)    │
    │  - Bulk CSV Cache     │
    │  - SQLite Fast Path   │
    │  - DOL API delta      │
    └───────────┬───────────┘
                │  List[OSHARecord]
    ┌───────────▼───────────┐
    │   grouped_search.py   │   ← Entity resolution
    │   (RapidFuzz + norm)  │
    └───────────┬───────────┘
                │  Grouped establishments
    ┌───────────▼──────────────────────────────────────────────┐
    │                   RiskAssessor                           │
    │  ┌────────────────────────┐  ┌────────────────────────┐ │
    │  │    MLRiskScorer        │  │ MultiTargetRiskScorer  │ │
    │  │  (Legacy GBR)          │  │  (v2 Probability)      │ │
    │  │  - 49-dim features     │  │  - Head 1: p_wr        │ │
    │  │  - Pop percentile rank │  │  - Head 3: p_injury    │ │
    │  │  - Feature importances │  │  - Head 5: p_pen_p95   │ │
    │  │  - TailCalibrator      │  │  - Head 2: E[penalty]  │ │
    │  │  - Site scores         │  │  - Isotonic calibration│ │
    │  └────────────────────────┘  └────────────────────────┘ │
    │                                                          │
    │  Bühlmann Credibility Aggregation (multi-site)          │
    │  Composite: w1×p_wr + w2×p_inj + w3×p_ext              │
    │  Evidence shrinkage + CDF percentile stretch            │
    │                                                          │
    │  ┌────────────────────────┐                             │
    │  │  score_outlook.py      │  ← 12-month forecast        │
    │  └────────────────────────┘                             │
    └─────────────────────────────────┬────────────────────────┘
                                      │ RiskAssessment
    ┌─────────────────────────────────▼─────┐
    │        VettingAgent.enhance_explanation│  ← Gemini LLM (optional)
    └─────────────────────────────────┬──────┘
                                      │
                              ┌───────▼────────┐
                              │   FastAPI       │
                              │  /api/assess    │
                              │  (SSE stream)   │
                              └───────┬─────────┘
                                      │
                              ┌───────▼────────┐
                              │  React Frontend│
                              │  (Vite + TS)   │
                              └────────────────┘
```

---

## Appendix: Key File Index

| File | Role |
|---|---|
| `src/scoring/ml_risk_scorer.py` | Legacy GBR scorer + population percentile engine |
| `src/scoring/multi_target_scorer.py` | v2 Multi-target probabilistic model |
| `src/scoring/features.py` | Shared 25-feature extraction (online path) |
| `src/scoring/multi_target_labeler.py` | 7-target training label construction |
| `src/scoring/labeling/temporal_labeler.py` | Temporal train/val split + real-label builder |
| `src/scoring/labeling/helpers.py` | Adverse outcome formula + CSV streaming |
| `src/scoring/labeling/inspection_propensity.py` | IPW bias correction |
| `src/scoring/penalty_percentiles.py` | NAICS P75/P90/P95 threshold computation |
| `src/scoring/industry_stats.py` | Industry z-score computation + normalization |
| `src/scoring/tail_calibrator.py` | Isotonic tail calibration for legacy GBR |
| `src/scoring/calibration.py` | Brier score, BSS, ECE utilities |
| `src/scoring/ranking_metrics.py` | Decile lift, top-k precision, KS statistic |
| `src/scoring/risk_assessor.py` | Assessment orchestrator + Bühlmann aggregation |
| `src/scoring/score_outlook.py` | 12-month compliance forecast |
| `src/agent/vetting_agent.py` | Top-level VettingAgent + Gemini LLM integration |
| `src/search/grouped_search.py` | Entity resolution + facility grouping |
| `src/data_retrieval/osha_client.py` | DOL API client + bulk CSV cache loader |
| `src/models/assessment.py` | `RiskAssessment` + `ProbabilisticRiskTargets` Pydantic models |
| `src/models/osha_record.py` | `OSHARecord`, `Violation`, `AccidentSummary` models |
| `api/main.py` | FastAPI REST + SSE endpoints |
| `api/schemas.py` | API response schemas |
| `scripts/train_multi_target.py` | Offline model training script |
| `scripts/build_cache.py` | Offline bulk CSV cache builder |
| `tests/validation/mt_shared.py` | Threshold constants + 2021 holdout empirical bounds |
| `ml_cache/penalty_percentiles.json` | NAICS P75/P90/P95 thresholds |
| `ml_cache/multi_target_model.pkl` | Serialized MultiTargetRiskScorer |
| `ml_cache/risk_model.pkl` | Serialized legacy GBR pipeline |
