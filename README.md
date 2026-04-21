# Manufacturer Compliance Intelligence & Vetting Agent

An AI-assisted manufacturer risk prediction and vetting system that evaluates the **future** compliance and safety risk of individual manufacturers using publicly available OSHA enforcement, incident, and inspection data.

The system moves beyond static summaries of past violations toward a **predictive, explainable** assessment of which manufacturers are more likely to experience serious compliance issues going forward.

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [Overall Architecture](#2-overall-architecture)
3. [Quick Start (Dev Setup)](#3-quick-start-dev-setup)
4. [Project Structure](#4-project-structure)
5. [Data Sources & Pipeline](#5-data-sources--pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [ML Models & Scoring Pipeline](#7-ml-models--scoring-pipeline)
8. [API Layer](#8-api-layer)
9. [Frontend](#9-frontend)
10. [Scripts Reference](#10-scripts-reference)
11. [Environment Variables](#11-environment-variables)
12. [Model Validation Metrics](#12-model-validation-metrics)
13. [Expansion Roadmap](#13-expansion-roadmap)

---

## 1. What the Project Does

Given a manufacturer name (with optional location metadata), the system:

1. **Resolves** the manufacturer name to one or more physical OSHA-registered establishments using fuzzy search and entity grouping.
2. **Retrieves** all publicly available OSHA enforcement history: inspections, citations, violation categories, penalties, and linked accident/injury reports.
3. **Extracts** a structured 49-dimensional feature vector capturing violation severity, penalty magnitude trends, recency, repeat/willful patterns, and industry-relative comparisons.
4. **Predicts** three calibrated probability targets for the next 12 months:
   - P(≥1 Serious/Willful/Repeat violation)
   - P(hospitalization or fatality event)
   - P(penalty ≥ industry 95th percentile)
5. **Produces** a single 0–100 composite risk score, a 3-tier recommendation (`Recommend` / `Proceed with Caution` / `Do Not Recommend`), and a 12-month forward compliance outlook.
6. **Explains** the score using feature weights, per-site breakdowns, industry percentile comparisons, and optionally a Gemini LLM-generated plain-English narrative.
7. **Answers follow-up questions** about the assessment through an interactive chat interface.

**Key principle:** Absence of enforcement records is surfaced as uncertainty — not interpreted as low risk.

---

## 2. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  React/TypeScript Frontend (Vite)          :5173                    │
│  SearchCard → SSE stream → RiskBanner, OutlookPanel, ViolationsGrid │
└────────────────────────┬────────────────────────────────────────────┘
                         │  /api/* (proxied by Vite)
┌────────────────────────▼────────────────────────────────────────────┐
│  FastAPI Backend                           :8000                    │
│  GET /api/health  GET /api/companies  GET /api/search               │
│  GET /api/locations   GET /api/assess (SSE stream)                  │
└──────┬──────────────────────────────────────────┬───────────────────┘
       │                                          │
┌──────▼──────────────┐              ┌────────────▼──────────────────┐
│  VettingAgent        │              │  OSHAClient                   │
│  (src/agent/)        │              │  (src/data_retrieval/)        │
│  - vet_manufacturer  │              │  - SQLite cache (osha_cache.db│
│  - vet_by_raw_names  │              │  - Company name normalisation │
│  - discuss_assessment│              │  - Entity key indexing        │
│  - Gemini LLM layer  │              └────────────────────────────────┘
└──────┬──────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│  RiskAssessor  (src/scoring/)                                        │
│  1. MLRiskScorer (legacy GBR)  — features, percentile rank          │
│  2. MultiTargetRiskScorer       — calibrated probability heads       │
│  3. Bühlmann credibility aggregation across sites                    │
│  4. 12-month Compliance Outlook (score_outlook.py)                  │
└──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│  ml_cache/                                                           │
│  osha_cache.db  risk_model.pkl  multi_target_model.pkl              │
│  inspections_bulk.csv  violations_bulk.csv  accidents_bulk.csv      │
│  penalty_percentiles.json  company_key_index.json                   │
└──────────────────────────────────────────────────────────────────────┘
       │  built by
┌──────▼──────────────────────────────────────────────────────────────┐
│  Data Pipeline  (scripts/)                                           │
│  build_cache.py  →  OshaData/ CSV chunks  →  ml_cache/              │
│  train_multi_target.py  →  ml_cache/multi_target_model.pkl          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start (Dev Setup)

### Prerequisites

- Python 3.10+
- Node.js / npm

### Install dependencies

```bash
# Python packages
pip install -r requirements.txt

# Frontend packages
cd frontend
npm install
cd ..
```

### Build the data cache (first-time, takes several minutes)

```bash
py scripts/build_cache.py
```

This reads all CSV chunks in `OshaData/` and writes consolidated files to `ml_cache/`, including the SQLite database `osha_cache.db`.

### Train the ML models (first-time, takes several minutes)

The base `MLRiskScorer` (`risk_model.pkl`) trains automatically on first use if no saved model is found. To build the multi-target probabilistic model explicitly:

```bash
py scripts/train_multi_target.py
# Or with a custom sample size:
py scripts/train_multi_target.py --sample-size 30000
# Force rebuild ignoring cached labels:
py scripts/train_multi_target.py --force
```

### Start the full stack

```bash
scripts\start_dev.bat
```

This starts:
- **FastAPI** on `http://localhost:8000` (with `--reload`)
- **Vite dev server** on `http://localhost:5173` (proxies `/api` to the backend)

Or start them separately:

**Terminal 1 — API:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Other entry points

| Command | Purpose |
|---|---|
| `py scripts/cli.py` | Interactive CLI vetting tool |
| `http://localhost:8000/docs` | Swagger UI for the API |
| `cd frontend && npm run build` | Production frontend build → `frontend/dist/` |

---

## 4. Project Structure

```
vetManufactures/
├── api/
│   ├── main.py              # FastAPI app — all routes, SSE streaming
│   └── schemas.py           # Pydantic response models
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Root component, tab/page state
│   │   ├── api/client.ts    # SSE client, API calls
│   │   ├── components/      # UI panels (see §9)
│   │   └── types/           # TypeScript types
│   └── package.json
├── src/
│   ├── agent/
│   │   └── vetting_agent.py # Orchestrates OSHAClient + RiskAssessor + Gemini
│   ├── data_retrieval/
│   │   ├── osha_client.py   # Loads/queries SQLite or CSV cache; DOL API calls
│   │   ├── naics_lookup.py  # NAICS industry name resolution
│   │   └── normalization/   # Company name cleaning & match-key generation
│   ├── models/
│   │   ├── assessment.py    # RiskAssessment, ProbabilisticRiskTargets dataclasses
│   │   ├── manufacturer.py  # Manufacturer dataclass
│   │   └── osha_record.py   # OSHARecord, Violation, AccidentSummary dataclasses
│   ├── scoring/
│   │   ├── features.py      # 25-element raw feature extraction
│   │   ├── ml_risk_scorer.py# Legacy GBR scorer; population percentile ranking
│   │   ├── multi_target_scorer.py # Primary probability-head model
│   │   ├── multi_target_labeler.py# Training label construction (temporal split)
│   │   ├── risk_assessor.py # Orchestrates both scorers + credibility aggregation
│   │   ├── score_outlook.py # 12-month forward compliance projection
│   │   ├── penalty_percentiles.py # NAICS-stratified P75/P90/P95 thresholds
│   │   ├── industry_stats.py# Industry mean/std for z-score features
│   │   ├── calibration.py   # Isotonic regression calibration helpers
│   │   ├── tail_calibrator.py
│   │   └── ranking_metrics.py
│   └── search/
│       └── grouped_search.py# Fuzzy entity grouping (rapidfuzz)
├── scripts/
│   ├── build_cache.py       # OSHA CSV → ml_cache/ pipeline
│   ├── train_multi_target.py# Train multi-target model
│   ├── cli.py               # Interactive CLI
│   ├── reoptimize_weights.py# Re-optimize composite weights only
│   ├── plot_feature_weights.py # Chart feature importances
│   ├── patch_company_keys.py# Recompute company_key column in SQLite
│   ├── _gen_metrics_table.py# Generate model metrics chart
│   ├── _gen_feature_table.py# Generate feature table chart
│   ├── start_dev.bat        # Start full stack
│   └── launch.bat
├── OshaData/                # Raw DOL API CSV chunks (input to build_cache.py)
│   ├── OSHA_inspection/
│   ├── OSHA_violation/
│   ├── OSHA_accident/
│   ├── OSHA_accident_abstract/
│   ├── OSHA_accident_injury/
│   ├── OSHA_violation_gen_duty_std/
│   └── ...
├── ml_cache/                # Processed cache (output of build_cache.py + training)
│   ├── osha_cache.db        # SQLite (primary lookup store)
│   ├── inspections_bulk.csv
│   ├── violations_bulk.csv
│   ├── accidents_bulk.csv
│   ├── accident_injuries_bulk.csv
│   ├── accident_abstracts_bulk.csv
│   ├── gen_duty_narratives_bulk.csv
│   ├── bulk_meta.json       # Cache timestamp metadata
│   ├── company_key_index.json
│   ├── penalty_percentiles.json
│   ├── population_data.json
│   ├── risk_model.pkl       # Legacy MLRiskScorer (GBR)
│   └── multi_target_model.pkl # Primary MultiTargetRiskScorer
├── tests/
├── fix_cdf.py               # Utility: rebuild score CDF in saved model
├── requirements.txt
├── DEV_SETUP.md
├── MODEL_REPORT.md          # Full technical model documentation
└── copilot-instructions.md
```

---

## 5. Data Sources & Pipeline

### Source

All data comes from the **U.S. Department of Labor OSHA Enforcement API (V4)**:
`https://apiprod.dol.gov/v4/get/OSHA`

Datasets used:

| OSHA Dataset | Content |
|---|---|
| `OSHA_inspection` | Inspection case details, date opened, establishment name, NAICS, employee count |
| `OSHA_violation` | Citation details, violation type, penalty amounts, gravity scores |
| `OSHA_accident` | Workplace accident event records |
| `OSHA_accident_abstract` | Multi-line accident narrative text |
| `OSHA_accident_injury` | Per-person injury records linked to accidents |
| `OSHA_violation_gen_duty_std` | General Duty Clause narrative text per citation |
| `OSHA_emphasis_codes` | Emphasis program codes |
| `OSHA_related_activity` | Related enforcement activities |

Raw data is downloaded as chunked CSV files into `OshaData/<dataset>/`.

### Cache Build (`scripts/build_cache.py`)

```bash
py scripts/build_cache.py
```

Steps performed:
1. Reads all CSV chunks from `OshaData/` sub-folders (natural-sorted)
2. Applies a **10-year rolling window** filter on inspections and violations (configurable via `CUTOFF_YEARS`)
3. Pre-joins multi-line accident abstract text into one row per accident
4. Pre-joins multi-line General Duty Clause narratives into one row per citation
5. Writes consolidated CSVs to `ml_cache/`
6. Builds `osha_cache.db` — a SQLite database with indexes on `company_key`, `estab_name`, `activity_nr`, enabling fast lookup without loading all CSVs into memory

**Why 10 years?** 3 years was too short (missed predictive pre-2020 violations). Full history (1972–present) was too long (reformed companies with old records score artificially high, degrading prediction quality).

### Company Name Normalization

`src/data_retrieval/normalization/company_names.py` performs:
- Strip legal suffixes (Inc, LLC, Corp, Ltd, etc.)
- Canonicalize common token variants (INTL→INTERNATIONAL, MFG→MANUFACTURING, etc.)
- Generate a `company_match_key` — a deduplicated, normalized token string used for fuzzy grouping

---

## 6. Feature Engineering

### Raw Feature Extraction (`src/scoring/features.py`)

`extract_establishment_features_raw()` produces **25 absolute features** per establishment from a list of `OSHARecord` objects:

| # | Feature | Description |
|---|---|---|
| 0 | `n_insp` | Total inspection count |
| 1 | `n_viols` | Total violation count |
| 2 | `serious_rate` | Serious violations / inspection |
| 3 | `willful_rate` | Willful violations / inspection |
| 4 | `repeat_rate` | Repeat violations / inspection |
| 5 | `total_pen` | Total penalty dollars |
| 6 | `avg_pen` | Mean penalty per violation |
| 7 | `max_pen` | Maximum single violation penalty |
| 8 | `recent_ratio` | Fraction of inspections within last 3 years |
| 9 | `severe_rate` | Inspections with accidents / total inspections |
| 10 | `vpi` | Violations per inspection |
| 11 | `acc_rate` | Accidents per inspection |
| 12 | `fat_rate` | Fatalities per inspection |
| 13 | `inj_rate` | Injuries per inspection |
| 14 | `avg_gravity` | Mean OSHA gravity score across all violations |
| 15 | `pen_per_insp` | Penalty dollars per inspection |
| 16 | `clean_ratio` | Fraction of inspections with zero violations |
| 17 | `time_adj_pen` | Time-decayed penalty (exp decay τ=3 years) |
| 18 | `recent_wr_rate` | Fraction of willful/repeat violations that are recent |
| 19 | `trend_delta` | VPI (all-time) minus VPI (recent) — positive = worsening |
| 20 | `log1p(willful_raw)` | Log-compressed absolute willful count |
| 21 | `log1p(repeat_raw)` | Log-compressed absolute repeat count |
| 22 | `fatality_flag` | 1 if any fatality ever recorded, 0 otherwise |
| 23 | `log1p(max_insp_pen)` | Log-compressed single worst inspection penalty |
| 24 | `log1p(median_estab_size)` | Log-compressed median establishment employee count |

### Full 49-Feature Vector (`src/scoring/ml_risk_scorer.py`)

`MLRiskScorer` extends the 25-feature raw vector to **49 features** by adding:

**Industry-relative z-scores (4 features):** — compare the establishment against its NAICS 2-digit industry cohort:
- `relative_violation_rate` (z-score)
- `relative_penalty` (z-score)
- `relative_serious_ratio` (z-score)
- `relative_willful_repeat` (z-score)

**2-digit NAICS sector one-hot encoding (24 sectors + unknown = 25 features):**
`naics_11` through `naics_92`, plus `naics_unknown`

Selected raw features are log-transformed before model input: `log_inspections`, `log_violations`, `accident_count`, `fatality_count`, `injury_count` (indices 0, 1, 11, 12, 13). Penalty amounts are intentionally **not** log-compressed to preserve high-end signal differentiation.

---

## 7. ML Models & Scoring Pipeline

### Two-tier architecture

#### Tier 1 — Legacy Base Scorer (`MLRiskScorer`)
- **Algorithm:** `sklearn.ensemble.GradientBoostingRegressor` in a `StandardScaler → GBR` pipeline
- **Role:** Provides population-relative percentile ranking, industry comparison, and per-site scores. Trained on OSHA population data so each score is relative to the broader inspected-establishment population.
- **Artifact:** `ml_cache/risk_model.pkl`

#### Tier 2 — Primary Multi-Target Scorer (`MultiTargetRiskScorer`)
- **Algorithm:** `sklearn.ensemble.HistGradientBoostingClassifier` per head, with early stopping, isotonic regression calibration, and Nelder-Mead weight optimization.
- **Artifact:** `ml_cache/multi_target_model.pkl`

**Prediction heads:**

| Head | Target | Algorithm |
|---|---|---|
| Head 1 | `p_serious_wr_event` — P(≥1 Serious/Willful/Repeat violation in 12mo) | HGBC + isotonic calibration |
| Head 2 | `p_injury_event` — P(hospitalization or fatality) | HGBC + isotonic calibration |
| Head 3 | `p_penalty_ge_p95` — P(penalty ≥ NAICS-specific P95 threshold) | HGBC + isotonic calibration |
| Head 4a | `p_penalty_ge_p75` — auxiliary tier (not in composite) | HGBC |
| Head 4b | `p_penalty_ge_p90` — auxiliary tier (not in composite) | HGBC |
| Reg 1 | `expected_penalty_usd_12m` — hurdle: binary × conditional log-penalty GBR | GBR + isotonic |
| Reg 2 | `gravity_score` — expected OSHA gravity | GBR + isotonic |

### Composite Score Formula

```
raw = w1 × p_serious_wr_event  +  w2 × p_injury_event  +  w3 × p_penalty_ge_p95

Default weights:  w1 = 0.578,  w2 = 0.256,  w3 = 0.165
```

**Post-processing pipeline:**
1. **Evidence shrinkage** toward population prior (15.0) using smooth exponential based on inspection count — prevents high scores from very sparse data.
2. **Percentile stretching** via `score = percentile^2.0 × 100` — concentrates most companies in the lower half, spreads high-risk companies apart at the top.
3. **Rescaling** to 0–100.

### Training Label Construction (`src/scoring/multi_target_labeler.py`)

- **Temporal cutoff:** 2022-01-01 (features built from pre-2022 OSHA history)
- **Outcome window:** 2022-01-01 → 2025-03-31 (3+ year post-COVID window; 2020–2021 excluded to remove COVID enforcement suppression)
- **IPW bias correction:** Inverse Probability Weighting corrects for OSHA's selection bias (establishments inspected more frequently tend to be higher-risk) so labels reflect expected rates, not sampling artifacts.
- Default sample size: 50,000 establishments

### Penalty Percentile Thresholds (`src/scoring/penalty_percentiles.py`)

NAICS 2-digit industry-stratified P75/P90/P95 penalty thresholds computed from pre-cutoff violations and saved to `ml_cache/penalty_percentiles.json`. Groups with fewer than 50 samples fall back to a `__global__` threshold.

### Multi-Site / Multi-Establishment Aggregation (`src/scoring/risk_assessor.py`)

For companies with multiple facilities, the system uses **Bühlmann credibility theory** (as used in NCCI workers' compensation experience rating):

```
Z_i = n_i / (n_i + K)              # K = 5  (credibility parameter)
credible_i = Z_i × score_i + (1 - Z_i) × portfolio_prior
```

A **tail-exposure blend** then weights the worst site more heavily when risk is concentrated:

```
final = (1 - α) × credibility_mean + α × max_credible
α = min(risk_concentration × 0.30, 0.30)
```

### Recommendation Thresholds

| Risk Score | Recommendation |
|---|---|
| Low | **Recommend** |
| Moderate | **Proceed with Caution** |
| High | **Do Not Recommend** |

### 12-Month Compliance Outlook (`src/scoring/score_outlook.py`)

Projects forward compliance metrics using per-inspection-rate features and the risk score:
- `expected_inspections_12m`
- `expected_violations_12m`
- `expected_penalties_usd_12m`
- `expected_serious_12m`
- `expected_willful_repeat_12m`
- `risk_band`: `"low"` / `"moderate"` / `"high"`

Score-based annual inspection frequency floors ensure projections don't collapse to zero for currently low-activity companies:
- Score ≥ 60 → 1.5 inspections/year floor
- Score 40–59 → 0.75/year
- Score 20–39 → 0.30/year
- Score < 20 → 0.10/year

### LLM Enhancement (Optional)

If `GOOGLE_API_KEY` is set, the `VettingAgent` passes the structured assessment through **Google Gemini** (`google-genai`) to generate a plain-English explanation narrative and to power the follow-up Q&A `discuss_assessment()` chat interface.

---

## 8. API Layer

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`

### Endpoints

#### `GET /api/health`
Health check.
```json
{"status": "ok"}
```

#### `GET /api/companies`
Returns a sorted, deduplicated list of all company names in the OSHA cache.
```json
["3M COMPANY", "ABBOTT LABORATORIES", ...]
```

#### `GET /api/search?q=<query>`
Fuzzy-searches company names and returns establishments grouped by likely parent entity. Uses `rapidfuzz` with a score cutoff of 55.

Response: `SearchResponse`
```json
{
  "query": "Amazon",
  "top_group": {
    "parent_name": "Amazon",
    "total_facilities": 47,
    "confidence": 0.92,
    "confidence_label": "High",
    "high_confidence": [...],
    "medium_confidence": [...],
    "low_confidence": [...]
  },
  "other_groups": [...],
  "unmatched": []
}
```

Each facility in the group includes `raw_name`, `display_name`, `facility_code`, `city`, `state`, `address`, `naics_code`, `confidence`, `confidence_label`.

#### `GET /api/locations?company=<name>`
Returns a list of address strings for a given company name.

#### `GET /api/assess` — **SSE Stream**
The primary assessment endpoint. Accepts either `raw_names` (comma-separated OSHA establishment names, preferred from the search UI) or `company` (name-resolution based).

**Query parameters:**
- `raw_names` — comma-separated raw OSHA estab names (e.g. `"AMAZON.COM SERVICES LLC,AMAZON (CMH1)"`)
- `company` — company name for resolution-based lookup
- `display_name` — label to use in the response
- `years_back` — how many years of history to use (default: 10)

**Server-Sent Events stream — event types:**

| Event type | Payload |
|---|---|
| `progress` | `{"message": "🔍 Searching OSHA records…"}` |
| `result` | Full `AssessmentResponse` JSON |
| `error` | `{"message": "..."}` |

**`AssessmentResponse` key fields:**

| Field | Type | Description |
|---|---|---|
| `manufacturer_name` | string | Resolved company name |
| `risk_score` | float | 0–100 composite risk score |
| `recommendation` | string | `"Recommend"` / `"Proceed with Caution"` / `"Do Not Recommend"` |
| `explanation` | string | AI-generated or rule-based explanation |
| `confidence_score` | float | 0–1 confidence in the score |
| `risk_confidence` | string | `"high"` / `"medium"` / `"low"` |
| `feature_weights` | dict | Per-feature contributions to the score |
| `percentile_rank` | float | Percentile vs. all OSHA-inspected establishments |
| `industry_label` | string | NAICS industry label |
| `industry_percentile` | float | Percentile within the same NAICS 2-digit group |
| `industry_comparison` | list[str] | Human-readable industry comparison bullets |
| `establishment_count` | int | Number of distinct facilities scored |
| `site_scores` | list | Per-site risk scores (`name`, `score`, `n_inspections`, `naics_code`, `city`, `state`) |
| `risk_concentration` | float | Gini-like measure of whether risk is concentrated in one site |
| `systemic_risk_flag` | bool | True if risk is broadly distributed across sites |
| `records` | list | Full `OSHARecordOut` objects with all violations and accidents |
| `record_count` | int | Number of inspection records returned |
| `outlook` | `ComplianceOutlook12M` | 12-month forward projection |
| `risk_targets` | `ProbabilisticRiskTargetsOut` | Raw probability head outputs |

**`ProbabilisticRiskTargetsOut` fields:**

| Field | Description |
|---|---|
| `p_serious_wr_event` | P(Serious/Willful/Repeat violation in 12mo) |
| `p_injury_event` | P(hospitalization or fatality event) |
| `p_penalty_ge_p95` | P(penalty ≥ industry P95) |
| `p_penalty_ge_p75` | P(penalty ≥ industry P75) |
| `p_penalty_ge_p90` | P(penalty ≥ industry P90) |
| `expected_penalty_usd_12m` | Model-predicted expected penalty dollars |
| `gravity_score` | Model-predicted OSHA gravity score |
| `composite_risk_score` | Raw composite before percentile stretching |

The frontend calls this endpoint via `POST /api/assess` using a JSON body (`raw_names` array, `display_name`, `years_back`) — internally the POST handler is also routed by `openAssessStream()` in `frontend/src/api/client.ts`.

---

## 9. Frontend

**Stack:** React 18, TypeScript, Vite, AG Grid Community, Lucide React icons.

**URL:** `http://localhost:5173`

### Pages

| Page | Description |
|---|---|
| **Assessment** | Main workflow — search, select facilities, run assessment, view results |
| **Model Report** | Static technical model documentation rendered as a printable/PDF report page |

### Assessment Page Layout

```
┌─────────────────────────────────────────────────────┐
│ Topbar: "Manufacturer Compliance Intelligence"       │
├──────────────┬──────────────────────────────────────┤
│  Sidebar     │  Content area                        │
│  SearchCard  │  ProgressStream (SSE messages)       │
│  (search,    │  RiskBanner (score + recommendation) │
│  select,     │  StatsGrid (key metrics)             │
│  Run button) │  Tabs:                               │
│              │    Overview → RiskTargetsPanel +     │
│              │               OutlookPanel + Summary │
│              │    Violations → ViolationsGrid (AG)  │
│              │    Sites → SiteBreakdown             │
│              │    Model Details → ExplanationPanel  │
│              │  ChatBox (Q&A with the agent)        │
└──────────────┴──────────────────────────────────────┘
```

### Components

| Component | Purpose |
|---|---|
| `SearchCard` | Debounced company search, facility selection with confidence tiers, "Run Assessment" button |
| `ProgressStream` | Renders streaming SSE `progress` messages with a live indicator |
| `RiskBanner` | Prominent coloured banner showing risk score (0–100) and recommendation tier |
| `StatsGrid` | Key metric tiles: record count, establishment count, industry percentile, confidence |
| `RiskTargetsPanel` | Gauge/bar display for the three primary probability heads |
| `OutlookPanel` | 12-month forward projection (inspections, violations, penalties, risk band) |
| `ExplanationPanel` | Feature weight breakdown, top drivers, industry comparison bullets |
| `ViolationsGrid` | AG Grid table of all violation records with filtering and sorting |
| `SiteBreakdown` | Per-establishment score cards, drop-high-risk-site recalculation control |
| `ChatBox` | Follow-up Q&A chat interface powered by the Gemini LLM layer |
| `ReportPage` | Printable model technical report with embedded SVG pipeline diagram and metrics charts |

### API Client (`frontend/src/api/client.ts`)

- `openAssessStream(request, onEvent)` — opens an SSE connection to `/api/assess`, dispatches `progress` / `result` / `error` events to the callback, returns a cleanup function.
- `recalculateDroppingHighRisk(result, threshold)` — client-side recalculation that drops sites above a score threshold and calls `reassess`.

---

## 10. Scripts Reference

| Script | Command | Purpose |
|---|---|---|
| `scripts/build_cache.py` | `py scripts/build_cache.py` | Consolidate `OshaData/` CSV chunks into `ml_cache/`. Builds SQLite DB. |
| `scripts/train_multi_target.py` | `py scripts/train_multi_target.py [--sample-size N] [--force]` | Build penalty percentile thresholds and train `multi_target_model.pkl`. |
| `scripts/cli.py` | `py scripts/cli.py` | Interactive CLI — enter a company name, get a risk report, ask follow-up questions. |
| `scripts/reoptimize_weights.py` | `py scripts/reoptimize_weights.py` | Re-optimize composite weights (`w1, w2, w3`) on existing trained heads via Nelder-Mead. Much faster than full retraining. |
| `scripts/plot_feature_weights.py` | `py scripts/plot_feature_weights.py [--top N] [--output path]` | Chart top feature importances for the three primary probability heads. Saves to `plots/`. |
| `scripts/patch_company_keys.py` | `py scripts/patch_company_keys.py` | Recompute `company_key` column in SQLite using the current normalization algorithm. Run after normalization logic changes; requires API server to be stopped. |
| `scripts/_gen_metrics_table.py` | `py scripts/_gen_metrics_table.py` | Generate model validation metrics bar chart (AUROC, Brier Skill, Lift). |
| `scripts/_gen_feature_table.py` | `py scripts/_gen_feature_table.py` | Generate feature engineering table chart. |
| `scripts/start_dev.bat` | `scripts\start_dev.bat` | Start FastAPI + Vite dev server concurrently. |
| `fix_cdf.py` | `py fix_cdf.py` | One-off utility: rebuild the score CDF in a saved `multi_target_model.pkl` without full retraining. Run if composite weights are changed externally. |

---

## 11. Environment Variables

Create a `.env` file in the workspace root (loaded by `python-dotenv`):

```env
# Required for LLM-powered explanations and follow-up chat
GOOGLE_API_KEY=your_google_gemini_api_key

# DOL API key (if rate-limit delta fetching is needed)
# DOL_API_KEY=your_dol_api_key
```

If `GOOGLE_API_KEY` is not set, the system operates fully without the LLM layer — risk scores, features, and probability predictions all work without it; only the plain-English explanation and interactive chat are disabled.

---

## 12. Model Validation Metrics

Evaluated on a 20% held-out validation fold (temporal split, pre-2022 features → 2022–2025 outcomes):

| Metric | Value |
|---|---|
| **WR/Serious Head** | |
| AUROC | 0.751 |
| Brier Skill Score | 0.1885 |
| PR-AUC / Average Precision | 0.721 |
| Top-Decile Lift | 2.145× |
| **Injury/Fatality Head** | |
| AUROC | 0.797 |
| Brier Skill Score | 0.0858 |
| Top-10% Capture Rate | 12.7% |
| **Penalty Tier Head (P95)** | |
| AUROC | 0.812 |
| PR-AUC / Average Precision | 0.634 |
| Top-Decile Lift | 3.2× |

**Overall test suite:** 113 tests passing, 1 skipped (as of April 2026).

For full technical details — training data construction, IPW bias correction, calibration methodology, adversarial validation, and per-head training hyperparameters — see [MODEL_REPORT.md](MODEL_REPORT.md).

---

## 13. Expansion Roadmap

| Phase | Scope |
|---|---|
| **Phase 1** *(current)* | Individual manufacturer OSHA risk prediction and vetting |
| **Phase 2** | Multi-source compliance intelligence: EPA/environmental enforcement, certification signals (ISO, quality standards), improved parent/subsidiary resolution |
| **Phase 3** | Comparative risk intelligence: industry peer benchmarking, cohort-based percentile comparisons, trend analysis across manufacturers |
| **Phase 4** | Monitoring & decision support: supplier alerting over time, historical vs. predicted risk tracking, operational integration |
