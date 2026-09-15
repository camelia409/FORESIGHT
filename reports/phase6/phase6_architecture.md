# PROJECT FORESIGHT — PHASE 6 SYSTEM ARCHITECTURE DOCUMENT

**Document ID:** `FORESIGHT-ARCH-P6-001`  
**Phase:** Phase 6 — Production Deployment, Pipeline Orchestration & Dashboard Integration  
**Status:** PRODUCTION COMPLETE & VERIFIED  
**Audience:** Enterprise Architecture, Data Engineering, ML Engineering, Operations  

---

## 1. Executive Summary & System Purpose

Project FORESIGHT is an enterprise-grade, AI-powered demand forecasting and inventory intelligence platform designed to eliminate stockouts, minimize working capital trapped in excess inventory, and automate replenishment decision support across enterprise retail/wholesale supply chains.

The platform processes multi-echelon sales and inventory signals, produces multi-horizon demand forecasts across 8 forward weeks using production machine learning models, executes physics-based inventory risk projections, and generates prioritized, deterministic operational directives for procurement buyers, inventory controllers, and logistics planners.

---

## 2. End-to-End System Architecture

The FORESIGHT architecture is organized into four loosely coupled, highly cohesive layers:

```
[ Data Ingestion & Storage ]
       │
       ▼
[ Pipeline Orchestration DAG (src/production_pipeline.py) ]
  ├── Stage 1: Ingestion & Input Verification
  ├── Stage 2: Validation & 50-SKU Universe Enforcement (150 Quarantined)
  ├── Stage 3: Model Loading & Lineage Verification
  ├── Stage 4: Multi-Horizon Forecasting (RF h=1, XGB h=2, Direct h=3..8)
  ├── Stage 5: Forward Inventory Position Projection (Policy B_LT)
  ├── Stage 6: Deterministic Risk Scoring (Stockout & Overstock)
  ├── Stage 7: Decision Support & Recommendation Engine
  ├── Stage 8: Governance Filtering (Option 1D Monetary Exclusion)
  └── Stage 9: Serving & Export (Parquet, CSV, Manifest JSON)
       │
       ├───────────────────────────────────────┐
       ▼                                       ▼
[ Real-Time Inference API ]          [ Executive Dashboard ]
(api/main.py, api/inference.py)       (app/streamlit_app.py)
  • Port 8000                           • Port 8501
  • FastAPI Async Server                • Multi-page Streamlit
  • 10 Standard Endpoints               • 5 Analytical Views
  • Sub-10ms Latency                    • Plotly Visualizations
```

---

## 3. Core Engine Components

### 3.1. Production Forecasting Engine (`src/models/` & `models/production/`)
- **Horizon 1 ($h=1$ week):** Tuned Random Forest regressor (`random_forest_h1.joblib`). Captures non-linear promotional and price interactions with minimum variance.
- **Horizon 2 ($h=2$ weeks):** Tuned XGBoost regressor (`xgboost_h2.joblib`). Optimized gradient-boosted decision trees delivering high accuracy on near-term demand shifts.
- **Horizons 3–8 ($h=3..8$ weeks):** Direct forecasting models incorporating 52-week seasonal naive baselines and trend adjustments to deliver stable medium-term demand planning horizons.

### 3.2. Inventory Risk Engine (`src/risk_engine.py` & `src/risk_scoring.py`)
- **Forward Inventory Trajectory:**
  $$\text{IP}_h = \text{Current\_Stock} + \sum_{i=1}^h \text{Arrival}_i - \sum_{i=1}^h \hat{y}_i$$
- **On-Order Arrival Inclusion (Ratified Option 2A — Policy $B_{LT}$):**
  Inbound purchase orders are recognized at $h=1$ if supplier lead time $\le 7$ days; for lead times between 8 and 14 days, inbound orders are recognized at $h=2$. No synthetic PO delivery dates are generated.
- **Stockout Score (0–100):** Continuous non-linear sigmoid risk metric combining weeks of cover, lead time demand, and projected stockout horizon.
- **Overstock Threshold (Ratified Option 3C — $N=8$ Weeks):**
  Identifies surplus coverage exceeding the 8-week planning ceiling:
  $$\text{Excess\_Units} = \max(0, \text{Current\_Stock} - (\text{Safety\_Stock} + 8 \times \bar{d}))$$

### 3.3. Decision Support & Recommendation Engine (`src/decision_support.py`)
- Evaluates operational priorities across 5 strict tiers:
  1. **Priority 1 (Critical):** `EXPEDITE_PO` — Stockout projected within supplier lead time.
  2. **Priority 2 (High):** `PLACE_PO` — Reorder point breached within lead time window.
  3. **Priority 3 (Medium):** `REVIEW_PIPELINE` — Reorder point breached with in-flight PO arriving.
  4. **Priority 4 (Low):** `FREEZE_REPLENISHMENT` — Surplus stock exceeds 8 weeks of cover.
  5. **Priority 5 (Informational):** `MAINTAIN_SCHEDULE` — Healthy inventory within target buffer.

### 3.4. Serving API Layer (`api/main.py`, `api/inference.py`, `api/schemas.py`)
Exposes 10 REST endpoints with sub-10ms response latency, full OpenAPI documentation (`/docs`), Pydantic V2 schema validation, CORS security middleware, and structured error responses.

### 3.5. Executive Dashboard (`app/streamlit_app.py` & `app/pages/`)
State-of-the-art multi-page Streamlit portal providing:
1. Executive Command Center (`streamlit_app.py`)
2. Fleet Health Overview (`01_executive_overview.py`)
3. Multi-Horizon Demand Forecasting (`02_demand_forecast.py`)
4. Inventory Risk Matrix (`03_inventory_risk.py`)
5. Tactical Action Center (`04_action_center.py`)
6. SKU 360° Intelligence Dossier (`05_sku_detail.py`)

---

## 4. Governance & Policy Enforcements

| Governance Decision | Ratified Option | Architecture Enforcement Mechanism |
| :--- | :--- | :--- |
| **Decision #1: Valuation Basis** | Option 1D — Explicit Exclusion | All monetary fields (`excess_inventory_value`, `inventory_value_at_risk`, `capital_at_risk`) are strictly `None`/`null`. Zero proxy pricing is used. Operations are 100% physical unit-based. |
| **Decision #2: On_Order Policy** | Option 2A — Policy $B_{LT}$ | Inbound PO arrivals are conditioned on verified supplier lead times ($\le 14$ days). No speculative receipt dates are injected. |
| **Decision #3: Overstock Threshold** | Option 3C — $N=8$ Weeks | Overstock evaluation horizon is fixed at 8 weeks (`overstock_threshold_status = "POLICY_RATIFIED"`), matching the ML forecast boundary. |
| **Production Universe** | 50 SKUs (`SKU001`–`SKU050`) | The 150 orphan SKUs (`SKU051`–`SKU200`) missing catalog master attributes are 100% quarantined in preprocessing and rejected with 404 by the API. |

---

## 5. Security, Reliability & Quality Assurance

- **Pre-Execution Cryptographic Integrity:** All 11 upstream model and data artifacts are SHA-256 hashed and verified before every release.
- **Regression Suite:** 257 automated tests covering data preprocessing, model inference, risk scoring, recommendation logic, API endpoints, error handling, and pipeline orchestration with a 100% pass rate (0 failures).
- **Stateless Operation:** Pipeline DAG executions are idempotent, writing timestamped run manifests and versioned outputs without modifying source historical datasets.
