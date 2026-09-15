# Project FORESIGHT — Phase 4B: Inventory Risk Engine Implementation & Validation Report

**Document ID:** `FORESIGHT-PHASE4B-REPORT`  
**Phase:** `Phase 4B — Inventory Risk Engine Implementation`  
**Status:** `IMPLEMENTATION & VALIDATION COMPLETE`  
**Author:** Production AI/ML Demand & Inventory Systems Team  
**Date:** `2026-09-15`  
**Target Codebase:** `f:\zidio\foresight\`  

---

## Executive Summary

Phase 4B implements the production **Inventory Risk Engine** (`src/risk_engine.py`) and batch scoring orchestrator (`src/risk_scoring.py`) for Project FORESIGHT. The engine consumes multi-horizon demand forecasts ($h=1..8$) from the production forecasting engine and point-in-time inventory snapshots from `inventory_snapshots.csv`, producing deterministic, unit-based risk assessments and operational action tiers.

The implementation adheres strictly to all authoritative Phase 4A governance decisions:
1. **Decision #1 (Valuation Basis):** UNRESOLVED. Monetary risk metrics (`excess_inventory_value`, `inventory_value_at_risk`, `capital_at_risk`) remain strictly **BLOCKED** and are emitted as `None` / `null`.
2. **Decision #2 (On_Order Inclusion):** Implements approved horizon-conditional **$Policy\ B_{LT}$**:
   $$IP(t, h) = \text{Current\_Stock}(t) + \text{On\_Order}(t) \cdot \mathbb{I}[\text{Lead\_Time\_Days}(t) \le 7h] - \sum_{k=1}^h \hat{y}(t+k)$$
   with explicit provenance metadata: `arrival_timing_confirmed = False`.
3. **Decision #3 (Overstock Threshold):** Implements the terminal-horizon clearance model at $N_{\text{weeks}} = 8$ weeks:
   $$\text{Excess Units: } \Delta_{\text{excess}}(t) = \max\left(0,\ IP(t, 8) - \text{Safety\_Stock}(t)\right)$$
   tagged with `overstock_threshold_status = "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"`.
4. **Orphan SKUs (`SKU051`–`SKU200`):** Quarantined and strictly excluded from scoring. The active universe consists exclusively of the 50 production SKUs (`SKU001`–`SKU050`).
5. **Temporal Safety:** Backward asof snapshot matching enforces zero future leakage ($\text{Snapshot\_Date} \le \text{origin\_date}$).

---

## 1. Governance & Authoritative Decision Enforcement

| Decision | Policy Name | Status | Engine Behavior |
|---|---|---|---|
| **#1 Valuation Basis** | Independent constant | **UNRESOLVED** | All monetary metrics (`excess_inventory_value`, `inventory_value_at_risk`, `capital_at_risk`) remain strictly `None`. Emits `valuation_basis_confirmed = False`. |
| **#2 On_Order Policy** | $Policy\ B_{LT}$ | **SUPPORTED** | On_Order included at horizon $h$ if and only if $\text{Lead\_Time\_Days} \le 7h$. Emits `arrival_timing_confirmed = False`. |
| **#3 Overstock Threshold** | Terminal 8-week horizon | **PENDING BUSINESS APPROVAL** | Evaluates surplus remaining after 8 weeks of demand plus safety stock ($N=8$). Emits status tag `ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL`. |
| **Orphan SKU Policy** | Quarantine | **ENFORCED** | Non-master inventory SKUs (`SKU051`–`SKU200`) are quarantined. Production universe is strictly `SKU001`–`SKU050`. |
| **Temporal Safety** | Backward Asof Merge | **ENFORCED** | Snapshots with $\text{Snapshot\_Date} > t$ are prohibited. Missing snapshots trigger `UNKNOWN` action tier and null scores. |

---

## 2. Mathematical Formulas Implemented

All formulas were implemented deterministically in `src/risk_engine.py`:

### 2.1 Projected Inventory Position $IP(t, h)$
$$IP(t, h) = \text{Current\_Stock}(t) + \text{On\_Order}(t) \cdot \mathbb{I}[\text{Lead\_Time\_Days}(t) \le 7h] - \sum_{k=1}^h \hat{y}(t+k), \quad h \in \{1..8\}$$
Negative values are preserved for deficit reporting.

### 2.2 Projected Stock Balance $SB(t, h)$ (Conservative Variant)
$$SB(t, h) = \text{Current\_Stock}(t) - \sum_{k=1}^h \hat{y}(t+k), \quad h \in \{1..8\}$$

### 2.3 Weeks of Cover ($WoC$) & Days of Supply ($DoS$)
$$WoC(t) = \frac{\text{Current\_Stock}(t) + \text{On\_Order}(t)}{\text{AvgWeeklyDemand}(t)}, \quad DoS(t) = WoC(t) \times 7.0$$
$$WoC_{\text{stock}}(t) = \frac{\text{Current\_Stock}(t)}{\text{AvgWeeklyDemand}(t)}, \quad DoS_{\text{stock}}(t) = WoC_{\text{stock}}(t) \times 7.0$$

### 2.4 Lead-Time Demand $LTD(t)$
$$LTD(t) = \sum_{k=1}^{\lfloor LT/7 \rfloor} \hat{y}(t+k) + \left(\frac{LT}{7} - \lfloor LT/7 \rfloor\right) \cdot \hat{y}(t+\lceil LT/7 \rceil)$$

### 2.5 Safety Stock Breach & Reorder Point Breach
$$SS\_BREACH(t, h) = 1 \text{ if any } k \in \{1..h\}: IP(t, k) < \text{Safety\_Stock}(t), \text{ else } 0$$
$$RP\_BREACH(t, h) = 1 \text{ if any } k \in \{1..h\}: IP(t, k) < \text{Reorder\_Point}(t), \text{ else } 0$$
Earliest breach weeks are tracked: $\min\{k : IP(t, k) < \text{Threshold}\}$.

### 2.6 Continuous Stockout Risk Score $SR(t)$
$$SR(t) = 1.0 - \min\left(1.0,\ \frac{WoC(t)}{\frac{\text{Lead\_Time\_Days}(t)}{7.0} + \frac{\text{Safety\_Stock}(t)}{\text{AvgWeeklyDemand}(t)}}\right)$$
Bounded in $[0.0, 1.0]$. $SR=0.0$ when demand is zero or pipeline coverage exceeds lead time plus safety buffer. $SR=1.0$ when total pipeline is depleted.

### 2.7 Terminal-Horizon Excess Inventory & Overstock Severity
$$\text{Available Supply at Horizon 8: } S(t, 8) = \text{Current\_Stock}(t) + \text{On\_Order}(t) \cdot \mathbb{I}[\text{Lead\_Time\_Days}(t) \le 56]$$
$$\text{Excess Units: } \Delta_{\text{excess}}(t) = \max\left(0,\ S(t, 8) - \sum_{k=1}^8 \hat{y}(t+k) - \text{Safety\_Stock}(t)\right)$$
$$\text{Excess Weeks of Cover: } WoC_{\text{excess}}(t) = \frac{\Delta_{\text{excess}}(t)}{\text{AvgWeeklyDemand}(t)}$$
$$\text{Continuous Overstock Score: } OS(t) = \frac{\Delta_{\text{excess}}(t)}{\text{Current\_Stock}(t) + \text{On\_Order}(t)}$$

---

## 3. Operational Taxonomies & Classification Logic

### 3.1 Operational Action Tiers
Mapped deterministically according to operational urgency:

| Priority | Action Tier | Trigger Condition | Operational Guidance |
|---|---|---|---|
| **1** | `CRITICAL REORDER` | $IP(t, k) \le 0$ for any $k \le \lceil LT/7 \rceil$ | Stockout projected within supplier lead time. Expedite delivery. |
| **2** | `REORDER` | $IP(t, k) < \text{Reorder\_Point}$ for any $k \le \lceil LT/7 \rceil$ | Reorder Point breach within supplier lead time. Place purchase order now. |
| **3** | `MONITOR` | $IP(t, k) < \text{Reorder\_Point}$ for any $k \le 8$ | Reorder Point breach projected within 8 weeks. Review pipeline. |
| **4** | `OVERSTOCK` | $WoC_{\text{excess}} > 0.0$ (no reorder breach) | Inventory exceeds 8-week clearance demand + safety stock. |
| **5** | `HEALTHY` | No breaches, $WoC_{\text{excess}} \le 0.0$ | Adequate coverage across 8-week horizon. |
| **Special**| `UNKNOWN` | No inventory snapshot at origin $t$ | Inventory data unavailable. Cannot score. |

### 3.2 Overstock Severity Cutoffs
- **HEALTHY:** $WoC_{\text{excess}} \le 0.0$ weeks
- **MONITOR:** $0.0 < WoC_{\text{excess}} \le 2.0$ weeks
- **HIGH:** $2.0 < WoC_{\text{excess}} \le 6.0$ weeks
- **CRITICAL:** $WoC_{\text{excess}} > 6.0$ weeks

---

## 4. Real Data Execution & Invariant Validation

The risk scoring pipeline (`src/risk_scoring.py`) executed across all 9 production test origins available in `final_predictions.parquet`:

- **Origins Scored:** 9 weekly test origins (`2025-03-04` through `2025-09-16`).
- **Universe:** 50 production SKUs (`SKU001`–`SKU050`).
- **Total Scored Panel Rows:** Exactly **450 rows** ($9 \times 50$).
- **Latest Origin Evaluated:** `2025-09-16` (50 SKUs).

### 4.1 Panel Action Tier Distribution (N = 450)
- `MONITOR`: **182 records (40.4%)**
- `REORDER`: **141 records (31.3%)**
- `CRITICAL REORDER`: **69 records (15.3%)**
- `OVERSTOCK`: **58 records (12.9%)**
- `HEALTHY`: **0 records (0.0%)** (all SKUs experienced either a reorder breach or excess within 8 weeks)

### 4.2 Panel Overstock Severity Distribution (N = 450)
- `HEALTHY` ($WoC_{\text{excess}} \le 0$): **337 records (74.89%)**
- `CRITICAL` ($WoC_{\text{excess}} > 6\text{w}$): **50 records (11.11%)**
- `HIGH` ($2 < WoC_{\text{excess}} \le 6\text{w}$): **33 records (7.33%)**
- `MONITOR` ($0 < WoC_{\text{excess}} \le 2\text{w}$): **30 records (6.67%)**

### 4.3 Latest Origin Assessment (`2025-09-16`, N = 50)
- `MONITOR`: 24 SKUs (48.0%)
- `REORDER`: 10 SKUs (20.0%)
- `OVERSTOCK`: 8 SKUs (16.0%)
- `CRITICAL REORDER`: 8 SKUs (16.0%)
- Overstock Severity: 37 `HEALTHY` (74.0%), 5 `CRITICAL` (10.0%), 4 `MONITOR` (8.0%), 4 `HIGH` (8.0%).

### 4.4 Invariant Verification Results
- **Grain Integrity:** Exactly 50 SKUs per origin; 0 duplicates.
- **Valuation Gating:** `excess_inventory_value` = `None` on 450/450 rows ($100\%$).
- **Governance Flags:** `valuation_basis_confirmed = False` on 450/450 rows ($100\%$).
- **Score Range:** $0.0 \le \text{stockout\_score} \le 1.0$ for all valid observations.
- **Surplus Non-negativity:** $\text{excess\_inventory\_units} \ge 0.0$ for all observations.

---

## 5. API Serving Integration (`POST /v1/risk`)

The FastAPI endpoint in `api/inference.py` was connected to `RiskEngine`:
- **Endpoint:** `POST /v1/risk`
- **Request Schema:** `RiskRequest(origin_date: date, skus: Optional[list[str]])`
- **Response Schema:** `RiskResponse(origin_date: date, n_skus: int, risk_records: list[RiskRecord])`
- **Automated Test:** `tests/test_inference.py::test_09_risk_endpoint_success` PASSED.

---

## 6. Verification & Test Results

The full repository test suite was executed:
- **Total Tests Passed:** **226 passed, 10 skipped, 0 failed** (27.06s).
- **Targeted Risk Engine Tests:**
  - `tests/test_risk_engine.py`: 26 passed (deterministic formulas, On_Order inclusion, boundary cuts, temporal safety, orphan quarantine, valuation gating).
  - `tests/test_risk_scoring.py`: 2 passed (real-data batch execution, parquet/JSON artifact creation, invariant validation).
  - `tests/test_inference.py`: 9 passed (including live `POST /v1/risk` endpoint test).

### Source Data & Model Artifact Hashes (Verified Unchanged)
- `data/raw/inventory_snapshots.csv`: `167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd` (VERIFIED)
- `data/raw/sku_master.csv`: `6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9` (VERIFIED)
- `data/processed/analysis_ready.parquet`: `f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427` (VERIFIED)
- `models/production/models/random_forest_h1.joblib`: `3c04dcbd4536433c9634c2a579a3def60ef34364d08d049c18e080349f3883d3` (VERIFIED)
- `models/production/models/xgboost_h2.joblib`: `3f07ba07ad12d05a79f4f029657f6b35704e991975f040a272534196367f461a` (VERIFIED)

---

## 7. Deliverables Created & Modified

### Modified Files
1. `src/risk_engine.py` — Complete implementation of deterministic risk calculation, action tier mapping, and governance gating.
2. `tests/test_risk_engine.py` — Activated and expanded to 26 unit and integration tests.
3. `api/inference.py` — Implemented live `POST /v1/risk` endpoint connected to `RiskEngine`.
4. `tests/test_inference.py` — Added integration test for `POST /v1/risk`.

### New Files Created
1. `src/risk_scoring.py` — Batch orchestrator for risk scoring execution and artifact generation.
2. `tests/test_risk_scoring.py` — End-to-end integration tests on real production data.
3. `artifacts/risk/risk_scores_panel.parquet` — 450-row multi-origin risk dataset.
4. `artifacts/risk/risk_scores_latest.parquet` — 50-row risk dataset for latest origin (`2025-09-16`).
5. `artifacts/risk/risk_latest.json` — Machine-readable risk payload for API/front-end consumption.
6. `reports/risk/risk_report_panel.csv` — Operational panel risk report across all evaluation origins.
7. `reports/risk/risk_report_latest.csv` — Operational risk report for origin `2025-09-16`.
8. `reports/risk/phase4b_risk_engine_report.md` — This authoritative Phase 4B report.

---

## 8. Exact Next Step

Phase 4B is **COMPLETE and FULLY VERIFIED**.

Per the project roadmap, the next phase is:
**Phase 5: Decision Support & Recommendation Engine (or Phase 4 Business Approval Gate sign-off).**
No further code changes are required for Phase 4B.
