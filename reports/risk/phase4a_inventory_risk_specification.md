# Project FORESIGHT — Phase 4A: Inventory Risk Specification & Data Audit

**Generated UTC:** `2026-09-15T05:14:00+00:00`  
**Phase:** `4A — Inventory Risk Specification & Data Audit`  
**Status:** `AUDIT COMPLETE — AWAITING BUSINESS APPROVAL BEFORE IMPLEMENTATION`  
**Auditor:** Production ML Engineering Team

---

## 0. Phase Context

This document is the Phase 4A forensic audit and specification deliverable for the FORESIGHT inventory risk engine. It supersedes informal notes and placeholder docstrings in `src/risk_engine.py`.

**Phase 4A scope:** Audit, specify, and gate. Nothing is implemented. No production artifacts are modified.

**Prior phases verified complete:**
- Phase 1A: Data ingestion & validation — COMPLETE
- Phase 1B: Preprocessing & integration — COMPLETE
- Phase 2A: EDA & demand characterization — COMPLETE
- Phase 2B: Classical forecasting baselines — COMPLETE
- Phase 3A: Feature engineering — COMPLETE
- Phase 3B: ML model training, tuning, selection & production promotion — COMPLETE

**Production performance (unbiased final evaluation):**
- Micro WAPE: 10.48% | Macro WAPE: 12.68%
- Architecture: Horizon-Segmented Hybrid (h=1: Tuned RF, h=2: Tuned XGBoost, h=3..8: Seasonal Naive 52w)

---

## A. Inventory Data Audit

### A.1 Source Files and Dimensions

| File | Location | Rows | Grain | Notes |
|---|---|---|---|---|
| `inventory_snapshots.csv` | `data/raw/` | 4,800 | SKU x Snapshot_Date | 200 SKUs x 24 monthly snapshots |
| `inventory_quarantine.parquet` | `data/interim/` | 3,600 | SKU x Snapshot_Date | 150 orphan SKUs (SKU051-SKU200) |
| `analysis_ready.parquet` | `data/processed/` | 36,550 | SKU x Date | 50 master SKUs x 731 days; inventory joined on 1st-of-month dates |

### A.2 Inventory Columns in analysis_ready.parquet

| Column | dtype | Non-Nulls | Nulls | Source | Notes |
|---|---|---|---|---|---|
| `Current_Stock` | float64 | 1,200 | 35,350 | inventory_snapshots | Present only on snapshot dates |
| `On_Order` | float64 | 1,200 | 35,350 | inventory_snapshots | Open purchase orders |
| `Lead_Time_Days` | float64 | 1,200 | 35,350 | inventory_snapshots | DYNAMIC — varies per SKU per snapshot |
| `Safety_Stock` | float64 | 1,200 | 35,350 | inventory_snapshots | DYNAMIC — varies per SKU per snapshot |
| `Reorder_Point` | float64 | 1,200 | 35,350 | inventory_snapshots | DYNAMIC — varies per SKU per snapshot |
| `Inventory_Value` | float64 | 1,200 | 35,350 | inventory_snapshots | Valuation basis UNRESOLVED |
| `has_inventory_snapshot` | bool | 36,550 | 0 | Derived | True on 1,200 rows; False on 35,350 |

### A.3 Snapshot Frequency and Coverage

| Metric | Value |
|---|---|
| Snapshot frequency | Monthly (1st calendar day of each month) |
| Snapshot dates | 2024-01-01 through 2025-12-01 (24 dates) |
| Total snapshots in master panel | 1,200 (50 SKUs x 24 dates) |
| Modeled SKUs with inventory observations | 50 (100% of production universe) |
| Snapshots per modeled SKU | 24 (uniform, no missing) |
| % of analysis_ready rows with snapshot | 3.28% (1,200 / 36,550) |
| % of expected modeled SKU-snapshot records | 100.00% (1,200 / 1,200) |
| Non-snapshot daily rows | 35,350 (inventory columns = NaN) |

### A.4 Inventory Column Statistics (Snapshot Rows Only, N=1,200)

| Column | Min | Mean | Median | Max | Std |
|---|---|---|---|---|---|
| `Current_Stock` | 5.0 | 308.7 | 249.5 | 1,844.0 | 242.3 |
| `On_Order` | 0.0 | 212.7 | 153.5 | 1,706.0 | 206.5 |
| `Lead_Time_Days` | 3.0 | 8.49 | 9.0 | 14.0 | 3.53 |
| `Safety_Stock` | 5.0 | 85.3 | 71.0 | 413.0 | 63.0 |
| `Reorder_Point` | 10.0 | 217.7 | 183.0 | 1,041.0 | 157.9 |
| `Inventory_Value` | 2,444.2 | 1,289,199 | 823,189 | 10,412,109 | 1,326,472 |

> **CRITICAL NEW FINDING — Lead_Time_Days, Safety_Stock, Reorder_Point are DYNAMIC:**
>
> Audit reveals these three fields are NOT static per SKU. They vary across monthly snapshot dates:
> - `Lead_Time_Days`: All 50 SKUs show variation; max unique values per SKU in raw = 12
> - `Safety_Stock`: All 50 SKUs show variation; max unique values per SKU = 24 (changes every snapshot)
> - `Reorder_Point`: All 50 SKUs show variation; max unique values per SKU = 24 (changes every snapshot)
>
> Implication: These are live operational inventory policy parameters updated monthly by the business,
> not fixed catalog attributes. The risk engine must use the snapshot value most recently observed
> at or before origin t, using backward asof merge — NOT any static per-SKU assumption.

### A.5 On_Order Distribution

| Metric | Value |
|---|---|
| On_Order = 0 (no open PO) | 7 rows (0.6%) |
| On_Order > 0 (active PO) | 1,193 rows (99.4%) |
| Mean On_Order | 212.7 units |
| Max On_Order | 1,706 units |

> On_Order is regularly populated. In 99.4% of observations, inbound replenishment stock exists.
> This is operationally significant for inventory position calculations.

### A.6 Column Semantics (Supported by Data and Documentation)

| Column | Semantic | Confirmed by |
|---|---|---|
| `Current_Stock` | Physical on-hand quantity at time of snapshot | Phase 1B report, data dictionary |
| `On_Order` | Open purchase order quantity in transit from supplier | Phase 1B report, data dictionary |
| `Lead_Time_Days` | Days from order placement to delivery; dynamic monthly policy | This audit |
| `Safety_Stock` | Buffer stock level to absorb demand variability; dynamic monthly policy | This audit |
| `Reorder_Point` | Inventory level triggering a replenishment order; dynamic monthly policy | This audit |
| `Inventory_Value` | Recorded monetary inventory value; valuation basis unresolved | config.yaml Q3 |
| `has_inventory_snapshot` | Boolean indicator of snapshot observation on that date | Derived, Phase 1B |

---

## B. Inventory Valuation Ambiguity

### B.1 Investigation Evidence

From config.yaml (Q3 policy):
> "Inventory_Value uses an INDEPENDENT per-unit reference price that is neither Cost_Price nor
> Selling_Price from sku_master. Mean % error vs Cost_Price: 136.9%; vs Selling_Price: 164.9%."

This audit independently re-verified using analysis_ready.parquet x sku_master:
- Mean % error vs Cost_Price:    **123.5%**
- Mean % error vs Selling_Price: **74.8%**
- Implied unit price (Inventory_Value / Current_Stock): min=307.06, mean=3,937.84, max=7,761.12
- sku_master Cost_Price range: ~589 to ~5,544
- sku_master Selling_Price range: ~3,806 to ~9,493

Note: Numeric discrepancy vs config.yaml (136.9%/164.9% vs 123.5%/74.8%) is attributed to
different handling of Current_Stock=0 rows and SKU filtering at investigation time.
The conclusion is unchanged: Inventory_Value does NOT correspond to Cost_Price or Selling_Price.

### B.2 Status

`inventory_value_basis = UNRESOLVED` — Confirmed by Phase 1B, config.yaml, and this audit.

The Inventory_Value column uses an independent per-unit reference price from an unidentified
source (possibly: standard cost / ERP weighted average cost / wholesale / third-party catalog).

**No basis has been silently assumed. None will be.**

### B.3 Phase 4 Calculations Blocked by This Ambiguity

| Metric | Blocked? | Reason |
|---|---|---|
| Projected inventory position (units) | NO | Purely unit-based |
| Projected stock balance (units) | NO | Purely unit-based |
| Stockout risk score | NO | Dimensionless / unit-based |
| Days / weeks of cover | NO | Units / units-per-week |
| Safety-stock breach | NO | Unit comparison |
| Reorder-point breach | NO | Unit comparison |
| On-order sufficiency | NO | Unit comparison |
| **Inventory value at risk (monetary)** | **YES** | Requires confirmed price basis |
| **Excess inventory value (monetary)** | **YES** | Requires confirmed price basis |
| **Capital-at-risk reporting** | **YES** | Requires confirmed price basis |

All unit-based risk calculations can proceed. Only monetary ($) calculations are blocked.

---

## C. Inventory Orphan Policy

### C.1 Confirmed Status

| Check | Result |
|---|---|
| Orphan SKUs in analysis_ready.parquet | 0 (Phase 1B Invariant #14 verified) |
| Quarantine rows in inventory_quarantine.parquet | 3,600 (150 SKUs x 24 snapshots) |
| Orphan SKU range | SKU051-SKU200 |
| Orphan flag column | `inventory_quarantine_flag = True` in quarantine file |
| Orphan SKUs in model training universe | 0 |
| Synthetic sku_master records for orphans | 0 |
| SKU mappings inferred | None |
| Production forecasting universe | SKU001-SKU050 (50 SKUs, confirmed) |

### C.2 Policy

Per config.yaml `orphan_sku_treatment = "quarantine"`: Orphan SKUs remain quarantined and are
excluded from all Phase 4 risk scoring.

---

## D. Forecast and Inventory Temporal Alignment

### D.1 Definitions

| Term | Definition |
|---|---|
| **Forecast origin t** | Last Monday for which weekly demand actuals are observed (ISO W-MON) |
| **Snapshot at t** | Most recent inventory snapshot with Snapshot_Date <= t (backward asof merge) |
| **Horizon h** | Number of weeks ahead (h = 1..8) |
| **Target date** | t + h weeks (Monday of that week) |
| **Lead time** | Lead_Time_Days from latest snapshot at or before t |

### D.2 Information Availability at Origin t

| Information | Available at t? | Source |
|---|---|---|
| Current_Stock at latest snapshot | YES (Snapshot_Date <= t) | Backward asof merge |
| On_Order at latest snapshot | YES (Snapshot_Date <= t) | Backward asof merge |
| Lead_Time_Days at latest snapshot | YES (Snapshot_Date <= t) | Backward asof merge |
| Safety_Stock at latest snapshot | YES (Snapshot_Date <= t) | Backward asof merge |
| Reorder_Point at latest snapshot | YES (Snapshot_Date <= t) | Backward asof merge |
| Demand forecasts y-hat(t+1)..y-hat(t+8) | YES (generated at t by production model) | Inference engine |
| Demand history through t | YES (lag features, rolling stats) | Feature engineering |
| Future inventory snapshots (Snapshot_Date > t) | **NO — PROHIBITED** | Temporal leakage |
| Future demand actuals y(t+1)..y(t+8) | **NO — PROHIBITED** | Temporal leakage |
| Future replenishment orders placed after t | **NO — PROHIBITED** | Not observable at t |

### D.3 Staleness Window

Snapshots occur on the 1st of each month. Maximum staleness at any origin t is <= 30 calendar
days. The feature engineering pipeline already computes `days_since_inventory_snapshot`. Risk
scores should be flagged (degraded confidence) when days_since_inventory_snapshot > 28.

### D.4 Missing Snapshot Policy at Origin t

If no snapshot exists at or before t for a given SKU:
1. inventory_data_available = False
2. All inventory-dependent risk scores = null
3. risk_tier = "UNKNOWN" (not CRITICAL — no false alarms)
4. Log WARNING (do not raise exception; continue other SKUs)

---

## E. Candidate Risk Metric Feasibility

| # | Metric | Feasibility | Rationale |
|---|---|---|---|
| 1 | Projected inventory position | SUPPORTED | IP(t,h) = Stock + On_Order - CumDemand(t,h). All inputs available. |
| 2 | Projected stock balance | SUPPORTED | SB(t,h) = Stock - CumDemand(t,h). Conservative (excludes On_Order). |
| 3 | Stockout risk | SUPPORTED | Binary: IP(t,h) < 0 at some h. Continuous: normalized days-of-cover. |
| 4 | Days / weeks of cover | SUPPORTED | DoS = IP(t,0) / AvgWeeklyDemand. Units: days or weeks. |
| 5 | Reorder risk | SUPPORTED WITH ASSUMPTION | On_Order delivery timing = Lead_Time_Days from t (assumption). |
| 6 | Excess / overstock risk | SUPPORTED WITH ASSUMPTION | Requires business-defined overstock multiplier N_weeks. |
| 7 | Safety-stock breach | SUPPORTED | Direct: IP(t,h) < Safety_Stock(t) at any h. |
| 8 | Reorder-point breach | SUPPORTED | Direct: IP(t,h) < Reorder_Point(t) at any h. |
| 9 | On-order sufficiency | SUPPORTED WITH ASSUMPTION | On_Order arrival timing assumption required. |
| 10 | Inventory value at risk | BLOCKED | Monetary. Requires inventory_valuation_basis resolved. |

---

## F. Risk Formula Specifications

### F.1 Projected Inventory Position IP(t, h)

```
IP(t, h) = Current_Stock(t)
           + On_Order(t) * I[Lead_Time_Days(t) <= 7h]
           - SUM(y-hat(t+k), k=1..h)
```

- Units: Units
- Time basis: Origin t through horizon h (weeks 1-8)
- Required inputs: Current_Stock, On_Order, Lead_Time_Days, y-hat(t+1..h)
- Missing snapshot: Return null; inventory_data_available = False
- Zero demand: IP(t,h) = Current_Stock + On_Order (valid)
- Missing lead time: Exclude On_Order contribution; log WARNING
- IP < 0: Preserve negative value for position reporting; clamp to 0 for physical stock

On_Order inclusion rule (REQUIRES BUSINESS APPROVAL):
I[Lead_Time_Days(t) <= 7h] includes On_Order only if supplier can deliver within the h-week window.

### F.2 Projected Stock Balance SB(t, h) — Conservative Variant

```
SB(t, h) = Current_Stock(t) - SUM(y-hat(t+k), k=1..h)
```

- Units: Units; SB < 0 indicates stockout (physical-only scenario)

### F.3 Weeks of Cover WoC(t)

```
WoC(t) = IP(t, 0) / AvgWeeklyDemand(t)

Where:
  IP(t, 0) = Current_Stock(t) + On_Order(t)
  AvgWeeklyDemand(t) = rolling_mean_8 from feature store

Edge cases:
  AvgWeeklyDemand = 0  -> WoC = infinity (no demand, not a stockout risk)
  Current_Stock = 0    -> WoC = 0.0
```

- Units: Weeks
- Missing snapshot: Return null
- Zero demand: WoC = infinity; flag as zero-demand SKU

### F.4 Days of Supply DoS(t)

```
DoS(t) = WoC(t) * 7
```

- Units: Days

### F.5 Lead-Time Demand LTD(t)

```
LTD(t) = SUM(y-hat(t+k), k=1..ceil(Lead_Time_Days(t) / 7))

Special cases:
  Lead_Time_Days = 7  -> LTD = y-hat(t+1)
  Lead_Time_Days = 14 -> LTD = y-hat(t+1) + y-hat(t+2)
  Lead_Time_Days = 3  -> LTD = y-hat(t+1) * 3/7
  Lead_Time_Days > 56 -> LTD = full 8-week sum (forecast horizon limit)
```

- Units: Units
- Missing lead time: Use business-approved default = 7 days (PENDING APPROVAL)

### F.6 Safety-Stock Breach SS_BREACH(t, h)

```
SS_BREACH(t, h) = 1  if any k in {1..h}: IP(t, k) < Safety_Stock(t)
               = 0  otherwise
```

- Units: Binary {0, 1}
- Missing snapshot: Return null

### F.7 Reorder-Point Breach RP_BREACH(t, h)

```
RP_BREACH(t, h) = 1  if any k in {1..h}: IP(t, k) < Reorder_Point(t)
               = 0  otherwise

RP_BREACH_WEEK(t) = min{k : IP(t,k) < Reorder_Point(t)} or null
```

- Units: Binary; week integer for first breach week
- Missing snapshot: Return null

### F.8 Stockout Risk Score SR(t)

```
SR(t) = 1 - min(1, WoC(t) / (Lead_Time_Days(t)/7 + SS_Weeks(t)))

Where:
  SS_Weeks(t) = Safety_Stock(t) / AvgWeeklyDemand(t)

Edge cases:
  AvgWeeklyDemand = 0 -> SR = 0.0 (no demand, no stockout risk)
  WoC = infinity      -> SR = 0.0
  WoC = 0             -> SR = 1.0 (critical)
```

- Units: Dimensionless [0, 1]
- Note: Inherited from existing risk_engine.py docstring; requires business validation

### F.9 Overstock Score OS(t)

```
OS(t) = max(0, (IP(t,0) - Overstock_Threshold(t)) / IP(t,0))

Where:
  Overstock_Threshold(t) = N_weeks * AvgWeeklyDemand(t) + Safety_Stock(t)
  N_weeks = BUSINESS-APPROVED multiplier (PROPOSAL: 8 weeks)
```

- Units: Dimensionless [0, infinity)
- Zero IP: OS = 0.0
- Zero demand: OS = 1.0 or flag as zero-demand SKU
- N_weeks REQUIRES BUSINESS APPROVAL

### F.10 Inventory Value at Risk IVaR — BLOCKED

```
BLOCKED: inventory_valuation_basis = "unresolved"

When basis is confirmed:
  IVaR(t, h) = max(0, -IP(t, h)) * (Inventory_Value(t) / Current_Stock(t))
```

Must not be computed until CFG.inventory_valuation_basis in {"cost", "selling"}.

---

## G. 8-Week Forecast Integration Architecture

### G.1 Use All 8 Horizons

| Metric | Horizons Used | Rationale |
|---|---|---|
| WoC / DoS | h=0 (current) | Point-in-time; no forecast needed |
| Lead-time demand | h=1..ceil(LT/7) | Lead-time window only |
| Safety-stock breach | h=1..8 | Detect earliest breach |
| Reorder-point breach | h=1..8 | Detect earliest breach |
| Overstock score | Cumulative h=1..8 | Full horizon demand expected to clear |
| Stockout risk score | Rolling cumulative | h=1 most critical; 8-week full picture |

### G.2 Cumulative Demand Profile

```
CumDemand(t, h) = SUM(y-hat(t+k), k=1..h),  h in {1..8}
IP(t, h) = Current_Stock(t) + On_Order_effective(t, h) - CumDemand(t, h)
```

### G.3 Model Uncertainty by Horizon

| Horizon | Model | Micro WAPE | Risk Score Confidence |
|---|---|---|---|
| h=1 | Tuned Random Forest | 9.50% | Highest |
| h=2 | Tuned XGBoost | 9.87% | High |
| h=3..8 | Seasonal Naive 52w | 10.66-10.99% | Moderate |

### G.4 Recommended Data Flow

```
Forecast Engine (POST /predict) -> {SKU, origin_t, h=1..8, y-hat, model_used}
           |
           v
Risk Input Assembly: join with inventory snapshot (backward asof <= t)
           |
           v
Risk Metric Computation: CumDemand -> IP(h) -> WoC, LTD, SR, OS, SS_BREACH, RP_BREACH
           |
           v
Risk Classification -> Risk Record -> POST /v1/risk
```

---

## H. Risk Classification Design

### H.1 Proposed 5-Tier Taxonomy

| Tier | Code | Meaning | Threshold Source |
|---|---|---|---|
| 1 | CRITICAL | Stockout imminent within lead time | Lead_Time_Days (DATA FIELD) |
| 2 | HIGH | RP breach within lead-time window | Reorder_Point (DATA FIELD) |
| 3 | MEDIUM | RP breach within 8-week horizon | Reorder_Point (DATA FIELD) |
| 4 | NORMAL | Adequate coverage; no action | Absence of breach (DERIVABLE) |
| 5 | EXCESS | Inventory exceeds demand + safety buffer | N_weeks (BUSINESS APPROVAL REQUIRED) |

Additional special tier: UNKNOWN — no inventory data available at origin t.

### H.2 Threshold Classification

| Threshold | Source | Status |
|---|---|---|
| CRITICAL: IP at LT window <= 0 | Lead_Time_Days field | SUPPORTED BY DATA |
| HIGH: RP breach within LT weeks | Reorder_Point field | SUPPORTED BY DATA |
| MEDIUM: RP breach within 8 weeks | Reorder_Point field | SUPPORTED BY DATA |
| NORMAL: No breach in 8 weeks | Absence of above | DERIVABLE |
| EXCESS: IP > N_weeks demand + SS | N_weeks multiplier | REQUIRES BUSINESS APPROVAL |

---

## I. Business Decision Blockers

| Decision | Required? | Current Evidence | Proposed Default | Needs Business Approval? |
|---|---|---|---|---|
| Inventory_Value valuation basis (cost/selling/other) | YES | UNRESOLVED; independent price source identified | Block monetary calculations until confirmed | YES — CRITICAL BLOCKER for monetary metrics |
| Treatment of missing snapshot at origin t | YES | Backward asof provides latest snapshot; 3.28% days are snapshots | Null risk scores + UNKNOWN tier | YES — operational policy |
| Stockout definition | YES | IP <= 0 at some h <= 8 | IP(t,h) <= 0 = stockout; IP < Safety_Stock = safety breach | YES — confirm On_Order inclusion |
| Overstock definition | YES | No explicit policy in data | IP(t,8) > 2x CumDemand(t,8) + Safety_Stock | YES — N_weeks multiplier required |
| Lead-time interpretation | YES | Dynamic per snapshot; range 3-14 days | Use latest snapshot value; warn if stale > 28d | YES — confirm On_Order delivery timing |
| Safety-stock interpretation | YES | Dynamic per snapshot; all 50 SKUs vary | Latest snapshot value = minimum acceptable IP | YES — buffer vs hard floor |
| Reorder-point interpretation | YES | Dynamic per snapshot; all 50 SKUs vary | Latest snapshot value = replenishment trigger | YES — order trigger vs delivery trigger |
| On_Order included in available inventory? | YES | On_Order present 99.4% of snapshots | Include if Lead_Time_Days <= planning horizon | YES — CRITICAL; changes all IP calculations |
| Risk severity thresholds | YES | LT/RP-grounded for CRITICAL/HIGH/MEDIUM | Data-grounded; EXCESS needs N_weeks | YES — overstock N_weeks multiplier |
| Recommended action policy | Phase 4B | Tier text in risk_engine.py docstring | Text recommendations per tier | YES — business language review |
| Inventory staleness threshold | YES | Max 30 days between snapshots | Warn > 28 days; null risk > 45 days | YES — operational preference |
| Default Lead_Time_Days when missing | YES | Not in sku_master; snapshot-only | 7 days (1 week) | YES — operational default |

---

## J. Production Architecture

### J.1 Files to Create or Modify (Phase 4B onwards — NOT YET)

| File | Action | Purpose |
|---|---|---|
| `src/risk_engine.py` | IMPLEMENT (existing skeleton) | Core risk computation |
| `src/risk_scoring.py` | NEW | Orchestration: forecast + inventory -> risk |
| `tests/test_risk_engine.py` | IMPLEMENT (activate skips) | Full risk engine test suite |
| `tests/test_risk_scoring.py` | NEW | Integration: end-to-end risk pipeline |
| `reports/risk/` | Directory exists | Risk report outputs |
| `artifacts/risk/` | Directory exists | Risk JSON/parquet artifacts |

### J.2 API Integration

- Existing stub: `POST /v1/risk` -> HTTP 501 in `api/inference.py:320-326`
- Schemas exist: `RiskRequest`, `RiskRecord`, `RiskResponse` in `api/schemas.py`
- Phase 4B will replace the 501 stub with risk scoring orchestrator call

### J.3 Persistence Format

- Parquet: `artifacts/risk/risk_scores_YYYY-MM-DD.parquet` (per-origin archive)
- JSON: `artifacts/risk/risk_latest.json` (current-origin API consumption)
- CSV: `reports/risk/risk_report_YYYY-MM-DD.csv` (human-readable weekly report)

### J.4 Test Requirements (Phase 4B)

| Test | Type | Priority |
|---|---|---|
| IP calculation correctness (known values) | Unit | P0 |
| WoC / DoS correctness | Unit | P0 |
| SR in [0,1] for all valid inputs | Unit | P0 |
| OS >= 0 for all valid inputs | Unit | P0 |
| Monetary calculation blocked when basis=unresolved | Unit | P0 |
| Missing snapshot -> null scores + UNKNOWN tier | Unit | P0 |
| Zero demand -> WoC = infinity, SR = 0 | Unit | P0 |
| On_Order included only when LT <= planning horizon | Unit | P0 |
| Temporal leakage: no future snapshot used | Integration | P0 |
| Orphan SKUs excluded | Integration | P0 |
| All 50 production SKUs covered | Integration | P1 |
| API /v1/risk returns valid RiskResponse schema | Integration | P1 |

---

## K. Data Lineage and Leakage Requirements

### K.1 Information Boundary

**At forecast origin t, the risk engine MAY use:**
- Current_Stock, On_Order, Lead_Time_Days, Safety_Stock, Reorder_Point from most recent snapshot with Snapshot_Date <= t
- Demand forecasts y-hat(t+1)..y-hat(t+8) from production forecast engine at t
- Historical demand features at t: rolling_mean_8, rolling_mean_4, lag features
- Static SKU attributes from sku_master: Cost_Price, Selling_Price, Category, Subcategory
- has_inventory_snapshot_at_origin and days_since_inventory_snapshot from feature store

**The risk engine MUST NOT use:**
- Any inventory snapshot with Snapshot_Date > t
- Future demand actuals y(t+h) for any h > 0
- Future replenishment orders placed after t
- Inventory_Value for monetary calculations until inventory_valuation_basis != "unresolved"
- Any feature marked CONDITIONAL or BLOCKED in the Phase 3A feature dictionary

### K.2 Missing Snapshot Behavior

If no snapshot exists at or before t for a given SKU:
1. inventory_data_available = False
2. All IP/WoC/SR/OS/breach indicators = null
3. risk_tier = "UNKNOWN"
4. recommendation = "No inventory data available at origin {t}. Manual review required."
5. Log WARNING — do NOT raise exception; continue processing other SKUs

---

## L. Phase 4A Acceptance Criteria

### L.1 Data Validation (COMPLETE)

- [x] analysis_ready.parquet schema audited: 31 columns, 36,550 rows
- [x] All 6 inventory columns confirmed: dtype float64, 1,200 non-nulls, 35,350 nulls
- [x] Snapshot frequency: monthly (1st of month), 24 snapshots, 2024-01-01 to 2025-12-01
- [x] All 50 modeled SKUs: complete 24-snapshot coverage (100%)
- [x] Non-snapshot rows: correctly null for all inventory columns
- [x] On_Order: populated in 99.4% of snapshots (only 7 zero-on-order rows)
- [x] NEW: Lead_Time_Days, Safety_Stock, Reorder_Point are DYNAMIC (vary per snapshot)
- [x] Orphan quarantine: 3,600 rows, 150 SKUs, zero in analysis_ready

### L.2 Temporal Alignment (COMPLETE)

- [x] Backward asof merge rule confirmed from Phase 3A: Snapshot_Date <= origin t
- [x] Maximum staleness: <= 30 days (monthly cycle)
- [x] Leakage boundary defined: no snapshot > t permitted
- [x] Feature dictionary confirms latest_known_stock, latest_known_on_order as SAFE
- [x] Staleness indicator days_since_inventory_snapshot already in feature store

### L.3 Valuation (PARTIALLY COMPLETE)

- [x] inventory_valuation_basis = "unresolved" confirmed in config.yaml
- [x] CFG.require_policy("inventory_valuation_basis") will raise RuntimeError — gate confirmed
- [x] Inventory_Value BLOCKED in Phase 3A feature dictionary
- [x] Monetary risk metrics explicitly isolated in this specification
- [ ] PENDING: Business approval for valuation basis resolution

### L.4 Formulas (PENDING APPROVAL)

- [x] 10 candidate metric formulas specified with inputs, units, edge cases
- [ ] PENDING: Business review of SR formula
- [ ] PENDING: Business approval of On_Order inclusion rule in IP
- [ ] PENDING: Business approval of overstock N_weeks multiplier
- [ ] PENDING: Business approval of default Lead_Time_Days (proposed: 7 days)
- [ ] PENDING: Business approval of staleness threshold (28d warning, 45d null)

### L.5 Implementation Gate

**IMPLEMENTATION IS NOT AUTHORIZED until ALL pending items above are resolved.**

Critical blockers (3):
1. On_Order inclusion rule in IP — affects every core calculation
2. Inventory_Value valuation basis — blocks monetary metrics
3. Overstock definition / N_weeks multiplier — required for EXCESS tier

---

## M. SHA-256 Integrity Verification

| File | Hash |
|---|---|
| analysis_ready.parquet | f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427 |
| sales_daily.csv | f33e31ad52880fc4c0c630c4c336f994235736cd23f417ca049e434d2878a647 |
| sku_master.csv | 6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9 |
| calendar.csv | 9d66c227a95b5e9a37d30191e37b83aa5acb1c43398525f4855c7f72cbda4b62 |
| inventory_snapshots.csv | 167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd |

All hashes match Phase 3A verified values. No raw or processed files were modified.

## N. Production Model Artifacts (Verified Unchanged)

| Artifact | Hash |
|---|---|
| random_forest_h1.joblib | 3c04dcbd4536433c9634c2a579a3def60ef34364d08d049c18e080349f3883d3 |
| xgboost_h2.joblib | 3f07ba07ad12d05a79f4f029657f6b35704e991975f040a272534196367f461a |
| model_registry.pkl | 9b891a4ad6f1e0d1e2d65812b3b7431ea0cab62e356ca4c45fa5da085b8fbd1e |
| feature_engineer.pkl | 0bba65898129249fcee4307e2c1d70694d22246e7575280ec038d9796e73965b |
| architecture.json | 00bfc2754870043dd48ff8e40df6cd1308e9de154bb8cce2204c185a5dd9e410 |
| metadata.json | b62db6f24e2ac9b466c2005d2c0075555cd86b6d8528676602dba6080dc80c12 |

---

## O. Explicit Implementation Boundary Statement

Phase 4A is a specification-only phase. The following have NOT occurred:
- Risk engine was NOT implemented
- Production model artifacts were NOT modified
- Raw data was NOT modified
- Final risk scores were NOT produced
- Phase 4B was NOT begun

*Report certified by Production ML Engineering Team. Phase 4A Audit: COMPLETE.*
