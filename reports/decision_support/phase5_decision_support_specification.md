# Project FORESIGHT — Phase 5: Decision Support & Recommendation Engine Specification

**Document ID:** `FORESIGHT-PHASE5-SPEC`  
**Phase:** `Phase 5 — Decision Support & Recommendation Engine`  
**Status:** `APPROVED SPECIFICATION FOR IMPLEMENTATION`  
**Author:** Lead ML / Inventory Intelligence Engineer  
**Date:** `2026-09-15`  
**Target Codebase:** `f:\zidio\foresight\`  

---

## 1. Executive Summary & Objective

Phase 5 builds the **Decision Support & Recommendation Engine** (`src/decision_support.py`) for Project FORESIGHT. 

The engine occupies the final operational layer in the FORESIGHT architecture:
```
Forecast Engine (Phase 3B)
        │
        ▼
Risk Engine (Phase 4B)
        │
        ▼
Risk Classification (Phase 4B)
        │
        ▼
Decision Support Engine (Phase 5)
        │
        ▼
Operational Recommendations (Phase 5)
```

**Core Principle:** Phase 5 **does not recalculate or modify** any demand forecasts or Phase 4B risk scores. It consumes the validated risk scores as read-only inputs, applies a deterministic decision ruleset, resolves conflicting operational signals, enforces strict governance boundaries, and outputs actionable, human-interpretable supply chain recommendations.

---

## 2. Input Contract

The Decision Support Engine accepts risk scoring data conforming to the Phase 4B schema:

| Column Name | Type | Required? | Source / Meaning |
|---|---|---|---|
| `origin_date` | `date` / `str` | **Yes** | Forecast origin Monday (ISO `W-MON`) |
| `SKU` | `str` | **Yes** | Master SKU identifier (`SKU001`–`SKU050`) |
| `Product_Name` | `str` | Optional | Catalog description |
| `Category` | `str` | Optional | Merchandise category |
| `Subcategory` | `str` | Optional | Merchandise subcategory |
| `inventory_data_available` | `bool` | **Yes** | True if snapshot observed $\le \text{origin\_date}$ |
| `snapshot_date` | `date` / `str` | Optional | Date of latest snapshot |
| `days_since_snapshot` | `int` | Optional | Snapshot staleness in days |
| `Current_Stock` | `float` | Optional | On-hand quantity |
| `On_Order` | `float` | Optional | Inbound purchase order quantity |
| `Lead_Time_Days` | `float` | Optional | Supplier delivery lead time |
| `Safety_Stock` | `float` | Optional | Dynamic safety buffer |
| `Reorder_Point` | `float` | Optional | Dynamic replenishment trigger |
| `avg_weekly_demand` | `float` | **Yes** | Expected weekly sales volume |
| `forecast_cum_8w` | `float` | **Yes** | 8-week cumulative forecast |
| `ip_h1` .. `ip_h8` | `float` | Optional | Projected inventory position trajectory |
| `sb_h1` .. `sb_h8` | `float` | Optional | Physical stock balance trajectory |
| `weeks_of_cover` | `float` | **Yes** | Total pipeline weeks of supply |
| `days_of_supply` | `float` | Optional | Total pipeline days of supply |
| `lead_time_demand` | `float` | Optional | Demand expected during supplier lead time |
| `is_stockout_risk` | `bool` | **Yes** | Stockout breach indicator ($\le 0$ within 8w) |
| `earliest_stockout_week` | `float` / `int` | Optional | Horizon week of earliest stockout |
| `estimated_stockout_date` | `date` / `str` | Optional | Monday date of earliest stockout |
| `is_safety_stock_breach` | `bool` | **Yes** | Safety stock breach indicator ($< SS$) |
| `earliest_ss_breach_week` | `float` / `int` | Optional | Horizon week of earliest safety stock breach |
| `is_reorder_point_breach` | `bool` | **Yes** | Reorder point breach indicator ($< RP$) |
| `earliest_rp_breach_week` | `float` / `int` | Optional | Horizon week of earliest RP breach |
| `stockout_score` | `float` | **Yes** | Normalized stockout risk in $[0.0, 1.0]$ |
| `excess_inventory_units` | `float` | **Yes** | Units exceeding 8w demand + safety stock |
| `excess_weeks_of_cover` | `float` | **Yes** | Surplus weeks of supply above safety buffer |
| `overstock_score` | `float` | **Yes** | Normalized surplus ratio in $[0.0, 1.0]$ |
| `overstock_tier` | `str` | **Yes** | `HEALTHY`, `MONITOR`, `HIGH`, `CRITICAL` |
| `action_tier` | `str` | **Yes** | `CRITICAL REORDER`, `REORDER`, `MONITOR`, `OVERSTOCK`, `HEALTHY`, `UNKNOWN` |
| `excess_inventory_value` | `None` | **Yes** | Must remain `None` / `null` |
| `valuation_basis_confirmed` | `bool` | **Yes** | Must remain `False` |
| `arrival_timing_confirmed` | `bool` | **Yes** | Must remain `False` |
| `on_order_policy` | `str` | **Yes** | Must equal `"Policy_B_LT"` |
| `overstock_threshold_weeks`| `int` | **Yes** | Must equal `8` |
| `overstock_threshold_status`| `str` | **Yes** | Must equal `"ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"` |

**Quarantine Invariant:** Any record with SKU outside `SKU001`–`SKU050` is rejected or quarantined.

---

## 3. Output Contract

Each generated recommendation row contains 16 standardized fields:

```json
{
  "origin_date": "2025-09-16",
  "sku": "SKU001",
  "product_name": "Product 001",
  "category": "Furniture",
  "subcategory": "Chair",
  "recommendation_code": "PLACE_PO",
  "recommendation_title": "Issue Replenishment Purchase Order",
  "recommended_action": "Place purchase order for 350 units to arrive by week 2.",
  "priority": "2 — HIGH",
  "priority_rank": 2,
  "rationale": "Projected inventory position (135 units) breaches Reorder Point (547 units) in week 1. Lead time is 13 days.",
  "triggering_risk": "REORDER_POINT_BREACH_LT",
  "supporting_metrics": {
    "current_stock": 226.0,
    "on_order": 257.0,
    "lead_time_days": 13.0,
    "safety_stock": 153.0,
    "reorder_point": 547.0,
    "avg_weekly_demand": 89.5,
    "weeks_of_cover": 5.4,
    "excess_weeks_of_cover": 0.0,
    "earliest_breach_week": 1,
    "projected_stockout_date": "2025-10-28"
  },
  "confidence_status": "MEDIUM_CONFIDENCE",
  "required_follow_up": "Verify supplier lead time and issue purchase order within 48 hours.",
  "governance": {
    "valuation_basis_confirmed": false,
    "arrival_timing_confirmed": false,
    "on_order_policy": "Policy_B_LT",
    "overstock_threshold_weeks": 8,
    "overstock_threshold_status": "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL",
    "monetary_valuation_blocked": true
  }
}
```

---

## 4. Recommendation Code & Action Taxonomy

| Code | Title | Target Situation | Priority | Operational Action |
|---|---|---|---|---|
| `EXPEDITE_PO` | Expedite Inbound Shipment | Stockout imminent within supplier lead time | **1 — CRITICAL** | Contact supplier immediately to expedite shipment or air-freight emergency lot. |
| `PLACE_PO` | Issue Purchase Order | RP breached within supplier lead time | **2 — HIGH** | Generate and release standard replenishment PO to prevent stockout. |
| `REVIEW_PIPELINE` | Review Replenishment Pipeline | RP breached beyond lead time (weeks 3–8) | **3 — MEDIUM** | Review open order delivery schedules; prepare PO requisition for next cycle. |
| `FREEZE_REPLENISHMENT` | Halt Inbound Replenishment | Overstock detected ($WoC_{\text{excess}} > 0$) | **4 — LOW** (or **3 — MEDIUM** if critical) | Defer new purchase orders for $\lfloor WoC_{\text{excess}} \rfloor$ weeks; evaluate promotional lift. |
| `MAINTAIN_SCHEDULE` | Maintain Normal Schedule | Healthy coverage across 8 weeks | **5 — INFORMATIONAL** | No supply chain intervention required. Monitor weekly demand signals. |
| `DATA_UNAVAILABLE` | Snapshot Data Unavailable | No inventory snapshot at origin | **0 — UNKNOWN** | Flag to inventory accounting; perform cycle count or verify ERP snapshot sync. |

---

## 5. Conflict Resolution & Priority Precedence

When multiple inventory risk indicators trigger simultaneously on the same SKU, the engine resolves them using a strict **precedence hierarchy**:

```
[Level 1: Imminent Stockout within Lead Time] ──────► EXPEDITE_PO (Priority 1)
                     │ (no)
[Level 2: Reorder Point Breach within Lead Time] ───► PLACE_PO (Priority 2)
                     │ (no)
[Level 3: Long-Horizon Reorder Breach (wks 3-8)] ───► REVIEW_PIPELINE (Priority 3)
                     │ (no)
[Level 4: Terminal Overstock (Excess WoC > 0)] ─────► FREEZE_REPLENISHMENT (Priority 3 or 4)
                     │ (no)
[Level 5: Adequate Balanced Coverage] ──────────────► MAINTAIN_SCHEDULE (Priority 5)
```

### Specific Conflict Scenarios:
1. **Simultaneous REORDER and OVERSTOCK signals:**
   - *Example:* Short-term dip below RP in week 1 (before inbound arrives), but large total supply produces $\Delta_{\text{excess}} > 0$ at week 8.
   - *Resolution:* **Shortage protection takes precedence.** The engine issues `EXPEDITE_PO` or `PLACE_PO`, but modally adjusts the guidance: *"Do not issue new large PO; expedite delivery of existing open order of X units, as pipeline supply beyond week 2 is already sufficient."*
2. **High Current Stock but Insufficient Projected Inventory:**
   - *Example:* Current stock is 400 units, but demand is 100 units/week with zero On_Order.
   - *Resolution:* Engine looks ahead through the 8-week trajectory. As soon as $IP(t, k) < RP$, the appropriate reorder code triggers regardless of whether current stock looks visually large today.
3. **Low Current Stock but Massive On_Order:**
   - *Example:* Current stock has 1 week of cover, but open PO has 10 weeks of cover arriving in 5 days.
   - *Resolution:* Since $LT \le 7$, Policy $B_{LT}$ credits the inbound shipment at $h=1$. The SKU is not flagged for reorder; instead, it is flagged as `FREEZE_REPLENISHMENT` with the instruction: *"Verify vendor delivery ETA for existing shipment; do not order additional units."*

---

## 6. Confidence & Explainability Requirements

### 6.1 Data Confidence Scoring
- **`HIGH_CONFIDENCE`:** Snapshot is fresh ($\text{days\_since\_snapshot} \le 14$ days).
- **`MEDIUM_CONFIDENCE`:** Snapshot is $15 \le \text{days\_since\_snapshot} \le 28$ days old.
- **`DEGRADED_CONFIDENCE`:** Snapshot is $> 28$ days old (approaching monthly refresh boundary).
- **`NO_DATA`:** Snapshot missing (`inventory_data_available = False`).

### 6.2 Plain-Language Explainability
Every recommendation must include a **deterministic, quantitative rationale** stating:
1. Current on-hand stock and weeks of physical cover.
2. Inbound open orders and supplier lead time.
3. The exact trigger week where inventory breaches safety stock or reorder point.
4. The exact surplus quantity and surplus weeks of cover if overstocked.

---

## 7. Governance & Regulatory Constraints

### 7.1 Monetary Metrics (BLOCKED)
Because Phase 4A Decision #1 determined `Inventory_Value` is an unresolved independent constant:
- **Rule:** Recommendations must **NEVER** quote dollar values, capital tied up, or financial savings.
- **Rule:** All order quantities and surplus quantities are stated strictly in **physical units** and **weeks of demand cover**.
- **Rule:** `monetary_valuation_blocked = True` and `valuation_basis_confirmed = False` are explicitly exported.

### 7.2 On_Order Arrival Timing (ESTIMATED)
Because Phase 4A Decision #2 proved purchase order delivery dates are unrecorded in ERP:
- **Rule:** The engine relies on $\text{Lead\_Time\_Days} \le 7h$ under Policy $B_{LT}$.
- **Rule:** Recommendations must **NEVER** state: *"Order guaranteed to arrive on September 22."*
- **Rule:** Recommendations must state: *"Based on standard lead time of X days, shipment expected around week h. Verify actual ETA with supplier."*
- **Rule:** `arrival_timing_confirmed = False` is explicitly exported.

### 7.3 Overstock Threshold ($N=8$ PENDING APPROVAL)
Because Phase 4A Decision #3 recommended $N=8$ weeks as an engineering baseline awaiting executive sign-off:
- **Rule:** Overstock recommendations must carry `overstock_threshold_status = "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"`.
- **Rule:** The rationale must note that surplus is evaluated against the 8-week production forecast horizon.

---

## 8. Idempotency & Determinism Requirements

1. **Pure Functionality:** Given identical Phase 4B risk inputs, the Decision Support Engine must produce bitwise-identical recommendation records.
2. **No Data Mutation:** The engine must never mutate the input DataFrames or objects.
3. **Deterministic Ordering:** Outputs must be sorted deterministically:
   - Primary: `priority_rank` ascending ($1 \to 5$).
   - Secondary: `earliest_stockout_week` ascending (earliest crises first).
   - Tertiary: `SKU` ascending.
