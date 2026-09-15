# FORESIGHT — On_Order Inclusion Policy Investigation

**Document type:** Phase 4A Business Decision #2 — Forensic Investigation
**Generated UTC:** 2026-09-15T05:55:00+00:00
**Scope:** Read-only. No production artifacts modified.
**Investigation method:** Empirical — field availability audit, numerical characterisation, policy consequence analysis.

---

## 1. Investigation Objective

Determine, from the actual dataset and production architecture alone, whether a technically defensible On_Order inclusion rule can be established without making an unsupported business assumption. Classify the conclusion and state whether business confirmation is required.

---

## 2. Data Sources Examined

| Source | Rows | Purpose |
|---|---|---|
| `data/raw/inventory_snapshots.csv` | 4,800 | Primary — contains On_Order, Lead_Time_Days, Current_Stock, Safety_Stock, Reorder_Point |
| `data/raw/sku_master.csv` | 50 | Reference — no inventory timing fields |
| `data/processed/analysis_ready.parquet` | 36,550 | Cross-verification |
| `configs/config.yaml` | — | Forecast horizon, policy settings |
| `src/risk_engine.py` | — | Production architecture intent |
| `reports/risk/phase4a_inventory_risk_specification.md` | — | Prior specification |
| `reports/eda/phase2a_eda_report.md` | — | EDA findings on inventory fields |
| `reports/data_quality/data_issue_investigation.md` | — | Phase 0b orphan and timing findings |

**SHA-256 hashes verified unchanged (pre- and post-investigation):**
- `inventory_snapshots.csv` → `167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd` ✅
- `sku_master.csv` → `6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9` ✅
- `analysis_ready.parquet` → `f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427` ✅

---

## 3. What On_Order Represents — Field Inventory

### 3.1 Available Columns

`inventory_snapshots.csv` columns:

```
Snapshot_Date | SKU | Current_Stock | On_Order | Lead_Time_Days
Safety_Stock  | Reorder_Point | Inventory_Value
```

**Key absence:** There is **no** arrival date, expected receipt date, order placement date, PO number, purchase order age, ETA, or delivery schedule column anywhere in the project data — not in `inventory_snapshots.csv`, not in `sku_master.csv`, not in `calendar.csv`, not in `sales_daily.csv`.

### 3.2 Semantic Interpretation of On_Order

Based on the available fields alone:

| What we know empirically | Evidence |
|---|---|
| `On_Order` is a quantity in units, not a date | dtype int/float, values 0–1706 |
| `On_Order > 0` in 99.4% of 1,200 master-SKU snapshot rows | Computed from data |
| `On_Order` changes every month for all 50 master SKUs (50/50 dynamic) | std per SKU = 147.6 units; range per SKU avg = 548.2 units |
| No arrival date column exists for On_Order anywhere in the project | Field audit confirmed |
| `Lead_Time_Days` is available and also changes monthly (Phase 4A dynamic finding) | 3–14 days range, std=3.53 |

**Engineering interpretation (inference, not demonstrated fact):** `On_Order` most likely represents the outstanding quantity of goods that have been ordered from a supplier and not yet received at the time of the snapshot. This is the standard inventory management meaning of the term "on order". However, **this meaning is not confirmed by any data dictionary or documentation in the project.**

---

## 4. Arrival Timing — Is It Known?

### 4.1 Critical Field Audit

| Field sought | In inventory_snapshots? | In sku_master? | In any file? |
|---|---|---|---|
| Order placement date | ❌ NO | ❌ NO | ❌ NO |
| Expected arrival date | ❌ NO | ❌ NO | ❌ NO |
| Purchase order age | ❌ NO | ❌ NO | ❌ NO |
| ETA / delivery schedule | ❌ NO | ❌ NO | ❌ NO |
| Lead_Time_Days | ✅ YES | ❌ NO | `inventory_snapshots.csv` only |

### 4.2 Conclusion on Arrival Timing

> **ARRIVAL TIMING IS UNKNOWN.**
>
> `Lead_Time_Days` tells us how long an order typically takes to arrive after it is placed. It does NOT tell us:
> - When the current On_Order was placed
> - How many days ago the order was placed
> - How many days remain until it arrives
>
> Without a placement date, Lead_Time_Days cannot be used to compute a specific horizon at which On_Order arrives. There is no basis to calculate: `arrival_horizon = order_placement_date + Lead_Time_Days`.

---

## 5. Numerical Evidence

### 5.1 On_Order Profile (1,200 master-SKU snapshot rows)

| Metric | Value |
|---|---|
| Min | 0 |
| Mean | 212.7 units |
| Median | 154 units |
| Max | 1,706 units |
| Std | 206.5 units |
| On_Order = 0 | 7 rows / 1,200 (0.6%) |
| On_Order > 0 | 1,193 rows / 1,200 (99.4%) |
| On_Order as % of mean Current_Stock | **68.9%** |

### 5.2 Lead_Time_Days Profile (all master rows)

| Metric | Value |
|---|---|
| Min | 3 days (0.43 weeks) |
| Mean | 8.49 days (1.21 weeks) |
| Median | 9 days |
| Max | 14 days (2.00 weeks) |
| Std | 3.53 days |
| Forecast horizon | **8 weeks = 56 days** |

**ALL observed Lead_Time_Days (3–14 days) fall within the first 2 weeks of the 8-week forecast horizon.** No SKU has a lead time exceeding 14 days. No lead time approaches or exceeds the 56-day forecast window.

### 5.3 On_Order Correlations

| Correlation | Value | Interpretation |
|---|---|---|
| On_Order vs Lead_Time_Days | 0.203 | Weak positive — longer lead times slightly correlate with larger orders |
| On_Order vs Current_Stock | 0.504 | Moderate — larger stock positions tend to have larger orders outstanding |
| On_Order vs Reorder_Point | 0.702 | Strong — On_Order tracks the reorder point closely (consistent with EOQ-style ordering) |
| On_Order vs Safety_Stock | 0.650 | Strong — On_Order tracks safety stock levels |

> **Interpretation:** The strong On_Order–Reorder_Point correlation (0.70) is consistent with a system where orders are placed when stock hits the reorder point, and the order size is calibrated to replenish to a target level. This is standard inventory planning behaviour but does not establish arrival timing.

### 5.4 On_Order vs Policy Parameters

| Comparison | Count | Percentage |
|---|---|---|
| On_Order > Safety_Stock | 954 / 1,200 | 79.5% |
| On_Order > Reorder_Point | 592 / 1,200 | 49.3% |
| On_Order > Current_Stock | 375 / 1,200 | 31.2% |

> **On_Order is a substantial, operationally significant quantity** — averaging 69% of the current stock level. Ignoring it entirely (Policy D) would systematically understate the inventory position and generate false CRITICAL alerts.

### 5.5 Impact on Stockout Assessment

The decision to include or exclude On_Order materially changes risk classification:

| Comparison | Rows affected |
|---|---|
| IP_without < Safety_Stock | 25 / 1,200 (2.1%) |
| IP_with < Safety_Stock | 2 / 1,200 (0.2%) |
| **Inclusion changes SS-breach classification** | **23 / 1,200 (1.9%)** |
| IP_without < Reorder_Point | 331 / 1,200 (27.6%) |
| IP_with < Reorder_Point | 62 / 1,200 (5.2%) |
| **Inclusion changes RP-breach classification** | **269 / 1,200 (22.4%)** |

> **Critical operational finding:** Whether On_Order is included changes the Reorder_Point breach classification for **22.4% of all snapshot rows**. This is a substantial business impact. Excluding On_Order (Policy D) would trigger false REORDER alerts on over one-quarter of snapshot rows.

---

## 6. Candidate Policy Comparison

### Policy A — Always Include On_Order

```
IP(t, h) = Current_Stock + On_Order    [for all h = 1..8]
```

| Dimension | Assessment |
|---|---|
| **Leakage risk** | None — both Current_Stock and On_Order are known at origin t (snapshot date) |
| **Inventory position interpretation** | Classical "Inventory Position" = on-hand + on-order. This is standard supply chain terminology and matches industry practice for IP calculations |
| **Stockout risk interpretation** | Optimistic — assumes On_Order arrives before any stockout can occur, which may not be true if there is a delivery delay |
| **Lead-time-demand interpretation** | On_Order is expected within 0.43–2.0 weeks; it should arrive before most forecast-horizon demand is consumed |
| **Conservativeness** | **OPTIMISTIC** — overstates IP if On_Order is delayed or if only part of On_Order has arrived at evaluation horizon h |
| **Operational realism** | High for fast suppliers (LT ≤ 7d). Reasonable for moderate suppliers (LT ≤ 14d, h ≥ 2) |
| **Can be implemented** | ✅ YES — all required fields available |
| **Production horizon impact** | IP appears inflated at h=1 for SKUs with LT > 7 days. By h=2 (14 days), all observed LTs are satisfied, so On_Order would be in-hand |

---

### Policy B1 — Include On_Order only when LT ≤ 7 days (within h=1)

```
IP(t, h=1) = Current_Stock + On_Order × I[Lead_Time_Days ≤ 7]
IP(t, h>1) = Current_Stock + On_Order    [once within horizon]
```

| Dimension | Assessment |
|---|---|
| **Leakage risk** | None |
| **Inventory position interpretation** | More cautious — for the first week, only fast-supplier orders are counted. This approximates the actual physical availability at h=1 |
| **Stockout risk interpretation** | Conservative for h=1 (good), but the threshold is arbitrary — why 7 days and not 10? |
| **Lead-time-demand interpretation** | Correct in principle — orders with LT ≤ 7d will arrive within the first week. Orders with LT 8–14d will arrive by week 2 |
| **Conservativeness** | **MODERATELY CONSERVATIVE** at h=1; normal at h≥2 |
| **Operational realism** | Highest — physically models what stock will be available at each horizon |
| **Can be implemented** | ✅ YES — Lead_Time_Days available per snapshot |
| **Production horizon impact** | Only 501/1200 (41.8%) rows include On_Order at h=1. All 1200 rows include it at h=2. Behaviour is stable from h=2 onward |
| **Critical weakness** | Lead_Time_Days is a policy parameter (typical lead time), not the confirmed arrival time of the specific current On_Order. The order may have been placed long ago and could already be in transit — possibly arriving much sooner than the stated lead time |

---

### Policy B_LT — Include On_Order when LT ≤ h × 7 (Phase 4A Spec formula)

```
IP(t, h) = Current_Stock + On_Order × I[Lead_Time_Days ≤ 7h]
```

| Dimension | Assessment |
|---|---|
| **Leakage risk** | None |
| **Inventory position interpretation** | Elegantly horizon-aware — as the horizon grows, more On_Order becomes "expected within horizon" |
| **Stockout risk interpretation** | Conservative at early horizons; increasingly optimistic at later horizons |
| **Conservativeness** | **GRADUATED** — most conservative at h=1, effectively equivalent to Policy A by h=2 (since all LTs ≤ 14d < 14d = 2×7d) |
| **Operational realism** | Highest of all implementable policies — correctly reflects that an order with LT=3d is more reliably available at h=1 than one with LT=14d |
| **Can be implemented** | ✅ YES — both Lead_Time_Days and horizon index h are available |
| **Critical weakness** | Same as B1: Lead_Time_Days is a policy estimate, not the actual remaining delivery time for the specific current order |
| **Practical convergence** | Since max LT=14 days, by h=2 (14 days), ALL On_Orders satisfy `LT ≤ 14 = 2×7`. So from h=2 onward, Policy B_LT = Policy A |

---

### Policy C — Allocate On_Order at specific arrival horizon (fine-grained timing)

```
IP(t, h) = Current_Stock + On_Order × I[arrival_horizon == h]   [cumulative version also possible]
```

| Dimension | Assessment |
|---|---|
| **Can be implemented** | ❌ **NO — DATA INSUFFICIENT** |
| **Why blocked** | Requires knowing when the current order was placed. `order_placement_date + Lead_Time_Days = arrival_date`. No order placement date exists in any project file |
| **Leakage risk** | N/A — cannot be computed |
| **Note** | This would be the most operationally precise policy but requires ERP integration to retrieve purchase order placement dates |

---

### Policy D — Never include On_Order

```
IP(t, h) = Current_Stock    [for all h = 1..8]
```

| Dimension | Assessment |
|---|---|
| **Leakage risk** | None |
| **Inventory position interpretation** | Physically conservative — represents only confirmed on-hand stock |
| **Stockout risk interpretation** | Most conservative — produces highest stockout risk scores |
| **Conservativeness** | **HIGHLY CONSERVATIVE — pessimistic** |
| **Operational realism** | Low — 99.4% of snapshots have On_Order > 0, averaging 212.7 units (69% of stock). Systematically ignoring this would make the system raise CRITICAL alerts for 27.6% of snapshot rows that are not truly critical |
| **Can be implemented** | ✅ YES |
| **Suitability** | Inappropriate as the primary policy. On_Order is a material, operationally real quantity that will arrive within 1–2 weeks. Excluding it permanently would render the risk engine unreliable for forward-looking planning |

---

## 7. Horizon-by-Horizon Analysis

With `forecast_horizon_weeks = 8` and observed `Lead_Time_Days = [3, 14]`:

| Horizon h | Days since origin | LT ≤ h×7 | On_Orders included (B_LT) | Notes |
|---|---|---|---|---|
| h=1 | 0–7 days | LT ≤ 7d | 501 / 1,200 (41.8%) | Only fastest suppliers' orders |
| h=2 | 7–14 days | LT ≤ 14d | **1,200 / 1,200 (100%)** | ALL observed lead times satisfied |
| h=3 | 14–21 days | LT ≤ 21d | 1,200 / 1,200 (100%) | No change from h=2 |
| h=4..8 | 21–56 days | LT ≤ 28d..56d | 1,200 / 1,200 (100%) | No change from h=2 |

**Key insight:** Because all observed Lead_Time_Days are ≤ 14 days (≤ 2 weeks), the B_LT formula produces exactly the same result as Policy A from h=2 onward. The only horizon where inclusion differs from Policy A is h=1.

---

## 8. Critical Warnings About Lead_Time_Days

The following must be clearly stated as engineering cautions — they cannot be resolved from the data:

> **Warning 1:** `Lead_Time_Days` is a **policy parameter** representing the typical expected lead time for future orders. It is NOT a record of how many days ago the current On_Order was placed, nor how many days remain until it arrives.
>
> **Warning 2:** Using `Lead_Time_Days` as a proxy for "time until On_Order arrives" is an **engineering assumption** — it could be wrong if:
> - The order was placed 12 days ago with LT=14d (arrives in 2 days, much sooner than LT suggests)
> - There is a supply chain delay (arrives later than LT suggests)
> - On_Order reflects partially received goods (unknown portion already in stock)
>
> **Warning 3:** The fact that Lead_Time_Days changes every month (dynamic policy parameter) means the stated LT reflects a current expected value, not a commitment for the current outstanding order.

---

## 9. Recommended Production-Safe Policy

### Recommendation: **Policy B_LT — Horizon-Conditional Inclusion**

```
IP(t, h) = Current_Stock + On_Order × I[Lead_Time_Days ≤ 7 × h]
```

**Classification: SUPPORTED ENGINEERING POLICY**

This policy is not "confirmed from data" because arrival timing cannot be proven. But it is the most defensible rule that:
1. Is fully computable from available fields (no missing data)
2. Is operationally realistic (includes On_Order when it is reasonably expected to arrive within the evaluation horizon)
3. Is most conservative at the most critical horizon (h=1, stockout-imminence detection)
4. Converges to standard IP definition by h=2 for all observed SKUs
5. Is consistent with the formula documented in `phase4a_inventory_risk_specification.md`

**Conservativeness classification: MODERATELY CONSERVATIVE at h=1; NEUTRAL at h≥2**

> At h=1 (the most safety-critical horizon), On_Order is only included when LT ≤ 7 days — i.e., the system only credits On_Order at h=1 when it is almost certain to arrive within the week. This is the most defensible choice given the absence of order placement dates.

### Secondary Fallback: Policy A (for sensitivity reporting)

Policy A (always include On_Order) should be computed alongside B_LT and reported as an "optimistic scenario" IP. The difference between B_LT IP and Policy A IP is the conservative adjustment at h=1.

### Why not Policy D?

Policy D (never include On_Order) would cause 22.4% of snapshot rows to incorrectly fail the Reorder_Point breach test. With a mean On_Order of 212.7 units (69% of stock), excluding it entirely would make the risk engine systematically and severely pessimistic — producing unreliable outputs and alert fatigue.

---

## 10. Whether Business Confirmation Is Required

**SUPPORTED ENGINEERING POLICY** — but business confirmation is recommended for two sub-decisions:

| Sub-decision | Engineering default | Business confirmation needed? |
|---|---|---|
| Base formula: IP = Stock + On_Order × I[LT ≤ 7h] | ✅ Defensible from data | **Recommended** — confirm semantics of On_Order and Lead_Time_Days |
| h=1 conservatism (LT ≤ 7 days threshold) | ✅ Defensible — aligns with h=1 weekly window | **Recommended** — business may prefer to always include On_Order at all horizons |
| What to do if LT is missing/null | Default: exclude On_Order | **Yes** — operational edge case |
| What to do if On_Order=0 at snapshot | IP = Current_Stock | No business decision needed |
| Fine-grained allocation (Policy C) | Blocked — no order date | **Yes** — requires ERP order date integration |

The key question to put to the business is:
> *"When `inventory_snapshots.csv` records On_Order = N units and Lead_Time_Days = L days at the 1st-of-month snapshot, does this mean an order for N units was placed recently and is expected to arrive in approximately L days? Or could it be an aggregate outstanding quantity from multiple orders placed at different times?"*

If On_Order is a single outstanding order placed recently, Policy B_LT is highly accurate. If it aggregates multiple orders with different placement dates, a more conservative approach (or Policy A with a business disclaimer) may be preferred.

---

## 11. Impact on Phase 4B Implementation

### Formula to implement:

```python
# In src/risk_engine.py — IP calculation
def compute_IP(current_stock, on_order, lead_time_days, horizon_h):
    """
    Compute projected inventory position at horizon h.
    Includes On_Order only when Lead_Time_Days suggests arrival within the horizon window.
    
    Parameters
    ----------
    current_stock   : float — units on hand at snapshot
    on_order        : float — units outstanding from supplier
    lead_time_days  : float — expected lead time in days (from snapshot)
    horizon_h       : int   — forecast horizon index (1..8, weeks)
    
    Returns
    -------
    float — projected inventory position at horizon h
    
    NOTE: This is a production-safe approximation. Lead_Time_Days is a policy
    estimate, not a confirmed arrival date. If order placement date becomes
    available from ERP, replace with: I[arrival_date <= origin_t + h*7]
    """
    on_order_included = float(on_order) if lead_time_days <= 7 * horizon_h else 0.0
    return float(current_stock) + on_order_included
```

### Fields required (all available in inventory_snapshots.csv):

| Field | Source | Available? |
|---|---|---|
| Current_Stock | inventory_snapshots.csv | ✅ |
| On_Order | inventory_snapshots.csv | ✅ |
| Lead_Time_Days | inventory_snapshots.csv | ✅ |
| horizon_h | Risk engine parameter (1..8) | ✅ |
| Order placement date (for Policy C) | Not in project | ❌ |

### Risk output metadata to always include:

```python
{
  "on_order_policy": "B_LT",   # I[LT <= 7h]
  "on_order_included_at_h1": bool,   # True if LT <= 7
  "on_order_included_at_h2": bool,   # True if LT <= 14 (always True given observed data)
  "arrival_timing_confirmed": False,  # Always False — no order placement date
  "lead_time_source": "inventory_snapshot_policy_estimate"
}
```

---

## 12. Deliverables Created

| File | Action | Contents |
|---|---|---|
| [`reports/risk/on_order_policy_investigation.md`](file:///f:/zidio/foresight/reports/risk/on_order_policy_investigation.md) | **CREATED** | This document |
| [`reports/risk/on_order_behavior_analysis.csv`](file:///f:/zidio/foresight/reports/risk/on_order_behavior_analysis.csv) | **CREATED** | Per-SKU On_Order and Lead_Time behaviour across 24 months |

---

## 13. Validation Results

| Check | Result |
|---|---|
| SHA-256 — inventory_snapshots.csv | ✅ UNCHANGED |
| SHA-256 — sku_master.csv | ✅ UNCHANGED |
| SHA-256 — analysis_ready.parquet | ✅ UNCHANGED |
| test_data_validation.py + test_preprocessing.py + test_risk_engine.py | ✅ **60 passed, 13 skipped, 0 failed** |
| src/risk_engine.py NotImplementedError stubs | ✅ UNCHANGED — 4 stubs intact |
| Production model artifacts | ✅ NOT TOUCHED |
| Phase 4B implementation | ✅ NOT STARTED |

---

*Investigation complete. No business assumptions made. No production artifacts modified. Phase 4B has not started.*
*Conclusion: SUPPORTED ENGINEERING POLICY — business confirmation recommended for On_Order semantics.*
