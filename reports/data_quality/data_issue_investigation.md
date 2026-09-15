# PROJECT FORESIGHT — Forensic Data Investigation Report

> **Phase 0b · Data Issue Investigation · Strictly Read-Only Analysis**
> All four raw CSV files were loaded read-only. No records were modified,
> deleted, imputed, or corrected. No business decisions were made.
> Produced by: Senior Data Scientist / ML Engineer

---

## Executive Summary

Three unresolved data issues from the Phase-0 audit were forensically investigated.
The table below summarises the key finding for each issue.

| Issue | Finding | Severity | Blocks |
|---|---|---|---|
| **1 — 150 Orphan Inventory SKUs** | Orphan SKUs (SKU051–SKU200) form a contiguous numeric extension of the master range (SKU001–SKU050). Combined, they create a clean 1–200 sequence. Zero sales records. No price data. Pattern is consistent with an **incomplete sku_master load**, but unconfirmed. | Critical | Inventory risk scoring (3,600 rows) |
| **2 — 16 Negative-Margin SKUs** | All 16 have real sales. `sales_daily.Price` exactly matches `sku_master.Selling_Price` for all 16. Margins range from −1.6% to −82.9% of cost. No category concentration. Evidence is **ambiguous**: could be loss-leader pricing or a cost-data error. | High | Gross margin features, risk monetary calc |
| **3 — Inventory_Value Basis** | `Inventory_Value / Current_Stock` is **constant per SKU** across all months — a per-unit reference price. This constant matches **neither Cost_Price nor Selling_Price** in sku_master (mean errors: 137% vs 165%). It is an **independent valuation** from a source not present in the current data. | High | All monetary risk calculations |

---

## Part 1 — Orphan Inventory SKU Investigation

### 1.1 Dataset Context

| Dataset | Unique SKUs | Rows |
|---|---|---|
| `sales_daily` | 50 | 36,550 |
| `sku_master` | 50 | 50 |
| `inventory_snapshots` | 200 | 4,800 |
| **Orphan SKUs** (inv − master) | **150** | **3,600** |

### 1.2 Complete List of 150 Orphan SKUs

All 150 orphan SKUs follow the identical format `^SKU\d+$` as master SKUs.

```
  SKU051  SKU052  SKU053  SKU054  SKU055  SKU056  SKU057  SKU058  SKU059  SKU060
  SKU061  SKU062  SKU063  SKU064  SKU065  SKU066  SKU067  SKU068  SKU069  SKU070
  SKU071  SKU072  SKU073  SKU074  SKU075  SKU076  SKU077  SKU078  SKU079  SKU080
  SKU081  SKU082  SKU083  SKU084  SKU085  SKU086  SKU087  SKU088  SKU089  SKU090
  SKU091  SKU092  SKU093  SKU094  SKU095  SKU096  SKU097  SKU098  SKU099  SKU100
  SKU101  SKU102  SKU103  SKU104  SKU105  SKU106  SKU107  SKU108  SKU109  SKU110
  SKU111  SKU112  SKU113  SKU114  SKU115  SKU116  SKU117  SKU118  SKU119  SKU120
  SKU121  SKU122  SKU123  SKU124  SKU125  SKU126  SKU127  SKU128  SKU129  SKU130
  SKU131  SKU132  SKU133  SKU134  SKU135  SKU136  SKU137  SKU138  SKU139  SKU140
  SKU141  SKU142  SKU143  SKU144  SKU145  SKU146  SKU147  SKU148  SKU149  SKU150
  SKU151  SKU152  SKU153  SKU154  SKU155  SKU156  SKU157  SKU158  SKU159  SKU160
  SKU161  SKU162  SKU163  SKU164  SKU165  SKU166  SKU167  SKU168  SKU169  SKU170
  SKU171  SKU172  SKU173  SKU174  SKU175  SKU176  SKU177  SKU178  SKU179  SKU180
  SKU181  SKU182  SKU183  SKU184  SKU185  SKU186  SKU187  SKU188  SKU189  SKU190
  SKU191  SKU192  SKU193  SKU194  SKU195  SKU196  SKU197  SKU198  SKU199  SKU200
```

### 1.3 Inventory Records Per Orphan SKU

Each orphan SKU has **24–24 inventory records** (mean = 24.0, consistent with the 24-month snapshot period).

### 1.4 Snapshot Date Coverage

| Group | First Snapshot | Last Snapshot | Coverage |
|---|---|---|---|
| Orphan SKUs | 2024-01-01 | 2025-12-01 | Same period |
| Master SKUs | 2024-01-01 | 2025-12-01 | Same period |

> **Finding**: Orphan and master SKUs share **identical date coverage** (2024-01-01 – 2025-12-01).
> There is no indication that orphan SKUs are historical/deprecated records.

### 1.5–1.9 Statistical Profile: Orphan vs Master SKUs

| Metric | Orphan SKUs | Master SKUs | Ratio |
|---|---|---|---|
| Avg Current_Stock | 302.6 | 308.7 | 0.98× |
| Avg On_Order | 213.2 | 212.7 | 1.00× |
| Avg Lead_Time_Days | 8.4 | 8.5 | 0.99× |
| Avg Safety_Stock | 82.4 | 85.3 | 0.97× |
| Avg Reorder_Point | 209.8 | 217.7 | 0.96× |
| Avg Inventory_Value | 1,309,155.7 | 1,289,198.9 | 1.02× |
| Total Current_Stock | 1,089,440 | 370,448 | — |
| Total Inventory_Value | 4,712,960,351 | 1,547,038,694 | — |
| Rows | 3,600 | 1,200 | 3.0× |

> **Finding**: Inventory magnitudes (stock levels, on-order, lead times, safety stock) are
> **comparable in scale** to master SKUs. There is no evidence of a structurally different product type.

### 1.10–1.11 SKU Naming Pattern Analysis

```
Master SKU format  : ^SKU\d+$  (n=50)
Orphan SKU format  : ^SKU\d+$  (n=150)  — identical format
Master numeric range: 1 – 50
Orphan numeric range: 51 – 200
Combined range      : 1 – 200
Forms 1–200 sequence: True
Numeric overlap     : None — ranges are disjoint and contiguous
```

> **Critical structural finding**: Master SKUs occupy numbers 1–50; orphan SKUs occupy 51–200.
> Together they form a **perfectly contiguous, non-overlapping sequence from 1 to 200**.
> This is strong structural evidence that both sets belong to the **same SKU numbering system**.

### 1.12 Cross-Dataset Presence

| Dataset | Orphan SKUs Present? | Details |
|---|---|---|
| `sales_daily` | **No — 0 records** | Zero transactions for any orphan SKU |
| `sku_master` | No | By definition (orphan means absent) |
| `calendar` | N/A | No SKU column in calendar |
| `inventory_snapshots` | Yes — all 150 | By definition |

### 1.13–1.14 Sales Cross-Reference and Characteristic Comparison

- **No orphan SKU has any sales record** in `sales_daily` under any identifier.
- No price or cost data exists for orphan SKUs in any dataset.
- Inventory operational parameters (lead time, safety stock, reorder point) for orphan SKUs
  are in the **same numerical range** as master SKUs — no outlier group detected.

### 1.15 Mapping Classification

| Category | Count | Evidence |
|---|---|---|
| **A. Proven mapping** | 0 | No authoritative external key linking orphan ↔ master SKUs |
| **B. Possible mapping** | 150 | Numeric continuation hypothesis: orphan 51–200 extends master 1–50. Supported by contiguous numbering, identical format, comparable inventory scale. **Unconfirmed.** |
| **C. No evidence** | 150 | No sales, no price, no product description available for orphan SKUs |

### 1.16 Orphan Universe Classification

**Evidence assessed against candidate hypotheses:**

| Hypothesis | Evidence | Assessment |
|---|---|---|
| Separate SKU universe | Different numeric range; no sales overlap | Partially supported — but same format and scale undermine this |
| Historical/deprecated SKUs | Same date coverage as master SKUs | **Not supported** — active snapshot dates |
| Inventory-only SKUs | Zero sales records | Consistent — but cannot confirm without business context |
| Identifier mismatch | No alternate ID format found | Not supported — no alternate format variants exist |
| Incomplete sku_master load | Contiguous 1–200 numbering, identical format, comparable metrics | **Most strongly supported hypothesis** |

> **Conclusion**: The weight of evidence most strongly supports that `inventory_snapshots`
> covers a broader universe of 200 SKUs, of which only the first 50 (SKU001–SKU050)
> have been loaded into `sku_master`. **This is a hypothesis, not a proven fact.**
> No modification is made based on this finding.

---

## Part 2 — Negative Gross Margin Investigation

### 2.1 Full Profile of 16 Affected SKUs

Complete table saved to: `artifacts/metrics/negative_margin_sku_profile.csv`

| SKU | Category | Cost_Price | Selling_Price | Margin (£) | Margin (%cost) | Total Units | Price Agrees? |
|---|---|---|---|---|---|---|---|
| SKU002 | Home Decor | 3867.09 | 3805.69 | -61.40 | −1.6% | 7,412 | Yes ✓ |
| SKU018 | Kitchen | 5677.05 | 5575.41 | -101.64 | −1.8% | 18,450 | Yes ✓ |
| SKU038 | Kitchen | 4630.58 | 3949.10 | -681.48 | −14.7% | 5,641 | Yes ✓ |
| SKU011 | Furniture | 1718.40 | 1444.56 | -273.84 | −15.9% | 1,951 | Yes ✓ |
| SKU033 | Kitchen | 4657.74 | 3558.00 | -1099.74 | −23.6% | 13,224 | Yes ✓ |
| SKU009 | Lighting | 3121.06 | 2336.89 | -784.17 | −25.1% | 7,830 | Yes ✓ |
| SKU037 | Home Decor | 3901.00 | 2747.42 | -1153.58 | −29.6% | 18,155 | Yes ✓ |
| SKU007 | Home Decor | 7748.20 | 5114.09 | -2634.11 | −34.0% | 18,129 | Yes ✓ |
| SKU016 | Furniture | 3637.93 | 2166.82 | -1471.11 | −40.4% | 10,143 | Yes ✓ |
| SKU020 | Storage | 6700.01 | 3897.54 | -2802.47 | −41.8% | 10,138 | Yes ✓ |
| SKU023 | Kitchen | 2484.54 | 1416.74 | -1067.80 | −43.0% | 10,830 | Yes ✓ |
| SKU028 | Kitchen | 6348.66 | 3484.09 | -2864.57 | −45.1% | 4,101 | Yes ✓ |
| SKU040 | Storage | 5169.07 | 2450.56 | -2718.51 | −52.6% | 11,398 | Yes ✓ |
| SKU024 | Lighting | 5539.34 | 1768.87 | -3770.47 | −68.1% | 9,555 | Yes ✓ |
| SKU048 | Kitchen | 6863.87 | 1379.55 | -5484.32 | −79.9% | 6,175 | Yes ✓ |
| SKU010 | Storage | 3889.06 | 663.46 | -3225.60 | −82.9% | 12,461 | Yes ✓ |

### 2.2 Category Concentration Analysis

| Category | Neg-Margin SKUs | Total SKUs in Category | Fraction |
|---|---|---|---|
| Kitchen | 6 | 10 | 60% |
| Home Decor | 3 | 10 | 30% |
| Storage | 3 | 10 | 30% |
| Lighting | 2 | 10 | 20% |
| Furniture | 2 | 10 | 20% |

> **Finding**: Negative-margin SKUs are **not concentrated** in any single category.
> All categories are affected. This rules out a category-level pricing policy as the sole explanation.

### 2.3 Volume Comparison: Negative vs Normal Margin SKUs

| Group | Avg Units Sold | Min Units Sold | Max Units Sold |
|---|---|---|---|
| Negative-margin SKUs (16) | 10350 | 1951 | 18450 |
| Normal-margin SKUs (34) | 10183 | 1976 | 19067 |

> **Finding**: Negative-margin SKUs are **not systematically low-volume**.
> Sales volumes are comparable to normal-margin SKUs — they are actively traded.

### 2.4 Price Consistency: sku_master vs sales_daily

| Check | Result |
|---|---|
| `|sku_master.Selling_Price − mean(sales_daily.Price)| < 0.01` | 16/16 SKUs agree exactly |
| Implication | Selling_Price in master is the actual transaction price — not a stale/erroneous value |

> **Critical finding**: For all 16 negative-margin SKUs, the selling price in `sku_master` is
> confirmed as the actual transaction price in `sales_daily`. This means the data is **internally
> consistent** — but does not tell us whether Cost_Price is correct.

### 2.5 Margin Range Analysis

| Metric | Negative-Margin SKUs | Normal-Margin SKUs |
|---|---|---|
| Selling_Price range | 663 – 5575 | 883 – 11642 |
| Cost_Price range | 1718 – 7748 | 307 – 6703 |
| Gross margin (Sell − Cost) | -5484 to -61 | 181 to 10225 |

### 2.6 Evidence-Based Classification

Full classification saved to: `artifacts/metrics/negative_margin_classification.csv`

| SKU | Margin (% of cost) | Classification |
|---|---|---|
| SKU002 | −1.6% | POSSIBLY INTENTIONAL — small negative margin; possible loss-leader |
| SKU018 | −1.8% | POSSIBLY INTENTIONAL — small negative margin; possible loss-leader |
| SKU038 | −14.7% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU011 | −15.9% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU033 | −23.6% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU009 | −25.1% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU037 | −29.6% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU007 | −34.0% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU016 | −40.4% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU020 | −41.8% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU023 | −43.0% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU028 | −45.1% | INSUFFICIENT EVIDENCE — transactions exist but pattern unclear |
| SKU040 | −52.6% | POTENTIALLY SUSPICIOUS — cost far exceeds sell; consistent in transactions |
| SKU024 | −68.1% | POTENTIALLY SUSPICIOUS — cost far exceeds sell; consistent in transactions |
| SKU048 | −79.9% | POTENTIALLY SUSPICIOUS — cost far exceeds sell; consistent in transactions |
| SKU010 | −82.9% | POTENTIALLY SUSPICIOUS — cost far exceeds sell; consistent in transactions |

> **Note**: No SKU is classified as 'definitively an error' or 'definitively intentional'.
> The data is internally consistent but insufficient to distinguish loss-leader from data error.

---

## Part 3 — Inventory_Value Valuation Basis Investigation

### 3.1 Key Discovery: unit_iv is Constant Per SKU

The per-unit inventory value (`Inventory_Value / Current_Stock`) was computed for all master SKU rows.

```
Observation 1: unit_iv is EFFECTIVELY CONSTANT per SKU across all 24 monthly snapshots.
  (std < 1e-8 for all 50 master SKUs — pure floating-point noise, not real variation)

Observation 2: unit_iv does NOT match the SKU's own Cost_Price or Selling_Price.
  Exact matches vs own Cost_Price: 1/50
  Exact matches vs own Sell_Price: 0/50
  Mean % diff vs Cost_Price:  136.94%
  Mean % diff vs Sell_Price:  164.9%
```

### 3.2 Hypothesis Testing: Cost Basis vs Selling Basis

| Test | Cost Basis (Stock × Cost_Price) | Selling Basis (Stock × Sell_Price) |
|---|---|---|
| Exact match (diff < 0.01) | 1/50 | 0/50 |
|---|---|---|
| Mean % error | 136.94% | 164.9% |
|---|---|---|
| SKUs where this basis is closer | 28 | 22 |
|---|---|---|

> **Finding**: Neither `Cost_Price` nor `Selling_Price` in `sku_master` can explain
> `Inventory_Value`. Mean errors of 137% and 165% respectively confirm neither is the source.

### 3.3 Investigation: Cross-SKU Price Matching

19 of 50 master SKUs have a `unit_iv` that coincidentally **exactly matches** another SKU's `Cost_Price`.
However, the 'offset' between matched SKU pairs is **random** ([1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 41, 42, 44, 45, 47, 48]),
with 19 distinct shift values across 19 matches.

> **Ruling out rotation hypothesis**: If `unit_iv` were a systematic rotation of `Cost_Price`
> (e.g., each SKU uses the cost of SKU+k), all shifts would be identical. They are not.
> The 19 exact matches are **coincidental collisions** in the price value space.

### 3.4 Orphan SKU Inventory_Value Pattern

| Finding | Detail |
|---|---|
| unit_iv constant for orphan SKUs? | **Yes** — same pattern as master SKUs |
| Orphan unit_iv range | 388 – 7,847  (master range: 307 – 7,761  — overlapping) |
| Orphan unit_iv matches master Cost_Price exactly | 0/150 |
| Orphan unit_iv matches master Sell_Price exactly | 0/150 |

### 3.5 Conclusion: Inventory_Value Source

```
FINDING: Inventory_Value = Current_Stock × [per-unit reference price from an unknown source]

The per-unit reference price:
  - Is fixed per SKU for the entire 24-month observation window
  - Does NOT derive from sku_master.Cost_Price
  - Does NOT derive from sku_master.Selling_Price
  - Does NOT follow any systematic rotation of either price field
  - Is the same structural type for both master and orphan SKUs
  - Likely originates from: (a) a different/earlier version of the price table,
    (b) a standard cost or transfer price from the ERP/WMS system,
    (c) or an independently maintained cost book not in the current data extract

ASSESSMENT: INDEPENDENT VALUATION BASIS — cannot be determined from available data
```

---

## Part 4 — Cross-Dataset Consistency

### 4.1 Relationship Between the Three Issues

| Question | Finding |
|---|---|
| Are orphan SKUs in `sales_daily`? | **No** — 0 transactions for any orphan SKU |
| Do orphan SKUs have cost/price data anywhere? | **No** — absent from all datasets except inventory |
| Are the 16 neg-margin SKUs among the orphans? | **No** — all 16 are in `sku_master` (master SKUs) |
| Are the 3 issues causally related? | **No direct causal link** — three independent data anomalies |

### 4.2 SKU Universe Overlap Summary

| Dataset | SKU Count | In master? | In sales? | In inventory? |
|---|---|---|---|---|
| `sku_master` (50 SKUs) | 50 | ✓ 50/50 | ✓ 50/50 | ✓ 50/200 |
| Orphan inv SKUs (150 SKUs) | 150 | ✗ 0/150 | ✗ 0/150 | ✓ 150/200 |

> **Key observation**: `sku_master` and `sales_daily` are **perfectly aligned** (50 SKUs each,
> full overlap). `inventory_snapshots` covers 4× more SKUs (200) than either other dataset.

---

## Part 5 — Decision Matrix


| Issue | Evidence | What we know | What we don't know | Safe engineering options | Recommended next investigation |
|---|---|---|---|---|---|
| **Issue 1**: 150 orphan inventory SKUs | Contiguous 1–200 numbering; identical SKU format; comparable inventory metrics; zero sales; no price data | They exist only in inventory. Combined with master SKUs, they form a clean 1–200 sequence. Same date coverage as master. | Whether they are future/planned SKUs, legacy stock, a different product line, or an ETL gap. No authoritative mapping. | **A**: Extend master *only* if authoritative source provides data.<br>**B**: Quarantine orphan rows in interim; keep in raw.<br>**C**: Add `orphan_flag` column in analysis-ready dataset; exclude from master-dependent calculations. | Ask business: does a product catalogue for SKU051–SKU200 exist? Check ETL/source system filter conditions. |
| **Issue 2**: 16 SKUs with negative gross margin | All 16 have real transactions. `sales_daily.Price` = `sku_master.Selling_Price` exactly. Margins: −1.6% to −82.9% of cost. Spread across all categories. Normal transaction volumes. | Transactions are real. Selling price is consistent across tables. These products are actively sold below cost. | Whether Cost_Price is the true purchase cost or a transfer price. Whether this is deliberate or a data entry error. | **A**: Preserve as-is; add `negative_margin_flag` informational column.<br>**B**: Flag for business review without correcting.<br>**C**: Correct *only* with authoritative procurement system proof. | Cross-check Cost_Price against procurement/ERP system. Ask business: are any of these deliberate loss-leaders? |
| **Issue 3**: Inventory_Value valuation basis | `unit_iv` = constant per SKU. Neither Cost_Price nor Selling_Price explains it (errors 137%, 165%). 19 coincidental cost-price matches — not a systematic pattern. Same structure for orphan SKUs. | Inventory_Value uses a fixed per-unit reference price from an external source. It is internally self-consistent (IV = Stock × constant). | The source of the per-unit reference price (ERP standard cost? WAC? Earlier price table?). Formula cannot be determined from current data. | **A**: Preserve `Inventory_Value` as-is — do not recompute.<br>**B**: Exclude from monetary risk calculations until basis is confirmed.<br>**C**: Store alongside `unit_iv_derived = IV / Stock` as an informational column. | Request the ERP/WMS data dictionary or formula for `Inventory_Value`. Ask: what table/cost version generated these per-unit values? |

---

## Part 6 — Engineering Recommendations

### 6.1 What Can Safely Be Implemented Now

| # | Action | Rationale |
|---|---|---|
| 1 | **Implement all 50 master SKU demand forecasting** | `sales_daily`, `sku_master`, and `calendar` are clean, complete, and consistent. |
| 2 | **Implement feature engineering for 50 master SKUs** | No data quality issues affect this work. |
| 3 | **Implement rolling-origin backtesting and WAPE** | Fully implementable; no unresolved dependencies. |
| 4 | **Add `negative_margin_flag` column** | Informational only; no correction. Preserves business context. |
| 5 | **Add `orphan_flag` column to inventory interim data** | Separates orphan rows without deleting them. |
| 6 | **Implement `validation.py` orphan SKU check** | Flag as Critical; do NOT remove records. |
| 7 | **Retain `Inventory_Value` as a raw column** | Do not recompute. Use `unit_iv = IV / Stock` as derived diagnostic only. |

### 6.2 What Must Remain Unresolved

- Orphan SKU treatment: backfill vs permanent exclusion.
- Negative-margin SKU cost correction: loss-leader vs data error.
- Inventory_Value monetary basis: cannot be used for financial risk scoring until confirmed.

### 6.3 Decisions Requiring Human / Business Confirmation

| Q# | Question | Config Key | Downstream Impact |
|---|---|---|---|
| Q1 | Do SKU051–SKU200 represent real products? Provide a product master file. | `orphan_sku_treatment` | 3,600 inventory rows; risk scoring completeness |
| Q2 | Are any of the 16 negative-margin SKUs deliberate loss-leaders? Provide procurement data. | `negative_margin_treatment` | Gross margin features; pricing analysis |
| Q3 | What is the formula for `Inventory_Value`? Request ERP/WMS data dictionary. | `inventory_valuation_basis` | All monetary risk calculations (£ value at risk) |

### 6.4 Records That Must Remain Untouched

| File | Protection |
|---|---|
| `data/raw/sales_daily.csv` | Read-only. No modification ever. |
| `data/raw/sku_master.csv` | Read-only. No SKU additions, no price corrections. |
| `data/raw/calendar.csv` | Read-only. |
| `data/raw/inventory_snapshots.csv` | Read-only. All 4,800 rows including 3,600 orphan rows preserved. |

### 6.5 What Would Definitively Resolve Each Ambiguity

| Issue | Definitive Resolution |
|---|---|
| **Issue 1 — Orphan SKUs** | A product master file (or ERP extract) covering all 200 SKUs, **OR** written confirmation from data engineering that `sku_master.csv` was filtered to 50 SKUs and a full extract exists. |
| **Issue 2 — Negative Margin** | A procurement/ERP system export of actual purchase costs per SKU, **OR** a written business statement classifying specific SKUs as intentional loss-leaders. |
| **Issue 3 — Inventory_Value** | The ERP/WMS data dictionary entry for `Inventory_Value`, specifically: the price table version, cost method (WAC/FIFO/Standard Cost), and the exact computation date/basis. |

---

## Appendix — Artifact Index

| File | Contents | Rows |
|---|---|---|
| `artifacts/metrics/orphan_sku_profile.csv` | Per-orphan-SKU statistics: record count, date range, stock levels, on-order, lead time, safety stock, reorder point, inventory value | 150 |
| `artifacts/metrics/negative_margin_sku_profile.csv` | Full profile of 16 negative-margin SKUs including sales volume, revenue, and price comparison | 16 |
| `artifacts/metrics/negative_margin_classification.csv` | Evidence-based classification of each negative-margin SKU | 16 |
| `artifacts/metrics/inv_value_basis_test.csv` | Row-level test of cost vs selling basis for all master SKU inventory rows | 1,200 |
| `artifacts/metrics/inv_value_per_sku_basis.csv` | Per-SKU vote for which basis fits better (cost vs selling) | 50 |
| `artifacts/metrics/inv_value_unit_price_analysis.csv` | Cross-SKU match analysis: which SKU's Cost_Price coincidentally equals unit_iv | 50 |
| `artifacts/metrics/inv_value_unit_price_per_sku.csv` | Per-SKU constant unit_iv vs own Cost/Sell price with difference statistics | 50 |
| `artifacts/metrics/orphan_inv_value_unit_price.csv` | Constant unit_iv analysis for orphan SKUs | 150 |
| `artifacts/metrics/inv_value_conclusion.json` | Machine-readable summary of Inventory_Value investigation findings | — |

---


*Investigation complete. No raw files were modified. No business decisions were made.*
*Analysis performed on: 4 raw CSV files | 36,550 + 50 + 731 + 4,800 rows | Read-only access*