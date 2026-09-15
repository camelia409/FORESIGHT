# Project FORESIGHT — Phase 1B Preprocessing & Data Integration Report

**Execution Timestamp (UTC):** `2026-09-15T09:04:01.591209+00:00`  
**Phase:** `1B — Preprocessing + Data Integration`  
**Status:** `COMPLETED — 16/16 INVARIANTS PASS`

---

## 1. Executive Summary
Phase 1B implements the production data normalisation, orphan inventory quarantine, safe multi-dataset integration, and invariant quality enforcement. The resulting analysis-ready dataset preserves the complete daily demand universe (**50 SKUs × 731 dates = 36,550 rows**) without row duplication, synthetic sales generation, loss of zero observations, or raw data modification.

No machine learning models have been trained, no EDA was conducted, and no lag or rolling features were generated.

---

## 2. Input & Output Dataset Accountability

| Dataset | Type | File / Path | Rows | Grain | Notes |
|---|---|---|---|---|---|
| `sales_daily.csv` | Raw Source | `data/raw/sales_daily.csv` | 36,550 | SKU + Date | 50 SKUs × 731 days complete panel |
| `sku_master.csv` | Raw Source | `data/raw/sku_master.csv` | 50 | SKU | SKU001–SKU050 catalog master |
| `calendar.csv` | Raw Source | `data/raw/calendar.csv` | 731 | Date | 2024-01-01 to 2025-12-31 daily |
| `inventory_snapshots.csv` | Raw Source | `data/raw/inventory_snapshots.csv` | 4,800 | SKU + Date | 200 SKUs × 24 monthly snapshots |
| `integrated_master.parquet` | Interim | `data/interim/integrated_master.parquet` | 36,550 | SKU + Date | Integrated intermediate staging |
| `inventory_quarantine.parquet` | Interim | `data/interim/inventory_quarantine.parquet` | 3,600 | SKU + Date | 150 orphan SKUs (SKU051–SKU200) quarantined |
| `analysis_ready.parquet` | Processed | `data/processed/analysis_ready.parquet` | 36,550 | SKU + Date | Final verified downstream demand universe |

---

## 3. Transformations Applied

1. **Sales Normalisation (`normalize_sales`)**:
   - `Date` converted to `datetime64[ns]`.
   - `SKU` stripped string representation.
   - Preserved `Units_Sold`, `Price`, and `Revenue` exactly.
   - Enforced non-negativity and verified `Revenue == Units_Sold × Price` with 0 invariant violations.
   - Retained all legitimate zero-sales observations (no row dropping).

2. **SKU Master Normalisation (`normalize_sku_master`)**:
   - `SKU` stripped string representation.
   - Preserved `Product_Name`, `Category`, `Subcategory`, `Cost_Price`, `Selling_Price`, `Launch_Date`, and `Gross_Margin_Per_Unit`.
   - Derived `negative_margin_flag = (Cost_Price > Selling_Price)`.
   - Preserved all cost and price values without alteration.

3. **Calendar Normalisation (`normalize_calendar`)**:
   - Standardised `date` column to `Date` (`datetime64[ns]`).
   - Preserved calendar properties and holiday/promotion flags.
   - Preserved structural missingness in `holiday` (723 nulls) and `promotion_event` (656 nulls) without arbitrary imputation.

4. **Inventory Normalisation & Quarantine (`normalize_inventory`)**:
   - Standardised `Snapshot_Date` to `Date` (`datetime64[ns]`).
   - Enforced numeric types on stock and reorder quantities.
   - Preserved raw `Inventory_Value` without recalculation.
   - Tagged records with `inventory_master_match_flag` and `inventory_quarantine_flag`.
   - Quarantined 3,600 orphan records (SKU051–SKU200) into `data/interim/inventory_quarantine.parquet`.

---

## 4. Integration & Join Safety Audit

The primary modeling dataset is built around `sales_daily × sku_master × calendar × inventory_master_aligned` with grain `(SKU, Date)`.

- **Pre-Join Grain Checks**:
  - `sales_daily`: 36,550 rows, 0 duplicate (SKU, Date) pairs.
  - `sku_master`: 50 rows, 0 duplicate SKU keys.
  - `calendar`: 731 rows, 0 duplicate Date keys.
  - `inventory_master_aligned`: 1,200 rows, 0 duplicate (SKU, Date) pairs (24 monthly snapshots × 50 SKUs).
- **Join Semantics**:
  - `sales LEFT JOIN sku_master ON SKU`: exactly 36,550 rows (0 unmatched SKUs).
  - `... LEFT JOIN calendar ON Date`: exactly 36,550 rows (0 unmatched dates).
  - `... LEFT JOIN inventory_master_aligned ON (SKU, Date)`: exactly 36,550 rows.
  - **Duplicate / Row Multiplication**: Exactly 0 extra rows created.
  - Added indicator `has_inventory_snapshot` (`True` on 1,200 snapshot days, `False` on 35,350 days).

---

## 5. Engineering Policy Enforcement

1. **Orphan SKUs (SKU051–SKU200)**:
   - Status: Quarantined into `data/interim/inventory_quarantine.parquet` (3,600 rows).
   - Zero orphan rows enter `analysis_ready.parquet`.
   - No synthetic product master records or speculative mappings were created.
2. **Negative Margins (16 SKUs)**:
   - Status: Preserved and flagged.
   - Exactly 16 SKUs carry `negative_margin_flag = True`.
   - Raw `Cost_Price` and `Selling_Price` were left unmodified.
3. **Inventory Valuation Basis**:
   - Status: Flagged as `unresolved`.
   - Raw `Inventory_Value` is preserved in full; no speculative recalculation was performed.
4. **Missing Values**:
   - Calendar `holiday` and `promotion_event` structural NaNs are preserved.
   - Daily non-snapshot inventory values (35,350 rows) remain unobserved (null) and are flagged via `has_inventory_snapshot = False`.
   - No global `fillna(0)` was applied.

---

## 6. Automated Data Quality Invariant Verification

**Result:** `16 / 16 Checks Passed (100%)`

| Invariant # | Check Name | Status | Details |
|---|---|---|---|
| 01 | `unique_sku_date_grain` | **PASS** | Duplicate (SKU, Date) count: 0 |
| 02 | `expected_row_count_36550` | **PASS** | Total rows: 36550 (expected 36,550) |
| 03 | `unique_sku_count_50` | **PASS** | Unique SKUs: 50 (expected 50) |
| 04 | `unique_date_count_731` | **PASS** | Unique Dates: 731 (expected 731) |
| 05 | `no_missing_sku` | **PASS** | Null SKU count: 0 |
| 06 | `no_missing_date` | **PASS** | Null Date count: 0 |
| 07 | `no_missing_product_name` | **PASS** | Null Product_Name count: 0 |
| 08 | `no_missing_category` | **PASS** | Null Category count: 0 |
| 09 | `revenue_equals_units_times_price` | **PASS** | Violations: 0, max diff: 0.000000 |
| 10 | `no_negative_units_sold` | **PASS** | Negative Units_Sold count: 0 |
| 11 | `no_negative_price` | **PASS** | Negative Price count: 0 |
| 12 | `no_negative_revenue` | **PASS** | Negative Revenue count: 0 |
| 13 | `negative_margin_skus_unchanged` | **PASS** | Negative-margin SKU count in panel: 16 (expected 16) |
| 14 | `no_orphan_sku_in_master_dataset` | **PASS** | Orphan SKUs in primary panel: 0 |
| 15 | `orphan_inventory_quarantined` | **PASS** | Quarantine rows: 3600 (expected 3600), SKUs: 150 (expected 150) |
| 16 | `raw_files_unchanged` | **PASS** | Raw hashes verified in pipeline orchestration. |

---

## 7. Raw SHA-256 Integrity Verification

All 4 source CSV files remain bit-for-bit identical to the initial ingestion baseline. No raw files were modified.

---

## 8. Confirmation of Stage Boundaries

- [x] NO exploratory data analysis (EDA) was performed.
- [x] NO feature engineering (lags, rolling stats, trend/seasonality decomposition) was performed.
- [x] NO weekly demand aggregation was performed.
- [x] NO machine learning or statistical forecasting models were trained.
- [x] NO risk engine scoring was executed.
- [x] Reproducible data layer successfully delivered.
