# PROJECT FORESIGHT — Phase 1A Validation Report
> Generated: 2026-09-13T08:36:38.108825 UTC
> Policy: orphan_sku_treatment=`quarantine`  negative_margin_treatment=`preserve_and_flag`  inventory_valuation_basis=`unresolved`

## Summary

| Status  | Count |
|---------|-------|
| PASS    | 44 |
| WARNING | 4 |
| FAIL    | 0 |
| **Total** | **48** |

> [!NOTE]
> No FAIL checks. All structural integrity checks passed.

## WARN Checks (4)

| Dataset | Check | Severity | Affected Rows | Message |
|---------|-------|----------|---------------|---------|
| sku_master | negative_margin_flag | WARNING | 16 | 16 SKU(s) have Cost_Price > Selling_Price. Policy: preserve_and_flag. These SKUs remain eligible for forecasting. Busine… |
| inventory_snapshots | orphan_sku_check | WARNING | 3600 | 150 inventory SKU(s) have no corresponding sku_master record. Affected rows: 3600. Policy: orphan_sku_treatment='quarant… |
| inventory_snapshots | inventory_value_basis | WARNING | 0 | Inventory_Value valuation basis is UNRESOLVED. Per Phase 0b investigation: unit_iv does not match Cost_Price or Selling_… |
| cross_dataset | inventory_sku_in_master | WARNING | 3600 | inventory_snapshots contains 150 SKU(s) absent from sku_master (3600 rows). Policy: 'quarantine'. These rows are preserv… |

## PASS Checks (44)

| Dataset | Check | Severity | Affected Rows | Message |
|---------|-------|----------|---------------|---------|
| sales_daily | required_columns | INFO | 0 | All 6 required columns present. |
| sales_daily | no_nat_Date | INFO | 0 | Column 'Date': 0 NaT values. |
| sales_daily | no_nulls_SKU | INFO | 0 | Column 'SKU': 0 nulls. |
| sales_daily | sku_format | INFO | 0 | All 50 unique SKUs match ^SKU[0-9]+$. |
| sales_daily | pk_uniqueness | INFO | 0 | Primary key (['Date', 'SKU']) is unique across all 36550 rows. |
| sales_daily | non_negative_Units_Sold | INFO | 0 | Column 'Units_Sold': all values >= 0. |
| sales_daily | positive_Price | INFO | 0 | Column 'Price': all values > 0. |
| sales_daily | non_negative_Revenue | INFO | 0 | Column 'Revenue': all values >= 0. |
| sales_daily | allowed_values_Promotion | INFO | 0 | Column 'Promotion': all values in {0, 1}. |
| sales_daily | revenue_consistency | INFO | 0 | Revenue = Units_Sold × Price for all 36550 rows (tol=0.01). |
| sku_master | required_columns | INFO | 0 | All 8 required columns present. |
| sku_master | no_nulls_SKU | INFO | 0 | Column 'SKU': 0 nulls. |
| sku_master | pk_uniqueness | INFO | 0 | Primary key (['SKU']) is unique across all 50 rows. |
| sku_master | sku_format | INFO | 0 | All 50 unique SKUs match ^SKU[0-9]+$. |
| sku_master | positive_Cost_Price | INFO | 0 | Column 'Cost_Price': all values > 0. |
| sku_master | positive_Selling_Price | INFO | 0 | Column 'Selling_Price': all values > 0. |
| sku_master | gross_margin_consistency | INFO | 0 | Gross_Margin_Per_Unit is consistent for all SKUs. |
| sku_master | no_nat_Launch_Date | INFO | 0 | Column 'Launch_Date': 0 NaT values. |
| calendar | required_columns | INFO | 0 | All 11 required columns present. |
| calendar | no_nat_date | INFO | 0 | Column 'date': 0 NaT values. |
| calendar | pk_uniqueness | INFO | 0 | Primary key (['date']) is unique across all 731 rows. |
| calendar | year_range | INFO | 0 | 'year': all values in {2024, 2025}. |
| calendar | month_range | INFO | 0 | 'month': all values in [1, 12]. |
| calendar | week_range | INFO | 0 | 'week': all values in [1, 53]. |
| calendar | allowed_values_is_weekend | INFO | 0 | Column 'is_weekend': all values in {0, 1}. |
| calendar | allowed_values_is_holiday | INFO | 0 | Column 'is_holiday': all values in {0, 1}. |
| calendar | holiday_nan_structural | INFO | 723 | 723 row(s) with holiday=NaN and is_holiday=0. Per audit finding: NaN = structural absence of event. Not an error. |
| calendar | holiday_nan_with_flag | INFO | 0 | No rows with is_holiday=1 and holiday=NaN. |
| calendar | promotion_event_nan | INFO | 656 | 656 row(s) with promotion_event=NaN. Per audit finding: NaN = no promotion on that date. Not an error. |
| inventory_snapshots | required_columns | INFO | 0 | All 8 required columns present. |
| inventory_snapshots | no_nat_Snapshot_Date | INFO | 0 | Column 'Snapshot_Date': 0 NaT values. |
| inventory_snapshots | no_nulls_SKU | INFO | 0 | Column 'SKU': 0 nulls. |
| inventory_snapshots | sku_format | INFO | 0 | All 200 unique SKUs match ^SKU[0-9]+$. |
| inventory_snapshots | pk_uniqueness | INFO | 0 | Primary key (['Snapshot_Date', 'SKU']) is unique across all 4800 rows. |
| inventory_snapshots | non_negative_Current_Stock | INFO | 0 | Column 'Current_Stock': all values >= 0. |
| inventory_snapshots | non_negative_On_Order | INFO | 0 | Column 'On_Order': all values >= 0. |
| inventory_snapshots | non_negative_Safety_Stock | INFO | 0 | Column 'Safety_Stock': all values >= 0. |
| inventory_snapshots | non_negative_Reorder_Point | INFO | 0 | Column 'Reorder_Point': all values >= 0. |
| inventory_snapshots | positive_Lead_Time_Days | INFO | 0 | Column 'Lead_Time_Days': all values > 0. |
| inventory_snapshots | non_negative_Inventory_Value | INFO | 0 | Column 'Inventory_Value': all values >= 0. |
| cross_dataset | sales_sku_in_master | INFO | 0 | All 50 sales SKUs present in sku_master. |
| cross_dataset | sales_date_in_calendar | INFO | 0 | All 731 sales dates present in calendar. |
| cross_dataset | inventory_date_in_calendar | INFO | 0 | All 24 inventory snapshot dates present in calendar. |
| sales_daily | panel_completeness | INFO | 0 | Panel is complete: 50 SKUs × 731 dates = 36,550 rows. |

---
*Phase 1A complete. No raw files were modified. No preprocessing performed.*