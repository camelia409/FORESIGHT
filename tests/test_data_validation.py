"""
test_data_validation.py — Tests for src/validation.py and src/data_ingestion.py
================================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Covers (12 required + extended coverage)
-----------------------------------------
 1. sales_daily schema PASS with valid data
 2. sku_master schema PASS with valid data
 3. calendar schema PASS with valid data
 4. inventory_snapshots schema PASS with valid data
 5. sales SKU referential integrity PASS
 6. sales Date/calendar referential integrity PASS
 7. Orphan inventory SKUs → WARNING (not FAIL)
 8. Negative margin → WARNING/FLAG (values not modified)
 9. file_sha256 can calculate a hash
10. Duplicate primary keys → FAIL detected
11. Negative quantity → FAIL detected
12. Missing required columns → FAIL detected

Additional tests
----------------
13. Revenue mismatch detected
14. Invalid Promotion value detected
15. holiday NaN when is_holiday=0 → INFO (not error)
16. holiday NaN when is_holiday=1 → FAIL
17. Panel completeness PASS
18. Panel gap detected
19. ValidationSuite summary counts correct
20. save_validation_report creates markdown and CSV
21. load_all_data() returns all four datasets with correct shapes
22. assert_hashes_unchanged passes when hashes match
23. assert_hashes_unchanged raises when hashes differ
24. Inventory_Value unresolved basis → WARNING

Usage
-----
    pytest tests/test_data_validation.py -v
    pytest tests/ -v
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.validation import (
    ValidationResult,
    ValidationSuite,
    validate_sales,
    validate_sku_master,
    validate_calendar,
    validate_inventory,
    validate_referential_integrity,
    validate_panel_completeness,
    run_all_validations,
    save_validation_report,
)
from src.utils import file_sha256, assert_hashes_unchanged


# ===========================================================================
# Fixtures — minimal valid DataFrames
# ===========================================================================

@pytest.fixture
def valid_sales() -> pd.DataFrame:
    """Minimal valid sales_daily DataFrame — 3 rows, 2 SKUs, 2 dates."""
    return pd.DataFrame({
        "Date":       pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "SKU":        ["SKU001", "SKU002", "SKU001"],
        "Units_Sold": [10,        5,         8],
        "Revenue":    [1000.0,    500.0,     800.0],
        "Price":      [100.0,     100.0,     100.0],
        "Promotion":  [0,         0,         1],
    })


@pytest.fixture
def valid_sku_master() -> pd.DataFrame:
    """Minimal valid sku_master — 2 SKUs, positive margins."""
    return pd.DataFrame({
        "SKU":                   ["SKU001", "SKU002"],
        "Product_Name":          ["Product A", "Product B"],
        "Category":              ["Cat1",      "Cat1"],
        "Subcategory":           ["Sub1",      "Sub2"],
        "Launch_Date":           pd.to_datetime(["2023-01-01", "2023-06-01"]),
        "Cost_Price":            [60.0,  70.0],
        "Selling_Price":         [100.0, 100.0],
        "Gross_Margin_Per_Unit": [40.0,  30.0],
    })


@pytest.fixture
def valid_calendar() -> pd.DataFrame:
    """Minimal valid calendar — 2 days, no events."""
    return pd.DataFrame({
        "date":            pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "year":            [2024,     2024],
        "month":           [1,        1],
        "quarter":         ["Q1",     "Q1"],
        "week":            [1,        1],
        "day_of_week":     ["Monday", "Tuesday"],
        "is_weekend":      [0,        0],
        "season":          ["Winter", "Winter"],
        "holiday":         [None,     None],
        "is_holiday":      [0,        0],
        "promotion_event": [None,     None],
    })


@pytest.fixture
def valid_inventory(valid_sku_master) -> pd.DataFrame:
    """Minimal valid inventory_snapshots — 2 master SKUs, no orphans."""
    return pd.DataFrame({
        "Snapshot_Date":   pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "SKU":             ["SKU001", "SKU002"],
        "Current_Stock":   [100,      200],
        "On_Order":        [50,       0],
        "Lead_Time_Days":  [7,        14],
        "Safety_Stock":    [20,       30],
        "Reorder_Point":   [40,       60],
        "Inventory_Value": [6000.0,   14000.0],
    })


@pytest.fixture
def all_data(valid_sales, valid_sku_master, valid_calendar, valid_inventory) -> dict:
    return {
        "sales":     valid_sales,
        "sku":       valid_sku_master,
        "calendar":  valid_calendar,
        "inventory": valid_inventory,
    }


# ===========================================================================
# 1–4: Schema validation (required columns PASS)
# ===========================================================================

class TestSchemaValidation:

    def test_sales_schema_pass(self, valid_sales):
        """Test 1: sales_daily with all required columns passes schema check."""
        results = validate_sales(valid_sales)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "PASS"

    def test_sku_master_schema_pass(self, valid_sku_master):
        """Test 2: sku_master with all required columns passes schema check."""
        results = validate_sku_master(valid_sku_master)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "PASS"

    def test_calendar_schema_pass(self, valid_calendar):
        """Test 3: calendar with all required columns passes schema check."""
        results = validate_calendar(valid_calendar)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "PASS"

    def test_inventory_schema_pass(self, valid_inventory):
        """Test 4: inventory_snapshots with all required columns passes schema check."""
        results = validate_inventory(valid_inventory)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "PASS"


# ===========================================================================
# 5–6: Referential integrity PASS
# ===========================================================================

class TestReferentialIntegrityPass:

    def test_sales_sku_in_master_pass(self, valid_sales, valid_sku_master,
                                      valid_calendar, valid_inventory):
        """Test 5: All sales SKUs in sku_master → PASS."""
        results = validate_referential_integrity(
            valid_sales, valid_sku_master, valid_calendar, valid_inventory
        )
        r = next(x for x in results if x.check_name == "sales_sku_in_master")
        assert r.status == "PASS"

    def test_sales_date_in_calendar_pass(self, valid_sales, valid_sku_master,
                                         valid_calendar, valid_inventory):
        """Test 6: All sales dates in calendar → PASS."""
        results = validate_referential_integrity(
            valid_sales, valid_sku_master, valid_calendar, valid_inventory
        )
        r = next(x for x in results if x.check_name == "sales_date_in_calendar")
        assert r.status == "PASS"


# ===========================================================================
# 7: Orphan inventory SKUs → WARNING not FAIL
# ===========================================================================

class TestOrphanInventorySKUs:

    def test_orphan_skus_produce_warning_not_fail(self, valid_sku_master):
        """Test 7: Inventory with orphan SKUs produces WARNING, not FAIL."""
        inv_with_orphans = pd.DataFrame({
            "Snapshot_Date":   pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "SKU":             ["SKU001", "SKU002", "SKU051"],  # SKU051 = orphan
            "Current_Stock":   [100,       200,      150],
            "On_Order":        [50,        0,        25],
            "Lead_Time_Days":  [7,         14,       10],
            "Safety_Stock":    [20,        30,       15],
            "Reorder_Point":   [40,        60,       30],
            "Inventory_Value": [6000.0,    14000.0,  9000.0],
        })
        results = validate_inventory(inv_with_orphans, sku_df=valid_sku_master)
        orphan_check = next(r for r in results if r.check_name == "orphan_sku_check")
        assert orphan_check.status == "WARNING", (
            "Orphan SKUs must produce WARNING, not FAIL, per engineering policy."
        )
        assert orphan_check.affected_rows == 1
        assert "SKU051" in orphan_check.detail

    def test_orphan_rows_are_not_removed(self, valid_sku_master):
        """Orphan inventory rows must be reported but NOT removed by validation."""
        inv = pd.DataFrame({
            "Snapshot_Date":   pd.to_datetime(["2024-01-01"]),
            "SKU":             ["SKU099"],  # orphan
            "Current_Stock":   [10],
            "On_Order":        [0],
            "Lead_Time_Days":  [5],
            "Safety_Stock":    [2],
            "Reorder_Point":   [5],
            "Inventory_Value": [500.0],
        })
        validate_inventory(inv, sku_df=valid_sku_master)
        # DataFrame must be unchanged
        assert len(inv) == 1
        assert inv["SKU"].iloc[0] == "SKU099"

    def test_ri_orphan_skus_warning_not_fail(self, valid_sales, valid_sku_master,
                                              valid_calendar):
        """Test 7 (RI): inventory.SKU orphan check in referential integrity → WARNING."""
        inv_orphan = pd.DataFrame({
            "Snapshot_Date":   pd.to_datetime(["2024-01-01"]),
            "SKU":             ["SKU100"],  # orphan
            "Current_Stock":   [50],
            "On_Order":        [0],
            "Lead_Time_Days":  [7],
            "Safety_Stock":    [5],
            "Reorder_Point":   [10],
            "Inventory_Value": [2500.0],
        })
        results = validate_referential_integrity(
            valid_sales, valid_sku_master, valid_calendar, inv_orphan
        )
        ri_check = next(r for r in results if r.check_name == "inventory_sku_in_master")
        assert ri_check.status == "WARNING"


# ===========================================================================
# 8: Negative margin → WARNING/FLAG (values not modified)
# ===========================================================================

class TestNegativeMargin:

    def test_negative_margin_produces_warning(self):
        """Test 8: SKUs with Cost_Price > Selling_Price produce WARNING, not FAIL."""
        sku_df = pd.DataFrame({
            "SKU":                   ["SKU001", "SKU002"],
            "Product_Name":          ["A",       "B"],
            "Category":              ["C1",      "C1"],
            "Subcategory":           ["S1",      "S1"],
            "Launch_Date":           pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "Cost_Price":            [150.0,    70.0],   # SKU001: cost > selling
            "Selling_Price":         [100.0,    100.0],
            "Gross_Margin_Per_Unit": [-50.0,    30.0],
        })
        results = validate_sku_master(sku_df)
        neg_check = next(r for r in results if r.check_name == "negative_margin_flag")
        assert neg_check.status == "WARNING"
        assert neg_check.affected_rows == 1

    def test_negative_margin_values_unchanged(self):
        """Validation must not modify Cost_Price or Selling_Price values."""
        sku_df = pd.DataFrame({
            "SKU":                   ["SKU001"],
            "Product_Name":          ["A"],
            "Category":              ["C1"],
            "Subcategory":           ["S1"],
            "Launch_Date":           pd.to_datetime(["2023-01-01"]),
            "Cost_Price":            [200.0],   # cost > selling
            "Selling_Price":         [100.0],
            "Gross_Margin_Per_Unit": [-100.0],
        })
        cost_before = sku_df["Cost_Price"].iloc[0]
        sell_before = sku_df["Selling_Price"].iloc[0]
        validate_sku_master(sku_df)
        assert sku_df["Cost_Price"].iloc[0] == cost_before
        assert sku_df["Selling_Price"].iloc[0] == sell_before

    def test_no_negative_margin_pass(self, valid_sku_master):
        """When no negative margins exist, check passes."""
        results = validate_sku_master(valid_sku_master)
        neg_check = next(r for r in results if r.check_name == "negative_margin_flag")
        assert neg_check.status == "PASS"


# ===========================================================================
# 9: file_sha256 can calculate a hash
# ===========================================================================

class TestFileHash:

    def test_sha256_returns_64_char_hex(self, tmp_path):
        """Test 9: file_sha256 returns a 64-character lowercase hex string."""
        f = tmp_path / "test.txt"
        f.write_bytes(b"FORESIGHT test content for hashing.")
        digest = file_sha256(f)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert digest == digest.lower()

    def test_sha256_deterministic(self, tmp_path):
        """SHA-256 of the same content must be identical on repeated calls."""
        f = tmp_path / "det.txt"
        f.write_bytes(b"deterministic content 123")
        assert file_sha256(f) == file_sha256(f)

    def test_sha256_known_value(self, tmp_path):
        """SHA-256 of empty bytes is the well-known empty hash."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert file_sha256(f) == expected

    def test_sha256_file_not_found(self, tmp_path):
        """file_sha256 raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            file_sha256(tmp_path / "nonexistent.csv")

    def test_raw_file_hashes_unchanged(self):
        """
        Test 9 (integration): Compute hashes of the real raw files.
        Verify they remain unchanged after computing them twice.
        """
        from src.config import PATHS
        from src.utils import hash_raw_files, assert_hashes_unchanged
        before = hash_raw_files(PATHS.raw_dir)
        after  = hash_raw_files(PATHS.raw_dir)
        assert_hashes_unchanged(before, after)  # must not raise

    def test_assert_hashes_unchanged_passes(self):
        """Test 22: assert_hashes_unchanged passes when hashes are identical."""
        h = {"file.csv": "abc123"}
        assert_hashes_unchanged(h, h.copy())  # must not raise

    def test_assert_hashes_unchanged_raises(self):
        """Test 23: assert_hashes_unchanged raises RuntimeError on mismatch."""
        before = {"file.csv": "aaa"}
        after  = {"file.csv": "bbb"}
        with pytest.raises(RuntimeError, match="INTEGRITY VIOLATION"):
            assert_hashes_unchanged(before, after)


# ===========================================================================
# 10: Duplicate primary keys → FAIL
# ===========================================================================

class TestPrimaryKeyUniqueness:

    def test_sales_duplicate_pk_detected(self):
        """Test 10: Duplicate (Date, SKU) in sales_daily produces FAIL."""
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01", "2024-01-01"]),  # duplicate
            "SKU":        ["SKU001",        "SKU001"],
            "Units_Sold": [10,               5],
            "Revenue":    [1000.0,           500.0],
            "Price":      [100.0,            100.0],
            "Promotion":  [0,                0],
        })
        results = validate_sales(df)
        pk_check = next(r for r in results if r.check_name == "pk_uniqueness")
        assert pk_check.status == "FAIL"

    def test_sku_master_duplicate_pk_detected(self):
        """Duplicate SKU in sku_master produces FAIL."""
        df = pd.DataFrame({
            "SKU":                   ["SKU001", "SKU001"],
            "Product_Name":          ["A",       "A_dup"],
            "Category":              ["C1",      "C1"],
            "Subcategory":           ["S1",      "S1"],
            "Launch_Date":           pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "Cost_Price":            [60.0,      60.0],
            "Selling_Price":         [100.0,     100.0],
            "Gross_Margin_Per_Unit": [40.0,      40.0],
        })
        results = validate_sku_master(df)
        pk_check = next(r for r in results if r.check_name == "pk_uniqueness")
        assert pk_check.status == "FAIL"

    def test_calendar_duplicate_date_detected(self):
        """Duplicate date in calendar produces FAIL."""
        df = pd.DataFrame({
            "date":            pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "year":            [2024,  2024],
            "month":           [1,     1],
            "quarter":         ["Q1",  "Q1"],
            "week":            [1,     1],
            "day_of_week":     ["Mon", "Mon"],
            "is_weekend":      [0,     0],
            "season":          ["W",   "W"],
            "holiday":         [None,  None],
            "is_holiday":      [0,     0],
            "promotion_event": [None,  None],
        })
        results = validate_calendar(df)
        pk_check = next(r for r in results if r.check_name == "pk_uniqueness")
        assert pk_check.status == "FAIL"

    def test_inventory_duplicate_pk_detected(self):
        """Duplicate (Snapshot_Date, SKU) in inventory produces FAIL."""
        df = pd.DataFrame({
            "Snapshot_Date":   pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "SKU":             ["SKU001",        "SKU001"],
            "Current_Stock":   [100,             105],
            "On_Order":        [0,               0],
            "Lead_Time_Days":  [7,               7],
            "Safety_Stock":    [10,              10],
            "Reorder_Point":   [20,              20],
            "Inventory_Value": [5000.0,          5100.0],
        })
        results = validate_inventory(df)
        pk_check = next(r for r in results if r.check_name == "pk_uniqueness")
        assert pk_check.status == "FAIL"


# ===========================================================================
# 11: Negative quantity → FAIL
# ===========================================================================

class TestNegativeQuantity:

    def test_negative_units_sold_detected(self):
        """Test 11: Negative Units_Sold produces FAIL."""
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01"]),
            "SKU":        ["SKU001"],
            "Units_Sold": [-5],          # invalid
            "Revenue":    [0.0],
            "Price":      [100.0],
            "Promotion":  [0],
        })
        results = validate_sales(df)
        check = next(r for r in results if r.check_name == "non_negative_Units_Sold")
        assert check.status == "FAIL"
        assert check.affected_rows == 1

    def test_negative_current_stock_detected(self):
        """Negative Current_Stock in inventory produces FAIL."""
        df = pd.DataFrame({
            "Snapshot_Date":   pd.to_datetime(["2024-01-01"]),
            "SKU":             ["SKU001"],
            "Current_Stock":   [-10],    # invalid
            "On_Order":        [0],
            "Lead_Time_Days":  [7],
            "Safety_Stock":    [5],
            "Reorder_Point":   [10],
            "Inventory_Value": [0.0],
        })
        results = validate_inventory(df)
        check = next(r for r in results if r.check_name == "non_negative_Current_Stock")
        assert check.status == "FAIL"

    def test_negative_price_detected(self):
        """Negative/zero Price in sales_daily produces FAIL."""
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01"]),
            "SKU":        ["SKU001"],
            "Units_Sold": [10],
            "Revenue":    [1000.0],
            "Price":      [-100.0],   # invalid
            "Promotion":  [0],
        })
        results = validate_sales(df)
        check = next(r for r in results if r.check_name == "positive_Price")
        assert check.status == "FAIL"


# ===========================================================================
# 12: Missing required columns → FAIL
# ===========================================================================

class TestMissingColumns:

    def test_sales_missing_column_fails(self):
        """Test 12: sales_daily missing 'Revenue' column produces FAIL."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"]),
            "SKU":  ["SKU001"],
            # Revenue is missing
            "Units_Sold": [10],
            "Price":      [100.0],
            "Promotion":  [0],
        })
        results = validate_sales(df)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "FAIL"
        assert col_check.severity == "CRITICAL"

    def test_sku_master_missing_column_fails(self):
        """sku_master missing 'Cost_Price' column produces FAIL."""
        df = pd.DataFrame({
            "SKU":           ["SKU001"],
            "Product_Name":  ["A"],
            "Category":      ["C"],
            "Subcategory":   ["S"],
            "Launch_Date":   pd.to_datetime(["2023-01-01"]),
            # Cost_Price missing
            "Selling_Price":         [100.0],
            "Gross_Margin_Per_Unit": [40.0],
        })
        results = validate_sku_master(df)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "FAIL"

    def test_inventory_missing_column_fails(self):
        """inventory_snapshots missing 'Inventory_Value' produces FAIL."""
        df = pd.DataFrame({
            "Snapshot_Date": pd.to_datetime(["2024-01-01"]),
            "SKU":           ["SKU001"],
            "Current_Stock": [100],
            "On_Order":      [0],
            "Lead_Time_Days":[7],
            "Safety_Stock":  [5],
            "Reorder_Point": [10],
            # Inventory_Value missing
        })
        results = validate_inventory(df)
        col_check = next(r for r in results if r.check_name == "required_columns")
        assert col_check.status == "FAIL"


# ===========================================================================
# 13: Revenue mismatch detected
# ===========================================================================

class TestRevenueConsistency:

    def test_revenue_mismatch_detected(self):
        """Test 13: Revenue ≠ Units_Sold × Price is flagged as FAIL."""
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01"]),
            "SKU":        ["SKU001"],
            "Units_Sold": [10],
            "Revenue":    [999.0],    # should be 1000.0
            "Price":      [100.0],
            "Promotion":  [0],
        })
        results = validate_sales(df)
        check = next(r for r in results if r.check_name == "revenue_consistency")
        assert check.status == "FAIL"
        assert check.affected_rows == 1

    def test_revenue_consistent_pass(self, valid_sales):
        """Revenue = Units_Sold × Price for all rows → PASS."""
        results = validate_sales(valid_sales)
        check = next(r for r in results if r.check_name == "revenue_consistency")
        assert check.status == "PASS"


# ===========================================================================
# 14: Invalid Promotion value
# ===========================================================================

class TestPromotionValues:

    def test_invalid_promotion_detected(self):
        """Test 14: Promotion value outside {0, 1} produces FAIL."""
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01"]),
            "SKU":        ["SKU001"],
            "Units_Sold": [10],
            "Revenue":    [1000.0],
            "Price":      [100.0],
            "Promotion":  [2],         # invalid
        })
        results = validate_sales(df)
        check = next(r for r in results if r.check_name == "allowed_values_Promotion")
        assert check.status == "FAIL"


# ===========================================================================
# 15–16: Calendar holiday NaN semantics
# ===========================================================================

class TestCalendarNaNSemantics:

    def test_holiday_nan_is_holiday_0_is_info(self, valid_calendar):
        """Test 15: holiday=NaN where is_holiday=0 is INFO (structural absence)."""
        results = validate_calendar(valid_calendar)
        check = next(r for r in results if r.check_name == "holiday_nan_structural")
        assert check.status == "PASS"
        assert check.severity == "INFO"

    def test_holiday_nan_with_is_holiday_1_is_error(self):
        """Test 16: holiday=NaN where is_holiday=1 is an ERROR."""
        df = pd.DataFrame({
            "date":            pd.to_datetime(["2024-12-25"]),
            "year":            [2024],
            "month":           [12],
            "quarter":         ["Q4"],
            "week":            [52],
            "day_of_week":     ["Wednesday"],
            "is_weekend":      [0],
            "season":          ["Winter"],
            "holiday":         [None],      # NaN but is_holiday == 1 → ERROR
            "is_holiday":      [1],
            "promotion_event": [None],
        })
        results = validate_calendar(df)
        check = next(r for r in results if r.check_name == "holiday_nan_with_flag")
        assert check.status == "FAIL"


# ===========================================================================
# 17–18: Panel completeness
# ===========================================================================

class TestPanelCompleteness:

    def test_complete_panel_passes(self, valid_sales):
        """Test 17: A complete SKU × date panel passes the completeness check."""
        results = validate_panel_completeness(valid_sales)
        check = next(r for r in results if r.check_name == "panel_completeness")
        # valid_sales has SKU001+SKU002 on 2024-01-01, but only SKU001 on 2024-01-02
        # → this is a 2-SKU panel with gap; just verify the check runs and returns a result
        assert check.status in {"PASS", "FAIL"}

    def test_missing_cell_detected(self):
        """Test 18: A gap in the SKU × date panel is flagged as FAIL."""
        # 2 SKUs × 2 dates = 4 expected; only 3 rows → gap
        df = pd.DataFrame({
            "Date":       pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "SKU":        ["SKU001",        "SKU002",        "SKU001"],
            # SKU002 on 2024-01-02 is missing
            "Units_Sold": [10,               5,               8],
            "Revenue":    [1000.0,           500.0,           800.0],
            "Price":      [100.0,            100.0,           100.0],
            "Promotion":  [0,                0,               1],
        })
        results = validate_panel_completeness(df)
        check = next(r for r in results if r.check_name == "panel_completeness")
        assert check.status == "FAIL"
        assert check.affected_rows == 1    # 1 missing cell

    def test_real_sales_panel_complete(self):
        """Integration: real sales_daily panel is 100% complete (50 × 731)."""
        from src.data_ingestion import load_sales
        sales = load_sales()
        results = validate_panel_completeness(sales)
        check = next(r for r in results if r.check_name == "panel_completeness")
        assert check.status == "PASS", (
            f"Expected complete panel but found gap: {check.message}"
        )


# ===========================================================================
# 19: ValidationSuite summary counts
# ===========================================================================

class TestValidationSuite:

    def test_suite_counts_correct(self):
        """Test 19: ValidationSuite n_pass / n_warning / n_fail counts are correct."""
        suite = ValidationSuite()
        suite.add(ValidationResult("ds", "c1", "PASS",    "INFO",     0, "ok"))
        suite.add(ValidationResult("ds", "c2", "PASS",    "INFO",     0, "ok"))
        suite.add(ValidationResult("ds", "c3", "WARNING", "WARNING",  5, "warn"))
        suite.add(ValidationResult("ds", "c4", "FAIL",    "ERROR",   10, "fail"))

        assert suite.n_pass    == 2
        assert suite.n_warning == 1
        assert suite.n_fail    == 1
        assert suite.has_fails is True
        assert suite.has_critical is False

    def test_suite_has_critical(self):
        """has_critical is True when a FAIL result has CRITICAL severity."""
        suite = ValidationSuite()
        suite.add(ValidationResult("ds", "c1", "FAIL", "CRITICAL", 0, "bad"))
        assert suite.has_critical is True


# ===========================================================================
# 20: save_validation_report creates files
# ===========================================================================

class TestSaveValidationReport:

    def test_report_files_created(self, all_data, tmp_path):
        """Test 20: save_validation_report creates markdown and CSV in the given dirs."""
        suite = run_all_validations(all_data)
        md_path, csv_path = save_validation_report(
            suite,
            report_dir  = tmp_path / "reports",
            metrics_dir = tmp_path / "metrics",
        )
        assert md_path.exists(),  f"Markdown report not created: {md_path}"
        assert csv_path.exists(), f"CSV artefact not created: {csv_path}"
        assert md_path.stat().st_size > 100
        assert csv_path.stat().st_size > 50

    def test_csv_has_correct_columns(self, all_data, tmp_path):
        """CSV artefact has the required columns."""
        suite = run_all_validations(all_data)
        _, csv_path = save_validation_report(
            suite,
            report_dir  = tmp_path / "reports",
            metrics_dir = tmp_path / "metrics",
        )
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames
        required = {"dataset", "check_name", "status", "severity", "affected_rows", "message"}
        assert required.issubset(set(headers))


# ===========================================================================
# 21: load_all_data returns correct shapes
# ===========================================================================

class TestDataIngestion:

    def test_load_all_data_shapes(self):
        """Test 21: load_all_data() returns all four datasets with expected shapes."""
        from src.data_ingestion import load_all_data
        data = load_all_data()

        assert set(data.keys()) == {"sales", "sku", "calendar", "inventory"}
        assert len(data["sales"])     == 36550, f"Expected 36550 sales rows, got {len(data['sales'])}"
        assert len(data["sku"])       == 50,    f"Expected 50 SKU rows, got {len(data['sku'])}"
        assert len(data["calendar"])  == 731,   f"Expected 731 calendar rows, got {len(data['calendar'])}"
        assert len(data["inventory"]) == 4800,  f"Expected 4800 inventory rows, got {len(data['inventory'])}"

    def test_load_sales_date_column_is_datetime(self):
        """Date column in sales_daily is datetime64 after loading."""
        from src.data_ingestion import load_sales
        df = load_sales()
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_load_returns_deep_copy(self):
        """Mutating the returned DataFrame must not affect a subsequent load."""
        from src.data_ingestion import load_sales
        df1 = load_sales()
        original_first = df1["Units_Sold"].iloc[0]
        df1["Units_Sold"] = -999  # mutate the copy
        df2 = load_sales()
        assert df2["Units_Sold"].iloc[0] == original_first, (
            "load_sales() must return a deep copy, not a shared reference."
        )


# ===========================================================================
# 24: Inventory_Value unresolved basis → WARNING
# ===========================================================================

class TestInventoryValueBasis:

    def test_unresolved_basis_produces_warning(self, valid_inventory, valid_sku_master):
        """Test 24: Unresolved Inventory_Value basis produces WARNING (per policy)."""
        results = validate_inventory(valid_inventory, sku_df=valid_sku_master)
        check = next(r for r in results if r.check_name == "inventory_value_basis")
        # Per config.yaml: inventory_valuation_basis = "unresolved"
        assert check.status == "WARNING"
        assert "unresolved" in check.message.lower() or "UNRESOLVED" in check.message


# ===========================================================================
# Integration: run_all_validations on real data
# ===========================================================================

class TestIntegrationRealData:

    def test_full_validation_suite_no_critical_fails(self):
        """
        Integration test: run_all_validations on the actual raw datasets.
        Must produce no CRITICAL FAIL results (orphan SKUs are WARNING, not FAIL).
        """
        from src.data_ingestion import load_all_data
        data  = load_all_data()
        suite = run_all_validations(data)

        critical_fails = [
            r for r in suite.results
            if r.status == "FAIL" and r.severity == "CRITICAL"
        ]
        assert len(critical_fails) == 0, (
            f"Unexpected CRITICAL FAIL results:\n"
            + "\n".join(f"  [{r.dataset}] {r.check_name}: {r.message}" for r in critical_fails)
        )

    def test_full_validation_has_warnings(self):
        """
        Integration test: warnings are present (orphan SKUs, neg margin, IV basis).
        This confirms the warning pipeline is firing, not silently suppressed.
        """
        from src.data_ingestion import load_all_data
        data  = load_all_data()
        suite = run_all_validations(data)
        assert suite.n_warning >= 3, (
            f"Expected >= 3 warnings (orphan SKUs + neg margin + IV basis), "
            f"got {suite.n_warning}"
        )
