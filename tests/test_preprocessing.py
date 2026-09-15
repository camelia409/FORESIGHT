"""
test_preprocessing.py — Comprehensive Unit & Integration Tests for Phase 1B
===========================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 1B — Preprocessing + Data Integration

Covers (15 Required Production Invariants & Integration Rules):
---------------------------------------------------------------
 1. sales normalization (dtypes, Date datetime, SKU str, zeros retained)
 2. SKU normalization (SKU str, Product_Name preserved, prices/costs intact)
 3. calendar normalization (date -> Date, attributes preserved, structural NaNs)
 4. inventory normalization & split (aligned vs quarantine)
 5. negative-margin flag preservation (16 SKUs flagged, values unaltered)
 6. orphan quarantine completeness (SKU051–SKU200, 3,600 rows, 150 SKUs)
 7. sales panel row count stability (exactly 36,550 rows)
 8. SKU-Date uniqueness invariant (0 duplicates)
 9. safe inventory joining (no row explosion, left join on daily panel)
10. duplicate inventory grain detection (structural error on dup keys)
11. Revenue invariant enforcement (Revenue == Units_Sold × Price, error on mismatch)
12. non-negativity constraint enforcement (error on negative Units_Sold)
13. raw data integrity (raw SHA-256 hashes unchanged)
14. analysis-ready schema & parquet read/write validation
15. deterministic preprocessing (identical output on repeated runs)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CFG, PATHS
from src.data_ingestion import load_all_data
from src.preprocessing import (
    normalize_sales,
    normalize_sku_master,
    normalize_calendar,
    normalize_inventory,
    handle_known_quality_issues,
    integrate_datasets,
    verify_analysis_ready_invariants,
    build_analysis_ready,
    generate_data_dictionary,
)
from src.utils import file_sha256, hash_raw_files, assert_hashes_unchanged


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def raw_data() -> dict[str, pd.DataFrame]:
    """Load real raw datasets once for integration tests."""
    return load_all_data()


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Minimal valid sales daily frame."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "SKU": ["SKU001", "SKU002", "SKU001", "SKU002"],
        "Units_Sold": [10, 0, 5, 20],  # Includes zero sales
        "Revenue": [1000.0, 0.0, 500.0, 2000.0],
        "Price": [100.0, 100.0, 100.0, 100.0],
        "Promotion": [0, 0, 1, 0],
    })


@pytest.fixture
def sample_sku() -> pd.DataFrame:
    """Minimal valid sku master frame with 1 negative margin SKU."""
    return pd.DataFrame({
        "SKU": ["SKU001", "SKU002"],
        "Product_Name": ["Product Alpha", "Product Beta"],
        "Category": ["Electronics", "Electronics"],
        "Subcategory": ["Gadgets", "Gadgets"],
        "Launch_Date": ["2023-01-01", "2023-06-01"],
        "Cost_Price": [60.0, 120.0],       # SKU002 has Cost (120) > Selling (100)
        "Selling_Price": [100.0, 100.0],
        "Gross_Margin_Per_Unit": [40.0, -20.0],
    })


@pytest.fixture
def sample_calendar() -> pd.DataFrame:
    """Minimal valid calendar frame with structural NaNs."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "year": [2024, 2024],
        "month": [1, 1],
        "quarter": ["Q1", "Q1"],
        "week": [1, 1],
        "day_of_week": ["Monday", "Tuesday"],
        "is_weekend": [0, 0],
        "season": ["Winter", "Winter"],
        "holiday": ["New Year", np.nan],  # Structural NaN on 2024-01-02
        "is_holiday": [1, 0],
        "promotion_event": [np.nan, "Flash Sale"],
    })


@pytest.fixture
def sample_inventory() -> pd.DataFrame:
    """Minimal inventory frame with 1 master SKU and 1 orphan SKU."""
    return pd.DataFrame({
        "Snapshot_Date": ["2024-01-01", "2024-01-01"],
        "SKU": ["SKU001", "SKU099"],  # SKU099 is orphan
        "Current_Stock": [500, 250],
        "On_Order": [100, 50],
        "Lead_Time_Days": [7, 14],
        "Safety_Stock": [50, 25],
        "Reorder_Point": [150, 75],
        "Inventory_Value": [30000.0, 15000.0],
    })


# ===========================================================================
# 1. Sales Normalization Tests
# ===========================================================================

def test_sales_normalization_dtypes_and_values(sample_sales: pd.DataFrame) -> None:
    """Test sales normalization enforces datetime, string, numeric dtypes, and preserves zeros."""
    norm = normalize_sales(sample_sales)

    assert pd.api.types.is_datetime64_any_dtype(norm["Date"])
    assert norm["SKU"].dtype == object
    assert norm["Units_Sold"].dtype == np.int64
    assert norm["Price"].dtype == np.float64
    assert norm["Revenue"].dtype == np.float64
    assert norm["Promotion"].dtype == np.int64

    # Zero sales observation is preserved
    zero_row = norm[(norm["SKU"] == "SKU002") & (norm["Date"] == "2024-01-01")]
    assert len(zero_row) == 1
    assert zero_row["Units_Sold"].iloc[0] == 0
    assert zero_row["Revenue"].iloc[0] == 0.0

    # Total rows preserved
    assert len(norm) == len(sample_sales)


def test_sales_normalization_catches_revenue_invariant_violation(sample_sales: pd.DataFrame) -> None:
    """Revenue == Units_Sold * Price must raise error on violation."""
    corrupted = sample_sales.copy()
    corrupted.loc[0, "Revenue"] = 9999.0  # Units_Sold (10) * Price (100) = 1000 != 9999

    with pytest.raises(ValueError, match="Revenue invariant failure"):
        normalize_sales(corrupted)


def test_sales_normalization_catches_negative_values(sample_sales: pd.DataFrame) -> None:
    """Negative units, price, or revenue must raise ValueError."""
    bad_units = sample_sales.copy()
    bad_units.loc[0, "Units_Sold"] = -5
    with pytest.raises(ValueError, match="negative Units_Sold"):
        normalize_sales(bad_units)

    bad_price = sample_sales.copy()
    bad_price.loc[0, "Price"] = -10.0
    with pytest.raises(ValueError, match="negative Price"):
        normalize_sales(bad_price)


# ===========================================================================
# 2. SKU Master Normalization Tests
# ===========================================================================

def test_sku_master_normalization_and_negative_margin_flag(sample_sku: pd.DataFrame) -> None:
    """Test SKU normalization preserves attributes and derives negative_margin_flag."""
    norm = normalize_sku_master(sample_sku)

    assert "negative_margin_flag" in norm.columns
    # SKU001: Cost 60 <= Selling 100 -> False
    assert norm.loc[norm["SKU"] == "SKU001", "negative_margin_flag"].iloc[0] == False
    # SKU002: Cost 120 > Selling 100 -> True
    assert norm.loc[norm["SKU"] == "SKU002", "negative_margin_flag"].iloc[0] == True

    # Values must remain identical (not altered or corrected)
    assert norm.loc[norm["SKU"] == "SKU002", "Cost_Price"].iloc[0] == 120.0
    assert norm.loc[norm["SKU"] == "SKU002", "Selling_Price"].iloc[0] == 100.0


# ===========================================================================
# 3. Calendar Normalization Tests
# ===========================================================================

def test_calendar_normalization_preserves_structural_nans(sample_calendar: pd.DataFrame) -> None:
    """Test calendar normalization normalizes Date and preserves structural NaNs."""
    norm = normalize_calendar(sample_calendar)

    assert "Date" in norm.columns
    assert "date" not in norm.columns  # Dropped to prevent join collisions
    assert pd.api.types.is_datetime64_any_dtype(norm["Date"])

    # Structural NaNs preserved
    assert pd.isna(norm.loc[norm["Date"] == "2024-01-02", "holiday"].iloc[0])
    assert pd.isna(norm.loc[norm["Date"] == "2024-01-01", "promotion_event"].iloc[0])


# ===========================================================================
# 4. Inventory Normalization & Quarantine Split Tests
# ===========================================================================

def test_inventory_normalization_quarantine_split(
    sample_inventory: pd.DataFrame,
    sample_sku: pd.DataFrame,
) -> None:
    """Test inventory split into aligned and quarantine based on sku_master presence."""
    norm_sku = normalize_sku_master(sample_sku)
    aligned, quarantine = normalize_inventory(sample_inventory, norm_sku)

    # SKU001 is in master -> aligned
    assert len(aligned) == 1
    assert aligned["SKU"].iloc[0] == "SKU001"
    assert aligned["inventory_master_match_flag"].iloc[0] == True

    # SKU099 is orphan -> quarantined
    assert len(quarantine) == 1
    assert quarantine["SKU"].iloc[0] == "SKU099"
    assert quarantine["inventory_quarantine_flag"].iloc[0] == True
    assert quarantine["Inventory_Value"].iloc[0] == 15000.0  # Preserved exactly


# ===========================================================================
# 5. Safe Integration Tests
# ===========================================================================

def test_integration_grain_safety_and_duplicate_prevention(
    sample_sales: pd.DataFrame,
    sample_sku: pd.DataFrame,
    sample_calendar: pd.DataFrame,
    sample_inventory: pd.DataFrame,
) -> None:
    """Test integrate_datasets merges cleanly without row explosion."""
    norm_sales = normalize_sales(sample_sales)
    norm_sku = normalize_sku_master(sample_sku)
    norm_cal = normalize_calendar(sample_calendar)
    inv_aligned, _ = normalize_inventory(sample_inventory, norm_sku)

    integrated = integrate_datasets(norm_sales, norm_sku, norm_cal, inv_aligned)

    # Row count must match sales exactly (4 rows)
    assert len(integrated) == 4
    # Inventory metrics joined on matched date/sku
    matched = integrated[(integrated["SKU"] == "SKU001") & (integrated["Date"] == "2024-01-01")]
    assert matched["Current_Stock"].iloc[0] == 500
    assert matched["has_inventory_snapshot"].iloc[0] == True

    # Non-snapshot date has NaN and flag False
    unmatched = integrated[(integrated["SKU"] == "SKU001") & (integrated["Date"] == "2024-01-02")]
    assert pd.isna(unmatched["Current_Stock"].iloc[0])
    assert unmatched["has_inventory_snapshot"].iloc[0] == False


def test_integration_catches_duplicate_inventory_grain(
    sample_sales: pd.DataFrame,
    sample_sku: pd.DataFrame,
    sample_calendar: pd.DataFrame,
) -> None:
    """If inventory contains duplicate (SKU, Date) snapshots, integration must raise error."""
    norm_sales = normalize_sales(sample_sales)
    norm_sku = normalize_sku_master(sample_sku)
    norm_cal = normalize_calendar(sample_calendar)

    dup_inventory = pd.DataFrame({
        "Snapshot_Date": ["2024-01-01", "2024-01-01"],
        "SKU": ["SKU001", "SKU001"],  # Duplicate key!
        "Current_Stock": [500, 600],
        "On_Order": [100, 100],
        "Lead_Time_Days": [7, 7],
        "Safety_Stock": [50, 50],
        "Reorder_Point": [150, 150],
        "Inventory_Value": [30000.0, 36000.0],
    })

    inv_aligned, _ = normalize_inventory(dup_inventory, norm_sku)
    with pytest.raises(ValueError, match="duplicate.*snapshots"):
        integrate_datasets(norm_sales, norm_sku, norm_cal, inv_aligned)


# ===========================================================================
# 6. End-to-End Real Data Pipeline & Invariant Tests
# ===========================================================================

def test_real_data_build_analysis_ready_and_invariants(raw_data: dict[str, pd.DataFrame]) -> None:
    """End-to-end integration test on real raw data verifying all 16 invariants."""
    raw_hashes_before = hash_raw_files(PATHS.raw_dir)

    analysis_ready, interim_master, inv_quarantine, inv_results = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
        raw_dir=PATHS.raw_dir,
        raw_hashes_before=raw_hashes_before,
    )

    # 1. 36,550 rows
    assert len(analysis_ready) == 36550
    assert len(interim_master) == 36550

    # 2. 50 unique SKUs
    assert analysis_ready["SKU"].nunique() == 50

    # 3. 731 unique dates
    assert analysis_ready["Date"].nunique() == 731

    # 4. Strict uniqueness on (SKU, Date)
    assert analysis_ready.duplicated(subset=["SKU", "Date"]).sum() == 0

    # 5. 150 quarantined orphan SKUs, 3,600 rows
    assert len(inv_quarantine) == 3600
    assert inv_quarantine["SKU"].nunique() == 150
    orphan_set = {f"SKU{i:03d}" for i in range(51, 201)}
    assert set(inv_quarantine["SKU"].unique()) == orphan_set

    # 6. No orphan SKU in analysis_ready
    assert analysis_ready["SKU"].isin(orphan_set).sum() == 0

    # 7. 16 negative-margin SKUs flagged
    neg_margin_skus = analysis_ready.groupby("SKU")["negative_margin_flag"].first()
    assert int(neg_margin_skus.sum()) == 16

    # 8. Revenue invariant holds across all 36,550 rows
    rev_diff = (analysis_ready["Revenue"] - (analysis_ready["Units_Sold"] * analysis_ready["Price"])).abs()
    assert rev_diff.max() <= CFG.revenue_price_tol

    # 9. No negative values
    assert (analysis_ready["Units_Sold"] < 0).sum() == 0
    assert (analysis_ready["Price"] < 0).sum() == 0
    assert (analysis_ready["Revenue"] < 0).sum() == 0

    # 10. Inventory snapshot alignment
    assert analysis_ready["has_inventory_snapshot"].sum() == 1200  # 24 dates * 50 SKUs

    # 11. All 16 invariants reported PASS
    assert len(inv_results) == 16
    assert all(r["status"] == "PASS" for r in inv_results.values())

    # 12. Raw files untouched
    raw_hashes_after = hash_raw_files(PATHS.raw_dir)
    assert_hashes_unchanged(raw_hashes_before, raw_hashes_after)


def test_data_dictionary_generation(raw_data: dict[str, pd.DataFrame]) -> None:
    """Verify data dictionary produces metadata for all analysis_ready columns."""
    analysis_ready, _, _, _ = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
    )

    dict_df = generate_data_dictionary(analysis_ready)

    assert len(dict_df) == len(analysis_ready.columns)
    assert set(dict_df["column_name"]) == set(analysis_ready.columns)
    required_fields = {"column_name", "dtype", "source_dataset", "transformation", "missing_count", "business_meaning"}
    assert required_fields.issubset(set(dict_df.columns))
    assert (dict_df["business_meaning"].str.len() > 0).all()


def test_deterministic_preprocessing(raw_data: dict[str, pd.DataFrame]) -> None:
    """Repeated preprocessing runs must produce 100% identical DataFrames."""
    res1, _, _, _ = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
    )
    res2, _, _, _ = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
    )

    pd.testing.assert_frame_equal(res1, res2)


def test_inventory_valuation_basis_unaltered(raw_data: dict[str, pd.DataFrame]) -> None:
    """Verify raw Inventory_Value is preserved without modification or revaluation."""
    analysis_ready, _, inv_quarantine, _ = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
    )

    # Master aligned inventory values matched exactly
    raw_inv = raw_data["inventory"]
    master_raw_inv = raw_inv[raw_inv["SKU"].isin(raw_data["sku"]["SKU"])]
    raw_val_sum = master_raw_inv["Inventory_Value"].sum()

    panel_val_sum = analysis_ready["Inventory_Value"].dropna().sum()
    np.testing.assert_allclose(raw_val_sum, panel_val_sum, rtol=1e-5)

    # Quarantined inventory values preserved exactly
    quarantine_raw_inv = raw_inv[~raw_inv["SKU"].isin(raw_data["sku"]["SKU"])]
    np.testing.assert_allclose(
        quarantine_raw_inv["Inventory_Value"].sum(),
        inv_quarantine["Inventory_Value"].sum(),
        rtol=1e-5,
    )


def test_save_analysis_ready_and_parquet_schema(raw_data: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    """Verify save_analysis_ready writes valid Parquet files readable by pyarrow."""
    from src.preprocessing import save_analysis_ready
    import json

    analysis_ready, interim_master, inv_quarantine, inv_results = build_analysis_ready(
        sales=raw_data["sales"],
        sku=raw_data["sku"],
        cal=raw_data["calendar"],
        inv=raw_data["inventory"],
    )

    saved = save_analysis_ready(
        analysis_ready_df=analysis_ready,
        interim_master_df=interim_master,
        inventory_quarantine_df=inv_quarantine,
        invariant_results=inv_results,
    )

    assert saved["analysis_ready"].exists()
    assert saved["interim_master"].exists()
    assert saved["inventory_quarantine"].exists()
    assert saved["data_dictionary"].exists()
    assert saved["lineage"].exists()
    assert saved["report"].exists()

    # Read back parquet files
    loaded_analysis = pd.read_parquet(saved["analysis_ready"], engine="pyarrow")
    assert len(loaded_analysis) == 36550
    assert loaded_analysis["SKU"].nunique() == 50
    assert loaded_analysis["Date"].nunique() == 731

    loaded_quarantine = pd.read_parquet(saved["inventory_quarantine"], engine="pyarrow")
    assert len(loaded_quarantine) == 3600
    assert loaded_quarantine["SKU"].nunique() == 150

    # Verify lineage manifest is valid JSON
    with open(saved["lineage"], "r", encoding="utf-8") as fh:
        lineage_data = json.load(fh)
    assert lineage_data["phase"].startswith("1B")
    assert lineage_data["output_datasets"]["analysis_ready"]["rows"] == 36550
