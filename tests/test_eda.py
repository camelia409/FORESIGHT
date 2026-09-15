"""
test_eda.py — Comprehensive Automated Tests for Phase 2A EDA
============================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 2A — Exploratory Data Analysis & Demand Characterization

Covers:
-------
 1. EDA input contains exactly 36,550 rows.
 2. Exactly 50 unique SKUs exist.
 3. Exactly 731 unique dates exist.
 4. SKU-Date uniqueness is strictly preserved (0 duplicates).
 5. Zero-demand observations are preserved (not dropped).
 6. Weekly aggregation is deterministic and sums reconcile with daily totals.
 7. SKU demand profile reconciles to source totals.
 8. Revenue totals reconcile with Units_Sold × Price.
 9. Promotion groups reconcile to total rows and demand.
10. No production dataset is modified by EDA (SHA-256 integrity).
11. No predictive lag/rolling feature is created in analysis_ready.parquet.
12. Inventory missingness is preserved (not converted to zero or forward-filled).
13. Orphan inventory remains separate in quarantine (3,600 rows).
14. All 9 required CSV artifacts and 14 plot files exist and are non-empty.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import CFG, PATHS, PROJECT_ROOT
from src.eda import (
    profile_dataset,
    analyze_demand_distribution,
    analyze_sku_demand_and_intermittency,
    analyze_temporal_patterns,
    investigate_weekly_aggregation,
    analyze_categories,
    analyze_promotions,
    analyze_price_demand,
    investigate_outliers,
    analyze_inventory_coverage,
    audit_feature_availability,
    run_eda,
)
from src.utils import file_sha256, hash_raw_files, assert_hashes_unchanged


@pytest.fixture(scope="module")
def analysis_ready_df() -> pd.DataFrame:
    """Load analysis_ready.parquet for EDA tests."""
    path = PATHS.processed_dir / "analysis_ready.parquet"
    assert path.exists(), f"analysis_ready.parquet missing at {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def quarantine_df() -> pd.DataFrame:
    """Load inventory_quarantine.parquet."""
    path = PATHS.interim_dir / "inventory_quarantine.parquet"
    assert path.exists(), f"inventory_quarantine.parquet missing at {path}"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_01_eda_input_row_count(analysis_ready_df: pd.DataFrame) -> None:
    """1. EDA input contains exactly 36,550 rows."""
    assert len(analysis_ready_df) == 36550


def test_02_unique_sku_count(analysis_ready_df: pd.DataFrame) -> None:
    """2. Exactly 50 SKUs exist."""
    assert analysis_ready_df["SKU"].nunique() == 50
    expected_skus = {f"SKU{i:03d}" for i in range(1, 51)}
    assert set(analysis_ready_df["SKU"].unique()) == expected_skus


def test_03_unique_date_count(analysis_ready_df: pd.DataFrame) -> None:
    """3. Exactly 731 dates exist."""
    assert analysis_ready_df["Date"].nunique() == 731
    assert str(analysis_ready_df["Date"].min().date()) == "2024-01-01"
    assert str(analysis_ready_df["Date"].max().date()) == "2025-12-31"


def test_04_sku_date_uniqueness(analysis_ready_df: pd.DataFrame) -> None:
    """4. SKU-Date uniqueness is maintained (0 duplicate pairs)."""
    assert analysis_ready_df.duplicated(subset=["SKU", "Date"]).sum() == 0


def test_05_zero_demand_preserved(analysis_ready_df: pd.DataFrame) -> None:
    """5. Zero-demand observations are preserved (not dropped)."""
    zero_count = (analysis_ready_df["Units_Sold"] == 0).sum()
    assert zero_count == 202
    assert (analysis_ready_df["Units_Sold"] >= 0).all()


def test_06_weekly_aggregation_deterministic(analysis_ready_df: pd.DataFrame) -> None:
    """6. Weekly aggregation is deterministic and sums reconcile with daily totals."""
    comp1, w1 = investigate_weekly_aggregation(analysis_ready_df)
    comp2, w2 = investigate_weekly_aggregation(analysis_ready_df)

    pd.testing.assert_frame_equal(w1, w2)
    assert w1["Weekly_Units"].sum() == analysis_ready_df["Units_Sold"].sum()
    assert len(w1) == 50 * 105  # 50 SKUs * 105 ISO weeks in 2024-2025


def test_07_sku_profile_reconciliation(analysis_ready_df: pd.DataFrame) -> None:
    """7. SKU demand profile reconciles to source totals."""
    sku_df = analyze_sku_demand_and_intermittency(analysis_ready_df)
    assert len(sku_df) == 50
    assert sku_df["Total_Units_Sold"].sum() == analysis_ready_df["Units_Sold"].sum()
    np.testing.assert_allclose(
        sku_df["Total_Revenue"].sum(),
        analysis_ready_df["Revenue"].sum(),
        rtol=1e-5,
    )
    # Check Syntetos-Boylan intermittency
    assert (sku_df["Intermittency_Class"] == "Smooth").all()
    assert (sku_df["ADI"] < 1.32).all()
    assert (sku_df["CV2"] < 0.49).all()


def test_08_revenue_totals_reconcile(analysis_ready_df: pd.DataFrame) -> None:
    """8. Revenue totals reconcile and Revenue == Units_Sold * Price holds."""
    rev_diff = (analysis_ready_df["Revenue"] - (analysis_ready_df["Units_Sold"] * analysis_ready_df["Price"])).abs()
    assert rev_diff.max() <= CFG.revenue_price_tol


def test_09_promotion_groups_reconcile(analysis_ready_df: pd.DataFrame) -> None:
    """9. Promotion groups reconcile to total rows and demand."""
    promo_df = analyze_promotions(analysis_ready_df)
    global_rows = promo_df[promo_df["Scope"] == "Global"]

    assert global_rows["Observations"].sum() == 36550
    assert global_rows["Total_Units"].sum() == analysis_ready_df["Units_Sold"].sum()

    # Check positive promotional lift
    p1_lift = global_rows.loc[global_rows["Group"] == "Promotion (1)", "Promotional_Lift_Pct"].iloc[0]
    assert p1_lift > 30.0  # Approx +38.1%


def test_10_production_data_unmodified_by_eda(analysis_ready_df: pd.DataFrame) -> None:
    """10. No production dataset is modified by EDA (SHA-256 integrity)."""
    raw_hashes_before = hash_raw_files(PATHS.raw_dir)
    processed_path = PATHS.processed_dir / "analysis_ready.parquet"
    proc_hash_before = file_sha256(processed_path)

    # Run EDA
    run_eda()

    # Assert hashes remain unchanged
    raw_hashes_after = hash_raw_files(PATHS.raw_dir)
    assert_hashes_unchanged(raw_hashes_before, raw_hashes_after)
    proc_hash_after = file_sha256(processed_path)
    assert proc_hash_before == proc_hash_after


def test_11_no_predictive_features_created(analysis_ready_df: pd.DataFrame) -> None:
    """11. No predictive lag/rolling feature is created in analysis_ready.parquet."""
    forbidden_substrings = ["lag_", "rolling_", "ma_", "moving_avg", "trend_", "seasonality_"]
    for col in analysis_ready_df.columns:
        for sub in forbidden_substrings:
            assert sub not in col.lower(), f"Found forbidden predictive feature column: {col}"


def test_12_inventory_missingness_preserved(analysis_ready_df: pd.DataFrame) -> None:
    """12. Inventory missingness is preserved (not converted to zero or forward-filled)."""
    # Exactly 1,200 snapshot dates
    assert analysis_ready_df["has_inventory_snapshot"].sum() == 1200
    # Exactly 35,350 days unobserved (null)
    assert analysis_ready_df["Current_Stock"].isna().sum() == 35350
    assert analysis_ready_df["Inventory_Value"].isna().sum() == 35350


def test_13_orphan_inventory_quarantine_integrity(quarantine_df: pd.DataFrame) -> None:
    """13. Orphan inventory remains separate in quarantine (3,600 rows)."""
    assert len(quarantine_df) == 3600
    assert quarantine_df["SKU"].nunique() == 150
    orphan_set = {f"SKU{i:03d}" for i in range(51, 201)}
    assert set(quarantine_df["SKU"].unique()) == orphan_set


def test_14_eda_artifacts_and_plots_exist() -> None:
    """14. All 9 required CSV artifacts and 14 plot files exist and are non-empty."""
    eda_artifacts_dir = PROJECT_ROOT / "artifacts" / "eda"
    plots_dir = eda_artifacts_dir / "plots"
    report_path = PATHS.reports_eda_dir / "phase2a_eda_report.md"

    expected_csvs = [
        "sku_demand_profile.csv",
        "demand_summary.csv",
        "temporal_summary.csv",
        "category_summary.csv",
        "promotion_summary.csv",
        "price_demand_summary.csv",
        "inventory_coverage_summary.csv",
        "outlier_candidates.csv",
        "feature_availability_audit.csv",
    ]

    for csv_name in expected_csvs:
        p = eda_artifacts_dir / csv_name
        assert p.exists(), f"Artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

    for i in range(1, 15):
        pattern = f"{i:02d}_*.png"
        matches = list(plots_dir.glob(pattern))
        assert len(matches) == 1, f"Missing plot #{i:02d} in {plots_dir}"
        assert matches[0].stat().st_size > 0, f"Plot is empty: {matches[0]}"

    assert report_path.exists(), f"Report missing: {report_path}"
    assert report_path.stat().st_size > 1000, f"Report is suspiciously short: {report_path}"
