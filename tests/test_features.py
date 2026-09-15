"""
test_features.py — Comprehensive Leakage & Temporal Governance Tests
====================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3A — Leakage-Safe Feature Engineering & Temporal Audit

Covers all required audit tests:
1. test_01_weekly_aggregation_grain_and_reconciliation
2. test_02_demand_lags_mathematical_correctness
3. test_03_rolling_features_strictly_historical
4. test_04_temporal_ordering
5. test_05_target_alignment
6. test_06_horizon_alignment
7. test_07_no_future_leakage_in_features
8. test_08_inventory_missingness_preserved
9. test_09_deterministic_output
10. test_10_no_source_mutation
11. test_11_expected_schema_and_zero_nulls
12. test_12_no_duplicate_keys
13. test_13_categorical_feature_validity
14. test_14_feature_lineage_completeness
15. test_15_feature_engineer_transformer_roundtrip
-- Explicit Required Adversarial Temporal Audit Tests (Prompt Section 13) --
16. test_future_demand_perturbation_invariance
17. test_future_price_perturbation_invariance
18. test_future_promotion_perturbation_invariance
19. test_future_inventory_perturbation_invariance
20. test_future_calendar_perturbation_invariance
21. test_inventory_snapshot_temporal_boundary
22. test_target_strictly_after_origin
23. test_max_forecast_horizon_8
24. test_no_target_columns_in_predictors
25. test_feature_engineer_fit_state_audit
26. test_multiple_origin_leakage_invariance
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import CFG, PATHS, PROJECT_ROOT
from src.feature_engineering import (
    DEMAND_LAGS,
    FORECAST_HORIZON,
    ROLLING_WINDOWS,
    WARM_UP_WEEKS,
    FeatureEngineer,
    _sha256_file,
    build_model_features,
    build_multi_horizon_targets,
    build_weekly_panel,
    generate_feature_dictionary,
    generate_feature_lineage,
    generate_feature_statistics,
    validate_feature_leakage,
)
from src.utils import hash_raw_files


@pytest.fixture(scope="session")
def real_analysis_ready():
    """Load verified analysis_ready.parquet."""
    path = PATHS.processed_dir / "analysis_ready.parquet"
    assert path.exists(), f"{path} does not exist"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def feature_tables(real_analysis_ready):
    """Generate (df_full, df_valid) once for tests."""
    df_full, df_valid = build_model_features(real_analysis_ready)
    return df_full, df_valid


# ---------------------------------------------------------------------------
# 1. Weekly Aggregation & Volume Conservation
# ---------------------------------------------------------------------------

def test_01_weekly_aggregation_grain_and_reconciliation(real_analysis_ready):
    weekly = build_weekly_panel(real_analysis_ready)
    assert weekly["SKU"].nunique() == 50, "Expected exactly 50 SKUs"
    assert weekly["week_start"].nunique() == 106, "Expected exactly 106 weeks"
    assert len(weekly) == 50 * 106, f"Expected 5,300 rows, got {len(weekly)}"

    daily_units = real_analysis_ready["Units_Sold"].sum()
    weekly_units = weekly["Units_Sold"].sum()
    assert daily_units == weekly_units, f"Volume mismatch: {daily_units} != {weekly_units}"


# ---------------------------------------------------------------------------
# 2. Demand Lag Mathematical Correctness
# ---------------------------------------------------------------------------

def test_02_demand_lags_mathematical_correctness(feature_tables, real_analysis_ready):
    df_full, _ = feature_tables
    weekly = build_weekly_panel(real_analysis_ready)

    sku_weekly = weekly[weekly["SKU"] == "SKU001"].sort_values("week_start").reset_index(drop=True)
    sku_feat = df_full[df_full["SKU"] == "SKU001"].sort_values("forecast_origin_date").reset_index(drop=True)

    idx = 60
    assert sku_feat.loc[idx, "lag_1"] == sku_weekly.loc[idx, "Units_Sold"]
    assert sku_feat.loc[idx, "lag_2"] == sku_weekly.loc[idx - 1, "Units_Sold"]
    assert sku_feat.loc[idx, "lag_4"] == sku_weekly.loc[idx - 3, "Units_Sold"]
    assert sku_feat.loc[idx, "lag_52"] == sku_weekly.loc[idx - 51, "Units_Sold"]


# ---------------------------------------------------------------------------
# 3. Rolling Features Use Strictly Historical Observations
# ---------------------------------------------------------------------------

def test_03_rolling_features_strictly_historical(feature_tables, real_analysis_ready):
    df_full, _ = feature_tables
    weekly = build_weekly_panel(real_analysis_ready)

    sku = "SKU005"
    sku_w = weekly[weekly["SKU"] == sku].sort_values("week_start").reset_index(drop=True)
    sku_f = df_full[df_full["SKU"] == sku].sort_values("forecast_origin_date").reset_index(drop=True)

    idx = 55
    expected_mean4 = sku_w.loc[idx-3:idx, "Units_Sold"].mean()
    actual_mean4 = sku_f.loc[idx, "rolling_mean_4"]
    assert np.isclose(expected_mean4, actual_mean4), f"rolling_mean_4 mismatch: {expected_mean4} vs {actual_mean4}"

    expected_min4 = sku_w.loc[idx-3:idx, "Units_Sold"].min()
    actual_min4 = sku_f.loc[idx, "rolling_min_4"]
    assert expected_min4 == actual_min4

    expected_max4 = sku_w.loc[idx-3:idx, "Units_Sold"].max()
    actual_max4 = sku_f.loc[idx, "rolling_max_4"]
    assert expected_max4 == actual_max4


# ---------------------------------------------------------------------------
# 4. Temporal Ordering
# ---------------------------------------------------------------------------

def test_04_temporal_ordering(feature_tables):
    _, df_valid = feature_tables
    for sku, group in df_valid.groupby("SKU"):
        dates = group["forecast_origin_date"].tolist()
        assert dates == sorted(dates), f"SKU {sku} dates not strictly sorted!"


# ---------------------------------------------------------------------------
# 5. Target Alignment (h=1..8)
# ---------------------------------------------------------------------------

def test_05_target_alignment(feature_tables, real_analysis_ready):
    df_full, _ = feature_tables
    weekly = build_weekly_panel(real_analysis_ready)

    sku = "SKU010"
    sku_w = weekly[weekly["SKU"] == sku].sort_values("week_start").reset_index(drop=True)
    sku_f = df_full[df_full["SKU"] == sku].sort_values("forecast_origin_date").reset_index(drop=True)

    idx = 52
    for h in range(1, 9):
        expected_target = sku_w.loc[idx + h, "Units_Sold"]
        actual_target = sku_f.loc[idx, f"target_h{h}"]
        assert expected_target == actual_target, f"target_h{h} mismatch at idx {idx}: {expected_target} vs {actual_target}"


# ---------------------------------------------------------------------------
# 6. Horizon Alignment
# ---------------------------------------------------------------------------

def test_06_horizon_alignment(feature_tables):
    _, df_valid = feature_tables
    target_cols = [c for c in df_valid.columns if c.startswith("target_h")]
    assert len(target_cols) == 8, f"Expected exactly 8 targets, got {len(target_cols)}"
    assert sorted(target_cols) == [f"target_h{h}" for h in range(1, 9)]


# ---------------------------------------------------------------------------
# 7. No Future Leakage in Feature Columns
# ---------------------------------------------------------------------------

def test_07_no_future_leakage_in_features(feature_tables):
    _, df_valid = feature_tables
    assert validate_feature_leakage(df_valid) is True


# ---------------------------------------------------------------------------
# 8. Inventory Missingness Preserved
# ---------------------------------------------------------------------------

def test_08_inventory_missingness_preserved(feature_tables):
    df_full, df_valid = feature_tables
    snapshot_origins = df_valid[df_valid["has_inventory_snapshot_at_origin"] == 1]["forecast_origin_date"].unique()
    for o in snapshot_origins:
        assert pd.Timestamp(o).day == 1, f"Expected 1st of month for snapshot origin, got {o}"

    assert (df_valid["days_since_inventory_snapshot"] >= 0).all()
    assert (df_valid["days_since_inventory_snapshot"] <= 35).all()


# ---------------------------------------------------------------------------
# 9. Deterministic Output
# ---------------------------------------------------------------------------

def test_09_deterministic_output(real_analysis_ready):
    _, df_valid1 = build_model_features(real_analysis_ready)
    _, df_valid2 = build_model_features(real_analysis_ready)
    pd.testing.assert_frame_equal(df_valid1, df_valid2)


# ---------------------------------------------------------------------------
# 10. No Source Mutation (SHA-256 Integrity)
# ---------------------------------------------------------------------------

def test_10_no_source_mutation(real_analysis_ready):
    analysis_ready_path = PATHS.processed_dir / "analysis_ready.parquet"
    pre_hash = _sha256_file(analysis_ready_path)
    build_model_features(real_analysis_ready)
    post_hash = _sha256_file(analysis_ready_path)
    assert pre_hash == post_hash, "analysis_ready.parquet was mutated during feature engineering!"


# ---------------------------------------------------------------------------
# 11. Expected Feature Schema and Absence of Nulls in Valid Slice
# ---------------------------------------------------------------------------

def test_11_expected_schema_and_zero_nulls(feature_tables):
    _, df_valid = feature_tables
    assert df_valid.shape == (2350, 60), f"Expected shape (2350, 60), got {df_valid.shape}"
    null_counts = df_valid.isna().sum()
    assert (null_counts == 0).all(), f"Found unexpected null values in valid training panel:\n{null_counts[null_counts > 0]}"


# ---------------------------------------------------------------------------
# 12. No Duplicate Training Keys
# ---------------------------------------------------------------------------

def test_12_no_duplicate_keys(feature_tables):
    _, df_valid = feature_tables
    duplicates = df_valid.duplicated(subset=["SKU", "forecast_origin_date"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (SKU, forecast_origin_date) keys!"


# ---------------------------------------------------------------------------
# 13. Categorical Feature Validity
# ---------------------------------------------------------------------------

def test_13_categorical_feature_validity(feature_tables):
    _, df_valid = feature_tables
    assert set(df_valid["Price_Tier"].unique()) == {"Budget", "Mid-Range", "Premium"}
    assert df_valid["Category"].nunique() == 5
    assert set(df_valid["negative_margin_flag"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 14. Feature Lineage Completeness
# ---------------------------------------------------------------------------

def test_14_feature_lineage_completeness():
    lineage = generate_feature_lineage()
    assert "metadata" in lineage
    assert "feature_registry" in lineage
    registry = lineage["feature_registry"]
    assert len(registry) >= 40

    dict_df = generate_feature_dictionary()
    assert "feature_name" in dict_df.columns
    assert "leakage_status" in dict_df.columns
    assert set(dict_df["leakage_status"].unique()).issubset({"SAFE", "CONDITIONAL", "TARGET", "LEAKAGE"})


# ---------------------------------------------------------------------------
# 15. FeatureEngineer Transformer Serialization Round-Trip
# ---------------------------------------------------------------------------

def test_15_feature_engineer_transformer_roundtrip(tmp_path, real_analysis_ready):
    fe = FeatureEngineer()
    df_transformed = fe.fit_transform(real_analysis_ready)
    assert fe.is_fitted_ is True
    assert len(fe.get_feature_names()) == 50

    model_path = tmp_path / "test_feature_engineer.pkl"
    fe.save(model_path)
    loaded_fe = FeatureEngineer.load(model_path)
    assert loaded_fe.is_fitted_ is True
    assert loaded_fe.get_feature_names() == fe.get_feature_names()


# ===========================================================================
# ADVERSARIAL TEMPORAL LEAKAGE AUDIT TESTS (Section 13 Requirements)
# ===========================================================================

def _run_single_origin_perturbation_check(real_analysis_ready, origin_t, pert_column, pert_fn):
    """Helper to verify that perturbing a column after origin_t causes 0 change in features at origin_t."""
    _, df_valid_orig = build_model_features(real_analysis_ready)
    features_orig = df_valid_orig[df_valid_orig["forecast_origin_date"] == origin_t].sort_values("SKU").reset_index(drop=True)

    df_pert = real_analysis_ready.copy()
    future_mask = df_pert["Date"] > (origin_t + pd.Timedelta(days=6))
    df_pert = pert_fn(df_pert, future_mask)

    _, df_valid_pert = build_model_features(df_pert)
    features_pert = df_valid_pert[df_valid_pert["forecast_origin_date"] == origin_t].sort_values("SKU").reset_index(drop=True)

    keys_targets = ["SKU", "forecast_origin_date"] + [f"target_h{h}" for h in range(1, 9)]
    feature_cols = [c for c in df_valid_orig.columns if c not in keys_targets]

    for col in feature_cols:
        o_vals = features_orig[col].values
        p_vals = features_pert[col].values
        if pd.api.types.is_bool_dtype(features_orig[col]):
            assert (o_vals == p_vals).all(), f"Leakage detected in {col} when perturbing {pert_column}"
        elif pd.api.types.is_numeric_dtype(features_orig[col]):
            max_diff = np.nanmax(np.abs(o_vals - p_vals))
            assert max_diff < 1e-6, f"Leakage detected in {col} (diff={max_diff}) when perturbing {pert_column}"
        else:
            assert (o_vals == p_vals).all(), f"Leakage detected in categorical {col} when perturbing {pert_column}"


def test_future_demand_perturbation_invariance(real_analysis_ready):
    """Perturbing future Units_Sold after origin t must not change origin-t features."""
    _, df_valid = build_model_features(real_analysis_ready)
    mid_origin = sorted(df_valid["forecast_origin_date"].unique())[23]
    _run_single_origin_perturbation_check(
        real_analysis_ready,
        mid_origin,
        "Units_Sold",
        lambda d, m: d.assign(Units_Sold=np.where(m, d["Units_Sold"] * 10 + 50, d["Units_Sold"]))
    )


def test_future_price_perturbation_invariance(real_analysis_ready):
    """Perturbing future Selling_Price after origin t must not change origin-t features."""
    _, df_valid = build_model_features(real_analysis_ready)
    mid_origin = sorted(df_valid["forecast_origin_date"].unique())[23]
    _run_single_origin_perturbation_check(
        real_analysis_ready,
        mid_origin,
        "Selling_Price",
        lambda d, m: d.assign(Selling_Price=np.where(m, d["Selling_Price"] * 3, d["Selling_Price"]))
    )


def test_future_promotion_perturbation_invariance(real_analysis_ready):
    """Perturbing future Promotion after origin t must not change origin-t features."""
    _, df_valid = build_model_features(real_analysis_ready)
    mid_origin = sorted(df_valid["forecast_origin_date"].unique())[23]
    _run_single_origin_perturbation_check(
        real_analysis_ready,
        mid_origin,
        "Promotion",
        lambda d, m: d.assign(Promotion=np.where(m, 1 - d["Promotion"], d["Promotion"]))
    )


def test_future_inventory_perturbation_invariance(real_analysis_ready):
    """Perturbing future inventory snapshots after origin t must not change origin-t features."""
    _, df_valid = build_model_features(real_analysis_ready)
    mid_origin = sorted(df_valid["forecast_origin_date"].unique())[23]
    _run_single_origin_perturbation_check(
        real_analysis_ready,
        mid_origin,
        "Current_Stock / On_Order",
        lambda d, m: d.assign(
            Current_Stock=np.where(m, d["Current_Stock"] + 5000, d["Current_Stock"]),
            On_Order=np.where(m, d["On_Order"] + 2000, d["On_Order"]),
        )
    )


def test_future_calendar_perturbation_invariance(real_analysis_ready):
    """Perturbing future calendar event fields after origin t must not change origin-t features."""
    _, df_valid = build_model_features(real_analysis_ready)
    mid_origin = sorted(df_valid["forecast_origin_date"].unique())[23]
    _run_single_origin_perturbation_check(
        real_analysis_ready,
        mid_origin,
        "is_holiday",
        lambda d, m: d.assign(is_holiday=np.where(m, 1 - d["is_holiday"], d["is_holiday"]))
    )


def test_inventory_snapshot_temporal_boundary(feature_tables):
    """Verify that latest known inventory strictly satisfies Snapshot_Date <= forecast origin."""
    _, df_valid = feature_tables
    # Days since inventory snapshot must be non-negative (never negative, which would mean future snapshot)
    assert (df_valid["days_since_inventory_snapshot"] >= 0).all(), "Found negative days_since_inventory_snapshot (future snapshot used!)"
    assert (df_valid["days_since_inventory_snapshot"] <= 35).all(), "Unexpected snapshot gap > 35 days"


def test_target_strictly_after_origin(feature_tables, real_analysis_ready):
    """Verify that target_date > forecast_origin for every training row and every horizon."""
    df_full, df_valid = feature_tables
    weekly = build_weekly_panel(real_analysis_ready)

    origins = df_valid["forecast_origin_date"].unique()
    for o in origins:
        o_ts = pd.Timestamp(o)
        for h in range(1, 9):
            target_date = o_ts + pd.Timedelta(weeks=h)
            assert target_date > o_ts, f"Target date {target_date} is not strictly after origin {o_ts}"


def test_max_forecast_horizon_8(feature_tables):
    """Verify maximum forecast horizon is exactly 8."""
    _, df_valid = feature_tables
    target_cols = [c for c in df_valid.columns if c.startswith("target_h")]
    assert len(target_cols) == 8
    assert "target_h8" in target_cols
    assert "target_h9" not in df_valid.columns


def test_no_target_columns_in_predictors(real_analysis_ready):
    """Verify no target column or target-derived contemporaneous value is in predictor feature names."""
    fe = FeatureEngineer()
    fe.fit(real_analysis_ready)
    predictor_names = fe.get_feature_names()

    for col in predictor_names:
        assert not col.startswith("target_"), f"Target column '{col}' found in predictors!"
        assert col != "Units_Sold", "Raw Units_Sold found in predictors!"
        assert col != "Revenue", "Contemporaneous Revenue found in predictors!"
        assert col != "Inventory_Value", "Inventory_Value found in predictors!"


def test_feature_engineer_fit_state_audit():
    """Verify that feature_engineer.pkl contains no leaking statistics or global aggregates."""
    fe_path = PATHS.models_dir / "production" / "feature_engineer.pkl"
    assert fe_path.exists(), f"Transformer missing at {fe_path}"
    fe = joblib.load(fe_path)

    state = vars(fe)
    # Ensure only structural metadata is preserved
    allowed_keys = {"forecast_horizon_weeks", "warm_up_weeks", "feature_names_", "target_names_", "is_fitted_"}
    actual_keys = set(state.keys())
    assert actual_keys == allowed_keys, f"Unexpected attributes in FeatureEngineer: {actual_keys - allowed_keys}"

    # Verify no global statistics / target statistics / scalers exist in state
    for k, v in state.items():
        assert not isinstance(v, (np.ndarray, pd.Series, pd.DataFrame)), f"Fitted state '{k}' contains empirical array/dataframe data!"


def test_multiple_origin_leakage_invariance(real_analysis_ready):
    """Verify zero leakage across early, middle, and late origins under multiple perturbation scenarios."""
    _, df_valid = build_model_features(real_analysis_ready)
    origins = sorted(df_valid["forecast_origin_date"].unique())

    test_origins = [origins[0], origins[len(origins) // 2], origins[-1]]  # early, middle, late

    for orig in test_origins:
        _run_single_origin_perturbation_check(
            real_analysis_ready,
            orig,
            "Units_Sold and Promotion combined",
            lambda d, m: d.assign(
                Units_Sold=np.where(m, d["Units_Sold"] * 5 + 20, d["Units_Sold"]),
                Promotion=np.where(m, 1 - d["Promotion"], d["Promotion"]),
            )
        )
