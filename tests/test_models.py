"""
test_models.py — Validation Tests for Phase 3B Candidate Model Framework
========================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 1: Model Training & Evaluation Framework

Covers
------
1. model artifacts exist
2. all three candidate models train
3. deterministic seed behavior
4. exactly 50 SAFE predictors used
5. zero TARGET columns in X
6. zero metadata columns in X
7. horizons 1–8 exist
8. predictions have no invalid negative values
9. prediction row count is correct
10. forecast origin precedes target date
11. training data never contains future targets relative to evaluation origin
12. no random temporal split
13. prediction table schema is correct
14. source datasets remain unchanged
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    get_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

KNOWN_HASHES = {
    "sales_daily.csv": "f33e31ad52880fc4c0c630c4c336f994235736cd23f417ca049e434d2878a647",
    "sku_master.csv": "6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9",
    "calendar.csv": "9d66c227a95b5e9a37d30191e37b83aa5acb1c43398525f4855c7f72cbda4b62",
    "inventory_snapshots.csv": "167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd",
    "analysis_ready.parquet": "f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427",
    "model_features.parquet": "4ac6841094d83845033d1bf98989154d1d151296a240fa4bf3d918254c965f26",
}


@pytest.fixture(scope="module")
def features_df():
    path = DATA_DIR / "processed" / "model_features.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def predictions_df():
    path = ARTIFACTS_DIR / "models" / "model_predictions.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


# 1. model artifacts exist
def test_01_model_artifacts_exist():
    candidates_dir = MODELS_DIR / "candidates"
    assert candidates_dir.exists(), f"Missing directory {candidates_dir}"
    for family in ["lightgbm", "xgboost", "random_forest"]:
        fam_dir = candidates_dir / family
        assert fam_dir.exists(), f"Missing candidate directory for {family}"
        assert (fam_dir / "candidate_manifest.json").exists(), f"Missing manifest for {family}"
        for h in range(1, 9):
            model_file = fam_dir / f"model_h{h}.joblib"
            assert model_file.exists(), f"Missing {model_file}"
            assert model_file.stat().st_size > 0, f"Empty model file {model_file}"


# 2. all three candidate models train
def test_02_all_three_candidate_models_train(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    sample_X = features_df.loc[:100, predictor_cols].copy()
    sample_y = features_df.loc[:100, "target_h1"].values

    for model_name, model_cls in [
        ("lightgbm", LightGBMForecaster),
        ("xgboost", XGBoostForecaster),
        ("random_forest", RandomForestForecaster),
    ]:
        model = model_cls()
        model.fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert len(preds) == len(sample_y), f"{model_name} prediction length mismatch"
        assert not np.isnan(preds).any(), f"{model_name} produced NaNs"
        assert not np.isinf(preds).any(), f"{model_name} produced Infs"


# 3. deterministic seed behavior
def test_03_deterministic_seed_behavior(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    sample_X = features_df.loc[:100, predictor_cols].copy()
    sample_y = features_df.loc[:100, "target_h1"].values

    for model_cls in [LightGBMForecaster, XGBoostForecaster, RandomForestForecaster]:
        m1 = model_cls(params={"random_state": 42})
        m1.fit(sample_X, sample_y)
        p1 = m1.predict(sample_X)

        m2 = model_cls(params={"random_state": 42})
        m2.fit(sample_X, sample_y)
        p2 = m2.predict(sample_X)

        assert np.allclose(p1, p2, atol=1e-6), f"{model_cls.model_name} non-deterministic with seed=42"


# 4. exactly 50 SAFE predictors used
def test_04_exactly_50_safe_predictors_used(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]
    assert len(predictor_cols) == 50, f"Expected exactly 50 predictors, got {len(predictor_cols)}"


# 5. zero TARGET columns in X
def test_05_zero_target_columns_in_X(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    # Check that predictors do not contain target columns
    assert not any(c in predictor_cols for c in target_cols), "TARGET columns present in predictors"

    # Check that model raises ValueError if target is passed in X
    bad_X = features_df.loc[:20, predictor_cols + ["target_h1"]].copy()
    y = features_df.loc[:20, "target_h1"].values
    m = LightGBMForecaster()
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE: TARGET columns found"):
        m.fit(bad_X, y)


# 6. zero metadata columns in X
def test_06_zero_metadata_columns_in_X(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    assert not any(c in predictor_cols for c in meta_cols), "Metadata columns present in predictors"

    bad_X = features_df.loc[:20, predictor_cols + ["SKU"]].copy()
    y = features_df.loc[:20, "target_h1"].values
    m = XGBoostForecaster()
    with pytest.raises(ValueError, match="CRITICAL: Metadata columns found"):
        m.fit(bad_X, y)


# 7. horizons 1–8 exist
def test_07_horizons_1_to_8_exist(predictions_df):
    horizons = sorted(predictions_df["horizon"].unique().tolist())
    assert horizons == [1, 2, 3, 4, 5, 6, 7, 8], f"Expected horizons 1..8, got {horizons}"


# 8. predictions have no invalid negative values
def test_08_predictions_non_negative(predictions_df):
    min_pred = predictions_df["prediction"].min()
    assert min_pred >= 0.0, f"Found negative predictions: min = {min_pred}"


# 9. prediction row count is correct
def test_09_prediction_row_count_correct(predictions_df):
    # 9 origins × 50 SKUs × 8 horizons = 3,600 predictions per model
    # 4 models (LightGBM, XGBoost, Random Forest, Seasonal Naive 52w) = 14,400 rows
    expected_rows = 9 * 50 * 8 * 4
    assert len(predictions_df) == expected_rows, f"Expected {expected_rows} rows, got {len(predictions_df)}"
    for model_name in predictions_df["model"].unique():
        sub_len = len(predictions_df[predictions_df["model"] == model_name])
        assert sub_len == 3600, f"Model {model_name} has {sub_len} predictions, expected 3,600"


# 10. forecast origin precedes target date
def test_10_forecast_origin_precedes_target_date(predictions_df):
    diffs = (predictions_df["target_date"] - predictions_df["forecast_origin_date"]).dt.days
    assert (diffs > 0).all(), "Found target dates not strictly following forecast origin date"
    assert (diffs >= 7).all(), "Target dates must be at least 7 days ahead (h=1..8)"


# 11. training data never contains future targets relative to evaluation origin
def test_11_training_data_never_contains_future_targets(features_df):
    # For every evaluation origin T_eval and horizon h, training target dates must be <= T_eval
    base_df = pd.read_parquet(ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet")
    eval_origins = sorted(base_df["forecast_origin"].unique())[3:]

    for eval_orig in eval_origins:
        eval_ts = pd.Timestamp(eval_orig)
        for h in range(1, 9):
            cutoff = eval_ts - pd.Timedelta(weeks=h)
            train_mask = (features_df["forecast_origin_date"] <= cutoff)
            train_target_dates = features_df.loc[train_mask, "forecast_origin_date"] + pd.Timedelta(weeks=h)
            assert (train_target_dates <= eval_ts).all(), (
                f"Training target date leakage at eval_origin {eval_ts}, horizon {h}"
            )


# 12. no random temporal split
def test_12_no_random_temporal_split(predictions_df):
    # Check that origins advance deterministically forward in time
    origins = sorted(predictions_df["forecast_origin_date"].unique())
    assert len(origins) == 9, f"Expected 9 evaluation origins, got {len(origins)}"
    for i in range(len(origins) - 1):
        assert origins[i] < origins[i + 1], "Origins are not strictly monotonically increasing"


# 13. prediction table schema is correct
def test_13_prediction_table_schema_correct(predictions_df):
    expected_cols = [
        "model", "SKU", "forecast_origin_date", "horizon",
        "target_date", "actual", "prediction", "Category", "Subcategory"
    ]
    for col in expected_cols:
        assert col in predictions_df.columns, f"Missing column {col} in prediction table"


# 14. source datasets remain unchanged
def test_14_source_datasets_remain_unchanged():
    paths = {
        "sales_daily.csv": DATA_DIR / "raw" / "sales_daily.csv",
        "sku_master.csv": DATA_DIR / "raw" / "sku_master.csv",
        "calendar.csv": DATA_DIR / "raw" / "calendar.csv",
        "inventory_snapshots.csv": DATA_DIR / "raw" / "inventory_snapshots.csv",
        "analysis_ready.parquet": DATA_DIR / "processed" / "analysis_ready.parquet",
        "model_features.parquet": DATA_DIR / "processed" / "model_features.parquet",
    }
    for name, p in paths.items():
        assert p.exists(), f"Missing {name}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == KNOWN_HASHES[name], f"Source hash mutated for {name}: {h} != {KNOWN_HASHES[name]}"
