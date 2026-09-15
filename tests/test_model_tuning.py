"""
test_model_tuning.py — Validation Tests for Phase 3B Step 2 Horizon-Aware Hyperparameter Tuning
================================================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 2: Horizon-Aware Hyperparameter Tuning

Covers:
1. tuning artifacts exist
2. best parameters exist for all required model/horizon pairs
3. all 8 horizons are represented
4. all 3 model families are represented
5. parameter values are valid
6. deterministic seed behavior
7. exactly 50 SAFE predictors are used
8. TARGET columns never enter X
9. metadata columns never enter X
10. validation remains temporal
11. final evaluation data does not influence tuning
12. prediction schema is correct
13. prediction count is correct
14. no duplicate prediction keys
15. no NaN/infinite predictions
16. forecast origin precedes target date
17. source data hashes unchanged
18. Phase 3A artifacts unchanged
19. Seasonal Naive benchmark remains unchanged
20. adversarial future perturbation produces zero leakage
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TUNING_DIR = ARTIFACTS_DIR / "models" / "tuning"

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
def tuned_predictions_df():
    path = ARTIFACTS_DIR / "models" / "tuned_predictions.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def best_params():
    path = TUNING_DIR / "best_parameters.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tuning_manifest():
    path = ARTIFACTS_DIR / "metrics" / "phase3b_tuning_manifest.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. tuning artifacts exist
def test_01_tuning_artifacts_exist():
    required_files = [
        TUNING_DIR / "tuning_results.csv",
        TUNING_DIR / "best_parameters.json",
        TUNING_DIR / "validation_predictions.parquet",
        TUNING_DIR / "tuning_manifest.json",
        ARTIFACTS_DIR / "models" / "tuned_predictions.parquet",
        ARTIFACTS_DIR / "models" / "tuned_summary.csv",
        ARTIFACTS_DIR / "models" / "tuned_by_horizon.csv",
        ARTIFACTS_DIR / "models" / "tuned_by_sku.csv",
        ARTIFACTS_DIR / "models" / "tuned_by_category.csv",
        ARTIFACTS_DIR / "metrics" / "phase3b_tuning_manifest.json",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required artifact: {rf}"
        assert rf.stat().st_size > 0, f"Empty artifact file: {rf}"

    # Tuned models directory structure
    for fam in ["lightgbm", "xgboost", "random_forest"]:
        fam_dir = MODELS_DIR / "tuned" / fam
        assert fam_dir.exists(), f"Missing tuned directory for {fam}"
        assert (fam_dir / "tuned_manifest.json").exists(), f"Missing tuned manifest for {fam}"
        for h in range(1, 9):
            model_file = fam_dir / f"model_h{h}.joblib"
            assert model_file.exists(), f"Missing {model_file}"
            assert model_file.stat().st_size > 0, f"Empty model file {model_file}"

    # Plots directory
    plot_dir = TUNING_DIR / "plots"
    assert plot_dir.exists(), f"Missing {plot_dir}"
    plots = list(plot_dir.glob("*.png"))
    assert len(plots) >= 5, f"Expected at least 5 plots, found {len(plots)}"


# 2. best parameters exist for all required model/horizon pairs
def test_02_best_parameters_exist_for_all_pairs(best_params):
    for fam in ["LightGBM", "XGBoost", "Random Forest"]:
        assert fam in best_params, f"Model family {fam} missing from best_parameters.json"
        for h in range(1, 9):
            h_str = str(h)
            assert h_str in best_params[fam], f"Horizon {h} missing for {fam}"
            entry = best_params[fam][h_str]
            assert "parameters" in entry, f"Parameters missing for {fam} h={h}"
            assert "validation_metrics" in entry, f"Validation metrics missing for {fam} h={h}"
            vm = entry["validation_metrics"]
            assert "Micro_WAPE" in vm and vm["Micro_WAPE"] > 0
            assert "Macro_WAPE" in vm and vm["Macro_WAPE"] > 0
            assert "MAE" in vm and vm["MAE"] > 0
            assert "RMSE" in vm and vm["RMSE"] > 0


# 3. all 8 horizons are represented
def test_03_all_8_horizons_represented(tuned_predictions_df):
    horizons = sorted(tuned_predictions_df["horizon"].unique().tolist())
    assert horizons == [1, 2, 3, 4, 5, 6, 7, 8], f"Expected horizons 1..8, got {horizons}"

    by_h_df = pd.read_csv(ARTIFACTS_DIR / "models" / "tuned_by_horizon.csv")
    for fam in ["Tuned LightGBM", "Tuned XGBoost", "Tuned Random Forest", "Seasonal Naive 52w"]:
        fam_horizons = sorted(by_h_df[by_h_df["model"] == fam]["horizon"].tolist())
        assert fam_horizons == [1, 2, 3, 4, 5, 6, 7, 8], f"Incomplete horizons for {fam}: {fam_horizons}"


# 4. all 3 model families are represented
def test_04_all_3_model_families_represented(tuned_predictions_df):
    models = tuned_predictions_df["model"].unique().tolist()
    assert "Tuned LightGBM" in models
    assert "Tuned XGBoost" in models
    assert "Tuned Random Forest" in models
    assert "Seasonal Naive 52w" in models


# 5. parameter values are valid
def test_05_parameter_values_are_valid(best_params):
    for h_str, entry in best_params["LightGBM"].items():
        p = entry["parameters"]
        assert p.get("learning_rate", 0.05) > 0, f"Invalid learning_rate for LGBM h={h_str}"
        assert p.get("num_leaves", 31) >= 10, f"Invalid num_leaves for LGBM h={h_str}"
        assert p.get("min_child_samples", 20) >= 5, f"Invalid min_child_samples for LGBM h={h_str}"

    for h_str, entry in best_params["XGBoost"].items():
        p = entry["parameters"]
        assert p.get("learning_rate", 0.05) > 0, f"Invalid learning_rate for XGB h={h_str}"
        assert p.get("max_depth", 6) in [3, 4, 5, 6, 7, 8], f"Invalid max_depth for XGB h={h_str}"
        assert p.get("min_child_weight", 1) >= 1, f"Invalid min_child_weight for XGB h={h_str}"

    for h_str, entry in best_params["Random Forest"].items():
        p = entry["parameters"]
        assert p.get("max_depth", 8) in [4, 5, 6, 8, 10, 12, None], f"Invalid max_depth for RF h={h_str}"
        assert p.get("min_samples_split", 2) >= 2, f"Invalid min_samples_split for RF h={h_str}"
        assert p.get("min_samples_leaf", 1) >= 1, f"Invalid min_samples_leaf for RF h={h_str}"


# 6. deterministic seed behavior
def test_06_deterministic_seed_behavior(features_df, best_params):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    sample_X = features_df.loc[:120, predictor_cols].copy()
    sample_y = features_df.loc[:120, "target_h1"].values

    for model_name, model_cls in [
        ("LightGBM", LightGBMForecaster),
        ("XGBoost", XGBoostForecaster),
        ("Random Forest", RandomForestForecaster),
    ]:
        p = best_params[model_name]["1"]["parameters"].copy()
        p["random_state"] = 42

        m1 = model_cls(params=p)
        m1.fit(sample_X, sample_y)
        preds1 = m1.predict(sample_X)

        m2 = model_cls(params=p)
        m2.fit(sample_X, sample_y)
        preds2 = m2.predict(sample_X)

        assert np.allclose(preds1, preds2, atol=1e-6), f"{model_name} non-deterministic under seed 42"


# 7. exactly 50 SAFE predictors are used
def test_07_exactly_50_safe_predictors_used(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]
    assert len(predictor_cols) == 50, f"Expected exactly 50 predictors, got {len(predictor_cols)}"


# 8. TARGET columns never enter X
def test_08_target_columns_never_enter_x(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    assert not any(c in predictor_cols for c in target_cols), "TARGET columns present in predictors"

    bad_X = features_df.loc[:20, predictor_cols + ["target_h1"]].copy()
    y = features_df.loc[:20, "target_h1"].values
    m = LightGBMForecaster()
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE: TARGET columns found"):
        m.fit(bad_X, y)


# 9. metadata columns never enter X
def test_09_metadata_columns_never_enter_x(features_df):
    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in features_df.columns if c not in target_cols and c not in meta_cols]

    assert not any(c in predictor_cols for c in meta_cols), "Metadata columns present in predictors"

    bad_X = features_df.loc[:20, predictor_cols + ["forecast_origin_date"]].copy()
    y = features_df.loc[:20, "target_h1"].values
    m = XGBoostForecaster()
    with pytest.raises(ValueError, match="CRITICAL: Metadata columns found"):
        m.fit(bad_X, y)


# 10. validation remains temporal
def test_10_validation_remains_temporal(tuning_manifest):
    val_origins = [pd.Timestamp(o) for o in tuning_manifest["validation_origins"]]
    assert len(val_origins) == 4, f"Expected 4 validation origins, got {len(val_origins)}"

    # Check validation origins are monotonically increasing
    for i in range(len(val_origins) - 1):
        assert val_origins[i] < val_origins[i + 1], "Validation origins are not strictly monotonically increasing"


# 11. final evaluation data does not influence tuning
def test_11_final_evaluation_data_does_not_influence_tuning(tuning_manifest):
    val_origins = set(tuning_manifest["validation_origins"])
    eval_origins = tuning_manifest["final_evaluation_origins"]

    # Final evaluation origins strictly include later periods not used in validation
    held_out_origins = set(eval_origins) - val_origins
    assert len(held_out_origins) == 5, f"Expected 5 held-out evaluation origins, got {len(held_out_origins)}"

    # Check validation predictions only contain the 4 validation origins
    val_preds_path = TUNING_DIR / "validation_predictions.parquet"
    assert val_preds_path.exists()
    val_preds = pd.read_parquet(val_preds_path)
    val_pred_origins = set(val_preds["forecast_origin_date"].astype(str).unique())
    assert val_pred_origins == val_origins, (
        f"Validation predictions contain unpermitted origins: {val_pred_origins - val_origins}"
    )


# 12. prediction schema is correct
def test_12_prediction_schema_correct(tuned_predictions_df):
    expected_cols = [
        "model", "SKU", "forecast_origin_date", "horizon",
        "target_date", "actual", "prediction", "Category", "Subcategory"
    ]
    for col in expected_cols:
        assert col in tuned_predictions_df.columns, f"Missing column {col} in tuned prediction table"


# 13. prediction count is correct
def test_13_prediction_count_correct(tuned_predictions_df):
    # 9 origins × 50 SKUs × 8 horizons = 3,600 predictions per model
    # 4 models (Tuned LightGBM, Tuned XGBoost, Tuned Random Forest, Seasonal Naive 52w) = 14,400 rows
    expected_rows = 9 * 50 * 8 * 4
    assert len(tuned_predictions_df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(tuned_predictions_df)}"
    )
    for model_name in tuned_predictions_df["model"].unique():
        sub_len = len(tuned_predictions_df[tuned_predictions_df["model"] == model_name])
        assert sub_len == 3600, f"Model {model_name} has {sub_len} predictions, expected 3,600"


# 14. no duplicate prediction keys
def test_14_no_duplicate_prediction_keys(tuned_predictions_df):
    key_cols = ["model", "SKU", "forecast_origin_date", "horizon"]
    duplicates = tuned_predictions_df.duplicated(subset=key_cols).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate prediction keys"


# 15. no NaN/infinite predictions
def test_15_no_nan_infinite_predictions(tuned_predictions_df):
    preds = tuned_predictions_df["prediction"].values
    assert not np.isnan(preds).any(), "Found NaN in predictions"
    assert not np.isinf(preds).any(), "Found Inf in predictions"
    assert (preds >= 0.0).all(), "Found negative predictions"


# 16. forecast origin precedes target date
def test_16_forecast_origin_precedes_target_date(tuned_predictions_df):
    origin = tuned_predictions_df["forecast_origin_date"]
    target = tuned_predictions_df["target_date"]
    diffs = (target - origin).dt.days
    assert (diffs > 0).all(), "Found target dates not strictly following forecast origin date"
    assert (diffs >= 7).all(), "Target dates must be at least 7 days ahead (h=1..8)"


# 17. source data hashes unchanged
def test_17_source_data_hashes_unchanged():
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


# 18. Phase 3A artifacts unchanged
def test_18_phase3a_artifacts_unchanged():
    feat_path = DATA_DIR / "processed" / "model_features.parquet"
    assert feat_path.exists()
    h = hashlib.sha256(feat_path.read_bytes()).hexdigest()
    assert h == KNOWN_HASHES["model_features.parquet"], f"model_features.parquet mutated: {h}"

    lineage_path = ARTIFACTS_DIR / "features" / "feature_lineage.json"
    assert lineage_path.exists(), "Missing Phase 3A feature_lineage.json"
    assert lineage_path.stat().st_size > 0, "Empty Phase 3A feature_lineage.json"

    stats_path = ARTIFACTS_DIR / "features" / "feature_statistics.csv"
    assert stats_path.exists(), "Missing Phase 3A feature_statistics.csv"
    assert stats_path.stat().st_size > 0, "Empty Phase 3A feature_statistics.csv"


# 19. Seasonal Naive benchmark remains unchanged
def test_19_seasonal_naive_benchmark_unchanged(tuned_predictions_df):
    sn_preds = tuned_predictions_df[tuned_predictions_df["model"] == "Seasonal Naive 52w"]
    total_actual = sn_preds["actual"].sum()
    total_abs_err = (sn_preds["actual"] - sn_preds["prediction"]).abs().sum()
    micro_wape = (total_abs_err / total_actual) * 100

    # Benchmark on identical 9 origins is exactly 10.674%
    assert abs(micro_wape - 10.674398) < 0.001, f"Seasonal Naive 9-origin WAPE mutated: {micro_wape:.4f}%"

    # Phase 2B baseline forecasts file check
    base_df = pd.read_parquet(ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet")
    sn_all = base_df[base_df["model"] == "seasonal_naive"]
    sn_all_actual = sn_all["actual"].sum()
    sn_all_err = (sn_all["actual"] - sn_all["forecast"]).abs().sum()
    overall_wape = (sn_all_err / sn_all_actual) * 100
    assert abs(overall_wape - 11.141005) < 0.001, f"Seasonal Naive 12-origin benchmark mutated: {overall_wape:.4f}%"


# 20. adversarial future perturbation produces zero leakage
def test_20_adversarial_future_perturbation_zero_leakage(tuning_manifest):
    audit_results = tuning_manifest["adversarial_leakage_audit"]
    assert len(audit_results) == 9, f"Expected 9 audit checks, got {len(audit_results)}"
    for check in audit_results:
        assert check["audit_passed"] is True, f"Leakage audit failed: {check}"
        assert check["max_absolute_difference"] < 1e-7, (
            f"Leakage difference too high: {check['max_absolute_difference']} for {check}"
        )
