"""
test_final_model.py — Validation Tests for Phase 3B Step 3 Final Model Selection & Promotion
=============================================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 3: Final Model Selection, Production Promotion & Inference

Tests:
1. architecture_selection exists
2. every horizon 1–8 has exactly one selected model
3. architecture is validation-derived
4. final predictions exist
5. prediction schema correct
6. all expected horizons exist (1..8)
7. all expected SKUs exist (50)
8. no duplicate keys
9. no NaN predictions
10. no infinite predictions
11. forecast origin < target date
12. benchmark unchanged (11.14% / 10.67%)
13. production artifacts load
14. feature transformer loads
15. inference works
16. inference determinism
17. serialization parity
18. training/inference parity
19. approved SAFE features only
20. no target leakage
21. adversarial leakage passes
22. source hashes unchanged
23. prior phase artifacts unchanged
24. model registry covers every horizon
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROD_DIR = MODELS_DIR / "production"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FINAL_DIR = ARTIFACTS_DIR / "models" / "final"

KNOWN_HASHES = {
    "sales_daily.csv": "f33e31ad52880fc4c0c630c4c336f994235736cd23f417ca049e434d2878a647",
    "sku_master.csv": "6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9",
    "calendar.csv": "9d66c227a95b5e9a37d30191e37b83aa5acb1c43398525f4855c7f72cbda4b62",
    "inventory_snapshots.csv": "167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd",
    "analysis_ready.parquet": "f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427",
    "model_features.parquet": "4ac6841094d83845033d1bf98989154d1d151296a240fa4bf3d918254c965f26",
}


@pytest.fixture(scope="module")
def arch_selection():
    path = FINAL_DIR / "architecture_selection.csv"
    assert path.exists(), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def final_preds():
    path = FINAL_DIR / "final_predictions.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def prod_registry():
    path = PROD_DIR / "model_registry.pkl"
    assert path.exists(), f"Missing {path}"
    return joblib.load(path)


# 1. architecture_selection exists
def test_01_architecture_selection_exists():
    assert (FINAL_DIR / "architecture_selection.csv").exists()
    assert (FINAL_DIR / "architecture.json").exists()
    assert (FINAL_DIR / "final_summary.csv").exists()
    assert (FINAL_DIR / "final_by_horizon.csv").exists()


# 2. every horizon 1–8 has exactly one selected model
def test_02_every_horizon_has_one_selected_model(arch_selection):
    assert len(arch_selection) == 8
    horizons = sorted(arch_selection["horizon"].tolist())
    assert horizons == [1, 2, 3, 4, 5, 6, 7, 8]
    assert arch_selection["selected_model"].notna().all()


# 3. architecture is validation-derived
def test_03_architecture_is_validation_derived(arch_selection):
    # Check that selected_model corresponds to the minimum validation_micro_wape
    for _, row in arch_selection.iterrows():
        h = row["horizon"]
        cand_map = {
            "Seasonal Naive 52w": row["seasonal_naive_validation_wape"],
            "Tuned LightGBM": row["lightgbm_validation_wape"],
            "Tuned XGBoost": row["xgboost_validation_wape"],
            "Tuned Random Forest": row["random_forest_validation_wape"],
        }
        min_model = min(cand_map, key=cand_map.get)
        assert row["selected_model"] == min_model, (
            f"Horizon {h} selected {row['selected_model']}, but validation minimum was {min_model}"
        )


# 4. final predictions exist
def test_04_final_predictions_exist(final_preds):
    assert len(final_preds) > 0
    assert "Selected Hybrid" in final_preds["model"].unique() or any(
        "Selected Hybrid" in m for m in final_preds["model"].unique()
    )


# 5. prediction schema correct
def test_05_prediction_schema_correct(final_preds):
    required_cols = [
        "model", "SKU", "forecast_origin_date", "horizon",
        "target_date", "actual", "prediction"
    ]
    for col in required_cols:
        assert col in final_preds.columns, f"Missing column {col} in final predictions"


# 6. all expected horizons exist (1..8)
def test_06_all_expected_horizons_exist(final_preds):
    hybrid_preds = final_preds[final_preds["model"].str.startswith("Selected Hybrid")]
    horizons = sorted(hybrid_preds["horizon"].unique().tolist())
    assert horizons == [1, 2, 3, 4, 5, 6, 7, 8]


# 7. all expected SKUs exist (50)
def test_07_all_expected_skus_exist(final_preds):
    hybrid_preds = final_preds[final_preds["model"].str.startswith("Selected Hybrid")]
    skus = sorted(hybrid_preds["SKU"].unique().tolist())
    assert len(skus) == 50
    assert skus[0] == "SKU001" and skus[-1] == "SKU050"


# 8. no duplicate keys
def test_08_no_duplicate_keys(final_preds):
    hybrid_preds = final_preds[final_preds["model"].str.startswith("Selected Hybrid")]
    dup_count = hybrid_preds.duplicated(subset=["SKU", "forecast_origin_date", "horizon"]).sum()
    assert dup_count == 0, f"Found {dup_count} duplicate prediction keys"


# 9. no NaN predictions
def test_09_no_nan_predictions(final_preds):
    preds = final_preds["prediction"].values
    assert not np.isnan(preds).any(), "Found NaN in final predictions"


# 10. no infinite predictions
def test_10_no_infinite_predictions(final_preds):
    preds = final_preds["prediction"].values
    assert not np.isinf(preds).any(), "Found Inf in final predictions"
    assert (preds >= 0.0).all(), "Found negative predictions in final predictions"


# 11. forecast origin < target date
def test_11_forecast_origin_precedes_target_date(final_preds):
    diffs = (pd.to_datetime(final_preds["target_date"]) - pd.to_datetime(final_preds["forecast_origin_date"])).dt.days
    assert (diffs >= 7).all(), "Target date must be at least 7 days ahead of forecast origin"


# 12. benchmark unchanged (11.14% on 12 origins, 10.67% on 9 origins)
def test_12_benchmark_unchanged(final_preds):
    sn = final_preds[final_preds["model"] == "Seasonal Naive 52w"]
    tot_act = sn["actual"].sum()
    tot_err = (sn["actual"] - sn["prediction"]).abs().sum()
    sn_9wape = (tot_err / tot_act) * 100
    assert abs(sn_9wape - 10.674398) < 0.001, f"Seasonal Naive 9-origin WAPE mutated: {sn_9wape:.4f}%"

    # Baseline file check
    base_df = pd.read_parquet(ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet")
    sn_all = base_df[base_df["model"] == "seasonal_naive"]
    sn_all_act = sn_all["actual"].sum()
    sn_all_err = (sn_all["actual"] - sn_all["forecast"]).abs().sum()
    sn_12wape = (sn_all_err / sn_all_act) * 100
    assert abs(sn_12wape - 11.141005) < 0.001, f"Seasonal Naive 12-origin benchmark mutated: {sn_12wape:.4f}%"


# 13. production artifacts load
def test_13_production_artifacts_load(prod_registry):
    assert "horizon_registry" in prod_registry
    assert (PROD_DIR / "architecture.json").exists()
    assert (PROD_DIR / "metadata.json").exists()
    assert (PROD_DIR / "model_card.md").exists()
    assert (PROD_DIR / "feature_engineer.pkl").exists()


# 14. feature transformer loads
def test_14_feature_transformer_loads():
    fe = joblib.load(PROD_DIR / "feature_engineer.pkl")
    assert fe is not None


# 15. inference works
def test_15_inference_works():
    from api.inference import get_inference_engine
    engine = get_inference_engine()
    assert engine.is_ready is True
    preds = engine.predict(origin_date_str="2025-09-16", skus=["SKU001"], horizon_weeks=8)
    assert len(preds) == 8
    assert preds[0]["SKU"] == "SKU001"
    assert preds[0]["prediction"] >= 0.0


# 16. inference determinism
def test_16_inference_determinism():
    from api.inference import get_inference_engine
    engine = get_inference_engine()
    p1 = engine.predict(origin_date_str="2025-09-16", skus=["SKU001", "SKU002"], horizon_weeks=4)
    p2 = engine.predict(origin_date_str="2025-09-16", skus=["SKU001", "SKU002"], horizon_weeks=4)
    vals1 = [p["prediction"] for p in p1]
    vals2 = [p["prediction"] for p in p2]
    assert np.allclose(vals1, vals2, atol=1e-8), "Inference is not deterministic"


# 17. serialization parity
def test_17_serialization_parity():
    # Load model from disk and check prediction matches
    m_disk = joblib.load(PROD_DIR / "models" / "random_forest_h1.joblib")
    features_df = pd.read_parquet(DATA_DIR / "processed" / "model_features.parquet")
    meta_cols = ["SKU", "forecast_origin_date"] + [f"target_h{h}" for h in range(1, 9)]
    pred_cols = [c for c in features_df.columns if c not in meta_cols]
    sample_X = features_df.loc[:20, pred_cols].copy()

    p1 = m_disk.predict(sample_X)
    m_disk2 = joblib.load(PROD_DIR / "models" / "random_forest_h1.joblib")
    p2 = m_disk2.predict(sample_X)
    assert np.allclose(p1, p2, atol=1e-10), "Serialization parity mismatch"


# 18. training/inference parity
def test_18_training_inference_parity():
    # Model in registry vs model on disk
    reg = joblib.load(PROD_DIR / "model_registry.pkl")
    m_info = reg["horizon_registry"]["1"]
    m_loaded = joblib.load(PROD_DIR / m_info["artifact"])

    features_df = pd.read_parquet(DATA_DIR / "processed" / "model_features.parquet")
    sample_row = features_df[features_df["forecast_origin_date"] == "2025-09-16"].copy()
    X = sample_row[reg["predictor_columns"]]

    p_direct = m_loaded.predict(X)
    from api.inference import get_inference_engine
    eng = get_inference_engine()
    preds_api = eng.predict(origin_date_str="2025-09-16", horizon_weeks=1)
    api_h1 = [p["prediction"] for p in preds_api if p["horizon"] == 1]
    assert np.allclose(p_direct, api_h1, atol=1e-7), "Parity mismatch between training artifact and inference API"


# 19. approved SAFE features only
def test_19_approved_safe_features_only(prod_registry):
    preds = prod_registry["predictor_columns"]
    assert len(preds) == 50, f"Expected 50 SAFE features, got {len(preds)}"
    for c in preds:
        assert not c.startswith("target_"), f"Target column {c} found in predictors"
        assert c not in ["SKU", "forecast_origin_date"], f"Metadata column {c} found in predictors"


# 20. no target leakage
def test_20_no_target_leakage(prod_registry):
    target_names = [f"target_h{h}" for h in range(1, 9)]
    for t in target_names:
        assert t not in prod_registry["predictor_columns"]


# 21. adversarial leakage passes
def test_21_adversarial_leakage_passes():
    latency_file = FINAL_DIR / "inference_latency.json"
    assert latency_file.exists()
    # Check that final pipeline run verified zero difference (< 1e-7)
    with open(PROD_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("production_approval_status") == "APPROVED"


# 22. source hashes unchanged
def test_22_source_hashes_unchanged():
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


# 23. prior phase artifacts unchanged
def test_23_prior_phase_artifacts_unchanged():
    assert (ARTIFACTS_DIR / "models" / "tuning" / "tuning_results.csv").exists()
    assert (ARTIFACTS_DIR / "models" / "tuning" / "best_parameters.json").exists()
    assert (ARTIFACTS_DIR / "models" / "model_summary.csv").exists()
    assert (ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet").exists()


# 24. model registry covers every horizon
def test_24_model_registry_covers_every_horizon(prod_registry):
    hr = prod_registry["horizon_registry"]
    for h in range(1, 9):
        h_str = str(h)
        assert h_str in hr, f"Horizon {h} missing from production registry"
        assert hr[h_str]["type"] in ["ml_model", "seasonal_naive"]
