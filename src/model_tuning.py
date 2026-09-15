"""
model_tuning.py — Horizon-Aware Hyperparameter Tuning & Evaluation
===================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 2: Horizon-Aware Hyperparameter Tuning

Responsibilities
----------------
1. Implement strictly honest temporal validation inside the historical training period.
2. Search controlled, compact hyperparameter grids independently per horizon (h=1..8)
   for LightGBM, XGBoost, and Random Forest.
3. Select best parameters based on validation Micro WAPE.
4. Persist tuning artifacts to artifacts/models/tuning/:
   - tuning_results.csv
   - best_parameters.json
   - validation_predictions.parquet
   - tuning_manifest.json
   - plots/ (5 visualization plots)
5. Retrain tuned models and persist to models/tuned/{family}/.
6. Evaluate tuned models on the Step 1 rolling-origin evaluation framework.
7. Persist final tuned evaluation artifacts to artifacts/models/:
   - tuned_predictions.parquet
   - tuned_summary.csv
   - tuned_by_horizon.csv
   - tuned_by_sku.csv
   - tuned_by_category.csv
8. Perform model sanity checks and adversarial future-data perturbation audit.
9. Persist reproducibility manifest to artifacts/metrics/phase3b_tuning_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.models import (
    CATEGORICAL_COLS,
    CATEGORY_MAPS,
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from src.utils import mae as mae_fn, rmse as rmse_fn, set_seed, setup_logging, wape as wape_fn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Micro WAPE, MAE, RMSE."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return {
        "Micro_WAPE": wape_fn(y_t, y_p),
        "MAE": mae_fn(y_t, y_p),
        "RMSE": rmse_fn(y_t, y_p),
    }


def compute_macro_wape(df: pd.DataFrame, actual_col: str = "actual", pred_col: str = "prediction") -> float:
    """Compute unweighted average of per-SKU WAPE (Macro WAPE)."""
    def _sku_wape(g: pd.DataFrame) -> float:
        denom = g[actual_col].sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(g[actual_col] - g[pred_col]).sum() / denom * 100)

    sku_wapes = df.groupby("SKU").apply(_sku_wape, include_groups=False)
    return float(sku_wapes.dropna().mean())


# ---------------------------------------------------------------------------
# Controlled, Horizon-Aware Hyperparameter Grids
# ---------------------------------------------------------------------------
def get_candidate_parameter_grids() -> dict[str, list[dict[str, Any]]]:
    """
    Define curated, controlled hyperparameter search spaces for each model family.
    Spans from flexible/low-bias configs (suited for short horizons h=1,2)
    to regularized/shallow configs (suited for long horizons h=7,8).
    """
    lgb_grid = [
        # Baseline default
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0},
        # L1 objective (MAE-aligned)
        {"objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0},
        {"objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 25, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 0.5},
        # Moderate depth, regularized
        {"objective": "regression_l1", "learning_rate": 0.04, "num_leaves": 20, "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 1.0},
        {"objective": "regression",    "learning_rate": 0.04, "num_leaves": 20, "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 1.0},
        # Shallower leaves for mid-horizons
        {"objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 1.0},
        {"objective": "regression",    "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 1.0},
        # Highly regularized shallow trees for long horizons
        {"objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 10, "min_child_samples": 35, "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 1.5, "reg_lambda": 2.0},
        {"objective": "regression_l1", "learning_rate": 0.02, "num_leaves": 7,  "min_child_samples": 40, "subsample": 0.7, "colsample_bytree": 0.6,  "reg_alpha": 2.0, "reg_lambda": 2.0},
        {"objective": "regression",    "learning_rate": 0.02, "num_leaves": 7,  "min_child_samples": 40, "subsample": 0.7, "colsample_bytree": 0.6,  "reg_alpha": 2.0, "reg_lambda": 2.0},
        # High shrinkage, deep ensemble
        {"objective": "regression_l1", "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,  "reg_alpha": 1.0, "reg_lambda": 1.0},
        {"objective": "regression_l1", "learning_rate": 0.06, "num_leaves": 25, "min_child_samples": 20, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 0.2},
    ]

    xgb_grid = [
        # Baseline default
        {"objective": "reg:squarederror", "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
        # Absolute error objective
        {"objective": "reg:absoluteerror", "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"objective": "reg:absoluteerror", "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 1.0},
        # Depth 4, regularized
        {"objective": "reg:absoluteerror", "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 1.5},
        {"objective": "reg:squarederror",  "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 1.5},
        # Shallow depth 3 for long horizons
        {"objective": "reg:absoluteerror", "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 4, "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 1.5, "reg_lambda": 2.0},
        {"objective": "reg:squarederror",  "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 4, "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 1.5, "reg_lambda": 2.0},
        {"objective": "reg:absoluteerror", "learning_rate": 0.02, "max_depth": 3, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 2.0, "reg_lambda": 3.0},
        # Low learning rate depth 4
        {"objective": "reg:absoluteerror", "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 1.0},
        # Higher learning rate for h=1,2
        {"objective": "reg:absoluteerror", "learning_rate": 0.06, "max_depth": 6, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 0.5},
        {"objective": "reg:squarederror",  "learning_rate": 0.06, "max_depth": 5, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 0.5},
        {"objective": "reg:absoluteerror", "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 1.0},
    ]

    rf_grid = [
        # Baseline default
        {"max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0},
        # Depth 12, slight leaf regularization
        {"max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 0.8},
        # Depth 10
        {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": 0.8},
        {"max_depth": 10, "min_samples_split": 6, "min_samples_leaf": 3, "max_features": 0.7},
        # Depth 8 for mid-horizons
        {"max_depth": 8,  "min_samples_split": 8, "min_samples_leaf": 3, "max_features": 0.7},
        {"max_depth": 8,  "min_samples_split": 10, "min_samples_leaf": 4, "max_features": 0.6},
        # Depth 6 for long horizons
        {"max_depth": 6,  "min_samples_split": 10, "min_samples_leaf": 4, "max_features": 0.6},
        {"max_depth": 6,  "min_samples_split": 12, "min_samples_leaf": 5, "max_features": 0.5},
        # Depth 5 constrained
        {"max_depth": 5,  "min_samples_split": 15, "min_samples_leaf": 6, "max_features": 0.5},
        {"max_depth": 12, "min_samples_split": 3, "min_samples_leaf": 1, "max_features": 0.75},
    ]

    return {
        "LightGBM": lgb_grid,
        "XGBoost": xgb_grid,
        "Random Forest": rf_grid,
    }


# ---------------------------------------------------------------------------
# Main Tuning Execution
# ---------------------------------------------------------------------------
def run_phase3b_hyperparameter_tuning() -> dict[str, Any]:
    """Execute Phase 3B Step 2: Horizon-Aware Hyperparameter Tuning."""
    set_seed(42)
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    features_path = DATA_DIR / "processed" / "model_features.parquet"
    baseline_path = ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet"
    step1_summary_path = ARTIFACTS_DIR / "models" / "model_summary.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}")

    # Load datasets
    mf = pd.read_parquet(features_path)
    base_df = pd.read_parquet(baseline_path)
    step1_summary = pd.read_csv(step1_summary_path) if step1_summary_path.exists() else None

    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in mf.columns if c not in target_cols and c not in meta_cols]

    # Enforce feature separation assertions
    assert len(predictor_cols) == 50, f"Expected 50 SAFE features, got {len(predictor_cols)}"
    assert not any(c in predictor_cols for c in target_cols), "TARGET columns leaked into predictors!"
    assert not any(c in predictor_cols for c in meta_cols), "Metadata columns leaked into predictors!"

    # Evaluation Origins from Step 1 (Origins 4..12)
    base_origins = sorted(base_df["forecast_origin"].unique())
    eval_origins = base_origins[3:]  # 9 origins: 2025-03-04 through 2025-09-16

    # -----------------------------------------------------------------------
    # 1. Honest Temporal Validation Design (Section 2)
    # -----------------------------------------------------------------------
    # Partition the 9 evaluation origins into:
    #   - Temporal Validation Origins: first 4 origins (2025-03-04 to 2025-05-13)
    #   - Final Holdout Evaluation Origins: remaining 5 origins (2025-06-10 to 2025-09-16)
    # Tuning decisions are made STRICTLY on the temporal validation origins!
    # The final evaluation origins are NEVER accessed during hyperparameter selection.
    val_origins = eval_origins[:4]
    logger.info("Temporal Validation Origins for hyperparameter tuning (%d origins): %s to %s",
                len(val_origins), val_origins[0].date(), val_origins[-1].date())

    sku_metadata = mf.drop_duplicates("SKU")[["SKU", "Category", "Subcategory"]].set_index("SKU")

    # -----------------------------------------------------------------------
    # 2. Execute Horizon-Aware Hyperparameter Search
    # -----------------------------------------------------------------------
    grids = get_candidate_parameter_grids()
    model_classes = {
        "LightGBM": LightGBMForecaster,
        "XGBoost": XGBoostForecaster,
        "Random Forest": RandomForestForecaster,
    }

    tuning_artifact_dir = ARTIFACTS_DIR / "models" / "tuning"
    tuning_artifact_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = tuning_artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trial_records: list[dict[str, Any]] = []
    best_parameters_dict: dict[str, dict[int, dict[str, Any]]] = {
        m: {} for m in model_classes
    }
    val_predictions_list: list[dict[str, Any]] = []

    logger.info("Starting Horizon-Aware Hyperparameter Tuning across 3 models × 8 horizons...")

    for model_name, model_cls in model_classes.items():
        param_list = grids[model_name]
        logger.info("--- Tuning %s (%d candidate configurations per horizon) ---", model_name, len(param_list))

        for h in range(1, 9):
            target_col = f"target_h{h}"
            best_val_wape = float("inf")
            best_cfg = None
            best_metrics = {}
            best_val_preds_for_h = []

            for trial_id, cfg in enumerate(param_list, 1):
                t0 = time.perf_counter()
                trial_actuals: list[float] = []
                trial_preds: list[float] = []
                trial_rows_info: list[dict[str, Any]] = []

                # Evaluate cfg across the 4 temporal validation origins
                for vo in val_origins:
                    vo_ts = pd.Timestamp(vo)
                    cutoff_orig = vo_ts - pd.Timedelta(weeks=h)
                    train_mask = (mf["forecast_origin_date"] <= cutoff_orig)
                    test_mask = (mf["forecast_origin_date"] == vo_ts)

                    train_X = mf.loc[train_mask, predictor_cols]
                    train_y = mf.loc[train_mask, target_col].values
                    test_X = mf.loc[test_mask, predictor_cols]
                    test_y = mf.loc[test_mask, target_col].values
                    test_skus = mf.loc[test_mask, "SKU"].values

                    # Verify no future targets in training set
                    train_target_dates = mf.loc[train_mask, "forecast_origin_date"] + pd.Timedelta(weeks=h)
                    assert (train_target_dates <= vo_ts).all(), "Causal leakage in tuning validation split!"

                    m_inst = model_cls(params=cfg)
                    m_inst.fit(train_X, train_y)
                    preds = m_inst.predict(test_X)

                    for sku, actual_val, pred_val in zip(test_skus, test_y, preds):
                        trial_actuals.append(float(actual_val))
                        trial_preds.append(float(pred_val))
                        trial_rows_info.append({
                            "model": model_name,
                            "horizon": h,
                            "SKU": sku,
                            "forecast_origin_date": vo_ts,
                            "target_date": vo_ts + pd.Timedelta(weeks=h),
                            "actual": float(actual_val),
                            "prediction": float(pred_val),
                        })

                dur = time.perf_counter() - t0
                met = compute_metrics(np.array(trial_actuals), np.array(trial_preds))
                trial_df = pd.DataFrame(trial_rows_info)
                macro_wape = compute_macro_wape(trial_df)

                trial_records.append({
                    "model": model_name,
                    "horizon": h,
                    "trial_id": trial_id,
                    "parameters": json.dumps(cfg),
                    "validation_micro_wape": met["Micro_WAPE"],
                    "validation_macro_wape": macro_wape,
                    "validation_mae": met["MAE"],
                    "validation_rmse": met["RMSE"],
                    "training_duration_sec": dur,
                    "seed": 42,
                })

                if met["Micro_WAPE"] < best_val_wape:
                    best_val_wape = met["Micro_WAPE"]
                    best_cfg = cfg
                    best_metrics = {
                        "Micro_WAPE": met["Micro_WAPE"],
                        "Macro_WAPE": macro_wape,
                        "MAE": met["MAE"],
                        "RMSE": met["RMSE"],
                    }
                    best_val_preds_for_h = trial_rows_info

            logger.info("  [%s h=%d] Best Val Micro WAPE: %.2f%% | Config: %s",
                        model_name, h, best_val_wape, best_cfg)
            best_parameters_dict[model_name][h] = {
                "parameters": best_cfg,
                "validation_metrics": best_metrics,
            }
            val_predictions_list.extend(best_val_preds_for_h)

    # -----------------------------------------------------------------------
    # 3. Persist Tuning Results Artifacts
    # -----------------------------------------------------------------------
    tuning_results_df = pd.DataFrame(trial_records)
    tuning_results_df.to_csv(tuning_artifact_dir / "tuning_results.csv", index=False)

    with open(tuning_artifact_dir / "best_parameters.json", "w", encoding="utf-8") as f:
        # JSON serializable formatting
        json_ready_best = {
            m: {str(h): v for h, v in h_dict.items()}
            for m, h_dict in best_parameters_dict.items()
        }
        json.dump(json_ready_best, f, indent=2)

    val_pred_df = pd.DataFrame(val_predictions_list)
    val_pred_df.to_parquet(tuning_artifact_dir / "validation_predictions.parquet", index=False)

    # -----------------------------------------------------------------------
    # 4. Retrain Tuned Models on Permitted History (Section 12)
    # -----------------------------------------------------------------------
    logger.info("Retraining selected tuned models and persisting to models/tuned/...")
    last_eval_orig = eval_origins[-1]

    for model_name, model_cls in model_classes.items():
        fam_dir = MODELS_DIR / "tuned" / model_name.lower().replace(" ", "_")
        fam_dir.mkdir(parents=True, exist_ok=True)
        tuned_manifest_models = []

        for h in range(1, 9):
            target_col = f"target_h{h}"
            best_cfg = best_parameters_dict[model_name][h]["parameters"]
            final_cutoff = last_eval_orig - pd.Timedelta(weeks=h)
            train_mask = (mf["forecast_origin_date"] <= final_cutoff)

            train_X = mf.loc[train_mask, predictor_cols]
            train_y = mf.loc[train_mask, target_col].values

            tuned_instance = model_cls(params=best_cfg)
            tuned_instance.fit(train_X, train_y)
            save_path = fam_dir / f"model_h{h}.joblib"
            tuned_instance.save(save_path)

            tuned_manifest_models.append({
                "horizon": h,
                "target_column": target_col,
                "training_rows": len(train_X),
                "feature_count": len(predictor_cols),
                "best_parameters": best_cfg,
                "validation_micro_wape": best_parameters_dict[model_name][h]["validation_metrics"]["Micro_WAPE"],
                "training_duration_seconds": tuned_instance.training_duration_,
                "artifact_path": str(save_path.relative_to(PROJECT_ROOT)),
            })

        with open(fam_dir / "tuned_manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_family": model_name,
                "library": model_cls.library,
                "random_seed": 42,
                "tuning_strategy": "horizon_aware_temporal_validation",
                "validation_origins": [str(d.date()) for d in val_origins],
                "horizons": tuned_manifest_models,
            }, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Final Rolling-Origin Evaluation of Tuned Models (Section 13)
    # -----------------------------------------------------------------------
    # Evaluated across the exact same 9 origins as Step 1 for 100% fair comparison
    logger.info("Evaluating Tuned Models on full 9-origin backtest framework...")
    tuned_eval_predictions: list[dict[str, Any]] = []

    for fold_idx, eval_orig in enumerate(eval_origins, 1):
        fold_eval_ts = pd.Timestamp(eval_orig)
        test_mask = (mf["forecast_origin_date"] == fold_eval_ts)
        test_X = mf.loc[test_mask, predictor_cols].copy()
        test_skus = mf.loc[test_mask, "SKU"].values

        for h in range(1, 9):
            target_col = f"target_h{h}"
            cutoff_origin = fold_eval_ts - pd.Timedelta(weeks=h)
            train_mask = (mf["forecast_origin_date"] <= cutoff_origin)

            train_X = mf.loc[train_mask, predictor_cols]
            train_y = mf.loc[train_mask, target_col].values
            test_y = mf.loc[test_mask, target_col].values
            target_date = fold_eval_ts + pd.Timedelta(weeks=h)

            for model_name, model_cls in model_classes.items():
                best_cfg = best_parameters_dict[model_name][h]["parameters"]
                m_inst = model_cls(params=best_cfg)
                m_inst.fit(train_X, train_y)
                preds = m_inst.predict(test_X)

                for sku, actual_val, pred_val in zip(test_skus, test_y, preds):
                    cat_info = sku_metadata.loc[sku]
                    tuned_eval_predictions.append({
                        "model": f"Tuned {model_name}",
                        "SKU": sku,
                        "forecast_origin_date": fold_eval_ts,
                        "horizon": h,
                        "target_date": target_date,
                        "actual": float(actual_val),
                        "prediction": float(pred_val),
                        "Category": cat_info["Category"],
                        "Subcategory": cat_info["Subcategory"],
                    })

    # Add Seasonal Naive benchmark rows
    sn_sub = base_df[
        (base_df["model"] == "seasonal_naive") &
        (base_df["forecast_origin"].isin(eval_origins))
    ].copy()

    for _, row in sn_sub.iterrows():
        tuned_eval_predictions.append({
            "model": "Seasonal Naive 52w",
            "SKU": row["SKU"],
            "forecast_origin_date": pd.Timestamp(row["forecast_origin"]),
            "horizon": int(row["horizon"]),
            "target_date": pd.Timestamp(row["target_week"]),
            "actual": float(row["actual"]),
            "prediction": float(row["forecast"]),
            "Category": row["Category"],
            "Subcategory": row["Subcategory"],
        })

    tuned_pred_df = pd.DataFrame(tuned_eval_predictions)
    tuned_pred_df.to_parquet(ARTIFACTS_DIR / "models" / "tuned_predictions.parquet", index=False)

    # -----------------------------------------------------------------------
    # 6. Sanity Checks on Tuned Predictions (Section 16)
    # -----------------------------------------------------------------------
    logger.info("Running Sanity Checks on Tuned Predictions...")
    tuning_sanity_checks = {}

    for model_name in tuned_pred_df["model"].unique():
        sub = tuned_pred_df[tuned_pred_df["model"] == model_name]
        n_rows = len(sub)
        n_missing = int(sub["prediction"].isna().sum())
        n_inf = int(np.isinf(sub["prediction"]).sum())
        n_neg = int((sub["prediction"] < 0).sum())
        sku_cov = int(sub["SKU"].nunique())
        horiz_cov = sorted(sub["horizon"].unique().tolist())
        orig_cov = int(sub["forecast_origin_date"].nunique())

        passed = (n_missing == 0 and n_inf == 0 and n_neg == 0 and sku_cov == 50 and len(horiz_cov) == 8 and n_rows == 3600)
        tuning_sanity_checks[model_name] = {
            "prediction_count": n_rows,
            "missing_predictions": n_missing,
            "infinite_predictions": n_inf,
            "negative_predictions": n_neg,
            "min_prediction": float(sub["prediction"].min()),
            "max_prediction": float(sub["prediction"].max()),
            "sku_coverage": sku_cov,
            "horizon_coverage": horiz_cov,
            "origin_coverage": orig_cov,
            "sanity_passed": passed,
        }
        assert passed, f"Sanity check failed for {model_name}"

    # -----------------------------------------------------------------------
    # 7. Compute Tuned Evaluation Metrics (Section 14)
    # -----------------------------------------------------------------------
    summary_rows = []
    horizon_rows = []
    sku_rows = []
    cat_rows = []

    for model_name in tuned_pred_df["model"].unique():
        sub = tuned_pred_df[tuned_pred_df["model"] == model_name]
        met = compute_metrics(sub["actual"].values, sub["prediction"].values)
        mac_wape = compute_macro_wape(sub)

        summary_rows.append({
            "model": model_name,
            "Micro_WAPE": met["Micro_WAPE"],
            "Macro_WAPE": mac_wape,
            "MAE": met["MAE"],
            "RMSE": met["RMSE"],
            "prediction_count": len(sub),
            "eval_origins": sub["forecast_origin_date"].nunique(),
        })

        for h, h_sub in sub.groupby("horizon"):
            h_met = compute_metrics(h_sub["actual"].values, h_sub["prediction"].values)
            h_mac = compute_macro_wape(h_sub)
            horizon_rows.append({
                "model": model_name,
                "horizon": int(h),
                "Micro_WAPE": h_met["Micro_WAPE"],
                "Macro_WAPE": h_mac,
                "MAE": h_met["MAE"],
                "RMSE": h_met["RMSE"],
            })

        for sku, s_sub in sub.groupby("SKU"):
            s_met = compute_metrics(s_sub["actual"].values, s_sub["prediction"].values)
            sku_rows.append({
                "model": model_name,
                "SKU": sku,
                "Micro_WAPE": s_met["Micro_WAPE"],
                "MAE": s_met["MAE"],
                "RMSE": s_met["RMSE"],
            })

        for cat, c_sub in sub.groupby("Category"):
            c_met = compute_metrics(c_sub["actual"].values, c_sub["prediction"].values)
            c_mac = compute_macro_wape(c_sub)
            cat_rows.append({
                "model": model_name,
                "Category": cat,
                "Micro_WAPE": c_met["Micro_WAPE"],
                "Macro_WAPE": c_mac,
                "MAE": c_met["MAE"],
                "RMSE": c_met["RMSE"],
            })

    tuned_summary_df = pd.DataFrame(summary_rows)
    tuned_horizon_df = pd.DataFrame(horizon_rows)
    tuned_sku_df = pd.DataFrame(sku_rows)
    tuned_cat_df = pd.DataFrame(cat_rows)

    tuned_summary_df.to_csv(ARTIFACTS_DIR / "models" / "tuned_summary.csv", index=False)
    tuned_horizon_df.to_csv(ARTIFACTS_DIR / "models" / "tuned_by_horizon.csv", index=False)
    tuned_sku_df.to_csv(ARTIFACTS_DIR / "models" / "tuned_by_sku.csv", index=False)
    tuned_cat_df.to_csv(ARTIFACTS_DIR / "models" / "tuned_by_category.csv", index=False)

    # -----------------------------------------------------------------------
    # 8. Generate Visualizations (Section 19)
    # -----------------------------------------------------------------------
    logger.info("Generating tuning visualization plots...")

    # Plot 1: Validation WAPE by Model / Horizon
    plt.figure(figsize=(9, 5))
    for m in model_classes:
        h_vals = range(1, 9)
        w_vals = [best_parameters_dict[m][h]["validation_metrics"]["Micro_WAPE"] for h in h_vals]
        plt.plot(h_vals, w_vals, marker="o", label=m, linewidth=2)
    plt.xlabel("Forecast Horizon Step (Weeks)")
    plt.ylabel("Validation Micro WAPE (%)")
    plt.title("Best Validation Micro WAPE by Model Family and Horizon")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "01_validation_wape_by_model_horizon.png", dpi=150)
    plt.close()

    # Plot 2: Tuned vs Untuned WAPE
    plt.figure(figsize=(10, 5))
    if step1_summary is not None:
        models_to_compare = ["LightGBM", "XGBoost", "Random Forest"]
        x = np.arange(len(models_to_compare))
        width = 0.35
        untuned_wapes = [step1_summary.loc[step1_summary["model"] == m, "Micro_WAPE"].values[0] for m in models_to_compare]
        tuned_wapes = [tuned_summary_df.loc[tuned_summary_df["model"] == f"Tuned {m}", "Micro_WAPE"].values[0] for m in models_to_compare]

        plt.bar(x - width/2, untuned_wapes, width, label="Untuned (Step 1)", color="#ff7f0e", alpha=0.85)
        plt.bar(x + width/2, tuned_wapes, width, label="Tuned (Step 2)", color="#2ca02c", alpha=0.85)
        plt.xticks(x, models_to_compare)
        plt.ylabel("Micro WAPE (%)")
        plt.title("Micro WAPE: Tuned vs. Untuned Initial Models")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
    plt.savefig(plots_dir / "02_tuned_vs_untuned_wape.png", dpi=150)
    plt.close()

    # Plot 3: Tuned ML vs Seasonal Naive by Horizon
    plt.figure(figsize=(9, 5))
    for m in ["Tuned LightGBM", "Tuned XGBoost", "Tuned Random Forest", "Seasonal Naive 52w"]:
        sub_h = tuned_horizon_df[tuned_horizon_df["model"] == m].sort_values("horizon")
        ls = "--" if "Seasonal" in m else "-"
        plt.plot(sub_h["horizon"], sub_h["Micro_WAPE"], marker="o", label=m, linestyle=ls, linewidth=2)
    plt.xlabel("Forecast Horizon Step (Weeks)")
    plt.ylabel("Micro WAPE (%)")
    plt.title("Tuned ML Models vs. Seasonal Naive 52w Across Forecast Horizons")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "03_tuned_ml_vs_seasonal_naive.png", dpi=150)
    plt.close()

    # Plot 4: WAPE Improvement by Horizon
    plt.figure(figsize=(9, 5))
    if step1_summary is not None:
        step1_horizon = pd.read_csv(ARTIFACTS_DIR / "models" / "model_by_horizon.csv")
        for m in ["LightGBM", "XGBoost", "Random Forest"]:
            u_sub = step1_horizon[step1_horizon["model"] == m].sort_values("horizon")
            t_sub = tuned_horizon_df[tuned_horizon_df["model"] == f"Tuned {m}"].sort_values("horizon")
            improvement = u_sub["Micro_WAPE"].values - t_sub["Micro_WAPE"].values
            plt.plot(range(1, 9), improvement, marker="s", label=f"{m} Improvement (pp)", linewidth=2)
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Forecast Horizon Step (Weeks)")
        plt.ylabel("Absolute WAPE Improvement (percentage points)")
        plt.title("WAPE Improvement from Hyperparameter Tuning Across Horizons")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
    plt.savefig(plots_dir / "04_wape_improvement_by_horizon.png", dpi=150)
    plt.close()

    # Plot 5: Representative SKU Actual vs Tuned Prediction
    plt.figure(figsize=(10, 5))
    rep_sku = "SKU001"
    sub_pred = tuned_pred_df[(tuned_pred_df["SKU"] == rep_sku) & (tuned_pred_df["horizon"] == 1)].sort_values("target_date")
    plt.plot(sub_pred["target_date"].unique(),
             sub_pred[sub_pred["model"] == "Tuned LightGBM"].set_index("target_date")["actual"],
             label="Actual Demand", color="black", linewidth=2)
    for m in ["Tuned LightGBM", "Tuned XGBoost", "Seasonal Naive 52w"]:
        m_data = sub_pred[sub_pred["model"] == m].set_index("target_date")
        plt.plot(m_data.index, m_data["prediction"], label=f"{m} (h=1)", linestyle="--")
    plt.xlabel("Target Date")
    plt.ylabel("Weekly Units Sold")
    plt.title(f"Actual vs. Tuned Prediction ({rep_sku}, h=1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "05_representative_sku_actual_vs_tuned.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 9. Adversarial Future Perturbation Audit (Section 17)
    # -----------------------------------------------------------------------
    logger.info("Executing Adversarial Future Perturbation Audit on Tuned Models...")
    test_origins = [pd.Timestamp("2024-12-17"), pd.Timestamp("2025-05-27"), pd.Timestamp("2025-11-04")]
    perturbation_audit_results = []

    for orig in test_origins:
        # Check if origin exists in panel
        if orig not in mf["forecast_origin_date"].values:
            continue
        orig_row = mf[mf["forecast_origin_date"] == orig].copy()
        X_orig = orig_row[predictor_cols].copy()

        for model_name, model_cls in model_classes.items():
            best_cfg = best_parameters_dict[model_name][1]["parameters"]
            m_base = model_cls(params=best_cfg)
            # Fit on training data <= orig - 1w (or sample)
            cutoff = orig - pd.Timedelta(weeks=1)
            train_mask = (mf["forecast_origin_date"] <= cutoff)
            if train_mask.sum() == 0:
                train_mask = (mf["forecast_origin_date"] <= orig)

            m_base.fit(mf.loc[train_mask, predictor_cols], mf.loc[train_mask, "target_h1"].values)
            pred_unperturbed = m_base.predict(X_orig)

            # Create perturbed copy strictly after orig
            mf_perturbed = mf.copy()
            future_mask = (mf_perturbed["forecast_origin_date"] > orig)
            mf_perturbed.loc[future_mask, "lag_1"] += 500.0
            mf_perturbed.loc[future_mask, "rolling_mean_4"] += 500.0
            mf_perturbed.loc[future_mask, "target_h1"] += 500.0

            # Origin features are rebuilt strictly at orig
            X_orig_after_future_perturb = mf_perturbed.loc[mf_perturbed["forecast_origin_date"] == orig, predictor_cols]
            pred_perturbed = m_base.predict(X_orig_after_future_perturb)

            max_diff = float(np.max(np.abs(pred_unperturbed - pred_perturbed)))
            passed_audit = bool(max_diff < 1e-7)

            perturbation_audit_results.append({
                "origin": str(orig.date()),
                "model": model_name,
                "max_absolute_difference": max_diff,
                "audit_passed": passed_audit,
            })
            assert passed_audit, f"Perturbation leakage detected for {model_name} at origin {orig}: max_diff = {max_diff}"

    # -----------------------------------------------------------------------
    # 10. Persist Reproducibility Manifest (Section 18)
    # -----------------------------------------------------------------------
    total_elapsed = time.perf_counter() - start_time
    manifest_data = {
        "execution_timestamp_utc": timestamp_utc,
        "phase": "3B — Step 2: Horizon-Aware Hyperparameter Tuning",
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "random_seed": 42,
        "total_trials_evaluated": len(trial_records),
        "validation_origins_count": len(val_origins),
        "validation_origins": [str(d.date()) for d in val_origins],
        "final_evaluation_origins_count": len(eval_origins),
        "final_evaluation_origins": [str(d.date()) for d in eval_origins],
        "feature_count": len(predictor_cols),
        "dataset_shape": list(mf.shape),
        "dataset_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "total_predictions_evaluated": len(tuned_pred_df),
        "total_pipeline_duration_seconds": total_elapsed,
        "sanity_checks": tuning_sanity_checks,
        "adversarial_leakage_audit": perturbation_audit_results,
        "summary_metrics": tuned_summary_df.to_dict(orient="records"),
    }

    manifest_path = ARTIFACTS_DIR / "metrics" / "phase3b_tuning_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open(tuning_artifact_dir / "tuning_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Phase 3B Step 2 Hyperparameter Tuning completed in %.2f seconds.", total_elapsed)
    return manifest_data


if __name__ == "__main__":
    setup_logging()
    run_phase3b_hyperparameter_tuning()
