"""
model_evaluation.py — Candidate Model Training & Rolling-Origin Evaluation
==========================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 1: Model Training & Evaluation Framework

Responsibilities
----------------
1. Load verified Phase 3A feature dataset (data/processed/model_features.parquet).
2. Enforce strict feature/target/metadata separation assertions.
3. Execute rolling-origin walk-forward evaluation across Phase 2B origins.
4. Train direct multi-horizon models (h=1..8) for:
   - LightGBM
   - XGBoost
   - Random Forest
5. Compare fairly against Phase 2B Seasonal Naive benchmark under identical conditions.
6. Run comprehensive model output sanity checks (missingness, NaNs, infinities, negatives, coverage).
7. Persist trained candidate models to models/candidates/.
8. Generate structured metrics artifacts under artifacts/models/:
   - model_summary.csv
   - model_by_horizon.csv
   - model_by_sku.csv
   - model_by_category.csv
   - model_predictions.parquet
9. Generate visualization plots under artifacts/models/plots/.
10. Persist reproducibility manifest artifacts/metrics/phase3b_initial_model_manifest.json.
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
    get_model,
)
from src.utils import mae as mae_fn, rmse as rmse_fn, set_seed, setup_logging, wape as wape_fn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard FORESIGHT metrics: Micro WAPE, MAE, RMSE."""
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


def run_phase3b_initial_models() -> dict[str, Any]:
    """
    Execute Phase 3B Step 1: Candidate Model Training & Evaluation.
    """
    set_seed(42)
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    features_path = DATA_DIR / "processed" / "model_features.parquet"
    baseline_path = ARTIFACTS_DIR / "baseline" / "baseline_forecasts.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing model_features.parquet at {features_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline_forecasts.parquet at {baseline_path}")

    # 1. Inspect and Validate Input Features
    mf = pd.read_parquet(features_path)
    logger.info("Loaded model_features.parquet: shape = %s", mf.shape)
    assert mf.shape == (2350, 60), f"Expected shape (2350, 60), got {mf.shape}"

    target_cols = [f"target_h{h}" for h in range(1, 9)]
    meta_cols = ["SKU", "forecast_origin_date"]
    predictor_cols = [c for c in mf.columns if c not in target_cols and c not in meta_cols]

    # Explicit assertions required by Section 4
    assert len(predictor_cols) == 50, f"Expected 50 SAFE features, got {len(predictor_cols)}"
    assert not any(c in predictor_cols for c in target_cols), "TARGET columns leaked into predictors!"
    assert not any(c in predictor_cols for c in meta_cols), "Metadata columns leaked into predictors!"
    assert all(c in mf.columns for c in predictor_cols), "Missing SAFE predictors in dataframe!"

    # 2. Setup Evaluation Origins matching Phase 2B
    base_df = pd.read_parquet(baseline_path)
    base_origins = sorted(base_df["forecast_origin"].unique())
    logger.info("Phase 2B Baseline origins (%d total): %s to %s", len(base_origins), base_origins[0], base_origins[-1])

    # Origins 4 to 12 (index 3 to 11) have full historical training coverage across all 8 horizons
    eval_origins = base_origins[3:]
    logger.info("Evaluation origins with complete h=1..8 training history: %d origins (%s to %s)",
                len(eval_origins), eval_origins[0].date(), eval_origins[-1].date())

    # Pre-extract category metadata for SKUs
    sku_metadata = mf.drop_duplicates("SKU")[["SKU", "Category", "Subcategory"]].set_index("SKU")

    # 3. Rolling-Origin Walk-Forward Model Training & Forecasting
    candidate_models = {
        "LightGBM": LightGBMForecaster,
        "XGBoost": XGBoostForecaster,
        "Random Forest": RandomForestForecaster,
    }

    all_predictions: list[dict[str, Any]] = []
    training_durations: dict[str, float] = {m: 0.0 for m in candidate_models}

    logger.info("Starting Rolling-Origin Walk-Forward Evaluation across %d origins and 8 horizons...", len(eval_origins))

    for fold_idx, eval_orig in enumerate(eval_origins, 1):
        fold_eval_ts = pd.Timestamp(eval_orig)
        logger.info("--- Fold %d/%d: Origin = %s ---", fold_idx, len(eval_origins), fold_eval_ts.date())

        test_mask = (mf["forecast_origin_date"] == fold_eval_ts)
        test_X = mf.loc[test_mask, predictor_cols].copy()
        test_skus = mf.loc[test_mask, "SKU"].values

        for h in range(1, 9):
            target_col = f"target_h{h}"
            # Strict causal temporal boundary: training target date must be <= eval_origin
            cutoff_origin = fold_eval_ts - pd.Timedelta(weeks=h)
            train_mask = (mf["forecast_origin_date"] <= cutoff_origin)

            train_X = mf.loc[train_mask, predictor_cols].copy()
            train_y = mf.loc[train_mask, target_col].values
            test_y = mf.loc[test_mask, target_col].values

            # Target strictly after origin assertion for test data
            target_date = fold_eval_ts + pd.Timedelta(weeks=h)
            assert target_date > fold_eval_ts, "Target date must strictly follow forecast origin"

            # Training data never contains future targets relative to evaluation origin
            train_target_dates = mf.loc[train_mask, "forecast_origin_date"] + pd.Timedelta(weeks=h)
            assert (train_target_dates <= fold_eval_ts).all(), "Training targets leak into future of evaluation origin!"

            for model_name, model_cls in candidate_models.items():
                m_instance = model_cls()
                t0 = time.perf_counter()
                m_instance.fit(train_X, train_y)
                dur = time.perf_counter() - t0
                training_durations[model_name] += dur

                preds = m_instance.predict(test_X)

                # Record predictions
                for sku, actual_val, pred_val in zip(test_skus, test_y, preds):
                    cat_info = sku_metadata.loc[sku]
                    all_predictions.append({
                        "model": model_name,
                        "SKU": sku,
                        "forecast_origin_date": fold_eval_ts,
                        "horizon": h,
                        "target_date": target_date,
                        "actual": float(actual_val),
                        "prediction": float(pred_val),
                        "Category": cat_info["Category"],
                        "Subcategory": cat_info["Subcategory"],
                    })

    # Join Seasonal Naive benchmark predictions for identical comparison
    sn_sub = base_df[
        (base_df["model"] == "seasonal_naive") &
        (base_df["forecast_origin"].isin(eval_origins))
    ].copy()

    for _, row in sn_sub.iterrows():
        all_predictions.append({
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

    pred_df = pd.DataFrame(all_predictions)

    # 4. Model Output Sanity Checks (Section 13)
    logger.info("Executing Model Output Sanity Checks...")
    sanity_checks = {}

    for model_name in pred_df["model"].unique():
        sub = pred_df[pred_df["model"] == model_name]
        n_rows = len(sub)
        n_missing = int(sub["prediction"].isna().sum())
        n_inf = int(np.isinf(sub["prediction"]).sum())
        n_neg = int((sub["prediction"] < 0).sum())
        min_p = float(sub["prediction"].min())
        max_p = float(sub["prediction"].max())
        sku_cov = int(sub["SKU"].nunique())
        horiz_cov = sorted(sub["horizon"].unique().tolist())
        orig_cov = int(sub["forecast_origin_date"].nunique())

        sanity_checks[model_name] = {
            "prediction_count": n_rows,
            "missing_predictions": n_missing,
            "infinite_predictions": n_inf,
            "negative_predictions": n_neg,
            "min_prediction": min_p,
            "max_prediction": max_p,
            "sku_coverage": sku_cov,
            "horizon_coverage": horiz_cov,
            "origin_coverage": orig_cov,
            "sanity_passed": (n_missing == 0 and n_inf == 0 and n_neg == 0 and sku_cov == 50 and len(horiz_cov) == 8),
        }
        assert sanity_checks[model_name]["sanity_passed"], f"Sanity check failed for {model_name}"

    # 5. Compute Detailed Metrics (Overall, by Horizon, by SKU, by Category)
    summary_rows = []
    horizon_rows = []
    sku_rows = []
    cat_rows = []

    for model_name in pred_df["model"].unique():
        sub = pred_df[pred_df["model"] == model_name]
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

        # By Horizon
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

        # By SKU
        for sku, s_sub in sub.groupby("SKU"):
            s_met = compute_metrics(s_sub["actual"].values, s_sub["prediction"].values)
            sku_rows.append({
                "model": model_name,
                "SKU": sku,
                "Micro_WAPE": s_met["Micro_WAPE"],
                "MAE": s_met["MAE"],
                "RMSE": s_met["RMSE"],
            })

        # By Category
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

    # Also add Phase 2B full 12-origin Seasonal Naive for reference
    sn_all = base_df[base_df["model"] == "seasonal_naive"]
    sn_all_met = compute_metrics(sn_all["actual"].values, sn_all["forecast"].values)
    sn_all_mac = compute_macro_wape(sn_all.rename(columns={"forecast": "prediction"}))
    summary_rows.append({
        "model": "Seasonal Naive 52w (All 12 Origins Phase 2B)",
        "Micro_WAPE": sn_all_met["Micro_WAPE"],
        "Macro_WAPE": sn_all_mac,
        "MAE": sn_all_met["MAE"],
        "RMSE": sn_all_met["RMSE"],
        "prediction_count": len(sn_all),
        "eval_origins": sn_all["forecast_origin"].nunique(),
    })

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    sku_df = pd.DataFrame(sku_rows)
    cat_df = pd.DataFrame(cat_rows)

    # 6. Persist Metrics and Prediction Artifacts
    models_artifact_dir = ARTIFACTS_DIR / "models"
    models_artifact_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = models_artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(models_artifact_dir / "model_summary.csv", index=False)
    horizon_df.to_csv(models_artifact_dir / "model_by_horizon.csv", index=False)
    sku_df.to_csv(models_artifact_dir / "model_by_sku.csv", index=False)
    cat_df.to_csv(models_artifact_dir / "model_by_category.csv", index=False)
    pred_df.to_parquet(models_artifact_dir / "model_predictions.parquet", index=False)
    logger.info("Persisted metric tables and model_predictions.parquet to %s", models_artifact_dir)

    # 7. Train and Persist Final Candidate Models (models/candidates/)
    logger.info("Training and persisting final candidate models to models/candidates/...")
    last_eval_orig = eval_origins[-1]

    for model_name, model_cls in candidate_models.items():
        sub_dir = MODELS_DIR / "candidates" / model_name.lower().replace(" ", "_")
        sub_dir.mkdir(parents=True, exist_ok=True)

        manifest_models = []
        for h in range(1, 9):
            target_col = f"target_h{h}"
            # Train using all historical data with known targets as of the final backtest cutoff
            final_cutoff = last_eval_orig - pd.Timedelta(weeks=h)
            train_mask = (mf["forecast_origin_date"] <= final_cutoff)
            train_X = mf.loc[train_mask, predictor_cols]
            train_y = mf.loc[train_mask, target_col].values

            m_inst = model_cls()
            m_inst.fit(train_X, train_y)
            save_path = sub_dir / f"model_h{h}.joblib"
            m_inst.save(save_path)

            manifest_models.append({
                "horizon": h,
                "target_column": target_col,
                "training_rows": len(train_X),
                "feature_count": len(predictor_cols),
                "training_duration_seconds": m_inst.training_duration_,
                "artifact_path": str(save_path.relative_to(PROJECT_ROOT)),
            })

        with open(sub_dir / "candidate_manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_family": model_name,
                "library": model_cls.library,
                "random_seed": 42,
                "baseline_hyperparameters": model_cls.DEFAULT_PARAMS,
                "horizons": manifest_models,
            }, f, indent=2)

    # 8. Generate Visualizations (Section 14)
    logger.info("Generating evaluation plots under artifacts/models/plots/...")
    
    # Plot 1: Model WAPE Comparison
    plt.figure(figsize=(9, 5))
    plot_summary = summary_df.set_index("model")
    x = np.arange(len(plot_summary))
    width = 0.35
    plt.bar(x - width/2, plot_summary["Micro_WAPE"], width, label="Micro WAPE (%)", color="#1f77b4")
    plt.bar(x + width/2, plot_summary["Macro_WAPE"], width, label="Macro WAPE (%)", color="#aec7e8")
    plt.xticks(x, [m.replace(" ", "\n") for m in plot_summary.index], fontsize=9)
    plt.ylabel("WAPE (%)")
    plt.title("Candidate Model WAPE Comparison vs. Seasonal Naive Benchmark")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "01_model_wape_comparison.png", dpi=150)
    plt.close()

    # Plot 2: WAPE by Horizon
    plt.figure(figsize=(9, 5))
    for model_name in horizon_df["model"].unique():
        sub_h = horizon_df[horizon_df["model"] == model_name].sort_values("horizon")
        plt.plot(sub_h["horizon"], sub_h["Micro_WAPE"], marker="o", label=model_name, linewidth=2)
    plt.xlabel("Forecast Horizon Step (Weeks)")
    plt.ylabel("Micro WAPE (%)")
    plt.title("Micro WAPE Progression by Forecast Horizon (h=1..8)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "02_wape_by_horizon.png", dpi=150)
    plt.close()

    # Plot 3: Actual vs Prediction for Representative SKU
    plt.figure(figsize=(10, 5))
    rep_sku = "SKU001"
    sub_pred = pred_df[(pred_df["SKU"] == rep_sku) & (pred_df["horizon"] == 1)].sort_values("target_date")
    plt.plot(sub_pred["target_date"].unique(),
             sub_pred[sub_pred["model"] == "LightGBM"].set_index("target_date")["actual"],
             label="Actual Units Sold", color="black", linewidth=2)
    for model_name in ["LightGBM", "XGBoost", "Random Forest", "Seasonal Naive 52w"]:
        m_data = sub_pred[sub_pred["model"] == model_name].set_index("target_date")
        plt.plot(m_data.index, m_data["prediction"], label=f"{model_name} (h=1)", linestyle="--")
    plt.xlabel("Target Date")
    plt.ylabel("Weekly Units Sold")
    plt.title(f"Actual vs. Prediction for Representative SKU ({rep_sku}, h=1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "03_actual_vs_prediction.png", dpi=150)
    plt.close()

    # Plot 4: Model Comparison by SKU
    plt.figure(figsize=(10, 5))
    lgb_sku = sku_df[sku_df["model"] == "LightGBM"].set_index("SKU")["Micro_WAPE"]
    sn_sku = sku_df[sku_df["model"] == "Seasonal Naive 52w"].set_index("SKU")["Micro_WAPE"]
    plt.scatter(sn_sku, lgb_sku, color="#2ca02c", alpha=0.8, edgecolors="black", s=60)
    lims = [0, max(sn_sku.max(), lgb_sku.max()) + 5]
    plt.plot(lims, lims, color="red", linestyle="--", label="1:1 Parity Line")
    plt.xlabel("Seasonal Naive 52w Micro WAPE (%)")
    plt.ylabel("LightGBM Micro WAPE (%)")
    plt.title("Per-SKU Error: LightGBM vs. Seasonal Naive 52w")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "04_model_comparison_by_sku.png", dpi=150)
    plt.close()

    # Plot 5: Residual Distribution
    plt.figure(figsize=(9, 5))
    for model_name in ["LightGBM", "XGBoost", "Random Forest", "Seasonal Naive 52w"]:
        m_sub = pred_df[pred_df["model"] == model_name]
        residuals = m_sub["actual"] - m_sub["prediction"]
        plt.hist(residuals, bins=40, alpha=0.4, label=model_name, density=True)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual (Actual - Prediction)")
    plt.ylabel("Density")
    plt.title("Residual Distribution Across Forecast Horizontally Pooled Predictions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "05_residual_distribution.png", dpi=150)
    plt.close()

    # 9. Persist Manifest (Section 16)
    total_elapsed = time.perf_counter() - start_time
    manifest_data = {
        "execution_timestamp_utc": timestamp_utc,
        "phase": "3B — Step 1: Model Training & Evaluation Framework",
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
        "feature_count": len(predictor_cols),
        "predictor_features": predictor_cols,
        "dataset_shape": list(mf.shape),
        "dataset_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "total_predictions_generated": len(pred_df),
        "evaluation_origins_count": len(eval_origins),
        "evaluation_origins": [str(d.date()) for d in eval_origins],
        "training_durations_seconds": training_durations,
        "total_pipeline_duration_seconds": total_elapsed,
        "sanity_checks": sanity_checks,
        "summary_metrics": summary_df.to_dict(orient="records"),
    }

    manifest_path = ARTIFACTS_DIR / "metrics" / "phase3b_initial_model_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Phase 3B Step 1 execution completed in %.2f seconds.", total_elapsed)
    return manifest_data


if __name__ == "__main__":
    setup_logging()
    run_phase3b_initial_models()
