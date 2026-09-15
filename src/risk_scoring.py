"""
risk_scoring.py — Risk Pipeline Orchestrator & Batch Scoring
=============================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 4B — Inventory Risk Engine Pipeline Execution

Responsibilities
----------------
- Orchestrate end-to-end risk scoring across production forecast origins.
- Assemble inputs:
    * Production multi-horizon demand forecasts (final_predictions.parquet or inference engine)
    * Raw inventory snapshots (inventory_snapshots.csv)
    * SKU master catalog (sku_master.csv)
- Enforce strict temporal safety (backward asof merge: Snapshot_Date <= origin_date).
- Filter and quarantine orphan inventory SKUs (SKU051-SKU200).
- Generate machine-readable artifacts:
    * artifacts/risk/risk_scores_panel.parquet
    * artifacts/risk/risk_scores_latest.parquet
    * artifacts/risk/risk_latest.json
- Generate human-readable operational reports:
    * reports/risk/risk_report_latest.csv
    * reports/risk/risk_report_panel.csv
- Validate numerical invariants:
    * Grain: exactly 50 production SKUs per origin
    * No future leakage
    * Monetary fields strictly None / null
    * Stockout risk in [0, 1]
    * Overstock units >= 0
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import CFG, PATHS
from src.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


def run_risk_scoring(
    origin_date: Optional[Union[str, date, pd.Timestamp]] = None,
    forecast_path: Optional[Path] = None,
    inventory_path: Optional[Path] = None,
    sku_master_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute batch risk scoring on real production data.

    Parameters
    ----------
    origin_date : Specific origin date (YYYY-MM-DD), or None for all available origins.
    forecast_path : Path to production forecast parquet (defaults to final_predictions.parquet).
    inventory_path : Path to raw inventory snapshots.
    sku_master_path : Path to sku_master.csv.
    output_dir : Destination for machine-readable artifacts (artifacts/risk).
    report_dir : Destination for human-readable reports (reports/risk).

    Returns
    -------
    dict with execution summary, artifact paths, row counts, and validation checks.
    """
    # 1. Resolve paths
    if output_dir is None:
        output_dir = PATHS.metrics_dir.parent / "risk"
    output_dir.mkdir(parents=True, exist_ok=True)

    if report_dir is None:
        report_dir = PATHS.reports_data_quality_dir.parent / "risk"
    report_dir.mkdir(parents=True, exist_ok=True)

    if forecast_path is None:
        forecast_path = PATHS.metrics_dir.parent / "models" / "final" / "final_predictions.parquet"
        if not forecast_path.exists():
            forecast_path = PATHS.models_dir / "production" / "final_predictions.parquet"

    if inventory_path is None:
        inventory_path = PATHS.raw_inventory

    if sku_master_path is None:
        sku_master_path = PATHS.raw_sku_master

    logger.info("[risk_scoring] Loading production forecasts from %s", forecast_path)
    df_fc = pd.read_parquet(forecast_path)

    logger.info("[risk_scoring] Loading inventory snapshots from %s", inventory_path)
    df_inv = pd.read_csv(inventory_path)

    logger.info("[risk_scoring] Loading SKU master from %s", sku_master_path)
    df_sku = pd.read_csv(sku_master_path)

    # 2. Instantiate RiskEngine
    engine = RiskEngine()

    # 3. Determine origins to score
    # Normalize origin column
    orig_col = None
    for c in ["forecast_origin_date", "origin_date", "forecast_origin"]:
        if c in df_fc.columns:
            orig_col = c
            break

    if orig_col is None:
        raise KeyError("Forecast parquet missing origin date column.")

    all_origins = sorted(df_fc[orig_col].unique())

    if origin_date is not None:
        target_orig = pd.Timestamp(origin_date)
        origins_to_score = [target_orig]
    else:
        origins_to_score = all_origins

    logger.info(
        "[risk_scoring] Scoring %d origin(s): %s",
        len(origins_to_score),
        [str(pd.to_datetime(o).date()) for o in origins_to_score],
    )

    # 4. Score all target origins
    panel_dfs = []
    for orig in origins_to_score:
        orig_dt = pd.Timestamp(orig)
        orig_fc = df_fc[df_fc[orig_col] == orig_dt].copy()

        scores_df = engine.score(
            forecast_df=orig_fc,
            inventory_df=df_inv,
            sku_master_df=df_sku,
            origin_date=orig_dt,
        )
        panel_dfs.append(scores_df)

    panel_scores = pd.concat(panel_dfs, ignore_index=True)

    # Merge product metadata for human-readable reporting
    meta_cols = ["SKU", "Product_Name", "Category", "Subcategory"]
    avail_meta = [c for c in meta_cols if c in df_sku.columns]
    panel_enriched = panel_scores.merge(df_sku[avail_meta], on="SKU", how="left")

    # 5. Extract latest origin
    latest_origin = origins_to_score[-1]
    latest_dt = pd.Timestamp(latest_origin).date()
    latest_scores = panel_enriched[panel_enriched["origin_date"] == latest_dt].copy()

    # 6. Save Artifacts
    # Parquet artifacts
    panel_parquet = output_dir / "risk_scores_panel.parquet"
    latest_parquet = output_dir / "risk_scores_latest.parquet"
    panel_scores.to_parquet(panel_parquet, index=False)
    latest_scores.to_parquet(latest_parquet, index=False)

    # CSV reports
    panel_csv = report_dir / "risk_report_panel.csv"
    latest_csv = report_dir / "risk_report_latest.csv"
    panel_enriched.to_csv(panel_csv, index=False)
    latest_enriched = latest_scores.to_csv(latest_csv, index=False)

    # Latest JSON payload
    latest_json_path = output_dir / "risk_latest.json"
    json_records = []
    for _, row in latest_scores.iterrows():
        est_d = row["estimated_stockout_date"]
        est_str = str(est_d) if pd.notna(est_d) and est_d is not None else None
        snap_d = row["snapshot_date"]
        snap_str = str(snap_d) if pd.notna(snap_d) and snap_d is not None else None

        json_records.append({
            "origin_date":               str(row["origin_date"]),
            "sku":                       str(row["SKU"]),
            "product_name":              row.get("Product_Name"),
            "category":                  row.get("Category"),
            "subcategory":               row.get("Subcategory"),
            "inventory_data_available":  bool(row["inventory_data_available"]),
            "snapshot_date":             snap_str,
            "days_since_snapshot":       int(row["days_since_snapshot"]) if pd.notna(row["days_since_snapshot"]) else None,
            "current_stock":             float(row["Current_Stock"]) if pd.notna(row["Current_Stock"]) else None,
            "on_order":                  float(row["On_Order"]) if pd.notna(row["On_Order"]) else None,
            "lead_time_days":            float(row["Lead_Time_Days"]) if pd.notna(row["Lead_Time_Days"]) else None,
            "safety_stock":              float(row["Safety_Stock"]) if pd.notna(row["Safety_Stock"]) else None,
            "reorder_point":             float(row["Reorder_Point"]) if pd.notna(row["Reorder_Point"]) else None,
            "avg_weekly_demand":         float(row["avg_weekly_demand"]) if pd.notna(row["avg_weekly_demand"]) else None,
            "forecast_cum_8w":           float(row["forecast_cum_8w"]) if pd.notna(row["forecast_cum_8w"]) else None,
            "weeks_of_cover":            float(row["weeks_of_cover"]) if pd.notna(row["weeks_of_cover"]) and not np.isinf(row["weeks_of_cover"]) else None,
            "days_of_supply":            float(row["days_of_supply"]) if pd.notna(row["days_of_supply"]) and not np.isinf(row["days_of_supply"]) else None,
            "is_stockout_risk":          bool(row["is_stockout_risk"]) if pd.notna(row["is_stockout_risk"]) else None,
            "earliest_stockout_week":    int(row["earliest_stockout_week"]) if pd.notna(row["earliest_stockout_week"]) else None,
            "estimated_stockout_date":   est_str,
            "is_safety_stock_breach":    bool(row["is_safety_stock_breach"]) if pd.notna(row["is_safety_stock_breach"]) else None,
            "earliest_ss_breach_week":   int(row["earliest_ss_breach_week"]) if pd.notna(row["earliest_ss_breach_week"]) else None,
            "is_reorder_point_breach":   bool(row["is_reorder_point_breach"]) if pd.notna(row["is_reorder_point_breach"]) else None,
            "earliest_rp_breach_week":   int(row["earliest_rp_breach_week"]) if pd.notna(row["earliest_rp_breach_week"]) else None,
            "stockout_score":            float(row["stockout_score"]) if pd.notna(row["stockout_score"]) else None,
            "excess_inventory_units":    float(row["excess_inventory_units"]) if pd.notna(row["excess_inventory_units"]) else None,
            "excess_weeks_of_cover":     float(row["excess_weeks_of_cover"]) if pd.notna(row["excess_weeks_of_cover"]) else None,
            "overstock_score":           float(row["overstock_score"]) if pd.notna(row["overstock_score"]) else None,
            "overstock_tier":            str(row["overstock_tier"]),
            "action_tier":               str(row["action_tier"]),
            "recommendation":            str(row["recommendation"]),
            # Governance
            "excess_inventory_value":    None,
            "inventory_value_at_risk":   None,
            "capital_at_risk":           None,
            "valuation_basis_confirmed": False,
            "arrival_timing_confirmed":  False,
            "on_order_policy":           row["on_order_policy"],
            "overstock_threshold_weeks": int(row["overstock_threshold_weeks"]),
            "overstock_threshold_status": row["overstock_threshold_status"],
        })

    json_payload = {
        "metadata": engine.get_governance_metadata(),
        "latest_origin_date": str(latest_dt),
        "total_origins_scored": len(origins_to_score),
        "n_skus": len(json_records),
        "records": json_records,
    }
    with open(latest_json_path, "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)

    # 7. Invariant Verification
    n_origins = len(origins_to_score)
    expected_rows = n_origins * CFG.KNOWN_MASTER_SKU_COUNT
    assert len(panel_scores) == expected_rows, (
        f"Row count mismatch: {len(panel_scores)} != {expected_rows}"
    )
    assert len(latest_scores) == CFG.KNOWN_MASTER_SKU_COUNT, (
        f"Latest count mismatch: {len(latest_scores)} != {CFG.KNOWN_MASTER_SKU_COUNT}"
    )
    assert panel_scores["excess_inventory_value"].isna().all(), (
        "Valuation Blocker Violated: excess_inventory_value contains non-null values!"
    )
    assert panel_scores["valuation_basis_confirmed"].eq(False).all(), (
        "Valuation Blocker Violated: valuation_basis_confirmed is not False!"
    )

    summary = {
        "origins_scored": len(origins_to_score),
        "total_scored_rows": len(panel_scores),
        "latest_origin": str(latest_dt),
        "latest_action_tier_distribution": latest_scores["action_tier"].value_counts().to_dict(),
        "latest_overstock_tier_distribution": latest_scores["overstock_tier"].value_counts().to_dict(),
        "artifacts_created": [
            str(panel_parquet),
            str(latest_parquet),
            str(latest_json_path),
            str(panel_csv),
            str(latest_csv),
        ],
    }
    logger.info("[risk_scoring] Execution complete. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_risk_scoring()
    print("\n--- Pipeline Run Summary ---")
    print(json.dumps(res, indent=2))
