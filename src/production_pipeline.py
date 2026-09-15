"""
production_pipeline.py — End-to-End Production Pipeline Orchestrator
====================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: Phase 6 — Production Deployment, Pipeline Orchestration & Dashboard Integration

Pipeline Stages (Sequential DAG):
1. INGEST           : Load raw datasets and verify input integrity.
2. VALIDATE         : Validate schemas, enforce production boundaries, quarantine orphan SKUs.
3. LOAD_MODELS      : Load production hybrid models (h=1 RF, h=2 XGB, h=3..8 Seasonal Naive).
4. FORECAST         : Generate multi-horizon forecasts (h=1..8) for 50 production SKUs.
5. INVENTORY_POSITION: Calculate projected inventory positions under Policy B_LT.
6. RISK_SCORING     : Compute stockout probabilities, buffer breaches, and terminal excess.
7. DECISION_SUPPORT : Generate operational recommendations and priority hierarchy.
8. GOVERNANCE_FILTER: Enforce Milestone 5.X ratified policies (Option 1D, Option 2A, Option 3C).
9. SERVE_EXPORT     : Persist run manifest and serving artifacts.

Ratified Governance Constraints:
- Decision #1: Option 1D — Monetary valuation explicitly excluded. Monetary metrics remain None.
- Decision #2: Option 2A — Policy B_LT ratified based on verified supplier lead times <= 14 days.
- Decision #3: Option 3C — Overstock threshold N = 8 weeks ratified (POLICY_RATIFIED).
- Production Boundary: SKU001–SKU050 only. SKU051–SKU200 strictly quarantined.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from src.config import CFG, PATHS, PROJECT_ROOT
from src.utils import setup_logging, hash_raw_files

logger = logging.getLogger("foresight.production_pipeline")

PROD_SKUS = [f"SKU{i:03d}" for i in range(1, 51)]
ORPHAN_SKUS = [f"SKU{i:03d}" for i in range(51, 201)]


class ProductionPipelineError(Exception):
    """Raised when a production pipeline stage fails critical validation."""
    pass


class ProductionPipeline:
    """
    Unified production pipeline orchestrator for Project FORESIGHT.
    Stateless, idempotent, and fully governed.
    """

    def __init__(self, origin_date: Optional[str] = None):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.origin_date_requested = origin_date
        self.actual_origin_date: Optional[str] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        self.dag_stages = [
            "INGEST",
            "VALIDATE",
            "LOAD_MODELS",
            "FORECAST",
            "INVENTORY_POSITION",
            "RISK_SCORING",
            "DECISION_SUPPORT",
            "GOVERNANCE_FILTER",
            "SERVE_EXPORT",
        ]

        self.manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "pipeline": "ProductionPipeline",
            "phase": "Phase 6 — Production Orchestration",
            "execution_start": datetime.now().isoformat(),
            "origin_date_requested": origin_date,
            "actual_origin_date": None,
            "stages_executed": 9,
            "skus_monitored": 50,
            "orphan_skus_quarantined": 150,
            "governance_policies": {
                "valuation_basis": "Option 1D (Monetary Valuation Excluded)",
                "on_order_policy": "Option 2A (Policy B_LT)",
                "overstock_threshold": "Option 3C (N=8 Weeks Ratified)",
                "decision_1_valuation_basis": "Option 1D — Explicit Exclusion of Monetary Valuation",
                "valuation_basis_confirmed": False,
                "monetary_valuation_blocked": True,
                "decision_2_on_order_policy": "Option 2A — Policy B_LT (Lead Time <= 14 Days)",
                "on_order_policy": "Policy_B_LT",
                "arrival_timing_confirmed": False,
                "decision_3_overstock_threshold": "Option 3C — N = 8 Weeks",
                "overstock_threshold_weeks": 8,
                "overstock_threshold_status": "POLICY_RATIFIED",
            },
            "stages": {},
            "metrics": {
                "production_skus_processed": 50,
                "orphan_skus_quarantined": 150,
                "forecast_records_generated": 0,
                "risk_records_generated": 0,
                "recommendations_generated": 0,
            },
            "artifacts_generated": [],
            "status": "INITIALIZED",
        }

        # Data buffers
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._filtered_inv: Optional[pd.DataFrame] = None
        self._sku_master: Optional[pd.DataFrame] = None
        self._forecasts_df: Optional[pd.DataFrame] = None
        self._risk_df: Optional[pd.DataFrame] = None
        self._recommendations_df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Stage 1: INGEST
    # -------------------------------------------------------------------------
    def stage_ingest(self) -> None:
        """Load raw datasets and record input baseline hashes."""
        logger.info("[INGEST] Loading raw datasets...")
        t0 = time.perf_counter()

        raw_dir = PATHS.raw_dir
        inv_path = raw_dir / "inventory_snapshots.csv"
        sku_path = raw_dir / "sku_master.csv"

        if not inv_path.exists():
            raise ProductionPipelineError(f"Missing inventory snapshots at {inv_path}")
        if not sku_path.exists():
            raise ProductionPipelineError(f"Missing SKU master at {sku_path}")

        self._raw_data["inventory"] = pd.read_csv(inv_path)
        self._raw_data["sku_master"] = pd.read_csv(sku_path)

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["INGEST"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "rows_loaded": {
                "inventory": len(self._raw_data["inventory"]),
                "sku_master": len(self._raw_data["sku_master"]),
            },
        }
        logger.info(f"[INGEST] Completed in {elapsed:.3f}s. Loaded {len(self._raw_data['inventory'])} inventory rows.")

    # -------------------------------------------------------------------------
    # Stage 2: VALIDATE
    # -------------------------------------------------------------------------
    def stage_validate(self) -> None:
        """Validate schemas, enforce 50-SKU production universe, and quarantine orphan SKUs."""
        logger.info("[VALIDATE] Enforcing production universe and quarantining orphan SKUs...")
        t0 = time.perf_counter()

        inv_df = self._raw_data["inventory"].copy()
        sku_df = self._raw_data["sku_master"].copy()

        # Schema assertion
        req_inv_cols = ["Snapshot_Date", "SKU", "Current_Stock", "On_Order", "Lead_Time_Days", "Safety_Stock", "Reorder_Point"]
        for c in req_inv_cols:
            if c not in inv_df.columns:
                raise ProductionPipelineError(f"Missing required inventory column: {c}")

        # Determine target origin date
        avail_dates = sorted(inv_df["Snapshot_Date"].unique())
        if self.origin_date_requested:
            if self.origin_date_requested in avail_dates:
                target_date = self.origin_date_requested
            else:
                # Find closest earlier date or latest
                past_dates = [d for d in avail_dates if d <= self.origin_date_requested]
                target_date = past_dates[-1] if past_dates else avail_dates[0]
        else:
            target_date = avail_dates[-1]

        self.actual_origin_date = target_date
        self.manifest["actual_origin_date"] = target_date

        # Slice to snapshot date
        snapshot_slice = inv_df[inv_df["Snapshot_Date"] == target_date].copy()

        # Quarantine check
        orphan_present = snapshot_slice[snapshot_slice["SKU"].isin(ORPHAN_SKUS)]
        n_orphans = len(orphan_present)
        self.manifest["metrics"]["orphan_skus_quarantined"] = n_orphans

        # Filter strictly to production SKUs
        prod_slice = snapshot_slice[snapshot_slice["SKU"].isin(PROD_SKUS)].copy()
        n_prod = len(prod_slice["SKU"].unique())

        if n_prod != 50:
            raise ProductionPipelineError(f"Expected exactly 50 production SKUs, found {n_prod}")

        self._filtered_inv = prod_slice
        self._sku_master = sku_df[sku_df["SKU"].isin(PROD_SKUS)].copy()
        self.manifest["metrics"]["production_skus_processed"] = n_prod

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["VALIDATE"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "target_snapshot_date": target_date,
            "production_skus": n_prod,
            "orphan_skus_quarantined": n_orphans,
        }
        logger.info(f"[VALIDATE] Completed in {elapsed:.3f}s. 50 production SKUs verified. {n_orphans} orphan SKUs quarantined.")

    # -------------------------------------------------------------------------
    # Stage 3 & 4: LOAD_MODELS & FORECAST
    # -------------------------------------------------------------------------
    def stage_forecast(self) -> None:
        """Generate direct multi-horizon forecasts using the production hybrid engine."""
        logger.info("[FORECAST] Generating 8-week forward demand forecasts...")
        t0 = time.perf_counter()

        from api.inference import get_inference_engine
        engine = get_inference_engine()

        if not engine.is_ready:
            raise ProductionPipelineError("Production inference engine failed to initialize.")

        preds = engine.predict(
            origin_date_str=self.actual_origin_date,
            skus=PROD_SKUS,
            horizon_weeks=8,
        )

        if not preds or len(preds) != (50 * 8):
            raise ProductionPipelineError(f"Expected {50 * 8} forecast records, received {len(preds) if preds else 0}")

        self._forecasts_df = pd.DataFrame(preds)
        self.manifest["metrics"]["forecast_records_generated"] = len(preds)

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["FORECAST"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "forecast_records": len(preds),
            "model_version": engine.registry.get("version", "1.0.0"),
            "architecture": engine.registry.get("architecture_name", "Horizon-Segmented Hybrid"),
        }
        logger.info(f"[FORECAST] Completed in {elapsed:.3f}s. {len(preds)} forecast points produced across h=1..8.")

    # -------------------------------------------------------------------------
    # Stage 5 & 6: INVENTORY_POSITION & RISK_SCORING
    # -------------------------------------------------------------------------
    def stage_risk_scoring(self) -> None:
        """Compute inventory positions under Policy B_LT and evaluate risk breaches."""
        logger.info("[RISK_SCORING] Evaluating inventory positions and safety buffer breaches...")
        t0 = time.perf_counter()

        from src.risk_engine import RiskEngine

        risk_eng = RiskEngine()
        risk_df = risk_eng.score(
            forecast_df=self._forecasts_df,
            inventory_df=self._filtered_inv,
            sku_master_df=self._sku_master,
            origin_date=self.actual_origin_date,
        )

        # Merge metadata
        if self._sku_master is not None:
            meta_cols = ["SKU", "Product_Name", "Category", "Subcategory"]
            avail_meta = [c for c in meta_cols if c in self._sku_master.columns]
            risk_df = risk_df.merge(self._sku_master[avail_meta], on="SKU", how="left")

        if len(risk_df) != 50:
            raise ProductionPipelineError(f"Expected exactly 50 scored risk records, got {len(risk_df)}")

        self._risk_df = risk_df
        self.manifest["metrics"]["risk_records_generated"] = len(risk_df)

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["RISK_SCORING"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "action_tier_counts": risk_df["action_tier"].value_counts().to_dict(),
        }
        logger.info(f"[RISK_SCORING] Completed in {elapsed:.3f}s. 50 SKUs scored.")

    # -------------------------------------------------------------------------
    # Stage 7 & 8: DECISION_SUPPORT & GOVERNANCE_FILTER
    # -------------------------------------------------------------------------
    def stage_decision_support(self) -> None:
        """Generate operational recommendations and enforce ratified governance policies."""
        logger.info("[DECISION_SUPPORT] Generating prioritized operational recommendations...")
        t0 = time.perf_counter()

        from src.decision_support import DecisionSupportEngine

        dse = DecisionSupportEngine()
        recs_df = dse.recommend(self._risk_df)

        # Enforce Milestone 5.X Ratified Governance Policies in output
        recs_df["valuation_basis_confirmed"] = False
        recs_df["monetary_valuation_blocked"] = True
        recs_df["on_order_policy"] = "Policy_B_LT"
        recs_df["arrival_timing_confirmed"] = False
        recs_df["overstock_threshold_weeks"] = 8
        recs_df["overstock_threshold_status"] = "POLICY_RATIFIED"

        # Explicitly ensure monetary exposure fields are blocked/null
        for m_col in ["excess_inventory_value", "inventory_value_at_risk", "capital_at_risk"]:
            if m_col in recs_df.columns:
                recs_df[m_col] = None

        if len(recs_df) != 50:
            raise ProductionPipelineError(f"Expected exactly 50 recommendations, got {len(recs_df)}")

        self._recommendations_df = recs_df
        self.manifest["metrics"]["recommendations_generated"] = len(recs_df)

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["DECISION_SUPPORT"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "recommendation_counts": recs_df["recommendation_code"].value_counts().to_dict(),
            "priority_counts": recs_df["priority"].value_counts().to_dict(),
        }
        logger.info(f"[DECISION_SUPPORT] Completed in {elapsed:.3f}s. Directives: {recs_df['recommendation_code'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # Stage 9: SERVE_EXPORT
    # -------------------------------------------------------------------------
    def stage_serve_export(self, export_csv: bool = True) -> None:
        """Persist Phase 6 operational artifacts and run manifest."""
        logger.info("[SERVE_EXPORT] Exporting Phase 6 production artifacts...")
        t0 = time.perf_counter()

        phase6_art_dir = PROJECT_ROOT / "artifacts" / "phase6"
        phase6_rep_dir = PROJECT_ROOT / "reports" / "phase6"
        phase6_art_dir.mkdir(parents=True, exist_ok=True)
        phase6_rep_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = phase6_art_dir / "production_latest_recommendations.parquet"
        self._recommendations_df.to_parquet(parquet_path, index=False)
        self.manifest["artifacts_generated"].append(str(parquet_path))

        if export_csv:
            csv_path = phase6_rep_dir / "production_latest_recommendations.csv"
            self._recommendations_df.to_csv(csv_path, index=False)
            self.manifest["artifacts_generated"].append(str(csv_path))

        elapsed = time.perf_counter() - t0
        self.manifest["stages"]["SERVE_EXPORT"] = {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 4),
            "artifacts_count": len(self.manifest["artifacts_generated"]),
        }

        # Finalize manifest
        self.end_time = time.perf_counter()
        total_time = self.end_time - self.start_time
        self.manifest["execution_end"] = datetime.now().isoformat()
        self.manifest["total_elapsed_seconds"] = round(total_time, 4)
        self.manifest["status"] = "COMPLETED"

        manifest_path = phase6_art_dir / "pipeline_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        self.manifest["artifacts_generated"].append(str(manifest_path))
        logger.info(f"[SERVE_EXPORT] Completed in {elapsed:.3f}s. Manifest saved to {manifest_path}.")

    # -------------------------------------------------------------------------
    # End-to-End Orchestrator
    # -------------------------------------------------------------------------
    def execute(self, export_csv: bool = True) -> Dict[str, Any]:
        """Execute all sequential DAG stages."""
        self.start_time = time.perf_counter()
        logger.info(f"============================================================")
        logger.info(f"  FORESIGHT PRODUCTION PIPELINE — RUN ID: {self.run_id}")
        logger.info(f"============================================================")

        try:
            self.stage_ingest()
            self.stage_validate()
            self.stage_forecast()
            self.stage_risk_scoring()
            self.stage_decision_support()
            self.stage_serve_export(export_csv=export_csv)
            logger.info(f"============================================================")
            logger.info(f"  PRODUCTION PIPELINE COMPLETED SUCCESSFULLY ({self.manifest['total_elapsed_seconds']:.2f}s)")
            logger.info(f"============================================================")
            return self.manifest
        except Exception as e:
            self.manifest["status"] = "FAILED"
            self.manifest["error"] = str(e)
            logger.critical(f"PRODUCTION PIPELINE FAILED: {e}", exc_info=True)
            raise

    def run_pipeline(self, export_csv: bool = True) -> Dict[str, Any]:
        """Alias for execute()."""
        return self.execute(export_csv=export_csv)


# Class alias for external orchestrator runners
ProductionPipelineOrchestrator = ProductionPipeline


def run_pipeline(origin_date: Optional[str] = None, export_csv: bool = True) -> Dict[str, Any]:
    """Convenience entry point for production pipeline execution."""
    pipeline = ProductionPipeline(origin_date=origin_date)
    return pipeline.execute(export_csv=export_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORESIGHT Production Pipeline Orchestrator")
    parser.add_argument("--origin", type=str, default=None, help="Forecast origin date (YYYY-MM-DD)")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV report export")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, name="foresight")

    try:
        manifest = run_pipeline(origin_date=args.origin, export_csv=not args.no_csv)
        print(f"\nPipeline Run Success! Run ID: {manifest['run_id']}")
        print(f"Origin Date: {manifest['actual_origin_date']}")
        print(f"SKUs Processed: {manifest['metrics']['production_skus_processed']}")
        print(f"Artifacts Generated: {len(manifest['artifacts_generated'])}")
    except Exception as exc:
        print(f"\nPipeline Error: {exc}")
        sys.exit(1)
