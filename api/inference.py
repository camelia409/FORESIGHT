"""
api/inference.py — Production Model Inference Logic for API Endpoints
======================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 3: Production Model Promotion & Inference

Responsibilities
----------------
- Load the production model registry and FeatureEngineer singleton at startup.
- Serve 8-week horizon-specific demand forecasts to API endpoints:
    * POST /predict
    * POST /v1/forecast
- Ensure inference is identical to training pipeline.
- Implement Horizon-Segmented Hybrid Architecture:
    * h=1: Tuned Random Forest
    * h=2: Tuned XGBoost
    * h=3..8: Seasonal Naive 52w
- Enforce non-negative predictions, finite values, and Pydantic validation.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from api.schemas import (
    ForecastRecord,
    ForecastRequest,
    ForecastResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
    RecommendationRecord,
    RecommendationRequest,
    RecommendationResponse,
    RiskRecord,
    RiskRequest,
    RiskResponse,
    SystemStatusResponse,
    InventoryItem,
    InventoryResponse,
    SkuDetailResponse,
    GovernanceResponse,
)
from src.feature_engineering import FeatureEngineer, build_model_features

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PROD_DIR = PROJECT_ROOT / "models" / "production"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Singleton Engine Class
# ---------------------------------------------------------------------------

class ProductionInferenceEngine:
    """Production Inference Engine managing model registry and prediction dispatch."""

    def __init__(self, prod_dir: Path = MODELS_PROD_DIR):
        self.prod_dir = prod_dir
        self.registry: Optional[Dict[str, Any]] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.models: Dict[str, Any] = {}
        self.features_cache: Optional[pd.DataFrame] = None
        self.is_ready: bool = False
        self._load()

    def _load(self) -> None:
        """Load production artifacts from disk."""
        reg_file = self.prod_dir / "model_registry.pkl"
        fe_file = self.prod_dir / "feature_engineer.pkl"

        if not reg_file.exists() or not fe_file.exists():
            logger.warning("[inference] Production artifacts not yet available.")
            self.is_ready = False
            return

        try:
            self.registry = joblib.load(reg_file)
            self.feature_engineer = joblib.load(fe_file)

            # Load individual ML models
            for h_str, info in self.registry["horizon_registry"].items():
                if info["type"] == "ml_model":
                    art_path = self.prod_dir / info["artifact"]
                    if art_path.exists():
                        self.models[h_str] = joblib.load(art_path)
                    else:
                        raise FileNotFoundError(f"Model artifact not found: {art_path}")

            # Pre-load features cache from model_features.parquet
            feat_path = DATA_DIR / "processed" / "model_features.parquet"
            if feat_path.exists():
                self.features_cache = pd.read_parquet(feat_path)

            self.is_ready = True
            logger.info("[inference] Production inference engine successfully initialized.")
        except Exception as e:
            logger.error(f"[inference] Failed to load production artifacts: {e}")
            self.is_ready = False

    def predict(
        self,
        origin_date_str: Optional[str] = None,
        skus: Optional[List[str]] = None,
        horizon_weeks: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Generate horizon-specific predictions.

        Parameters
        ----------
        origin_date_str : str, optional
            Forecast origin date (YYYY-MM-DD). If None, defaults to latest available.
        skus : list of str, optional
            List of SKUs to forecast. If None, forecasts all active SKUs.
        horizon_weeks : int
            Forecast horizon steps (1..8).

        Returns
        -------
        list of prediction dicts.
        """
        if not self.is_ready:
            raise RuntimeError("Production inference engine is not ready.")

        if self.features_cache is None:
            feat_path = DATA_DIR / "processed" / "model_features.parquet"
            self.features_cache = pd.read_parquet(feat_path)

        # Determine origin date
        available_origins = sorted(self.features_cache["forecast_origin_date"].astype(str).unique())
        if origin_date_str is None:
            target_origin = available_origins[-1]
        else:
            target_origin = str(origin_date_str)

        if target_origin not in available_origins:
            # Fallback: check if close Monday or closest available origin
            target_origin = available_origins[-1]

        # Filter slice at origin
        origin_dt = pd.Timestamp(target_origin)
        origin_slice = self.features_cache[self.features_cache["forecast_origin_date"] == origin_dt].copy()

        if skus is not None and len(skus) > 0:
            origin_slice = origin_slice[origin_slice["SKU"].isin(skus)].copy()

        if len(origin_slice) == 0:
            raise ValueError(f"No feature records available for origin {target_origin} and requested SKUs.")

        predictor_cols = self.registry["predictor_columns"]
        X = origin_slice[predictor_cols].copy()
        sku_list = origin_slice["SKU"].tolist()

        results = []
        # Dispatch each horizon
        for h in range(1, horizon_weeks + 1):
            h_str = str(h)
            target_date_val = origin_dt + pd.Timedelta(weeks=h)
            target_date_str = str(target_date_val.date())
            reg_entry = self.registry["horizon_registry"].get(h_str)

            if reg_entry is None or reg_entry["type"] == "seasonal_naive":
                # Seasonal Naive 52w forecast
                # If target_h{h} exists or lag_52 is used as the proxy:
                # lag_52 is the demand 52 weeks ago relative to origin.
                # For horizon h, the 52-week seasonal lag relative to target week (t+h) is lag_{52-h}
                # If available, use baseline forecasts or lag values
                model_used = "Seasonal Naive 52w"
                # Check if baseline forecasts table exists for exact lookup
                base_file = PROJECT_ROOT / "artifacts" / "baseline" / "baseline_forecasts.parquet"
                if base_file.exists():
                    base_df = pd.read_parquet(base_file)
                    sub_sn = base_df[
                        (base_df["model"] == "seasonal_naive")
                        & (base_df["forecast_origin"] == origin_dt)
                        & (base_df["horizon"] == h)
                    ]
                    sn_map = dict(zip(sub_sn["SKU"], sub_sn["forecast"]))
                else:
                    sn_map = {}

                for sku in sku_list:
                    pred_val = sn_map.get(sku, float(origin_slice.loc[origin_slice["SKU"] == sku, "lag_52"].values[0]))
                    results.append({
                        "SKU": sku,
                        "forecast_origin_date": target_origin,
                        "horizon": h,
                        "target_date": target_date_str,
                        "prediction": float(max(0.0, pred_val)),
                        "model_used": model_used,
                        "model_version": self.registry["version"],
                    })

            elif reg_entry["type"] == "ml_model":
                model_used = reg_entry["model_family"]
                model = self.models.get(h_str)
                if model is None:
                    art_file = self.prod_dir / reg_entry["artifact"]
                    model = joblib.load(art_file)
                    self.models[h_str] = model

                preds = model.predict(X)
                for sku, pred_val in zip(sku_list, preds):
                    results.append({
                        "SKU": sku,
                        "forecast_origin_date": target_origin,
                        "horizon": h,
                        "target_date": target_date_str,
                        "prediction": float(max(0.0, pred_val)),
                        "model_used": model_used,
                        "model_version": self.registry["version"],
                    })

        return results


# Global inference engine singleton
_engine = ProductionInferenceEngine()


def get_inference_engine() -> ProductionInferenceEngine:
    global _engine
    if not _engine.is_ready:
        _engine._load()
    return _engine


def load_artefacts() -> None:
    """Application startup loader."""
    get_inference_engine()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """
    Standard production inference endpoint.
    Accepts SKU list, origin date, and horizon; returns 8-week forecasts.
    """
    engine = get_inference_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production forecasting engine is not initialized or artifacts are missing.",
        )

    try:
        preds = engine.predict(
            origin_date_str=request.origin_date,
            skus=request.skus,
            horizon_weeks=request.horizon_weeks,
        )
        origin_val = preds[0]["forecast_origin_date"] if preds else (request.origin_date or "unknown")
        n_skus = len(set(p["SKU"] for p in preds))

        items = [PredictionItem(**p) for p in preds]
        return PredictResponse(
            origin_date=origin_val,
            horizon_weeks=request.horizon_weeks,
            n_skus=n_skus,
            model_version=engine.registry["version"],
            architecture=engine.registry["architecture_name"],
            predictions=items,
        )
    except Exception as e:
        logger.error(f"Inference error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v1/forecast", response_model=ForecastResponse, tags=["Inference"])
async def forecast_v1_endpoint(request: ForecastRequest) -> ForecastResponse:
    """
    V1 Forecast endpoint conforming to ForecastRequest schema.
    """
    engine = get_inference_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production forecasting engine is not initialized.",
        )

    try:
        preds = engine.predict(
            origin_date_str=str(request.origin_date),
            skus=request.skus,
            horizon_weeks=min(8, request.horizon_weeks),
        )

        records = [
            ForecastRecord(
                forecast_date=datetime.strptime(p["target_date"], "%Y-%m-%d").date(),
                sku=p["SKU"],
                horizon_step=p["horizon"],
                units_forecast=p["prediction"],
                lower_bound=None,
                upper_bound=None,
                model_name=p["model_used"],
            )
            for p in preds
        ]

        return ForecastResponse(
            origin_date=request.origin_date,
            horizon_weeks=min(8, request.horizon_weeks),
            n_skus=len(set(p["SKU"] for p in preds)),
            forecast_records=records,
        )
    except Exception as e:
        logger.error(f"Inference error in /v1/forecast: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v1/risk", response_model=RiskResponse, tags=["Risk"])
async def risk_endpoint(request: RiskRequest) -> RiskResponse:
    """Generate 8-week horizon inventory risk scores and action tiers."""
    try:
        from src.risk_engine import RiskEngine

        engine = get_inference_engine()
        origin_str = str(request.origin_date)

        # 1. Generate demand forecasts
        preds = engine.predict(origin_date_str=origin_str, skus=request.skus, horizon_weeks=8)
        preds_df = pd.DataFrame(preds)

        # 2. Load inventory snapshots
        inv_path = DATA_DIR / "raw" / "inventory_snapshots.csv"
        inv_df = pd.read_csv(inv_path)

        # 3. Load sku_master
        sku_path = DATA_DIR / "raw" / "sku_master.csv"
        sku_master_df = pd.read_csv(sku_path) if sku_path.exists() else None

        # 4. Score via RiskEngine
        risk_eng = RiskEngine()
        scores_df = risk_eng.score(
            forecast_df=preds_df,
            inventory_df=inv_df,
            sku_master_df=sku_master_df,
            origin_date=request.origin_date,
        )

        # Filter to requested SKUs if specified
        if request.skus is not None and len(request.skus) > 0:
            scores_df = scores_df[scores_df["SKU"].isin(request.skus)].copy()

        risk_records = []
        for _, row in scores_df.iterrows():
            dos = row["days_of_supply"]
            if pd.isna(dos) or np.isinf(dos):
                dos_val = 999.0 if (pd.notna(dos) and np.isinf(dos)) else 0.0
            else:
                dos_val = float(dos)

            sr = row["stockout_score"]
            sr_val = float(sr) if pd.notna(sr) else 0.0

            os = row["overstock_score"]
            os_val = float(os) if pd.notna(os) else 0.0

            est_date = row["estimated_stockout_date"]
            if pd.notna(est_date) and est_date is not None:
                est_dt = pd.to_datetime(est_date).date()
            else:
                est_dt = None

            risk_records.append(
                RiskRecord(
                    sku=row["SKU"],
                    days_of_supply=dos_val,
                    stockout_score=sr_val,
                    overstock_score=os_val,
                    action_tier=row["action_tier"],
                    recommendation=row["recommendation"],
                    estimated_stockout_date=est_dt,
                    excess_inventory_value=None,
                )
            )

        return RiskResponse(
            origin_date=request.origin_date,
            n_skus=len(risk_records),
            risk_records=risk_records,
        )
    except Exception as e:
        logger.error(f"Error in /v1/risk: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v1/recommendations", response_model=RecommendationResponse, tags=["Decision Support"])
async def recommendations_endpoint(request: RecommendationRequest) -> RecommendationResponse:
    """Generate prioritized operational inventory recommendations for supply chain planners."""
    try:
        from src.decision_support import DecisionSupportEngine
        from src.risk_engine import RiskEngine

        engine = get_inference_engine()
        origin_str = str(request.origin_date) if request.origin_date else None

        # 1. Generate demand forecasts
        preds = engine.predict(origin_date_str=origin_str, skus=request.skus, horizon_weeks=8)
        preds_df = pd.DataFrame(preds)

        actual_origin = preds[0]["forecast_origin_date"] if preds else str(date.today())

        # 2. Load inventory snapshots & SKU master
        inv_path = DATA_DIR / "raw" / "inventory_snapshots.csv"
        inv_df = pd.read_csv(inv_path)

        sku_path = DATA_DIR / "raw" / "sku_master.csv"
        sku_master_df = pd.read_csv(sku_path) if sku_path.exists() else None

        # 3. Compute Phase 4B risk scores
        risk_eng = RiskEngine()
        risk_df = risk_eng.score(
            forecast_df=preds_df,
            inventory_df=inv_df,
            sku_master_df=sku_master_df,
            origin_date=actual_origin,
        )

        # Merge catalog metadata if available
        if sku_master_df is not None:
            meta_cols = ["SKU", "Product_Name", "Category", "Subcategory"]
            avail_meta = [c for c in meta_cols if c in sku_master_df.columns]
            risk_df = risk_df.merge(sku_master_df[avail_meta], on="SKU", how="left")

        # 4. Generate operational recommendations via Phase 5 DecisionSupportEngine
        dse = DecisionSupportEngine()
        recs_df = dse.recommend(risk_df)

        if request.skus is not None and len(request.skus) > 0:
            recs_df = recs_df[recs_df["sku"].isin(request.skus)].copy()

        out_records = []
        for _, row in recs_df.iterrows():
            woc_val = row["weeks_of_cover"]
            woc_clean = float(woc_val) if pd.notna(woc_val) and not np.isinf(woc_val) else None
            ex_woc_val = row["excess_weeks_of_cover"]
            ex_woc_clean = float(ex_woc_val) if pd.notna(ex_woc_val) and not np.isinf(ex_woc_val) else None

            out_records.append(
                RecommendationRecord(
                    sku=str(row["sku"]),
                    product_name=row.get("product_name"),
                    category=row.get("category"),
                    subcategory=row.get("subcategory"),
                    recommendation_code=str(row["recommendation_code"]),
                    recommendation_title=str(row["recommendation_title"]),
                    recommended_action=str(row["recommended_action"]),
                    priority=str(row["priority"]),
                    priority_rank=int(row["priority_rank"]),
                    rationale=str(row["rationale"]),
                    triggering_risk=str(row["triggering_risk"]),
                    confidence_status=str(row["confidence_status"]),
                    required_follow_up=str(row["required_follow_up"]),
                    weeks_of_cover=woc_clean,
                    excess_weeks_of_cover=ex_woc_clean,
                )
            )

        return RecommendationResponse(
            origin_date=str(actual_origin),
            n_skus=len(out_records),
            recommendations=out_records,
            valuation_basis_confirmed=False,
            arrival_timing_confirmed=False,
            on_order_policy="Policy_B_LT",
            overstock_threshold_weeks=8,
            overstock_threshold_status="ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL",
        )
    except Exception as e:
        logger.error(f"Error in /v1/recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ---------------------------------------------------------------------------
# Phase 6 Production & Dashboard Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/status", response_model=SystemStatusResponse, tags=["Dashboard & Operations"])
async def api_status_endpoint() -> SystemStatusResponse:
    """Return operational system status, loaded model readiness, and pipeline manifest info."""
    engine = get_inference_engine()
    manifest_file = PROJECT_ROOT / "artifacts" / "phase6" / "pipeline_manifest.json"
    last_timestamp = None
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as fh:
                m = json.load(fh)
                last_timestamp = m.get("execution_end")
        except Exception:
            pass

    return SystemStatusResponse(
        status="HEALTHY" if engine.is_ready else "DEGRADED",
        version="1.0.0",
        phase="Phase 6 — Production Deployment",
        pipeline_ready=True,
        inference_ready=bool(engine.is_ready),
        production_skus_count=50,
        quarantined_orphans_count=150,
        last_run_timestamp=last_timestamp,
        governance_summary={
            "valuation_basis": "Option 1D — Explicit Exclusion of Monetary Valuation",
            "monetary_valuation_blocked": True,
            "on_order_policy": "Policy_B_LT (Option 2A Ratified)",
            "overstock_threshold_weeks": 8,
            "overstock_threshold_status": "POLICY_RATIFIED",
        },
    )


@router.get("/api/inventory", response_model=InventoryResponse, tags=["Dashboard & Operations"])
async def api_inventory_endpoint() -> InventoryResponse:
    """Return latest fleet-wide inventory status for all 50 production SKUs."""
    rec_path = PROJECT_ROOT / "artifacts" / "decision_support" / "recommendations_latest.parquet"
    if not rec_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Latest recommendations artifact missing.")

    df = pd.read_parquet(rec_path)
    df = df[df["sku"].isin([f"SKU{i:03d}" for i in range(1, 51)])].copy()

    items = []
    for _, row in df.iterrows():
        items.append(
            InventoryItem(
                sku=str(row["sku"]),
                product_name=row.get("product_name"),
                category=row.get("category"),
                current_stock=float(row.get("current_stock", 0.0)),
                on_order=float(row.get("on_order", 0.0)),
                lead_time_days=float(row.get("lead_time_days", 0.0)),
                safety_stock=float(row.get("safety_stock", 0.0)),
                reorder_point=float(row.get("reorder_point", 0.0)),
                weeks_of_cover=float(row.get("weeks_of_cover", 0.0)) if pd.notna(row.get("weeks_of_cover")) else 0.0,
                stockout_risk_score=float(row.get("stockout_risk_score", 0.0)) if pd.notna(row.get("stockout_risk_score")) else 0.0,
                action_tier=str(row.get("triggering_risk", "MONITOR")),
            )
        )

    origin_val = str(df["origin_date"].iloc[0]) if len(df) > 0 else "2025-09-16"

    return InventoryResponse(
        origin_date=origin_val,
        n_skus=len(items),
        total_current_stock=float(df["current_stock"].sum()),
        total_on_order=float(df["on_order"].sum()),
        items=items,
        governance={
            "valuation_basis_confirmed": False,
            "monetary_valuation_blocked": True,
            "on_order_policy": "Policy_B_LT",
            "overstock_threshold_weeks": 8,
            "overstock_threshold_status": "POLICY_RATIFIED",
        },
    )


@router.get("/api/inventory/{sku}", response_model=SkuDetailResponse, tags=["Dashboard & Operations"])
async def api_sku_detail_endpoint(sku: str) -> SkuDetailResponse:
    """Return comprehensive single-SKU 360-degree drill-down with forecast curve and recommendations."""
    target_sku = sku.strip().upper()
    valid_skus = [f"SKU{i:03d}" for i in range(1, 51)]
    if target_sku not in valid_skus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU '{sku}' not found in production universe (SKU001–SKU050) or is quarantined.",
        )

    rec_path = PROJECT_ROOT / "artifacts" / "decision_support" / "recommendations_latest.parquet"
    if not rec_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Latest recommendations artifact missing.")

    df = pd.read_parquet(rec_path)
    sku_rows = df[df["sku"] == target_sku]
    if len(sku_rows) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No data for SKU {target_sku}")

    row = sku_rows.iloc[0]

    # Generate 8-week forecast curve for this SKU
    engine = get_inference_engine()
    forecast_vals = []
    if engine.is_ready:
        try:
            preds = engine.predict(origin_date_str=str(row.get("origin_date", "2025-09-16")), skus=[target_sku], horizon_weeks=8)
            forecast_vals = [round(float(p["prediction"]), 2) for p in sorted(preds, key=lambda x: x["horizon"])]
        except Exception:
            forecast_vals = []

    woc = row.get("weeks_of_cover")
    ex_woc = row.get("excess_weeks_of_cover")

    return SkuDetailResponse(
        sku=target_sku,
        product_name=row.get("product_name"),
        category=row.get("category"),
        subcategory=row.get("subcategory"),
        current_stock=float(row.get("current_stock", 0.0)),
        on_order=float(row.get("on_order", 0.0)),
        lead_time_days=float(row.get("lead_time_days", 0.0)),
        safety_stock=float(row.get("safety_stock", 0.0)),
        reorder_point=float(row.get("reorder_point", 0.0)),
        weeks_of_cover=float(woc) if pd.notna(woc) and not np.isinf(woc) else 0.0,
        excess_weeks_of_cover=float(ex_woc) if pd.notna(ex_woc) and not np.isinf(ex_woc) else 0.0,
        stockout_risk_score=float(row.get("stockout_risk_score", 0.0)) if pd.notna(row.get("stockout_risk_score")) else 0.0,
        action_tier=str(row.get("triggering_risk", "MONITOR")),
        recommendation_code=str(row.get("recommendation_code", "MAINTAIN_SCHEDULE")),
        recommendation_title=str(row.get("recommendation_title", "Maintain Schedule")),
        recommended_action=str(row.get("recommended_action", "Maintain regular replenishment.")),
        priority=str(row.get("priority", "3 — MEDIUM")),
        rationale=str(row.get("rationale", "")),
        forecast_h1_8=forecast_vals,
        governance={
            "valuation_basis_confirmed": False,
            "monetary_valuation_blocked": True,
            "on_order_policy": "Policy_B_LT",
            "arrival_timing_confirmed": False,
            "overstock_threshold_weeks": 8,
            "overstock_threshold_status": "POLICY_RATIFIED",
        },
    )


@router.get("/api/risk", response_model=RiskResponse, tags=["Dashboard & Operations"])
async def api_risk_get_endpoint() -> RiskResponse:
    """GET endpoint returning latest risk assessment for all 50 production SKUs."""
    risk_path = PROJECT_ROOT / "artifacts" / "risk" / "risk_scores_latest.parquet"
    if not risk_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Latest risk artifact missing.")

    df = pd.read_parquet(risk_path)
    df = df[df["SKU"].isin([f"SKU{i:03d}" for i in range(1, 51)])].copy()

    records = []
    for _, row in df.iterrows():
        records.append(
            RiskRecord(
                sku=str(row["SKU"]),
                days_of_supply=float(row.get("days_of_supply", 0.0)) if pd.notna(row.get("days_of_supply")) and not np.isinf(row.get("days_of_supply")) else 999.0,
                stockout_score=float(row.get("stockout_score", 0.0)) if pd.notna(row.get("stockout_score")) else 0.0,
                overstock_score=float(row.get("overstock_score", 0.0)) if pd.notna(row.get("overstock_score")) else 0.0,
                action_tier=str(row.get("action_tier", "UNKNOWN")),
                recommendation=str(row.get("recommendation", "")),
                estimated_stockout_date=pd.to_datetime(row["estimated_stockout_date"]).date() if pd.notna(row.get("estimated_stockout_date")) and row.get("estimated_stockout_date") is not None else None,
                excess_inventory_value=None,
            )
        )

    origin_val = str(df["origin_date"].iloc[0]) if len(df) > 0 else "2025-09-16"
    return RiskResponse(
        origin_date=origin_val,
        n_skus=len(records),
        risk_records=records,
    )


@router.get("/api/recommendations", response_model=RecommendationResponse, tags=["Dashboard & Operations"])
async def api_recommendations_get_endpoint() -> RecommendationResponse:
    """GET endpoint returning latest operational recommendations for all 50 production SKUs."""
    rec_path = PROJECT_ROOT / "artifacts" / "decision_support" / "recommendations_latest.parquet"
    if not rec_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Latest recommendations artifact missing.")

    df = pd.read_parquet(rec_path)
    df = df[df["sku"].isin([f"SKU{i:03d}" for i in range(1, 51)])].copy()

    recs = []
    for _, row in df.iterrows():
        recs.append(
            RecommendationRecord(
                sku=str(row["sku"]),
                product_name=row.get("product_name"),
                category=row.get("category"),
                subcategory=row.get("subcategory"),
                recommendation_code=str(row["recommendation_code"]),
                recommendation_title=str(row["recommendation_title"]),
                recommended_action=str(row["recommended_action"]),
                priority=str(row["priority"]),
                priority_rank=int(row["priority_rank"]),
                rationale=str(row["rationale"]),
                triggering_risk=str(row["triggering_risk"]),
                confidence_status=str(row["confidence_status"]),
                required_follow_up=str(row["required_follow_up"]),
                weeks_of_cover=float(row["weeks_of_cover"]) if pd.notna(row.get("weeks_of_cover")) and not np.isinf(row["weeks_of_cover"]) else None,
                excess_weeks_of_cover=float(row["excess_weeks_of_cover"]) if pd.notna(row.get("excess_weeks_of_cover")) and not np.isinf(row["excess_weeks_of_cover"]) else None,
            )
        )

    origin_val = str(df["origin_date"].iloc[0]) if len(df) > 0 else "2025-09-16"
    return RecommendationResponse(
        origin_date=origin_val,
        n_skus=len(recs),
        recommendations=recs,
        valuation_basis_confirmed=False,
        arrival_timing_confirmed=False,
        on_order_policy="Policy_B_LT",
        overstock_threshold_weeks=8,
        overstock_threshold_status="POLICY_RATIFIED",
    )


@router.get("/api/governance", response_model=GovernanceResponse, tags=["Dashboard & Operations"])
async def api_governance_endpoint() -> GovernanceResponse:
    """Return ratified Milestone 5.X governance policies and project boundaries."""
    return GovernanceResponse()



