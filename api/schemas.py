"""
api/schemas.py — Pydantic Request & Response Schemas
=====================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Define all Pydantic models for API request validation and response serialisation.
- Provide strong typing for all API payloads to catch errors at the boundary.
- Document field constraints and examples for OpenAPI documentation.

Schemas (to be expanded in Phase 6)
-------------------------------------
Request:
    ForecastRequest     — POST /v1/forecast
    RiskRequest         — POST /v1/risk

Response:
    ForecastRecord      — single SKU × week forecast
    ForecastResponse    — list of ForecastRecords
    RiskRecord          — single SKU risk assessment
    RiskResponse        — list of RiskRecords
    HealthResponse      — GET /health
    ErrorResponse       — structured error body

Status
------
SKELETON — schemas defined as minimal stubs.  Expand in Phase 6.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:           str = Field(..., example="healthy")
    inference_ready:  bool = Field(..., example=False)


class ErrorResponse(BaseModel):
    error:   str = Field(..., example="not_implemented")
    detail:  str = Field(..., example="This endpoint is not yet active.")


# ---------------------------------------------------------------------------
# Forecast schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """
    Request body for POST /v1/forecast.

    Fields
    ------
    origin_date   : The last date for which actuals are available.
                    Forecasts begin from origin_date + 1 week.
    skus          : Optional list of SKUs to forecast.
                    If None, all active SKUs are forecast.
    horizon_weeks : Forecast horizon in weeks (default: 8).
    """
    origin_date:   date            = Field(..., example="2025-01-06")
    skus:          Optional[list[str]] = Field(None, example=["SKU001", "SKU002"])
    horizon_weeks: int             = Field(8, ge=1, le=52, example=8)

    class Config:
        json_schema_extra = {
            "example": {
                "origin_date":   "2025-01-06",
                "skus":          ["SKU001", "SKU002"],
                "horizon_weeks": 8,
            }
        }


class ForecastRecord(BaseModel):
    """A single forecast data point for one SKU and one week."""
    forecast_date:   date  = Field(..., example="2025-01-13")
    sku:             str   = Field(..., example="SKU001")
    horizon_step:    int   = Field(..., ge=1, example=1)
    units_forecast:  float = Field(..., ge=0.0, example=125.5)
    lower_bound:     Optional[float] = Field(None, example=100.0)
    upper_bound:     Optional[float] = Field(None, example=155.0)
    model_name:      str   = Field(..., example="lightgbm")


class ForecastResponse(BaseModel):
    """Response body for POST /v1/forecast."""
    origin_date:      date
    horizon_weeks:    int
    n_skus:           int
    forecast_records: list[ForecastRecord]


# ---------------------------------------------------------------------------
# Risk schemas
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    """
    Request body for POST /v1/risk.

    Fields
    ------
    origin_date  : Date of the latest available forecast and inventory data.
    skus         : Optional SKU filter (None = all SKUs).
    """
    origin_date: date               = Field(..., example="2025-01-06")
    skus:        Optional[list[str]] = Field(None, example=["SKU001"])

    class Config:
        json_schema_extra = {
            "example": {"origin_date": "2025-01-06", "skus": None}
        }


class RiskRecord(BaseModel):
    """Inventory risk assessment for a single SKU."""
    sku:                    str   = Field(..., example="SKU001")
    days_of_supply:         float = Field(..., example=45.0)
    stockout_score:         float = Field(..., ge=0.0, le=1.0, example=0.12)
    overstock_score:        float = Field(..., ge=0.0,          example=0.0)
    action_tier:            str   = Field(..., example="HEALTHY")
    recommendation:         str   = Field(..., example="No action required. Monitor weekly.")
    estimated_stockout_date: Optional[date] = Field(None, example=None)
    # Monetary fields — only populated when inventory_valuation_basis is configured
    excess_inventory_value: Optional[float] = Field(None, example=None)


class RiskResponse(BaseModel):
    """Response body for POST /v1/risk."""
    origin_date:   date
    n_skus:        int
    risk_records:  list[RiskRecord]
    # Metadata
    valuation_basis_confirmed: bool = Field(
        False,
        description="True only when CFG.inventory_valuation_basis has been set."
    )
    orphan_skus_included: bool = Field(
        False,
        description="True only when CFG.orphan_sku_treatment has been set."
    )


# ---------------------------------------------------------------------------
# Production Predict schemas (Phase 3B Step 3)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Request body for POST /predict.

    Fields
    ------
    origin_date   : The forecast origin date (YYYY-MM-DD). Default: latest available.
    skus          : Optional list of SKUs to forecast (None = all active SKUs).
    horizon_weeks : Forecast horizon in weeks (1 to 8, default: 8).
    """
    origin_date:   Optional[str]       = Field(None, example="2025-09-16")
    skus:          Optional[list[str]] = Field(None, example=["SKU001", "SKU002"])
    horizon_weeks: int                 = Field(8, ge=1, le=8, example=8)


class PredictionItem(BaseModel):
    """A single production forecast item."""
    SKU:                  str   = Field(..., example="SKU001")
    forecast_origin_date: str   = Field(..., example="2025-09-16")
    horizon:              int   = Field(..., ge=1, le=8, example=1)
    target_date:          str   = Field(..., example="2025-09-23")
    prediction:           float = Field(..., ge=0.0, example=120.5)
    model_used:           str   = Field(..., example="Tuned Random Forest")
    model_version:        str   = Field(..., example="1.0.0")


class PredictResponse(BaseModel):
    """Response body for POST /predict."""
    origin_date:      str
    horizon_weeks:    int
    n_skus:           int
    model_version:    str
    architecture:     str
    predictions:      list[PredictionItem]


# ---------------------------------------------------------------------------
# Decision Support & Recommendation schemas (Phase 5)
# ---------------------------------------------------------------------------

class RecommendationRecord(BaseModel):
    """A single operational recommendation for one SKU."""
    sku:                   str             = Field(..., example="SKU001")
    product_name:          Optional[str]   = Field(None, example="Product 001")
    category:              Optional[str]   = Field(None, example="Furniture")
    subcategory:           Optional[str]   = Field(None, example="Chair")
    recommendation_code:   str             = Field(..., example="PLACE_PO")
    recommendation_title:  str             = Field(..., example="Issue Replenishment Purchase Order")
    recommended_action:    str             = Field(..., example="Place purchase order for 350 units.")
    priority:              str             = Field(..., example="2 — HIGH")
    priority_rank:         int             = Field(..., example=2)
    rationale:             str             = Field(..., example="Reorder Point breach in week 1.")
    triggering_risk:       str             = Field(..., example="REORDER_POINT_BREACH_LT")
    confidence_status:     str             = Field(..., example="HIGH_CONFIDENCE")
    required_follow_up:    str             = Field(..., example="Issue purchase order within 48 hours.")
    weeks_of_cover:        Optional[float] = Field(None, example=5.4)
    excess_weeks_of_cover: Optional[float] = Field(None, example=0.0)


class RecommendationRequest(BaseModel):
    """Request body for POST /v1/recommendations."""
    origin_date: Optional[str]       = Field(None, example="2025-09-16")
    skus:        Optional[list[str]] = Field(None, example=["SKU001", "SKU002"])


class RecommendationResponse(BaseModel):
    """Response body for POST /v1/recommendations."""
    origin_date:                str
    n_skus:                     int
    recommendations:            list[RecommendationRecord]
    valuation_basis_confirmed:  bool = False
    arrival_timing_confirmed:   bool = False
    on_order_policy:            str  = "Policy_B_LT"
    overstock_threshold_weeks:  int  = 8
    overstock_threshold_status: str  = "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"


# ---------------------------------------------------------------------------
# Phase 6 Production & Dashboard Schemas
# ---------------------------------------------------------------------------

class SystemStatusResponse(BaseModel):
    """System status and operational readiness response for GET /api/status."""
    status:                    str             = Field(..., example="HEALTHY")
    version:                   str             = Field(..., example="1.0.0")
    phase:                     str             = Field(..., example="Phase 6 — Production Deployment")
    pipeline_ready:            bool            = Field(..., example=True)
    inference_ready:           bool            = Field(..., example=True)
    production_skus_count:     int             = Field(..., example=50)
    quarantined_orphans_count: int             = Field(..., example=150)
    last_run_timestamp:        Optional[str]   = Field(None, example="2026-09-15T12:30:00")
    governance_summary:        dict            = Field(default_factory=dict)


class InventoryItem(BaseModel):
    """Inventory status record for a single production SKU."""
    sku:                 str             = Field(..., example="SKU001")
    product_name:        Optional[str]   = Field(None, example="Valve Pro")
    category:            Optional[str]   = Field(None, example="Industrial")
    current_stock:       float           = Field(..., example=250.0)
    on_order:            float           = Field(..., example=120.0)
    lead_time_days:      float           = Field(..., example=7.0)
    safety_stock:        float           = Field(..., example=85.0)
    reorder_point:       float           = Field(..., example=180.0)
    weeks_of_cover:      float           = Field(..., example=3.5)
    stockout_risk_score: float           = Field(..., example=0.12)
    action_tier:         str             = Field(..., example="MONITOR")


class InventoryResponse(BaseModel):
    """Response body for GET /api/inventory."""
    origin_date:         str
    n_skus:              int
    total_current_stock: float
    total_on_order:      float
    items:               list[InventoryItem]
    governance:          dict


class SkuDetailResponse(BaseModel):
    """Comprehensive drill-down response for GET /api/inventory/{sku}."""
    sku:                   str             = Field(..., example="SKU001")
    product_name:          Optional[str]   = Field(None, example="Product 001")
    category:              Optional[str]   = Field(None, example="Industrial")
    subcategory:           Optional[str]   = Field(None, example="Valves")
    current_stock:         float           = Field(..., example=250.0)
    on_order:              float           = Field(..., example=120.0)
    lead_time_days:        float           = Field(..., example=7.0)
    safety_stock:          float           = Field(..., example=85.0)
    reorder_point:         float           = Field(..., example=180.0)
    weeks_of_cover:        float           = Field(..., example=3.5)
    excess_weeks_of_cover: float           = Field(..., example=0.0)
    stockout_risk_score:   float           = Field(..., example=0.12)
    action_tier:           str             = Field(..., example="MONITOR")
    recommendation_code:   str             = Field(..., example="REVIEW_PIPELINE")
    recommendation_title:  str             = Field(..., example="Review In-Flight Replenishment")
    recommended_action:    str             = Field(..., example="Confirm PO delivery milestones.")
    priority:              str             = Field(..., example="3 — MEDIUM")
    rationale:             str             = Field(..., example="In-flight PO covers lead-time demand.")
    forecast_h1_8:         list[float]     = Field(default_factory=list)
    governance:            dict            = Field(default_factory=dict)


class GovernanceResponse(BaseModel):
    """Audit response for GET /api/governance exposing all ratified decisions."""
    milestone:                 str  = "Milestone 5.X — Governance Gate Management"
    phase_5_status:            str  = "GOVERNANCE_RATIFIED"
    phase_6_status:            str  = "AUTHORIZED_TO_COMMENCE"
    valuation_basis:           str  = "Option 1D — Explicit Exclusion of Monetary Valuation"
    valuation_basis_confirmed: bool = False
    monetary_valuation_blocked:bool = True
    on_order_policy:           str  = "Policy_B_LT (Option 2A Ratified)"
    arrival_timing_confirmed:  bool = False
    overstock_threshold_weeks: int  = 8
    overstock_threshold_status:str  = "POLICY_RATIFIED"
    production_sku_universe:   str  = "SKU001–SKU050 (50 SKUs)"
    orphan_sku_quarantine:     str  = "SKU051–SKU200 (150 SKUs Quarantined)"

