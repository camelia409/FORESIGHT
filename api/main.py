"""
api/main.py — FastAPI Application Entry-Point
=============================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 3: Production Model Promotion & Inference

Responsibilities
----------------
- Expose a REST API for on-demand SKU demand forecasting.
- Endpoints:
    * GET /health : health check returning healthy status and inference_ready flag.
    * GET /version : version information.
    * POST /predict : production multi-horizon forecasting endpoint.
    * POST /v1/forecast : V1 forecast endpoint.
    * POST /v1/risk : placeholder for Phase 4 risk scoring.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Internal imports
from api import schemas  # noqa: F401
from api.inference import get_inference_engine, load_artefacts, router as inference_router
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load production models at startup."""
    load_artefacts()
    yield


app = FastAPI(
    title="FORESIGHT Production Forecasting API",
    description=(
        "AI-Powered Demand Forecasting & Inventory Risk API for Project FORESIGHT.\n\n"
        "**Status**: Production Horizon-Segmented Hybrid Forecasting Engine Active."
    ),
    version="1.0.0",
    contact={"name": "FORESIGHT Team"},
    license_info={"name": "Private — Internal Use Only"},
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register Inference Routers
# ---------------------------------------------------------------------------
app.include_router(inference_router)


# ---------------------------------------------------------------------------
# Core Monitoring Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Status"])
async def root():
    """Root endpoint — confirm the API is reachable."""
    return {
        "service": "FORESIGHT Production Forecasting API",
        "version": "1.0.0",
        "status": "active",
        "architecture": "Horizon-Segmented Hybrid",
    }


@app.get("/health", tags=["Status"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    engine = get_inference_engine()
    return HealthResponse(
        status="healthy",
        inference_ready=bool(engine.is_ready),
    )


@app.get("/version", tags=["Status"])
async def version():
    """Return API version information."""
    return {
        "version": "1.0.0",
        "phase": "3B — Step 3: Production Promoted",
        "architecture": "Horizon-Segmented Hybrid",
    }
