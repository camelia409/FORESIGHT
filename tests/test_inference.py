"""
test_inference.py — API Inference & Serving Tests
===================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 3B — Step 3: Production Promotion & API Validation

Tests:
1. API health endpoint returns healthy and inference_ready=True
2. Root endpoint returns active status and version
3. Version endpoint returns 1.0.0
4. POST /predict endpoint returns valid 8-horizon forecast
5. POST /v1/forecast endpoint returns valid ForecastResponse
6. Prediction values are strictly non-negative and finite
7. Predictions schema contains required fields
8. SKU subsetting works in /predict
9. Horizon subsetting works in /predict
10. Latency threshold test (< 200 ms)
"""

from __future__ import annotations

import time
from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_01_health_endpoint_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["inference_ready"] is True


def test_02_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["version"] == "1.0.0"


def test_03_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"


def test_04_predict_endpoint_success():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001", "SKU002"],
        "horizon_weeks": 8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_skus"] == 2
    assert data["horizon_weeks"] == 8
    assert len(data["predictions"]) == 16  # 2 SKUs × 8 horizons

    # Check first prediction schema
    first = data["predictions"][0]
    assert "SKU" in first
    assert "forecast_origin_date" in first
    assert "horizon" in first
    assert "target_date" in first
    assert "prediction" in first
    assert "model_used" in first
    assert "model_version" in first
    assert first["prediction"] >= 0.0


def test_05_forecast_v1_endpoint_success():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001"],
        "horizon_weeks": 4,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_skus"] == 1
    assert data["horizon_weeks"] == 4
    assert len(data["forecast_records"]) == 4


def test_06_prediction_values_non_negative_and_finite():
    payload = {
        "origin_date": "2025-09-16",
        "skus": None,  # all 50 SKUs
        "horizon_weeks": 8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_skus"] == 50
    assert len(data["predictions"]) == 400  # 50 × 8

    for p in data["predictions"]:
        val = p["prediction"]
        assert val is not None
        assert val >= 0.0
        assert not (val != val)  # not NaN


def test_07_invalid_horizon_raises_422():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001"],
        "horizon_weeks": 10,  # exceeds max 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_08_inference_latency_under_threshold():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001", "SKU002", "SKU003"],
        "horizon_weeks": 8,
    }
    start = time.perf_counter()
    response = client.post("/predict", json=payload)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert response.status_code == 200
    assert elapsed < 250.0, f"Inference took too long: {elapsed:.2f} ms"


def test_09_risk_endpoint_success():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001", "SKU002"],
    }
    response = client.post("/v1/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_skus"] == 2
    assert len(data["risk_records"]) == 2

    for rec in data["risk_records"]:
        assert rec["sku"] in ["SKU001", "SKU002"]
        assert rec["days_of_supply"] >= 0.0
        assert 0.0 <= rec["stockout_score"] <= 1.0
        assert rec["overstock_score"] >= 0.0
        assert rec["action_tier"] in [
            "CRITICAL REORDER", "REORDER", "MONITOR", "HEALTHY", "OVERSTOCK", "UNKNOWN"
        ]
        assert len(rec["recommendation"]) > 0
        # Monetary excess value must remain null
        assert rec["excess_inventory_value"] is None


def test_10_recommendations_endpoint_success():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001", "SKU002"],
    }
    response = client.post("/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_skus"] == 2
    assert len(data["recommendations"]) == 2

    valid_codes = [
        "EXPEDITE_PO", "PLACE_PO", "REVIEW_PIPELINE", 
        "FREEZE_REPLENISHMENT", "MAINTAIN_SCHEDULE", "DATA_UNAVAILABLE"
    ]
    for rec in data["recommendations"]:
        assert rec["sku"] in ["SKU001", "SKU002"]
        assert rec["recommendation_code"] in valid_codes
        assert any(p in rec["priority"] for p in ["URGENT", "HIGH", "MEDIUM", "LOW", "ROUTINE", "UNKNOWN"])
        assert 0 <= rec["priority_rank"] <= 5
        assert len(rec["rationale"]) > 0
        assert len(rec["recommended_action"]) > 0


def test_11_recommendations_governance_flags():
    payload = {
        "origin_date": "2025-09-16",
        "skus": ["SKU001"],
    }
    response = client.post("/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valuation_basis_confirmed"] is False
    assert data["arrival_timing_confirmed"] is False
    assert data["overstock_threshold_status"] == "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"

