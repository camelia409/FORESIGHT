"""
tests/test_phase6_production.py — Production Phase 6 Verification Suite
========================================================================
Validates the complete production deployment infrastructure:
  1. Pipeline Orchestrator DAG execution & manifest generation.
  2. Production SKU universe enforcement (50 production SKUs, 150 quarantined orphans).
  3. Strict governance policy compliance:
     - Decision #1 (Option 1D): Zero monetary exposure metrics.
     - Decision #2 (Option 2A): Policy B_LT on-order accounting.
     - Decision #3 (Option 3C): N=8 weeks ratified overstock horizon.
  4. Extended REST API endpoints (/api/status, /api/inventory, /api/risk, /api/recommendations, /api/governance).
  5. Error handling and 404 validation for quarantined and invalid SKUs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.production_pipeline import ProductionPipelineOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Pipeline Orchestrator & DAG Tests
# ---------------------------------------------------------------------------

def test_pipeline_orchestrator_initialization():
    """Verify production pipeline initializes with valid configuration."""
    pipeline = ProductionPipelineOrchestrator(origin_date="2025-09-16")
    assert pipeline.origin_date_requested == "2025-09-16"
    assert len(pipeline.dag_stages) == 9
    assert pipeline.dag_stages[0] == "INGEST"
    assert pipeline.dag_stages[-1] == "SERVE_EXPORT"


def test_pipeline_execution_and_manifest():
    """Verify end-to-end execution of production pipeline and manifest output."""
    pipeline = ProductionPipelineOrchestrator(origin_date="2025-09-16")
    manifest = pipeline.run_pipeline()

    assert manifest["status"] == "COMPLETED"
    assert manifest["stages_executed"] == 9
    assert manifest["skus_monitored"] == 50
    assert manifest["orphan_skus_quarantined"] == 150
    assert "Option 1D" in manifest["governance_policies"]["decision_1_valuation_basis"]
    assert "Policy_B_LT" in manifest["governance_policies"]["on_order_policy"]
    assert manifest["governance_policies"]["overstock_threshold_status"] == "POLICY_RATIFIED"

    # Verify manifest file on disk
    manifest_path = PROJECT_ROOT / "artifacts" / "phase6" / "pipeline_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "COMPLETED"


def test_production_output_artifacts_and_quarantine():
    """Verify output parquet has exactly 50 production SKUs and zero orphans."""
    parquet_path = PROJECT_ROOT / "artifacts" / "phase6" / "production_latest_recommendations.parquet"
    assert parquet_path.is_file()
    df = pd.read_parquet(parquet_path)

    assert len(df) == 50
    skus = df["sku"].tolist()
    assert all(sku.startswith("SKU0") for sku in skus)
    # Confirm no orphan SKUs SKU051 - SKU200
    for i in range(51, 201):
        orphan = f"SKU{i:03d}"
        assert orphan not in skus


# ---------------------------------------------------------------------------
# 2. Governance Policy Compliance Tests
# ---------------------------------------------------------------------------

def test_decision_1_monetary_exclusion():
    """Verify Decision #1 Option 1D: No monetary valuation metrics are exposed."""
    parquet_path = PROJECT_ROOT / "artifacts" / "phase6" / "production_latest_recommendations.parquet"
    df = pd.read_parquet(parquet_path)

    # Monetary risk columns must NOT be present as non-null numeric values
    for col in ["excess_inventory_value", "inventory_value_at_risk", "capital_at_risk"]:
        if col in df.columns:
            assert df[col].isna().all(), f"Column {col} must be strictly null"
    assert df["monetary_valuation_blocked"].all()


def test_decision_2_on_order_policy_blt():
    """Verify Decision #2 Option 2A: Policy B_LT is applied across recommendations."""
    parquet_path = PROJECT_ROOT / "artifacts" / "phase6" / "production_latest_recommendations.parquet"
    df = pd.read_parquet(parquet_path)

    assert (df["on_order_policy"] == "Policy_B_LT").all()


def test_decision_3_overstock_threshold_ratified():
    """Verify Decision #3 Option 3C: N=8 weeks is ratified."""
    parquet_path = PROJECT_ROOT / "artifacts" / "phase6" / "production_latest_recommendations.parquet"
    df = pd.read_parquet(parquet_path)

    assert (df["overstock_threshold_weeks"] == 8).all()
    assert (df["overstock_threshold_status"] == "POLICY_RATIFIED").all()


# ---------------------------------------------------------------------------
# 3. Extended Production API Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_status_endpoint(client: TestClient):
    """Test GET /api/status endpoint returns valid production system telemetry."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("HEALTHY", "OPERATIONAL")
    assert data["production_skus_count"] == 50
    assert data["quarantined_orphans_count"] == 150
    assert "Option 1D" in data["governance_summary"]["valuation_basis"]
    assert "Policy_B_LT" in data["governance_summary"]["on_order_policy"]
    assert data["governance_summary"]["overstock_threshold_status"] == "POLICY_RATIFIED"


def test_api_inventory_endpoint(client: TestClient):
    """Test GET /api/inventory returns 50 production items."""
    response = client.get("/api/inventory")
    assert response.status_code == 200
    data = response.json()

    assert data["n_skus"] == 50
    items = data["items"]
    assert len(items) == 50
    first = items[0]
    assert "sku" in first
    assert "current_stock" in first
    assert "safety_stock" in first
    assert "reorder_point" in first
    assert first["current_stock"] >= 0


def test_api_inventory_single_sku(client: TestClient):
    """Test GET /api/inventory/{sku} for valid SKU."""
    response = client.get("/api/inventory/SKU010")
    assert response.status_code == 200
    data = response.json()

    assert data["sku"] == "SKU010"
    assert "lead_time_days" in data
    assert "current_stock" in data
    assert len(data["forecast_h1_8"]) == 8


def test_api_inventory_orphan_sku_quarantine(client: TestClient):
    """Test GET /api/inventory/{sku} for quarantined orphan SKU returns 404."""
    response = client.get("/api/inventory/SKU055")
    assert response.status_code == 404
    detail = response.json().get("detail", "").lower()
    assert "quarantined" in detail or "not found" in detail


def test_api_inventory_nonexistent_sku(client: TestClient):
    """Test GET /api/inventory/{sku} for completely non-existent SKU returns 404."""
    response = client.get("/api/inventory/SKU999")
    assert response.status_code == 404


def test_api_risk_endpoint(client: TestClient):
    """Test GET /api/risk returns risk posture across production fleet."""
    response = client.get("/api/risk")
    assert response.status_code == 200
    data = response.json()

    assert data["origin_date"] is not None
    assert data["n_skus"] == 50
    assert len(data["risk_records"]) == 50
    first_record = data["risk_records"][0]
    assert 0.0 <= first_record["stockout_score"] <= 100.0


def test_api_recommendations_endpoint(client: TestClient):
    """Test GET /api/recommendations returns ratified production recommendations."""
    response = client.get("/api/recommendations")
    assert response.status_code == 200
    data = response.json()

    assert data["n_skus"] == 50
    assert len(data["recommendations"]) == 50
    assert data["overstock_threshold_status"] == "POLICY_RATIFIED"
    assert data["on_order_policy"] == "Policy_B_LT"

    # Check first recommendation
    rec = data["recommendations"][0]
    assert rec["priority_rank"] in [1, 2, 3, 4, 5]
    assert rec["recommendation_code"] in [
        "EXPEDITE_PO", "PLACE_PO", "REVIEW_PIPELINE",
        "FREEZE_REPLENISHMENT", "MAINTAIN_SCHEDULE", "DATA_UNAVAILABLE"
    ]


def test_api_governance_endpoint(client: TestClient):
    """Test GET /api/governance returns full ratified governance policy metadata."""
    response = client.get("/api/governance")
    assert response.status_code == 200
    data = response.json()

    assert data["phase_5_status"] == "GOVERNANCE_RATIFIED"
    assert data["phase_6_status"] == "AUTHORIZED_TO_COMMENCE"
    assert "1D" in data["valuation_basis"]
    assert "Policy_B_LT" in data["on_order_policy"]
    assert data["overstock_threshold_weeks"] == 8
    assert data["overstock_threshold_status"] == "POLICY_RATIFIED"
    assert data["monetary_valuation_blocked"] is True
    assert "SKU001" in data["production_sku_universe"]
    assert "Quarantined" in data["orphan_sku_quarantine"]
