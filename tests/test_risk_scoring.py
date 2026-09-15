"""
test_risk_scoring.py — Integration Tests for src/risk_scoring.py
=================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 4B — Inventory Risk Scoring Orchestration Validation

Tests:
1. run_risk_scoring executes on real production forecast & inventory data.
2. Verified output row counts: 50 SKUs per origin.
3. Machine-readable artifacts generated (parquet, JSON, CSV).
4. Monetary valuation blocker strictly enforced on real batch outputs.
5. Action tiers and overstock tiers strictly within taxonomies.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.risk_scoring import run_risk_scoring
from src.config import CFG, PATHS


def test_01_run_risk_scoring_real_data(tmp_path):
    """Execute risk scoring on real production data and verify summary."""
    output_dir = tmp_path / "artifacts" / "risk"
    report_dir = tmp_path / "reports" / "risk"

    summary = run_risk_scoring(
        output_dir=output_dir,
        report_dir=report_dir,
    )

    assert summary["origins_scored"] == 9
    assert summary["total_scored_rows"] == 9 * 50  # 450 rows
    assert summary["latest_origin"] == "2025-09-16"

    # Verify files created
    for p in summary["artifacts_created"]:
        assert Path(p).exists(), f"Artifact missing: {p}"

    # Verify latest JSON payload
    json_path = output_dir / "risk_latest.json"
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["n_skus"] == 50
    assert len(data["records"]) == 50
    assert data["metadata"]["valuation_basis_confirmed"] is False
    assert data["metadata"]["arrival_timing_confirmed"] is False

    # Verify monetary blocker on real outputs
    for rec in data["records"]:
        assert rec["excess_inventory_value"] is None
        assert rec["inventory_value_at_risk"] is None
        assert rec["capital_at_risk"] is None
        assert rec["action_tier"] in [
            "CRITICAL REORDER", "REORDER", "MONITOR", "HEALTHY", "OVERSTOCK", "UNKNOWN"
        ]


def test_02_panel_scores_invariants(tmp_path):
    """Verify invariants on panel parquet output."""
    output_dir = tmp_path / "artifacts" / "risk"
    report_dir = tmp_path / "reports" / "risk"

    run_risk_scoring(
        output_dir=output_dir,
        report_dir=report_dir,
    )

    df_panel = pd.read_parquet(output_dir / "risk_scores_panel.parquet")
    assert len(df_panel) == 450
    assert df_panel["SKU"].nunique() == 50

    # Valuation blocker
    assert df_panel["excess_inventory_value"].isna().all()
    assert df_panel["valuation_basis_confirmed"].eq(False).all()

    # Numerical validity
    valid_mask = df_panel["inventory_data_available"] == True
    assert df_panel.loc[valid_mask, "stockout_score"].between(0.0, 1.0).all()
    assert (df_panel.loc[valid_mask, "excess_inventory_units"] >= 0.0).all()
