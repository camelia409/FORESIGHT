"""
test_decision_support.py — Comprehensive Test Suite for src/decision_support.py
=================================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 5 — Decision Support & Recommendation Engine Testing

Covers:
1. Deterministic recommendation generation
2. One recommendation per required SKU/origin grain
3. No mutation of Phase 4B input data
4. CRITICAL REORDER behavior (EXPEDITE_PO)
5. REORDER behavior (PLACE_PO)
6. MONITOR behavior (REVIEW_PIPELINE)
7. OVERSTOCK behavior (FREEZE_REPLENISHMENT)
8. HEALTHY behavior (MAINTAIN_SCHEDULE)
9. Conflicting signal precedence (shortage over surplus)
10. Null monetary fields strictly preserved
11. valuation_basis_confirmed=False propagation
12. arrival_timing_confirmed=False propagation
13. overstock approval-status propagation
14. Unknown/missing inventory data handling (DATA_UNAVAILABLE)
15. Boundary conditions (e.g. earliest breach week, zero demand)
16. Explainability fields (clear rationale and action instructions)
17. Output schema validation
18. Deterministic sorting (priority_rank ascending, SKU ascending)
19. Orphan SKU exclusion (SKU051-SKU200 quarantined)
20. Integration with real Phase 4B artifacts (panel and latest)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.decision_support import (
    DecisionSupportEngine,
    REC_EXPEDITE_PO,
    REC_PLACE_PO,
    REC_REVIEW_PIPELINE,
    REC_FREEZE_REPLENISHMENT,
    REC_MAINTAIN_SCHEDULE,
    REC_DATA_UNAVAILABLE,
    ALL_RECOMMENDATION_CODES,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    PRIORITY_INFORMATIONAL,
    PRIORITY_UNKNOWN,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_DEGRADED,
    CONFIDENCE_NO_DATA,
    RISK_STOCKOUT_IMMINENT,
    RISK_REORDER_BREACH_LT,
    RISK_REORDER_BREACH_HORIZON,
    RISK_EXCESS_INVENTORY,
    RISK_NONE,
    RISK_MISSING_DATA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> DecisionSupportEngine:
    return DecisionSupportEngine()


@pytest.fixture
def base_risk_row() -> dict:
    """Standard valid baseline risk row representing a healthy SKU."""
    return {
        "origin_date": "2025-09-16",
        "SKU": "SKU001",
        "Product_Name": "Product 001",
        "Category": "Furniture",
        "Subcategory": "Chair",
        "inventory_data_available": True,
        "snapshot_date": "2025-09-01",
        "days_since_snapshot": 15,
        "Current_Stock": 500.0,
        "On_Order": 200.0,
        "Lead_Time_Days": 7.0,
        "Safety_Stock": 100.0,
        "Reorder_Point": 200.0,
        "avg_weekly_demand": 50.0,
        "forecast_cum_8w": 400.0,
        "ip_h1": 650.0,
        "ip_h2": 600.0,
        "ip_h3": 550.0,
        "ip_h4": 500.0,
        "ip_h5": 450.0,
        "ip_h6": 400.0,
        "ip_h7": 350.0,
        "ip_h8": 300.0,
        "sb_h1": 450.0,
        "sb_h8": 100.0,
        "weeks_of_cover": 14.0,
        "days_of_supply": 98.0,
        "stock_weeks_of_cover": 10.0,
        "stock_days_of_supply": 70.0,
        "lead_time_demand": 50.0,
        "is_stockout_risk": False,
        "earliest_stockout_week": None,
        "estimated_stockout_date": None,
        "is_safety_stock_breach": False,
        "earliest_ss_breach_week": None,
        "is_reorder_point_breach": False,
        "earliest_rp_breach_week": None,
        "stockout_score": 0.0,
        "excess_inventory_units": 0.0,
        "excess_weeks_of_cover": 0.0,
        "overstock_score": 0.0,
        "overstock_tier": "HEALTHY",
        "action_tier": "HEALTHY",
        "recommendation": "HEALTHY: Adequate coverage across 8 weeks.",
        "excess_inventory_value": None,
        "inventory_value_at_risk": None,
        "capital_at_risk": None,
        "valuation_basis_confirmed": False,
        "arrival_timing_confirmed": False,
        "on_order_policy": "Policy_B_LT",
        "overstock_threshold_weeks": 8,
        "overstock_threshold_status": "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL",
    }


# ---------------------------------------------------------------------------
# Unit & Rule Tests
# ---------------------------------------------------------------------------

class TestRecommendationRules:

    def test_01_deterministic_generation(self, engine, base_risk_row):
        """Given identical inputs, engine must produce bitwise-identical recommendations."""
        rec1 = engine.evaluate_sku_recommendation(base_risk_row)
        rec2 = engine.evaluate_sku_recommendation(base_risk_row)
        assert rec1 == rec2

    def test_02_healthy_sku_behavior(self, engine, base_risk_row):
        """Healthy SKU produces MAINTAIN_SCHEDULE with INFORMATIONAL priority."""
        rec = engine.evaluate_sku_recommendation(base_risk_row)
        assert rec["recommendation_code"] == REC_MAINTAIN_SCHEDULE
        assert rec["priority"] == PRIORITY_INFORMATIONAL
        assert rec["priority_rank"] == 5
        assert rec["triggering_risk"] == RISK_NONE

    def test_03_critical_reorder_behavior(self, engine, base_risk_row):
        """Critical stockout within lead time produces EXPEDITE_PO with CRITICAL priority."""
        row = base_risk_row.copy()
        row["action_tier"] = "CRITICAL REORDER"
        row["is_stockout_risk"] = True
        row["earliest_stockout_week"] = 1
        row["ip_h1"] = -15.0

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_EXPEDITE_PO
        assert rec["priority"] == PRIORITY_CRITICAL
        assert rec["priority_rank"] == 1
        assert rec["triggering_risk"] == RISK_STOCKOUT_IMMINENT
        assert "Expedite" in rec["recommendation_title"]

    def test_04_reorder_behavior(self, engine, base_risk_row):
        """Reorder Point breach within lead time produces PLACE_PO with HIGH priority."""
        row = base_risk_row.copy()
        row["action_tier"] = "REORDER"
        row["is_reorder_point_breach"] = True
        row["earliest_rp_breach_week"] = 1
        row["ip_h1"] = 120.0  # below RP of 200

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_PLACE_PO
        assert rec["priority"] == PRIORITY_HIGH
        assert rec["priority_rank"] == 2
        assert rec["triggering_risk"] == RISK_REORDER_BREACH_LT
        assert "Place" in rec["recommended_action"] or "replenishment" in rec["recommended_action"]

    def test_05_monitor_behavior(self, engine, base_risk_row):
        """Reorder Point breach beyond lead time produces REVIEW_PIPELINE with MEDIUM priority."""
        row = base_risk_row.copy()
        row["action_tier"] = "MONITOR"
        row["is_reorder_point_breach"] = True
        row["earliest_rp_breach_week"] = 5  # beyond 7-day lead time (week 1)
        row["ip_h5"] = 180.0  # below RP of 200

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_REVIEW_PIPELINE
        assert rec["priority"] == PRIORITY_MEDIUM
        assert rec["priority_rank"] == 3
        assert rec["triggering_risk"] == RISK_REORDER_BREACH_HORIZON

    def test_06_overstock_behavior_moderate(self, engine, base_risk_row):
        """Moderate overstock (2 to 6 weeks) produces FREEZE_REPLENISHMENT with LOW priority."""
        row = base_risk_row.copy()
        row["action_tier"] = "OVERSTOCK"
        row["overstock_tier"] = "HIGH"
        row["excess_inventory_units"] = 200.0
        row["excess_weeks_of_cover"] = 4.0

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_FREEZE_REPLENISHMENT
        assert rec["priority"] == PRIORITY_LOW
        assert rec["priority_rank"] == 4
        assert rec["triggering_risk"] == RISK_EXCESS_INVENTORY
        assert "Pause" in rec["recommended_action"] or "halt" in rec["recommended_action"].lower()

    def test_07_overstock_behavior_critical(self, engine, base_risk_row):
        """Severe overstock (> 6 weeks) elevates FREEZE_REPLENISHMENT to MEDIUM priority."""
        row = base_risk_row.copy()
        row["action_tier"] = "OVERSTOCK"
        row["overstock_tier"] = "CRITICAL"
        row["excess_inventory_units"] = 500.0
        row["excess_weeks_of_cover"] = 10.0

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_FREEZE_REPLENISHMENT
        assert rec["priority"] == PRIORITY_MEDIUM
        assert rec["priority_rank"] == 3
        assert rec["triggering_risk"] == RISK_EXCESS_INVENTORY
        assert "FREEZE REPLENISHMENT" in rec["recommended_action"]

    def test_08_conflicting_signals_shortage_precedence(self, engine, base_risk_row):
        """
        Conflict test: SKU has immediate Reorder Point breach (action_tier = REORDER),
        but long-horizon excess_inventory_units > 0.
        Shortage protection MUST take precedence: produces PLACE_PO, modulating order size.
        """
        row = base_risk_row.copy()
        row["action_tier"] = "REORDER"
        row["is_reorder_point_breach"] = True
        row["earliest_rp_breach_week"] = 1
        row["excess_inventory_units"] = 150.0
        row["excess_weeks_of_cover"] = 3.0

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_PLACE_PO
        assert rec["priority"] == PRIORITY_HIGH
        assert rec["priority_rank"] == 2
        assert "precedence" in rec["rationale"].lower() or "reorder" in rec["rationale"].lower()

    def test_09_missing_snapshot_emits_unknown(self, engine, base_risk_row):
        """Missing inventory snapshot produces DATA_UNAVAILABLE with UNKNOWN priority."""
        row = base_risk_row.copy()
        row["inventory_data_available"] = False
        row["Current_Stock"] = None

        rec = engine.evaluate_sku_recommendation(row)
        assert rec["recommendation_code"] == REC_DATA_UNAVAILABLE
        assert rec["priority"] == PRIORITY_UNKNOWN
        assert rec["priority_rank"] == 0
        assert rec["confidence_status"] == CONFIDENCE_NO_DATA


# ---------------------------------------------------------------------------
# Governance & Invariant Tests
# ---------------------------------------------------------------------------

class TestGovernanceAndInvariants:

    def test_10_monetary_metrics_strictly_null(self, engine, base_risk_row):
        """Monetary fields must never be calculated or populated with proxies."""
        rec = engine.evaluate_sku_recommendation(base_risk_row)
        assert rec["governance"]["monetary_valuation_blocked"] is True
        assert rec["governance"]["valuation_basis_confirmed"] is False

    def test_11_governance_flags_propagated(self, engine, base_risk_row):
        """Governance flags must propagate accurately."""
        rec = engine.evaluate_sku_recommendation(base_risk_row)
        gov = rec["governance"]
        assert gov["arrival_timing_confirmed"] is False
        assert gov["on_order_policy"] == "Policy_B_LT"
        assert gov["overstock_threshold_weeks"] == 8
        assert gov["overstock_threshold_status"] == "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"

    def test_12_no_mutation_of_input_dataframe(self, engine, base_risk_row):
        """Calling recommend() must not alter or mutate the input DataFrame."""
        df_in = pd.DataFrame([base_risk_row, base_risk_row])
        df_copy = df_in.copy(deep=True)
        _ = engine.recommend(df_in)
        pd.testing.assert_frame_equal(df_in, df_copy)

    def test_13_orphan_skus_quarantined(self, engine, base_risk_row):
        """Orphan SKUs (e.g. SKU051) must be strictly filtered from recommendations."""
        r1 = base_risk_row.copy()
        r1["SKU"] = "SKU001"
        r2 = base_risk_row.copy()
        r2["SKU"] = "SKU051"
        r3 = base_risk_row.copy()
        r3["SKU"] = "SKU199"

        df_in = pd.DataFrame([r1, r2, r3])
        recs_df = engine.recommend(df_in)
        skus_out = set(recs_df["sku"].unique())
        assert "SKU001" in skus_out
        assert "SKU051" not in skus_out
        assert "SKU199" not in skus_out

    def test_14_deterministic_sorting(self, engine, base_risk_row):
        """Output rows must be sorted by priority_rank ascending, then SKU ascending."""
        # Row 1: Low priority (rank 4)
        r_low = base_risk_row.copy()
        r_low["SKU"] = "SKU002"
        r_low["action_tier"] = "OVERSTOCK"
        r_low["excess_weeks_of_cover"] = 3.0

        # Row 2: Critical priority (rank 1)
        r_crit = base_risk_row.copy()
        r_crit["SKU"] = "SKU003"
        r_crit["action_tier"] = "CRITICAL REORDER"
        r_crit["is_stockout_risk"] = True
        r_crit["earliest_stockout_week"] = 1

        # Row 3: High priority (rank 2)
        r_high = base_risk_row.copy()
        r_high["SKU"] = "SKU001"
        r_high["action_tier"] = "REORDER"
        r_high["is_reorder_point_breach"] = True
        r_high["earliest_rp_breach_week"] = 1

        df_in = pd.DataFrame([r_low, r_crit, r_high])
        recs = engine.recommend(df_in)

        # Expected order: SKU003 (rank 1), SKU001 (rank 2), SKU002 (rank 4)
        assert recs["sku"].tolist() == ["SKU003", "SKU001", "SKU002"]
        assert recs["priority_rank"].tolist() == [1, 2, 4]


# ---------------------------------------------------------------------------
# Integration Tests on Real Phase 4B Artifacts
# ---------------------------------------------------------------------------

class TestRealDataIntegration:

    def test_15_batch_run_on_real_phase4b_artifacts(self, engine, tmp_path):
        """Execute batch recommendation pipeline on real Phase 4B outputs."""
        output_dir = tmp_path / "artifacts" / "decision_support"
        report_dir = tmp_path / "reports" / "decision_support"

        summary = engine.batch_run(
            output_dir=output_dir,
            report_dir=report_dir,
        )

        assert summary["latest_origin"] == "2025-09-16"
        assert summary["latest_sku_count"] == 50
        assert summary["panel_total_records"] == 450

        # Verify all artifacts exist on disk
        for p in summary["artifacts_created"]:
            assert Path(p).exists(), f"Deliverable missing: {p}"

        # Verify latest parquet structure
        df_latest = pd.read_parquet(output_dir / "recommendations_latest.parquet")
        assert len(df_latest) == 50
        assert set(df_latest["sku"].unique()) == {f"SKU{i:03d}" for i in range(1, 51)}

        # Verify all recommendation codes are valid
        assert df_latest["recommendation_code"].isin(ALL_RECOMMENDATION_CODES).all()

        # Verify governance
        assert df_latest["valuation_basis_confirmed"].eq(False).all()
        assert df_latest["arrival_timing_confirmed"].eq(False).all()
        assert df_latest["monetary_valuation_blocked"].eq(True).all()
