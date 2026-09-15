"""
test_risk_engine.py — Comprehensive Test Suite for src/risk_engine.py
====================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 4B — Inventory Risk Engine Testing

Validates:
- Exact deterministic unit risk formulas.
- Horizon-conditional On_Order inclusion under approved Policy B_LT.
- Terminal-horizon excess inventory calculation (Phase 4A Decision #3).
- Overstock severity boundary cutoffs exactly at 0, 2, and 6 weeks.
- Action tier classification hierarchy (CRITICAL REORDER > REORDER > MONITOR > OVERSTOCK > HEALTHY).
- Missing snapshot handling & UNKNOWN tier assignment.
- Strict orphan SKU quarantine (SKU051-SKU200 excluded).
- Monetary valuation blocker enforcement (valuation_basis_confirmed=False, monetary fields=None).
- Policy provenance metadata integrity.
- Adversarial temporal leakage prevention (Snapshot_Date > origin must never be read).
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from src.risk_engine import (
    RiskEngine,
    ALL_TIERS,
    ALL_OVERSTOCK_TIERS,
    TIER_CRITICAL_REORDER,
    TIER_REORDER,
    TIER_MONITOR,
    TIER_HEALTHY,
    TIER_OVERSTOCK,
    TIER_UNKNOWN,
    OVERSTOCK_TIER_HEALTHY,
    OVERSTOCK_TIER_MONITOR,
    OVERSTOCK_TIER_HIGH,
    OVERSTOCK_TIER_CRITICAL,
    OVERSTOCK_TIER_UNKNOWN,
    ON_ORDER_POLICY_NAME,
    OVERSTOCK_THRESHOLD_WEEKS_DEFAULT,
    OVERSTOCK_THRESHOLD_STATUS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine()


@pytest.fixture
def deterministic_forecast() -> pd.DataFrame:
    """
    8-week flat forecast of 50 units/week for SKU001 and SKU002 at origin 2025-01-06.
    Cumulative 8-week demand = 400 units. Average weekly demand = 50.0 units.
    """
    rows = []
    origin = pd.Timestamp("2025-01-06")
    for sku in ["SKU001", "SKU002"]:
        for h in range(1, 9):
            rows.append({
                "SKU": sku,
                "forecast_origin_date": origin,
                "horizon": h,
                "target_date": origin + pd.Timedelta(weeks=h),
                "prediction": 50.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def deterministic_inventory() -> pd.DataFrame:
    """
    Snapshot observed on 2025-01-01 (valid for origin 2025-01-06).
    SKU001:
      Current_Stock = 200, On_Order = 100, Lead_Time_Days = 7, Safety_Stock = 50, Reorder_Point = 100
      Total Pipeline = 300
      Avg Demand = 50 u/wk
      WoC = 300 / 50 = 6.0 weeks
      IP at h=1: 200 + 100 (arrives in week 1 since LT=7 <= 7) - 50 = 250
      IP at h=8: 200 + 100 - 400 = -100
      Excess: max(0, 300 - 400 - 50) = 0
    SKU002:
      Current_Stock = 800, On_Order = 200, Lead_Time_Days = 14, Safety_Stock = 100, Reorder_Point = 150
      Total Pipeline = 1000
      Avg Demand = 50 u/wk
      WoC = 1000 / 50 = 20.0 weeks
      IP at h=1: 800 + 0 (LT=14 > 7) - 50 = 750
      IP at h=2: 800 + 200 (LT=14 <= 14) - 100 = 900
      IP at h=8: 800 + 200 - 400 = 600
      Excess Units: max(0, 1000 - 400 - 100) = 500 units
      Excess WoC: 500 / 50 = 10.0 weeks (> 6 weeks -> CRITICAL overstock)
    """
    return pd.DataFrame({
        "Snapshot_Date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "SKU": ["SKU001", "SKU002"],
        "Current_Stock": [200.0, 800.0],
        "On_Order": [100.0, 200.0],
        "Lead_Time_Days": [7.0, 14.0],
        "Safety_Stock": [50.0, 100.0],
        "Reorder_Point": [100.0, 150.0],
        "Inventory_Value": [50000.0, 250000.0],
    })


# ---------------------------------------------------------------------------
# Unit Formula & Deterministic Arithmetic Tests
# ---------------------------------------------------------------------------

class TestDeterministicRiskFormulas:

    def test_horizon_conditional_on_order_inclusion(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        Policy B_LT test:
        SKU002 has Lead_Time_Days = 14.
        For h=1 (7 days): 14 <= 7 is FALSE -> On_Order (200) MUST NOT be included.
          IP_h1 = Current_Stock (800) + 0 - y_hat_h1 (50) = 750.
        For h=2 (14 days): 14 <= 14 is TRUE -> On_Order (200) MUST be included.
          IP_h2 = Current_Stock (800) + 200 - (50 + 50) = 900.
        """
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku2 = scores[scores["SKU"] == "SKU002"].iloc[0]

        assert row_sku2["ip_h1"] == pytest.approx(750.0)
        assert row_sku2["ip_h2"] == pytest.approx(900.0)

    def test_stock_balance_conservative(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """SB(t, h) = Current_Stock - CumDemand (excludes On_Order entirely)."""
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku1 = scores[scores["SKU"] == "SKU001"].iloc[0]
        # Current_Stock = 200, CumDemand at h=1 is 50, at h=8 is 400
        assert row_sku1["sb_h1"] == pytest.approx(150.0)
        assert row_sku1["sb_h8"] == pytest.approx(-200.0)

    def test_weeks_of_cover_and_days_of_supply(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """WoC = (Current_Stock + On_Order) / AvgWeeklyDemand, DoS = WoC * 7."""
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku1 = scores[scores["SKU"] == "SKU001"].iloc[0]
        # Total pipeline = 300, avg demand = 50 -> WoC = 6.0, DoS = 42.0
        assert row_sku1["weeks_of_cover"] == pytest.approx(6.0)
        assert row_sku1["days_of_supply"] == pytest.approx(42.0)
        # Stock only: 200 / 50 = 4.0 weeks
        assert row_sku1["stock_weeks_of_cover"] == pytest.approx(4.0)
        assert row_sku1["stock_days_of_supply"] == pytest.approx(28.0)

    def test_lead_time_demand_exact(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        SKU001: LT = 7 days -> 1 week -> LTD = 50.0.
        SKU002: LT = 14 days -> 2 weeks -> LTD = 100.0.
        """
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku1 = scores[scores["SKU"] == "SKU001"].iloc[0]
        row_sku2 = scores[scores["SKU"] == "SKU002"].iloc[0]
        assert row_sku1["lead_time_demand"] == pytest.approx(50.0)
        assert row_sku2["lead_time_demand"] == pytest.approx(100.0)

    def test_stockout_risk_score_calculation(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        SR(t) = 1 - min(1, WoC / (LT_weeks + SS_weeks)).
        SKU001: WoC = 6.0, LT_weeks = 1.0, SS_weeks = 50/50 = 1.0. Denom = 2.0.
        WoC / Denom = 6.0 / 2.0 = 3.0. min(1, 3.0) = 1.0 -> SR = 0.0.
        """
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku1 = scores[scores["SKU"] == "SKU001"].iloc[0]
        assert row_sku1["stockout_score"] == pytest.approx(0.0)

    def test_breaches_and_earliest_weeks(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        SKU001:
          ip_h: [250, 200, 150, 100, 50, 0, -50, -100]
          Safety_Stock = 50 -> ip_h5 = 50 (< 50 is false), ip_h6 = 0 (< 50 is true) -> earliest_ss = 6
          Reorder_Point = 100 -> ip_h4 = 100, ip_h5 = 50 (< 100 is true) -> earliest_rp = 5
          Stockout: ip_h6 = 0 (<= 0 is true) -> earliest_stockout = 6
        """
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku1 = scores[scores["SKU"] == "SKU001"].iloc[0]
        assert row_sku1["is_safety_stock_breach"] is True
        assert row_sku1["earliest_ss_breach_week"] == 6
        assert row_sku1["is_reorder_point_breach"] is True
        assert row_sku1["earliest_rp_breach_week"] == 5
        assert row_sku1["is_stockout_risk"] is True
        assert row_sku1["earliest_stockout_week"] == 6


# ---------------------------------------------------------------------------
# Core Excess Inventory & Severity Boundary Tests
# ---------------------------------------------------------------------------

class TestExcessInventoryAndSeverityBoundaries:

    def test_core_excess_formula_exact(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        Excess_Units = max(0, Inventory_Position(t, 8) - Sum(Forecast) - Safety_Stock)
        SKU002: Available supply = 800 + 200 = 1000. Sum(Forecast) = 400. Safety_Stock = 100.
        Excess_Units = 1000 - 400 - 100 = 500.
        Excess_WoC = 500 / 50 = 10.0 weeks.
        Overstock_Score = 500 / 1000 = 0.5.
        """
        scores = risk_engine.score(
            deterministic_forecast,
            deterministic_inventory,
            origin_date="2025-01-06",
        )
        row_sku2 = scores[scores["SKU"] == "SKU002"].iloc[0]
        assert row_sku2["excess_inventory_units"] == pytest.approx(500.0)
        assert row_sku2["excess_weeks_of_cover"] == pytest.approx(10.0)
        assert row_sku2["overstock_score"] == pytest.approx(0.5)
        assert row_sku2["overstock_tier"] == OVERSTOCK_TIER_CRITICAL

    @pytest.mark.parametrize("excess_woc,expected_tier", [
        (-1.0, OVERSTOCK_TIER_HEALTHY),
        (0.0,  OVERSTOCK_TIER_HEALTHY),
        (0.01, OVERSTOCK_TIER_MONITOR),
        (1.0,  OVERSTOCK_TIER_MONITOR),
        (2.0,  OVERSTOCK_TIER_MONITOR),
        (2.01, OVERSTOCK_TIER_HIGH),
        (4.0,  OVERSTOCK_TIER_HIGH),
        (6.0,  OVERSTOCK_TIER_HIGH),
        (6.01, OVERSTOCK_TIER_CRITICAL),
        (12.0, OVERSTOCK_TIER_CRITICAL),
    ])
    def test_severity_boundaries_exact(self, risk_engine, excess_woc, expected_tier):
        """
        Test exact mathematical cutoffs:
        HEALTHY:  WoC <= 0
        MONITOR:  0 < WoC <= 2
        HIGH:     2 < WoC <= 6
        CRITICAL: WoC > 6
        """
        # Setup synthetic scenario
        demand = 50.0
        cum_8w = 400.0
        ss = 100.0
        # excess_units = excess_woc * demand
        target_excess = max(0.0, excess_woc * demand)
        # supply = target_excess + cum_8w + ss
        supply = target_excess + cum_8w + ss if excess_woc >= 0 else (cum_8w + ss - 50.0)

        inv_df = pd.DataFrame([{
            "Snapshot_Date": "2025-01-01",
            "SKU": "SKU001",
            "Current_Stock": supply,
            "On_Order": 0.0,
            "Lead_Time_Days": 7.0,
            "Safety_Stock": ss,
            "Reorder_Point": 150.0,
            "Inventory_Value": 1000.0,
        }])
        fc_df = pd.DataFrame([{
            "SKU": "SKU001",
            "forecast_origin_date": "2025-01-06",
            "horizon": h,
            "prediction": demand,
        } for h in range(1, 9)])

        scores = risk_engine.score(fc_df, inv_df, origin_date="2025-01-06")
        assert scores["overstock_tier"].iloc[0] == expected_tier


# ---------------------------------------------------------------------------
# Action Tier Hierarchy & Edge Cases
# ---------------------------------------------------------------------------

class TestActionTierHierarchy:

    def test_critical_reorder_takes_precedence(self, risk_engine):
        """Stockout within lead time window triggers CRITICAL REORDER regardless of other metrics."""
        inv_df = pd.DataFrame([{
            "Snapshot_Date": "2025-01-01",
            "SKU": "SKU001",
            "Current_Stock": 10.0,  # very low stock
            "On_Order": 500.0,
            "Lead_Time_Days": 14.0, # arrival in week 2
            "Safety_Stock": 50.0,
            "Reorder_Point": 100.0,
            "Inventory_Value": 1000.0,
        }])
        fc_df = pd.DataFrame([{
            "SKU": "SKU001",
            "forecast_origin_date": "2025-01-06",
            "horizon": h,
            "prediction": 50.0,
        } for h in range(1, 9)])
        # In week 1: Current_Stock=10, Demand=50 -> IP=-40 (stockout within LT of 14d)
        scores = risk_engine.score(fc_df, inv_df, origin_date="2025-01-06")
        assert scores["action_tier"].iloc[0] == TIER_CRITICAL_REORDER

    def test_reorder_tier_within_lead_time(self, risk_engine):
        """RP breach within lead time triggers REORDER."""
        inv_df = pd.DataFrame([{
            "Snapshot_Date": "2025-01-01",
            "SKU": "SKU001",
            "Current_Stock": 70.0,
            "On_Order": 0.0,
            "Lead_Time_Days": 7.0, # 1 week
            "Safety_Stock": 20.0,
            "Reorder_Point": 50.0, # RP breach when IP < 50
            "Inventory_Value": 1000.0,
        }])
        fc_df = pd.DataFrame([{
            "SKU": "SKU001",
            "forecast_origin_date": "2025-01-06",
            "horizon": h,
            "prediction": 30.0, # at h=1: IP = 70 - 30 = 40 < RP(50)
        } for h in range(1, 9)])
        scores = risk_engine.score(fc_df, inv_df, origin_date="2025-01-06")
        assert scores["action_tier"].iloc[0] == TIER_REORDER

    def test_overstock_tier_assigned_when_excess_present(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """SKU002 has large surplus and no stockout/RP breach -> OVERSTOCK."""
        scores = risk_engine.score(deterministic_forecast, deterministic_inventory, origin_date="2025-01-06")
        assert scores[scores["SKU"] == "SKU002"]["action_tier"].iloc[0] == TIER_OVERSTOCK


# ---------------------------------------------------------------------------
# Temporal Safety & Missing Snapshot Tests
# ---------------------------------------------------------------------------

class TestTemporalSafetyAndMissingSnapshots:

    def test_missing_snapshot_emits_unknown_tier(self, risk_engine, deterministic_forecast):
        """When no snapshot exists for a SKU, risk engine must set UNKNOWN tier and null scores."""
        empty_inv = pd.DataFrame(columns=[
            "Snapshot_Date", "SKU", "Current_Stock", "On_Order",
            "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "Inventory_Value"
        ])
        scores = risk_engine.score(deterministic_forecast, empty_inv, origin_date="2025-01-06")
        row = scores.iloc[0]
        assert bool(row["inventory_data_available"]) is False
        assert row["action_tier"] == TIER_UNKNOWN
        assert row["overstock_tier"] == OVERSTOCK_TIER_UNKNOWN
        assert pd.isna(row["stockout_score"])
        assert pd.isna(row["excess_inventory_units"])

    def test_adversarial_temporal_leakage_prevented(self, risk_engine, deterministic_forecast):
        """
        If a snapshot exists ONLY in the future (Snapshot_Date > origin_date),
        it MUST NOT be read. The engine must treat inventory as unavailable.
        """
        future_inv = pd.DataFrame([{
            "Snapshot_Date": "2025-01-15", # strictly after origin 2025-01-06
            "SKU": "SKU001",
            "Current_Stock": 500.0,
            "On_Order": 100.0,
            "Lead_Time_Days": 7.0,
            "Safety_Stock": 50.0,
            "Reorder_Point": 100.0,
            "Inventory_Value": 1000.0,
        }])
        scores = risk_engine.score(deterministic_forecast, future_inv, origin_date="2025-01-06")
        row = scores[scores["SKU"] == "SKU001"].iloc[0]
        assert bool(row["inventory_data_available"]) is False
        assert row["action_tier"] == TIER_UNKNOWN


# ---------------------------------------------------------------------------
# Orphan Quarantine & Governance Integrity Tests
# ---------------------------------------------------------------------------

class TestOrphanQuarantineAndGovernance:

    def test_orphan_skus_strictly_excluded(self, risk_engine, deterministic_forecast):
        """
        Orphan inventory SKUs (SKU051-SKU200) must be quarantined and excluded
        from scoring output.
        """
        inv_with_orphans = pd.DataFrame([
            {"Snapshot_Date": "2025-01-01", "SKU": "SKU001", "Current_Stock": 200, "On_Order": 0, "Lead_Time_Days": 7, "Safety_Stock": 50, "Reorder_Point": 100, "Inventory_Value": 1000},
            {"Snapshot_Date": "2025-01-01", "SKU": "SKU051", "Current_Stock": 500, "On_Order": 0, "Lead_Time_Days": 7, "Safety_Stock": 50, "Reorder_Point": 100, "Inventory_Value": 2000},
            {"Snapshot_Date": "2025-01-01", "SKU": "SKU199", "Current_Stock": 300, "On_Order": 0, "Lead_Time_Days": 7, "Safety_Stock": 50, "Reorder_Point": 100, "Inventory_Value": 3000},
        ])
        scores = risk_engine.score(deterministic_forecast, inv_with_orphans, origin_date="2025-01-06")
        scored_skus = set(scores["SKU"].unique())
        assert "SKU051" not in scored_skus
        assert "SKU199" not in scored_skus

    def test_monetary_metrics_strictly_blocked(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """
        Decision #1 compliance: Monetary valuation basis is unresolved.
        Monetary fields must remain None / null across all records.
        """
        scores = risk_engine.score(deterministic_forecast, deterministic_inventory, origin_date="2025-01-06")
        for _, row in scores.iterrows():
            assert row["excess_inventory_value"] is None
            assert row["inventory_value_at_risk"] is None
            assert row["capital_at_risk"] is None
            assert row["valuation_basis_confirmed"] is False

    def test_governance_metadata_presence(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """Verify presence and accuracy of Phase 4A governance metadata."""
        scores = risk_engine.score(deterministic_forecast, deterministic_inventory, origin_date="2025-01-06")
        row = scores.iloc[0]
        assert bool(row["arrival_timing_confirmed"]) is False
        assert bool(row["valuation_basis_confirmed"]) is False
        assert row["on_order_policy"] == ON_ORDER_POLICY_NAME
        assert row["overstock_threshold_weeks"] == OVERSTOCK_THRESHOLD_WEEKS_DEFAULT
        assert row["overstock_threshold_status"] == OVERSTOCK_THRESHOLD_STATUS

    def test_generate_report_end_to_end(self, risk_engine, deterministic_forecast, deterministic_inventory):
        """Test full generate_report pipeline with classify and recommend steps."""
        report = risk_engine.generate_report(
            origin_date="2025-01-06",
            forecast_df=deterministic_forecast,
            inventory_df=deterministic_inventory,
        )
        assert not report.empty
        assert "action_tier" in report.columns
        assert "recommendation" in report.columns
        for tier in report["action_tier"]:
            assert tier in ALL_TIERS
