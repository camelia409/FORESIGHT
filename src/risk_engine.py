"""
risk_engine.py — Stockout / Overstock Risk Scoring & Decision Engine
=====================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 4B — Inventory Risk Engine Implementation

Responsibilities
----------------
- Consume 8-week demand forecasts from production forecasting engine / inference pipeline.
- Consume inventory snapshots (Current_Stock, On_Order, Lead_Time_Days, Safety_Stock, Reorder_Point).
- Enforce strict temporal safety: only snapshots observed at or before forecast origin t (Snapshot_Date <= t).
- Apply approved Phase 4A Policy B_LT for horizon-conditional On_Order inclusion:
      IP(t, h) = Current_Stock(t) + On_Order(t) * I[Lead_Time_Days(t) <= 7 * h] - CumDemand(t, h)
- Compute deterministic, unit-based risk metrics for h=1..8:
    * Projected Inventory Position IP(t, h)
    * Projected Stock Balance SB(t, h) (conservative physical stock)
    * Days of Supply (DoS) / Weeks of Cover (WoC)
    * Lead-Time Demand (LTD)
    * Safety Stock breach (SS_BREACH) and earliest breach week
    * Reorder Point breach (RP_BREACH) and earliest breach week
    * Continuous Stockout Risk score SR(t) in [0.0, 1.0]
    * Terminal-horizon excess inventory units Delta_excess(t)
    * Excess weeks of cover WoC_excess(t)
    * Continuous overstock score OS(t)
    * Estimated stockout date
- Classify SKUs into operational action tiers:
    * CRITICAL REORDER : stockout imminent within lead time window
    * REORDER          : reorder point breach within lead time window
    * MONITOR          : reorder point breach within 8-week planning horizon
    * OVERSTOCK        : excess inventory exceeding 8-week clearance demand + safety stock
    * HEALTHY          : adequate coverage, no breach, no excess
    * UNKNOWN          : missing inventory snapshot at origin t
- Classify excess inventory into 4-tier overstock severity:
    * HEALTHY  : WoC_excess <= 0.0
    * MONITOR  : 0.0 < WoC_excess <= 2.0
    * HIGH     : 2.0 < WoC_excess <= 6.0
    * CRITICAL : WoC_excess > 6.0
- Enforce Phase 4A Governance & Blockers:
    * valuation_basis_confirmed = False (Inventory_Value is independent constant)
    * arrival_timing_confirmed = False (On_Order arrival dates unobserved)
    * Monetary excess and capital-at-risk fields remain strictly None / NaN
    * overstock_threshold_weeks = 8 (engineering recommendation pending business approval)
    * Orphan SKUs (SKU051-SKU200) strictly quarantined and excluded.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import CFG, PATHS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Taxonomies
# ---------------------------------------------------------------------------

# Action Tier Taxonomy (Phase 4A Section H.1)
TIER_CRITICAL_REORDER = "CRITICAL REORDER"
TIER_REORDER          = "REORDER"
TIER_MONITOR          = "MONITOR"
TIER_HEALTHY          = "HEALTHY"
TIER_OVERSTOCK        = "OVERSTOCK"
TIER_UNKNOWN          = "UNKNOWN"

ALL_TIERS = [
    TIER_CRITICAL_REORDER,
    TIER_REORDER,
    TIER_MONITOR,
    TIER_HEALTHY,
    TIER_OVERSTOCK,
    TIER_UNKNOWN,
]

# Overstock Severity Tiers (Phase 4A Section H / Decision #3)
OVERSTOCK_TIER_HEALTHY  = "HEALTHY"
OVERSTOCK_TIER_MONITOR  = "MONITOR"
OVERSTOCK_TIER_HIGH     = "HIGH"
OVERSTOCK_TIER_CRITICAL = "CRITICAL"
OVERSTOCK_TIER_UNKNOWN  = "UNKNOWN"

ALL_OVERSTOCK_TIERS = [
    OVERSTOCK_TIER_HEALTHY,
    OVERSTOCK_TIER_MONITOR,
    OVERSTOCK_TIER_HIGH,
    OVERSTOCK_TIER_CRITICAL,
    OVERSTOCK_TIER_UNKNOWN,
]

# Policy Governance Constants
ON_ORDER_POLICY_NAME = "Policy_B_LT"
OVERSTOCK_THRESHOLD_WEEKS_DEFAULT = 8
OVERSTOCK_THRESHOLD_STATUS = "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"


class RiskEngine:
    """
    Inventory risk scoring, classification, and recommendation engine.

    Consumes 8-week forward demand forecasts and point-in-time inventory snapshots
    to compute deterministic inventory health metrics. Adheres strictly to
    Phase 4A governance gates: monetary metrics remain gated as None.
    """

    def __init__(
        self,
        overstock_threshold_weeks: int = OVERSTOCK_THRESHOLD_WEEKS_DEFAULT,
        on_order_policy: str = ON_ORDER_POLICY_NAME,
    ):
        self.overstock_threshold_weeks = int(overstock_threshold_weeks)
        self.on_order_policy = str(on_order_policy)
        self._config_check()

    def _config_check(self) -> None:
        """Log governance status and policy warnings."""
        if CFG.inventory_valuation_basis == "unresolved":
            logger.info(
                "[risk_engine] Inventory valuation basis is UNRESOLVED. "
                "Monetary metrics (Excess_Inventory_Value, IVaR) remain gated as None."
            )
        if CFG.orphan_sku_treatment == "quarantine":
            logger.debug(
                "[risk_engine] Orphan SKU treatment is 'quarantine'. "
                "Non-master inventory SKUs will be excluded from production scoring."
            )

    @staticmethod
    def get_governance_metadata() -> Dict[str, Any]:
        """Return provenance metadata and governance flags."""
        return {
            "valuation_basis_confirmed": False,
            "arrival_timing_confirmed": False,
            "on_order_policy": ON_ORDER_POLICY_NAME,
            "overstock_threshold_weeks": OVERSTOCK_THRESHOLD_WEEKS_DEFAULT,
            "overstock_threshold_status": OVERSTOCK_THRESHOLD_STATUS,
            "monetary_metrics_blocked": True,
            "production_skus_expected": CFG.KNOWN_MASTER_SKU_COUNT,
        }

    # -----------------------------------------------------------------------
    # Core Computation Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_forecasts(
        forecast_df: pd.DataFrame,
        horizon_weeks: int = 8,
    ) -> Dict[Tuple[Any, str], List[float]]:
        """
        Normalize forecast input into a dictionary keyed by (origin_date, SKU) -> [y_hat_1..y_hat_8].
        Supports long format (SKU, horizon, prediction) and wide format.
        """
        f_df = forecast_df.copy()

        # Identify SKU column
        sku_col = None
        for c in ["SKU", "sku"]:
            if c in f_df.columns:
                sku_col = c
                break
        if sku_col is None:
            raise KeyError("Forecast DataFrame must contain a 'SKU' or 'sku' column.")

        # Identify origin column
        origin_col = None
        for c in ["forecast_origin_date", "origin_date", "forecast_origin", "origin"]:
            if c in f_df.columns:
                origin_col = c
                break

        # Identify horizon and prediction columns if long format
        h_col = None
        for c in ["horizon", "horizon_step", "h"]:
            if c in f_df.columns:
                h_col = c
                break

        pred_col = None
        for c in ["prediction", "units_forecast", "forecast", "pred", "Units_Sold"]:
            if c in f_df.columns:
                pred_col = c
                break

        forecast_map: Dict[Tuple[Any, str], List[float]] = {}

        if h_col is not None and pred_col is not None:
            # Long format
            if origin_col is None:
                # Assign dummy origin if not present
                f_df["_origin"] = pd.Timestamp("2025-01-01").date()
                origin_col = "_origin"

            f_df["_origin_norm"] = pd.to_datetime(f_df[origin_col]).dt.date
            grouped = f_df.groupby(["_origin_norm", sku_col])

            for (orig, sku), grp in grouped:
                grp_sorted = grp.sort_values(h_col)
                preds = grp_sorted[pred_col].astype(float).tolist()
                # Ensure exactly horizon_weeks length
                if len(preds) < horizon_weeks:
                    last_val = preds[-1] if preds else 0.0
                    preds.extend([last_val] * (horizon_weeks - len(preds)))
                forecast_map[(orig, str(sku))] = [max(0.0, p) for p in preds[:horizon_weeks]]

        else:
            # Check for wide format columns (e.g. h1..h8 or target_h1..target_h8)
            wide_cols = []
            for h in range(1, horizon_weeks + 1):
                candidates = [f"h{h}", f"target_h{h}", f"pred_h{h}", f"forecast_h{h}"]
                found = False
                for cand in candidates:
                    if cand in f_df.columns:
                        wide_cols.append(cand)
                        found = True
                        break
                if not found:
                    break

            if len(wide_cols) == horizon_weeks:
                if origin_col is None:
                    f_df["_origin"] = pd.Timestamp("2025-01-01").date()
                    origin_col = "_origin"
                f_df["_origin_norm"] = pd.to_datetime(f_df[origin_col]).dt.date

                for _, row in f_df.iterrows():
                    orig = row["_origin_norm"]
                    sku = str(row[sku_col])
                    preds = [max(0.0, float(row[col])) for col in wide_cols]
                    forecast_map[(orig, sku)] = preds
            else:
                raise ValueError(
                    "Forecast DataFrame must contain either long-format [SKU, horizon, prediction] "
                    "or wide-format [h1..h8] forecast columns."
                )

        return forecast_map

    @staticmethod
    def _find_latest_snapshot(
        inventory_df: pd.DataFrame,
        sku: str,
        origin_dt: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """
        Retrieve the latest inventory snapshot observed at or before origin_dt for sku.
        Strict temporal safety: Snapshot_Date <= origin_dt.
        """
        sku_snaps = inventory_df[
            (inventory_df["SKU"] == sku) &
            (inventory_df["Snapshot_Date"] <= origin_dt)
        ]
        if sku_snaps.empty:
            return None
        return sku_snaps.sort_values("Snapshot_Date", ascending=False).iloc[0]

    # -----------------------------------------------------------------------
    # Public API: score()
    # -----------------------------------------------------------------------

    def score(
        self,
        forecast_df:   pd.DataFrame,
        inventory_df:  pd.DataFrame,
        sku_master_df: Optional[pd.DataFrame] = None,
        origin_date:   Optional[Union[str, date, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """
        Compute per-SKU inventory risk scores using demand forecasts and inventory snapshots.

        Parameters
        ----------
        forecast_df   : Weekly SKU demand forecasts (h=1..8).
        inventory_df  : Inventory snapshots (Current_Stock, On_Order, Lead_Time_Days, Safety_Stock, Reorder_Point).
        sku_master_df : Optional SKU master table (for orphan filtering & category enrichment).
        origin_date   : Optional forecast origin filter. If None, derives from forecast_df.

        Returns
        -------
        pd.DataFrame containing deterministic unit risk metrics per SKU.
        """
        inv = inventory_df.copy()
        inv["Snapshot_Date"] = pd.to_datetime(inv["Snapshot_Date"])

        # Orphan SKU quarantine enforcement
        valid_skus = None
        if sku_master_df is not None and "SKU" in sku_master_df.columns:
            valid_skus = set(sku_master_df["SKU"].unique())
        else:
            # Fallback to known production master SKU range SKU001-SKU050
            valid_skus = {f"SKU{i:03d}" for i in range(1, CFG.KNOWN_MASTER_SKU_COUNT + 1)}

        # Filter inventory and forecast to master universe
        inv = inv[inv["SKU"].isin(valid_skus)].copy()

        # Parse forecasts
        f_map = self._normalize_forecasts(forecast_df, horizon_weeks=8)

        # Determine evaluation origins
        if origin_date is not None:
            target_origins = [pd.to_datetime(origin_date).date()]
        else:
            target_origins = sorted(list(set(orig for (orig, _) in f_map.keys())))

        # Extract master SKU set
        all_skus = sorted(list(valid_skus))

        rows = []
        for orig in target_origins:
            orig_ts = pd.Timestamp(orig)

            for sku in all_skus:
                # 1. Check forecast availability
                forecasts = f_map.get((orig, sku))
                if forecasts is None:
                    # If origin not in dict, check if a single origin was parsed
                    single_orig_matches = [v for (o, s), v in f_map.items() if s == sku]
                    if len(single_orig_matches) == 1:
                        forecasts = single_orig_matches[0]

                # 2. Check snapshot availability (backward asof, Snapshot_Date <= orig_ts)
                snap = self._find_latest_snapshot(inv, sku, orig_ts)

                if snap is None:
                    # Missing inventory snapshot: emit null record with UNKNOWN tier
                    rows.append({
                        "origin_date":               orig,
                        "SKU":                       sku,
                        "inventory_data_available":  False,
                        "snapshot_date":             None,
                        "days_since_snapshot":       None,
                        "Current_Stock":             None,
                        "On_Order":                  None,
                        "Lead_Time_Days":            None,
                        "Safety_Stock":              None,
                        "Reorder_Point":             None,
                        "avg_weekly_demand":         float(np.mean(forecasts)) if forecasts else None,
                        "forecast_cum_8w":           float(np.sum(forecasts)) if forecasts else None,
                        "ip_h1":                     None,
                        "ip_h2":                     None,
                        "ip_h3":                     None,
                        "ip_h4":                     None,
                        "ip_h5":                     None,
                        "ip_h6":                     None,
                        "ip_h7":                     None,
                        "ip_h8":                     None,
                        "sb_h1":                     None,
                        "sb_h8":                     None,
                        "weeks_of_cover":            None,
                        "days_of_supply":            None,
                        "stock_weeks_of_cover":      None,
                        "stock_days_of_supply":      None,
                        "lead_time_demand":          None,
                        "is_stockout_risk":          None,
                        "earliest_stockout_week":    None,
                        "estimated_stockout_date":   None,
                        "is_safety_stock_breach":    None,
                        "earliest_ss_breach_week":   None,
                        "is_reorder_point_breach":   None,
                        "earliest_rp_breach_week":   None,
                        "stockout_score":            None,
                        "excess_inventory_units":    None,
                        "excess_weeks_of_cover":     None,
                        "overstock_score":           None,
                        "overstock_tier":            OVERSTOCK_TIER_UNKNOWN,
                        "action_tier":               TIER_UNKNOWN,
                        "recommendation":            f"UNKNOWN: No inventory snapshot available at or before {orig}.",
                        # Governance fields
                        "excess_inventory_value":    None,
                        "inventory_value_at_risk":   None,
                        "capital_at_risk":           None,
                        "valuation_basis_confirmed": False,
                        "arrival_timing_confirmed":  False,
                        "on_order_policy":           self.on_order_policy,
                        "overstock_threshold_weeks": self.overstock_threshold_weeks,
                        "overstock_threshold_status": OVERSTOCK_THRESHOLD_STATUS,
                    })
                    continue

                # 3. Process available snapshot
                curr_stock   = float(snap["Current_Stock"])
                on_order     = float(snap["On_Order"])
                lt_days      = float(snap["Lead_Time_Days"])
                safety_stock = float(snap["Safety_Stock"])
                reorder_pt   = float(snap["Reorder_Point"])
                snap_date    = pd.to_datetime(snap["Snapshot_Date"]).date()
                days_since_snap = (orig - snap_date).days

                # Forecast vector (h=1..8)
                if forecasts is None:
                    forecasts = [0.0] * 8
                cum_forecasts = np.cumsum(forecasts).tolist()
                cum_8w = float(cum_forecasts[-1])
                avg_weekly_d = float(np.mean(forecasts))

                # 4. Projected Inventory Position IP(t, h) under Policy B_LT
                # IP(t, h) = Current_Stock + On_Order * I[Lead_Time_Days <= 7*h] - CumDemand(t, h)
                ip_h = []
                sb_h = []
                for h in range(1, 9):
                    on_order_effective = on_order if (lt_days <= 7.0 * h) else 0.0
                    ip_val = curr_stock + on_order_effective - cum_forecasts[h - 1]
                    sb_val = curr_stock - cum_forecasts[h - 1]
                    ip_h.append(float(ip_val))
                    sb_h.append(float(sb_val))

                # 5. Weeks of Cover & Days of Supply
                total_pipeline = curr_stock + on_order
                if avg_weekly_d > 0.0:
                    woc_ip    = total_pipeline / avg_weekly_d
                    dos_ip    = woc_ip * 7.0
                    woc_stock = curr_stock / avg_weekly_d
                    dos_stock = woc_stock * 7.0
                else:
                    woc_ip    = float("inf") if total_pipeline > 0 else 0.0
                    dos_ip    = float("inf") if total_pipeline > 0 else 0.0
                    woc_stock = float("inf") if curr_stock > 0 else 0.0
                    dos_stock = float("inf") if curr_stock > 0 else 0.0

                # 6. Lead-Time Demand LTD(t)
                # Pro-rata for fractional weeks
                lt_weeks = lt_days / 7.0
                full_weeks = int(math.floor(lt_weeks))
                frac_week = lt_weeks - full_weeks
                ltd = 0.0
                for k in range(min(full_weeks, 8)):
                    ltd += forecasts[k]
                if full_weeks < 8 and frac_week > 0:
                    ltd += forecasts[full_weeks] * frac_week

                # 7. Breaches & Stockout Checks across h=1..8
                stockout_weeks = [h for h, val in enumerate(ip_h, start=1) if val <= 0.0]
                is_stockout    = len(stockout_weeks) > 0
                earliest_stockout_w = stockout_weeks[0] if is_stockout else None

                ss_breach_weeks = [h for h, val in enumerate(ip_h, start=1) if val < safety_stock]
                is_ss_breach    = len(ss_breach_weeks) > 0
                earliest_ss_w   = ss_breach_weeks[0] if is_ss_breach else None

                rp_breach_weeks = [h for h, val in enumerate(ip_h, start=1) if val < reorder_pt]
                is_rp_breach    = len(rp_breach_weeks) > 0
                earliest_rp_w   = rp_breach_weeks[0] if is_rp_breach else None

                # Estimated stockout date
                est_stockout_date = None
                if earliest_stockout_w is not None:
                    est_stockout_date = orig + timedelta(weeks=earliest_stockout_w)

                # 8. Continuous Stockout Risk Score SR(t) in [0.0, 1.0]
                if avg_weekly_d <= 0.0:
                    sr_score = 0.0
                elif total_pipeline <= 0.0:
                    sr_score = 1.0
                else:
                    ss_weeks = safety_stock / avg_weekly_d
                    denom = lt_weeks + ss_weeks
                    ratio = woc_ip / denom if denom > 0 else 1.0
                    sr_score = float(np.clip(1.0 - min(1.0, ratio), 0.0, 1.0))

                # 9. Terminal Excess Inventory Formula (Phase 4A Decision #3)
                # Excess_Units(t) = max(0, Inventory_Position(t, 8) - Sum(Forecast(t+1..t+8)) - Safety_Stock(t))
                # where Inventory_Position(t, 8) = Current_Stock + On_Order * I[LT <= 56]
                on_order_at_h8 = on_order if (lt_days <= 56.0) else 0.0
                available_supply_h8 = curr_stock + on_order_at_h8
                excess_units = max(0.0, available_supply_h8 - cum_8w - safety_stock)

                if avg_weekly_d > 0.0:
                    excess_woc = excess_units / avg_weekly_d
                else:
                    excess_woc = 0.0

                overstock_score = (excess_units / total_pipeline) if total_pipeline > 0 else 0.0

                # 10. Overstock Severity Tiering (0, 2, 6 weeks boundaries)
                if excess_woc <= 0.0:
                    os_tier = OVERSTOCK_TIER_HEALTHY
                elif excess_woc <= 2.0:
                    os_tier = OVERSTOCK_TIER_MONITOR
                elif excess_woc <= 6.0:
                    os_tier = OVERSTOCK_TIER_HIGH
                else:
                    os_tier = OVERSTOCK_TIER_CRITICAL

                # 11. Operational Action Tier Classification
                # Priority: CRITICAL REORDER -> REORDER -> MONITOR -> OVERSTOCK -> HEALTHY
                lt_horizon_ceil = min(8, max(1, int(math.ceil(lt_weeks))))

                stockout_in_lt = any(ip_h[k - 1] <= 0.0 for k in range(1, lt_horizon_ceil + 1))
                rp_breach_in_lt = any(ip_h[k - 1] < reorder_pt for k in range(1, lt_horizon_ceil + 1))

                if stockout_in_lt:
                    action_tier = TIER_CRITICAL_REORDER
                elif rp_breach_in_lt:
                    action_tier = TIER_REORDER
                elif is_rp_breach:
                    action_tier = TIER_MONITOR
                elif excess_woc > 0.0:
                    action_tier = TIER_OVERSTOCK
                else:
                    action_tier = TIER_HEALTHY

                # 12. Human-Readable Recommendation Text
                if action_tier == TIER_CRITICAL_REORDER:
                    recom = (
                        f"CRITICAL REORDER: Stockout projected in week {earliest_stockout_w} "
                        f"(within lead time of {lt_days:.0f} days). Expedite replenishment immediately."
                    )
                elif action_tier == TIER_REORDER:
                    recom = (
                        f"REORDER: Projected inventory breaches Reorder Point ({reorder_pt:.0f} units) "
                        f"in week {earliest_rp_w} (within lead time of {lt_days:.0f} days). Place PO now."
                    )
                elif action_tier == TIER_MONITOR:
                    recom = (
                        f"MONITOR: Reorder Point breach projected in week {earliest_rp_w}. "
                        f"Review replenishment pipeline."
                    )
                elif action_tier == TIER_OVERSTOCK:
                    recom = (
                        f"OVERSTOCK ({os_tier}): {excess_units:.0f} excess units ({excess_woc:.1f} weeks of cover "
                        f"above safety buffer). Defer future orders or review promotional clearance."
                    )
                else:
                    recom = (
                        f"HEALTHY: Adequate inventory coverage across 8-week horizon "
                        f"({woc_ip:.1f} weeks of cover). No action required."
                    )

                rows.append({
                    "origin_date":               orig,
                    "SKU":                       sku,
                    "inventory_data_available":  True,
                    "snapshot_date":             snap_date,
                    "days_since_snapshot":       days_since_snap,
                    "Current_Stock":             curr_stock,
                    "On_Order":                  on_order,
                    "Lead_Time_Days":            lt_days,
                    "Safety_Stock":              safety_stock,
                    "Reorder_Point":             reorder_pt,
                    "avg_weekly_demand":         avg_weekly_d,
                    "forecast_cum_8w":           cum_8w,
                    "ip_h1":                     ip_h[0],
                    "ip_h2":                     ip_h[1],
                    "ip_h3":                     ip_h[2],
                    "ip_h4":                     ip_h[3],
                    "ip_h5":                     ip_h[4],
                    "ip_h6":                     ip_h[5],
                    "ip_h7":                     ip_h[6],
                    "ip_h8":                     ip_h[7],
                    "sb_h1":                     sb_h[0],
                    "sb_h8":                     sb_h[7],
                    "weeks_of_cover":            woc_ip,
                    "days_of_supply":            dos_ip,
                    "stock_weeks_of_cover":      woc_stock,
                    "stock_days_of_supply":      dos_stock,
                    "lead_time_demand":          ltd,
                    "is_stockout_risk":          is_stockout,
                    "earliest_stockout_week":    earliest_stockout_w,
                    "estimated_stockout_date":   est_stockout_date,
                    "is_safety_stock_breach":    is_ss_breach,
                    "earliest_ss_breach_week":   earliest_ss_w,
                    "is_reorder_point_breach":   is_rp_breach,
                    "earliest_rp_breach_week":   earliest_rp_w,
                    "stockout_score":            sr_score,
                    "excess_inventory_units":    excess_units,
                    "excess_weeks_of_cover":     excess_woc,
                    "overstock_score":           overstock_score,
                    "overstock_tier":            os_tier,
                    "action_tier":               action_tier,
                    "recommendation":            recom,
                    # Governance & Valuation Blocker fields
                    "excess_inventory_value":    None,
                    "inventory_value_at_risk":   None,
                    "capital_at_risk":           None,
                    "valuation_basis_confirmed": False,
                    "arrival_timing_confirmed":  False,
                    "on_order_policy":           self.on_order_policy,
                    "overstock_threshold_weeks": self.overstock_threshold_weeks,
                    "overstock_threshold_status": OVERSTOCK_THRESHOLD_STATUS,
                })

        scores_df = pd.DataFrame(rows)
        return scores_df

    # -----------------------------------------------------------------------
    # Public API: classify() & recommend()
    # -----------------------------------------------------------------------

    def classify(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with guaranteed 'action_tier' column in ALL_TIERS.
        If action_tier already computed by score(), validates and returns.
        """
        df = scores_df.copy()
        if "action_tier" not in df.columns:
            raise KeyError("scores_df must contain 'action_tier'. Call score() first.")
        # Ensure all tiers are valid
        invalid = df[~df["action_tier"].isin(ALL_TIERS)]
        if not invalid.empty:
            raise ValueError(f"Found invalid action tiers: {invalid['action_tier'].unique()}")
        return df

    def recommend(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with structured 'recommendation' text column.
        If recommendation already computed by score(), validates and returns.
        """
        df = classified_df.copy()
        if "recommendation" not in df.columns:
            raise KeyError("classified_df must contain 'recommendation'. Call score() first.")
        return df

    # -----------------------------------------------------------------------
    # Public API: generate_report()
    # -----------------------------------------------------------------------

    def generate_report(
        self,
        origin_date:   Union[str, date, pd.Timestamp],
        forecast_df:   Optional[pd.DataFrame] = None,
        inventory_df:  Optional[pd.DataFrame] = None,
        sku_master_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Orchestrate end-to-end risk report for origin_date:
        1. Load default production datasets if not provided.
        2. Compute unit risk scores, classifications, and recommendations.
        3. Validate schema and numerical invariants.
        4. Return clean, production-ready report.
        """
        orig_ts = pd.Timestamp(origin_date)

        # 1. Load forecast_df if not provided
        if forecast_df is None:
            final_pred_path = PATHS.models_dir / "production" / "final_predictions.parquet"
            if not final_pred_path.exists():
                final_pred_path = PATHS.metrics_dir.parent / "models" / "final" / "final_predictions.parquet"
            if final_pred_path.exists():
                forecast_df = pd.read_parquet(final_pred_path)
            else:
                raise FileNotFoundError(f"Production forecast file not found: {final_pred_path}")

        # 2. Load inventory_df if not provided
        if inventory_df is None:
            inventory_df = pd.read_csv(PATHS.raw_inventory)

        # 3. Load sku_master_df if not provided
        if sku_master_df is None:
            sku_master_df = pd.read_csv(PATHS.raw_sku_master)

        scores_df = self.score(
            forecast_df=forecast_df,
            inventory_df=inventory_df,
            sku_master_df=sku_master_df,
            origin_date=orig_ts,
        )

        classified_df = self.classify(scores_df)
        report_df = self.recommend(classified_df)
        return report_df


__all__ = [
    "RiskEngine",
    "TIER_CRITICAL_REORDER",
    "TIER_REORDER",
    "TIER_MONITOR",
    "TIER_HEALTHY",
    "TIER_OVERSTOCK",
    "TIER_UNKNOWN",
    "ALL_TIERS",
    "OVERSTOCK_TIER_HEALTHY",
    "OVERSTOCK_TIER_MONITOR",
    "OVERSTOCK_TIER_HIGH",
    "OVERSTOCK_TIER_CRITICAL",
    "OVERSTOCK_TIER_UNKNOWN",
    "ALL_OVERSTOCK_TIERS",
    "ON_ORDER_POLICY_NAME",
    "OVERSTOCK_THRESHOLD_WEEKS_DEFAULT",
    "OVERSTOCK_THRESHOLD_STATUS",
]
