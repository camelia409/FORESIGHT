"""
decision_support.py — Production Decision Support & Recommendation Engine
===========================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Phase: 5 — Decision Support & Operational Recommendation Engine

Responsibilities
----------------
- Consume validated Phase 4B risk outputs (risk_scores_panel.parquet, risk_scores_latest.parquet).
- Never recalculate or modify demand forecasts or Phase 4B risk scores.
- Apply a deterministic, rule-based decision support hierarchy to map inventory risks to
  actionable operational recommendations.
- Resolve conflicting inventory signals with strict shortage-over-surplus precedence.
- Enforce Phase 4A/4B governance gates:
    * valuation_basis_confirmed = False (monetary fields remain strictly None)
    * arrival_timing_confirmed = False (On_Order arrival dates are estimated)
    * overstock_threshold_status = "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL"
    * Orphan SKUs (SKU051-SKU200) strictly quarantined.
- Generate operational supply chain guidance:
    * recommendation_code: machine-readable action identifier (EXPEDITE_PO, PLACE_PO, etc.)
    * recommendation_title: human-readable action title
    * recommended_action: concrete operational instructions (order quantity, timing, holds)
    * priority: standardized priority label (1 — CRITICAL, 2 — HIGH, etc.) and priority_rank
    * rationale: quantitative plain-language justification
    * triggering_risk: primary risk condition prompting the action
    * supporting_metrics: dictionary of relevant scalar metrics
    * confidence_status: data freshness rating (HIGH, MEDIUM, DEGRADED, NO_DATA)
    * required_follow_up: specific operational next step
    * governance: dictionary of policy metadata
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import CFG, PATHS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recommendation Codes & Action Taxonomy
# ---------------------------------------------------------------------------

REC_EXPEDITE_PO          = "EXPEDITE_PO"
REC_PLACE_PO             = "PLACE_PO"
REC_REVIEW_PIPELINE      = "REVIEW_PIPELINE"
REC_FREEZE_REPLENISHMENT = "FREEZE_REPLENISHMENT"
REC_MAINTAIN_SCHEDULE    = "MAINTAIN_SCHEDULE"
REC_DATA_UNAVAILABLE     = "DATA_UNAVAILABLE"

ALL_RECOMMENDATION_CODES = [
    REC_EXPEDITE_PO,
    REC_PLACE_PO,
    REC_REVIEW_PIPELINE,
    REC_FREEZE_REPLENISHMENT,
    REC_MAINTAIN_SCHEDULE,
    REC_DATA_UNAVAILABLE,
]

# Priority Tiers & Numerical Ranks (1 = highest urgency)
PRIORITY_CRITICAL      = "1 — CRITICAL"
PRIORITY_HIGH          = "2 — HIGH"
PRIORITY_MEDIUM        = "3 — MEDIUM"
PRIORITY_LOW           = "4 — LOW"
PRIORITY_INFORMATIONAL = "5 — INFORMATIONAL"
PRIORITY_UNKNOWN       = "0 — UNKNOWN"

PRIORITY_RANK_MAP = {
    PRIORITY_CRITICAL: 1,
    PRIORITY_HIGH: 2,
    PRIORITY_MEDIUM: 3,
    PRIORITY_LOW: 4,
    PRIORITY_INFORMATIONAL: 5,
    PRIORITY_UNKNOWN: 0,
}

# Data Confidence Levels
CONFIDENCE_HIGH     = "HIGH_CONFIDENCE"       # <= 14 days old
CONFIDENCE_MEDIUM   = "MEDIUM_CONFIDENCE"     # 15 to 28 days old
CONFIDENCE_DEGRADED = "DEGRADED_CONFIDENCE"   # > 28 days old
CONFIDENCE_NO_DATA  = "NO_DATA"               # missing inventory snapshot

# Triggering Risk Taxonomy
RISK_STOCKOUT_IMMINENT        = "STOCKOUT_IMMINENT"
RISK_REORDER_BREACH_LT        = "REORDER_POINT_BREACH_LT"
RISK_REORDER_BREACH_HORIZON   = "REORDER_POINT_BREACH_HORIZON"
RISK_EXCESS_INVENTORY         = "EXCESS_INVENTORY"
RISK_NONE                     = "NONE"
RISK_MISSING_DATA             = "MISSING_INVENTORY_DATA"


class DecisionSupportEngine:
    """
    Deterministic Decision Support & Operational Recommendation Engine.

    Consumes Phase 4B risk outputs and converts them into prioritized,
    actionable supply chain recommendations without recalculating upstream risk scores.
    """

    def __init__(self):
        self._validate_runtime_governance()

    def _validate_runtime_governance(self) -> None:
        """Verify governance configuration before engine operation."""
        if CFG.inventory_valuation_basis != "unresolved":
            logger.warning(
                "[decision_support] Expected inventory_valuation_basis='unresolved', got '%s'",
                CFG.inventory_valuation_basis,
            )

    # -----------------------------------------------------------------------
    # Core Recommendation Evaluation for a Single SKU Observation
    # -----------------------------------------------------------------------

    @staticmethod
    def evaluate_sku_recommendation(row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate and construct a deterministic recommendation record for a single SKU.

        Parameters
        ----------
        row : pd.Series or dict conforming to Phase 4B risk output schema.

        Returns
        -------
        dict containing standardized recommendation attributes.
        """
        # 1. Identity fields
        orig_val = row.get("origin_date")
        if isinstance(orig_val, (pd.Timestamp, datetime)):
            orig_date_str = str(orig_val.date())
        elif isinstance(orig_val, date):
            orig_date_str = str(orig_val)
        else:
            orig_date_str = str(orig_val)

        sku = str(row.get("SKU", row.get("sku", "")))
        prod_name = row.get("Product_Name")
        category = row.get("Category")
        subcategory = row.get("Subcategory")

        # 2. Check inventory availability
        inv_avail = bool(row.get("inventory_data_available", False))

        if not inv_avail:
            # Case: Snapshot missing / unavailable
            return {
                "origin_date":          orig_date_str,
                "sku":                  sku,
                "product_name":         prod_name,
                "category":             category,
                "subcategory":          subcategory,
                "recommendation_code":  REC_DATA_UNAVAILABLE,
                "recommendation_title": "Inventory Snapshot Missing",
                "recommended_action":   (
                    "Request immediate physical cycle count or verify ERP snapshot synchronization. "
                    "Operational replenishment risk cannot be evaluated without stock observations."
                ),
                "priority":             PRIORITY_UNKNOWN,
                "priority_rank":        0,
                "rationale":            f"No inventory snapshot was recorded at or before origin date {orig_date_str}.",
                "triggering_risk":      RISK_MISSING_DATA,
                "supporting_metrics": {
                    "current_stock":           None,
                    "on_order":                None,
                    "lead_time_days":          None,
                    "safety_stock":            None,
                    "reorder_point":           None,
                    "avg_weekly_demand":       float(row.get("avg_weekly_demand")) if pd.notna(row.get("avg_weekly_demand")) else None,
                    "weeks_of_cover":          None,
                    "excess_weeks_of_cover":   None,
                    "earliest_breach_week":    None,
                    "earliest_stockout_week":  None,
                    "projected_stockout_date": None,
                },
                "confidence_status":    CONFIDENCE_NO_DATA,
                "required_follow_up":   "Contact inventory accounting to restore automated snapshot synchronization.",
                "governance": {
                    "valuation_basis_confirmed": False,
                    "arrival_timing_confirmed":  False,
                    "on_order_policy":           str(row.get("on_order_policy", "Policy_B_LT")),
                    "overstock_threshold_weeks": int(row.get("overstock_threshold_weeks", 8)),
                    "overstock_threshold_status": str(row.get("overstock_threshold_status", "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL")),
                    "monetary_valuation_blocked": True,
                },
            }

        # 3. Extract verified Phase 4B metrics
        curr_stock   = float(row.get("Current_Stock", 0.0))
        on_order     = float(row.get("On_Order", 0.0))
        lt_days      = float(row.get("Lead_Time_Days", 7.0))
        safety_stock = float(row.get("Safety_Stock", 0.0))
        reorder_pt   = float(row.get("Reorder_Point", 0.0))
        avg_demand   = float(row.get("avg_weekly_demand", 0.0))
        woc          = float(row.get("weeks_of_cover", 0.0))
        excess_units = float(row.get("excess_inventory_units", 0.0))
        excess_woc   = float(row.get("excess_weeks_of_cover", 0.0))
        action_tier  = str(row.get("action_tier", ""))
        overstock_tier = str(row.get("overstock_tier", ""))

        earliest_stockout_w = row.get("earliest_stockout_week")
        earliest_rp_w       = row.get("earliest_rp_breach_week")
        earliest_ss_w       = row.get("earliest_ss_breach_week")
        est_stockout_date   = row.get("estimated_stockout_date")
        days_since_snap     = row.get("days_since_snapshot")

        # 4. Determine Data Confidence Level
        if days_since_snap is None or pd.isna(days_since_snap):
            conf_status = CONFIDENCE_MEDIUM
        else:
            d_val = int(days_since_snap)
            if d_val <= 14:
                conf_status = CONFIDENCE_HIGH
            elif d_val <= 28:
                conf_status = CONFIDENCE_MEDIUM
            else:
                conf_status = CONFIDENCE_DEGRADED

        # 5. Determine Supporting Metrics Dict
        supporting_metrics = {
            "current_stock":           curr_stock,
            "on_order":                on_order,
            "lead_time_days":          lt_days,
            "safety_stock":            safety_stock,
            "reorder_point":           reorder_pt,
            "avg_weekly_demand":       avg_demand,
            "weeks_of_cover":          woc if not np.isinf(woc) else 999.0,
            "excess_weeks_of_cover":   excess_woc,
            "earliest_breach_week":    int(earliest_rp_w) if pd.notna(earliest_rp_w) and earliest_rp_w is not None else None,
            "earliest_stockout_week":  int(earliest_stockout_w) if pd.notna(earliest_stockout_w) and earliest_stockout_w is not None else None,
            "projected_stockout_date": str(est_stockout_date) if pd.notna(est_stockout_date) and est_stockout_date is not None else None,
        }

        # 6. Conflict Resolution & Action Logic (Strict Precedence Hierarchy)
        # Hierarchy:
        # Priority 1: CRITICAL REORDER (Stockout imminent within lead time)
        # Priority 2: REORDER (RP breach within lead time)
        # Priority 3: MONITOR (RP breach beyond lead time, or critical overstock)
        # Priority 4: OVERSTOCK (Excess inventory, no shortage)
        # Priority 5: HEALTHY (Balanced coverage)

        lt_weeks = lt_days / 7.0
        lt_horizon_ceil = min(8, max(1, int(math.ceil(lt_weeks))))

        if action_tier == "CRITICAL REORDER":
            rec_code = REC_EXPEDITE_PO
            rec_title = "Expedite Inbound Shipment"
            priority = PRIORITY_CRITICAL
            trig_risk = RISK_STOCKOUT_IMMINENT
            follow_up = "Contact supplier within 24 hours to expedite shipment and obtain tracking confirmation."

            stk_w = int(earliest_stockout_w) if pd.notna(earliest_stockout_w) and earliest_stockout_w is not None else 1
            ip_val = row.get(f"ip_h{stk_w}", 0.0)

            if on_order > 0:
                action_text = (
                    f"Expedite existing inbound purchase order of {on_order:.0f} units immediately or arrange emergency "
                    f"split-shipment. Physical stock of {curr_stock:.0f} units will be depleted in week {stk_w}."
                )
                rationale_text = (
                    f"Projected inventory position drops to {ip_val:.0f} units (stockout) in week {stk_w}, which is within "
                    f"the supplier lead time of {lt_days:.0f} days. Open orders exist but require delivery acceleration."
                )
            else:
                target_emergency_qty = max(1, int(round(max(reorder_pt - curr_stock, 4.0 * avg_demand))))
                action_text = (
                    f"Issue emergency purchase order for ~{target_emergency_qty} units with air/priority freight. "
                    f"Zero inbound orders exist and physical stock of {curr_stock:.0f} units exhausts in week {stk_w}."
                )
                rationale_text = (
                    f"Stockout projected in week {stk_w} ({ip_val:.0f} units balance) within lead time of {lt_days:.0f} days "
                    f"with no open replenishment orders in the pipeline."
                )

        elif action_tier == "REORDER":
            rec_code = REC_PLACE_PO
            rec_title = "Issue Replenishment Purchase Order"
            priority = PRIORITY_HIGH
            trig_risk = RISK_REORDER_BREACH_LT
            follow_up = "Create purchase order requisition and transmit to vendor within 48 hours."

            rp_w = int(earliest_rp_w) if pd.notna(earliest_rp_w) and earliest_rp_w is not None else 1
            ip_at_rp = row.get(f"ip_h{rp_w}", 0.0)
            ltd_val = row.get("lead_time_demand", lt_weeks * avg_demand)

            # Order sizing rule (purely unit-based):
            rec_order_qty = max(1, int(round(max(reorder_pt - ip_at_rp + ltd_val, 4.0 * avg_demand))))

            # Conflict resolution: Simultaneous reorder breach and long-horizon overstock?
            if excess_units > 0:
                action_text = (
                    f"Issue a calibrated replenishment order for ~{rec_order_qty} units to protect week {rp_w} Reorder Point breach. "
                    f"Note: Long-horizon inventory indicates {excess_woc:.1f} weeks of terminal surplus; do not exceed required batch size."
                )
                rationale_text = (
                    f"Inventory position breaches Reorder Point ({reorder_pt:.0f} units) in week {rp_w} (lead time is {lt_days:.0f} days). "
                    f"Reorder takes strict precedence over terminal surplus to prevent intermediate stockout."
                )
            else:
                action_text = (
                    f"Place replenishment purchase order for ~{rec_order_qty} units. "
                    f"Target supplier delivery on or before week {rp_w}."
                )
                rationale_text = (
                    f"Projected inventory position ({ip_at_rp:.0f} units) breaches Reorder Point ({reorder_pt:.0f} units) "
                    f"in week {rp_w}, inside the supplier lead time window of {lt_days:.0f} days."
                )

        elif action_tier == "MONITOR":
            rec_code = REC_REVIEW_PIPELINE
            rec_title = "Review Replenishment Pipeline"
            priority = PRIORITY_MEDIUM
            trig_risk = RISK_REORDER_BREACH_HORIZON
            follow_up = "Review open purchase order ETAs during weekly inventory planning meeting."

            rp_w = int(earliest_rp_w) if pd.notna(earliest_rp_w) and earliest_rp_w is not None else lt_horizon_ceil + 1
            ip_at_rp = row.get(f"ip_h{rp_w}", 0.0)
            order_by_week = max(1, rp_w - lt_horizon_ceil)

            action_text = (
                f"Prepare replenishment purchase order for week {order_by_week}. "
                f"Current pipeline is adequate through week {rp_w - 1}, with Reorder Point breach projected in week {rp_w}."
            )
            rationale_text = (
                f"Inventory position remains above Reorder Point during immediate lead time, but declines to {ip_at_rp:.0f} units "
                f"(below Reorder Point of {reorder_pt:.0f} units) in week {rp_w}."
            )

        elif action_tier == "OVERSTOCK" or excess_woc > 0.0:
            rec_code = REC_FREEZE_REPLENISHMENT
            rec_title = "Halt Inbound Replenishment"
            trig_risk = RISK_EXCESS_INVENTORY

            if overstock_tier == "CRITICAL" or excess_woc > 6.0:
                priority = PRIORITY_MEDIUM  # Elevated priority due to extreme capital tie-up
                action_text = (
                    f"FREEZE REPLENISHMENT: Severe overstock detected ({excess_units:.0f} excess units, {excess_woc:.1f} weeks of cover "
                    f"above safety buffer). Suspend all new purchase orders for {int(math.ceil(excess_woc))} weeks. "
                    f"Evaluate immediate promotional markdowns, bundle offers, or stock transfers."
                )
                follow_up = "Escalate to merchandise planning for promotional markdown or supplier return assessment."
            elif overstock_tier == "HIGH" or excess_woc > 2.0:
                priority = PRIORITY_LOW
                action_text = (
                    f"Pause purchase orders for {max(1, int(math.ceil(excess_woc)))} weeks. "
                    f"Total pipeline provides {woc:.1f} weeks of cover ({excess_units:.0f} units surplus above 8-week horizon)."
                )
                follow_up = "Monitor weekly demand consumption and confirm no automated replenishment POs are released."
            else:
                priority = PRIORITY_LOW
                action_text = (
                    f"Elevated stock: pipeline holds {excess_units:.0f} units ({excess_woc:.1f} weeks) above 8-week horizon target. "
                    f"Defer upcoming order placements."
                )
                follow_up = "Review inventory position at next monthly snapshot."

            rationale_text = (
                f"Projected inventory position at horizon h=8 exceeds required safety stock by {excess_units:.0f} units "
                f"({excess_woc:.1f} surplus weeks of demand). Evaluated against approved 8-week production forecast horizon."
            )

        else:
            # Case: HEALTHY / Balanced
            rec_code = REC_MAINTAIN_SCHEDULE
            rec_title = "Maintain Normal Schedule"
            priority = PRIORITY_INFORMATIONAL
            trig_risk = RISK_NONE
            action_text = (
                f"Inventory coverage is well-balanced across the 8-week horizon ({woc:.1f} weeks of cover). "
                f"No supply chain intervention required."
            )
            rationale_text = (
                f"Projected inventory position remains above Reorder Point throughout planning horizon and clears "
                f"naturally without excess stock."
            )
            follow_up = "Continue weekly forecast and sales monitoring."

        return {
            "origin_date":          orig_date_str,
            "sku":                  sku,
            "product_name":         prod_name,
            "category":             category,
            "subcategory":          subcategory,
            "recommendation_code":  rec_code,
            "recommendation_title": rec_title,
            "recommended_action":   action_text,
            "priority":             priority,
            "priority_rank":        PRIORITY_RANK_MAP.get(priority, 99),
            "rationale":            rationale_text,
            "triggering_risk":      trig_risk,
            "supporting_metrics":   supporting_metrics,
            "confidence_status":    conf_status,
            "required_follow_up":   follow_up,
            "governance": {
                "valuation_basis_confirmed": False,
                "arrival_timing_confirmed":  False,
                "on_order_policy":           str(row.get("on_order_policy", "Policy_B_LT")),
                "overstock_threshold_weeks": int(row.get("overstock_threshold_weeks", 8)),
                "overstock_threshold_status": str(row.get("overstock_threshold_status", "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL")),
                "monetary_valuation_blocked": True,
            },
        }

    # -----------------------------------------------------------------------
    # Public Batch API: recommend()
    # -----------------------------------------------------------------------

    def recommend(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate operational recommendations for an entire DataFrame of risk scores.

        Parameters
        ----------
        risk_df : DataFrame containing Phase 4B risk outputs.

        Returns
        -------
        DataFrame of recommendation records, deterministically sorted by priority_rank and SKU.
        """
        # Guard: do not mutate input DataFrame
        df = risk_df.copy()

        # Enforce orphan quarantine: restrict to SKU001-SKU050
        valid_skus = {f"SKU{i:03d}" for i in range(1, CFG.KNOWN_MASTER_SKU_COUNT + 1)}
        sku_col = "SKU" if "SKU" in df.columns else "sku"
        if sku_col in df.columns:
            df = df[df[sku_col].isin(valid_skus)].copy()

        recs = []
        for _, row in df.iterrows():
            rec = self.evaluate_sku_recommendation(row)
            # Flatten governance for tabular DataFrame representation
            rec_row = {
                "origin_date":                rec["origin_date"],
                "sku":                        rec["sku"],
                "product_name":               rec["product_name"],
                "category":                   rec["category"],
                "subcategory":                rec["subcategory"],
                "recommendation_code":        rec["recommendation_code"],
                "recommendation_title":       rec["recommendation_title"],
                "recommended_action":         rec["recommended_action"],
                "priority":                   rec["priority"],
                "priority_rank":              rec["priority_rank"],
                "rationale":                  rec["rationale"],
                "triggering_risk":            rec["triggering_risk"],
                "confidence_status":          rec["confidence_status"],
                "required_follow_up":         rec["required_follow_up"],
                # Supporting metrics unpacked
                "current_stock":              rec["supporting_metrics"]["current_stock"],
                "on_order":                   rec["supporting_metrics"]["on_order"],
                "lead_time_days":             rec["supporting_metrics"]["lead_time_days"],
                "safety_stock":               rec["supporting_metrics"]["safety_stock"],
                "reorder_point":              rec["supporting_metrics"]["reorder_point"],
                "avg_weekly_demand":          rec["supporting_metrics"]["avg_weekly_demand"],
                "weeks_of_cover":             rec["supporting_metrics"]["weeks_of_cover"],
                "excess_weeks_of_cover":      rec["supporting_metrics"]["excess_weeks_of_cover"],
                "earliest_breach_week":       rec["supporting_metrics"]["earliest_breach_week"],
                "earliest_stockout_week":     rec["supporting_metrics"]["earliest_stockout_week"],
                "projected_stockout_date":    rec["supporting_metrics"]["projected_stockout_date"],
                # Governance flags
                "valuation_basis_confirmed":  rec["governance"]["valuation_basis_confirmed"],
                "arrival_timing_confirmed":   rec["governance"]["arrival_timing_confirmed"],
                "on_order_policy":            rec["governance"]["on_order_policy"],
                "overstock_threshold_weeks":  rec["governance"]["overstock_threshold_weeks"],
                "overstock_threshold_status": rec["governance"]["overstock_threshold_status"],
                "monetary_valuation_blocked": rec["governance"]["monetary_valuation_blocked"],
            }
            recs.append(rec_row)

        recs_df = pd.DataFrame(recs)

        # Deterministic sort: origin_date ascending, priority_rank ascending, sku ascending
        sort_cols = [c for c in ["origin_date", "priority_rank", "sku"] if c in recs_df.columns]
        if sort_cols:
            recs_df = recs_df.sort_values(sort_cols, ascending=[True, True, True]).reset_index(drop=True)

        return recs_df

    # -----------------------------------------------------------------------
    # Batch Pipeline Execution & Artifact Generation
    # -----------------------------------------------------------------------

    def batch_run(
        self,
        risk_panel_path:  Optional[Path] = None,
        risk_latest_path: Optional[Path] = None,
        output_dir:       Optional[Path] = None,
        report_dir:       Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute batch decision support on Phase 4B risk outputs and save deliverables.

        Parameters
        ----------
        risk_panel_path  : Path to artifacts/risk/risk_scores_panel.parquet.
        risk_latest_path : Path to artifacts/risk/risk_scores_latest.parquet.
        output_dir       : Destination for machine-readable parquet/JSON artifacts.
        report_dir       : Destination for human-readable CSV/markdown reports.

        Returns
        -------
        dict containing execution metrics, counts, and paths of generated files.
        """
        if output_dir is None:
            output_dir = PATHS.metrics_dir.parent / "decision_support"
        output_dir.mkdir(parents=True, exist_ok=True)

        if report_dir is None:
            report_dir = PATHS.reports_data_quality_dir.parent / "decision_support"
        report_dir.mkdir(parents=True, exist_ok=True)

        if risk_panel_path is None:
            risk_panel_path = PATHS.metrics_dir.parent / "risk" / "risk_scores_panel.parquet"

        if risk_latest_path is None:
            risk_latest_path = PATHS.metrics_dir.parent / "risk" / "risk_scores_latest.parquet"

        logger.info("[decision_support] Reading Phase 4B panel from %s", risk_panel_path)
        panel_risk = pd.read_parquet(risk_panel_path)

        logger.info("[decision_support] Reading Phase 4B latest from %s", risk_latest_path)
        latest_risk = pd.read_parquet(risk_latest_path)

        # Generate recommendations
        panel_recs = self.recommend(panel_risk)
        latest_recs = self.recommend(latest_risk)

        # Artifact paths
        panel_parquet  = output_dir / "recommendations_panel.parquet"
        latest_parquet = output_dir / "recommendations_latest.parquet"
        summary_json   = output_dir / "recommendation_summary.json"

        panel_csv  = report_dir / "recommendations_panel.csv"
        latest_csv = report_dir / "recommendations_latest.csv"

        # Save Parquet & CSV
        panel_recs.to_parquet(panel_parquet, index=False)
        latest_recs.to_parquet(latest_parquet, index=False)
        panel_recs.to_csv(panel_csv, index=False)
        latest_recs.to_csv(latest_csv, index=False)

        # Generate recommendation_summary.json
        summary_payload = {
            "metadata": {
                "engine": "DecisionSupportEngine",
                "phase": "Phase 5 — Decision Support & Recommendation Engine",
                "execution_timestamp": datetime.now().isoformat(),
                "governance": {
                    "valuation_basis_confirmed": False,
                    "arrival_timing_confirmed": False,
                    "on_order_policy": "Policy_B_LT",
                    "overstock_threshold_weeks": 8,
                    "overstock_threshold_status": "ENGINEERING_RECOMMENDATION_PENDING_BUSINESS_APPROVAL",
                    "monetary_valuation_blocked": True,
                },
            },
            "latest_origin": str(latest_recs["origin_date"].iloc[0]),
            "latest_sku_count": len(latest_recs),
            "latest_priority_distribution": latest_recs["priority"].value_counts().to_dict(),
            "latest_recommendation_distribution": latest_recs["recommendation_code"].value_counts().to_dict(),
            "panel_total_records": len(panel_recs),
            "panel_priority_distribution": panel_recs["priority"].value_counts().to_dict(),
            "panel_recommendation_distribution": panel_recs["recommendation_code"].value_counts().to_dict(),
            "artifacts_created": [
                str(panel_parquet),
                str(latest_parquet),
                str(summary_json),
                str(panel_csv),
                str(latest_csv),
            ],
        }

        with open(summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)

        logger.info("[decision_support] Batch run complete. Latest breakdown: %s", summary_payload["latest_recommendation_distribution"])
        return summary_payload


__all__ = [
    "DecisionSupportEngine",
    "REC_EXPEDITE_PO",
    "REC_PLACE_PO",
    "REC_REVIEW_PIPELINE",
    "REC_FREEZE_REPLENISHMENT",
    "REC_MAINTAIN_SCHEDULE",
    "REC_DATA_UNAVAILABLE",
    "ALL_RECOMMENDATION_CODES",
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_MEDIUM",
    "PRIORITY_LOW",
    "PRIORITY_INFORMATIONAL",
    "PRIORITY_UNKNOWN",
    "PRIORITY_RANK_MAP",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_DEGRADED",
    "CONFIDENCE_NO_DATA",
    "RISK_STOCKOUT_IMMINENT",
    "RISK_REORDER_BREACH_LT",
    "RISK_REORDER_BREACH_HORIZON",
    "RISK_EXCESS_INVENTORY",
    "RISK_NONE",
    "RISK_MISSING_DATA",
]
