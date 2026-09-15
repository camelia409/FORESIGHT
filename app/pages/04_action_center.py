"""
app/pages/04_action_center.py — Tactical Action Center & Decision Support
========================================================================
Operational decision console providing prioritized, deterministic action recommendations
for procurement buyers, inventory controllers, and supply chain planners.

Strict Governance:
  - Decision #1 (Option 1D): ZERO monetary valuation metrics. Unit-based only.
  - Decision #2 (Option 2A): Policy B_LT supplier lead time on-order inclusion.
  - Decision #3 (Option 3C): N=8 weeks ratified overstock horizon.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path for Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np

try:
    from app.data_loader import (
        load_latest_recommendations,
    )
except ModuleNotFoundError:
    from data_loader import (
        load_latest_recommendations,
    )


st.set_page_config(
    page_title="Action Center | FORESIGHT",
    page_icon="⚡",
    layout="wide",
)

st.markdown("# ⚡ Tactical Action Center")
st.markdown("Deterministic, prioritized inventory replenishment, expediting, and surplus management recommendations.")

st.divider()

# Load data
df_recs = load_latest_recommendations()

if df_recs.empty:
    st.error("⚠️ No recommendation data available. Run the production pipeline.")
    st.stop()

# ---------------------------------------------------------------------------
# Action Summary Metrics
# ---------------------------------------------------------------------------
expedite_count = len(df_recs[df_recs["recommendation_code"] == "EXPEDITE_PO"])
place_po_count = len(df_recs[df_recs["recommendation_code"] == "PLACE_PO"])
review_count   = len(df_recs[df_recs["recommendation_code"] == "REVIEW_PIPELINE"])
freeze_count   = len(df_recs[df_recs["recommendation_code"] == "FREEZE_REPLENISHMENT"])
maintain_count = len(df_recs[df_recs["recommendation_code"] == "MAINTAIN_SCHEDULE"])

a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Expedite PO (P1)", f"{expedite_count} SKUs", "Immediate Supplier Contact", delta_color="inverse")
a2.metric("Place New PO (P2)", f"{place_po_count} SKUs", "Reorder Required", delta_color="inverse")
a3.metric("Review Pipeline (P3)", f"{review_count} SKUs", "Order Arriving Post-Breach", delta_color="off")
a4.metric("Freeze Order (P4)", f"{freeze_count} SKUs", "Surplus > 8w Coverage", delta_color="off")
a5.metric("Maintain (P5)", f"{maintain_count} SKUs", "Healthy Schedule")

st.divider()

# ---------------------------------------------------------------------------
# Role-Based Worklist Filtering
# ---------------------------------------------------------------------------
st.markdown("### 🎯 Operational Role Views")

role_tab_all, role_tab_procure, role_tab_inv, role_tab_ops = st.tabs([
    "🌐 Complete Enterprise Worklist (50)",
    "🛒 Procurement & Sourcing (P1 & P2)",
    "📦 Inventory Control & Surplus (P3 & P4)",
    "🚚 Operations & Warehouse (All Scheduled)",
])

with role_tab_procure:
    st.info("Showing immediate purchasing actions: Expedite Inbound POs and Place New Purchase Orders.")
    df_procure = df_recs[df_recs["priority_rank"].isin([1, 2])].sort_values("priority_rank")
    st.dataframe(
        df_procure[[
            "sku", "product_name", "category", "recommendation_code",
            "current_stock", "on_order", "reorder_point", "lead_time_days",
            "recommended_action", "required_follow_up"
        ]],
        use_container_width=True,
        hide_index=True,
    )

with role_tab_inv:
    st.info("Showing pipeline reviews and surplus freezes to prevent unnecessary working capital commitment.")
    df_inv = df_recs[df_recs["priority_rank"].isin([3, 4])].sort_values("priority_rank")
    st.dataframe(
        df_inv[[
            "sku", "product_name", "category", "recommendation_code",
            "current_stock", "weeks_of_cover", "excess_weeks_of_cover",
            "recommended_action", "required_follow_up"
        ]],
        use_container_width=True,
        hide_index=True,
    )

with role_tab_ops:
    st.info("Showing healthy stock maintaining regular operational schedules.")
    df_ops = df_recs[df_recs["priority_rank"] == 5].sort_values("sku")
    st.dataframe(
        df_ops[[
            "sku", "product_name", "category", "current_stock",
            "weeks_of_cover", "avg_weekly_demand", "recommended_action"
        ]],
        use_container_width=True,
        hide_index=True,
    )

with role_tab_all:
    # Interactive multi-filters
    f_c1, f_c2, f_c3 = st.columns([2, 2, 1])
    with f_c1:
        all_cats = sorted(df_recs["category"].dropna().unique())
        f_cat = st.multiselect("Filter Category", options=all_cats, key="all_cat_filter")
    with f_c2:
        all_codes = sorted(df_recs["recommendation_code"].dropna().unique())
        f_code = st.multiselect("Filter Action Code", options=all_codes, key="all_code_filter")
    with f_c3:
        st.write("")
        st.write("")
        csv_bytes = df_recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Action List",
            data=csv_bytes,
            file_name="foresight_action_recommendations.csv",
            mime="text/csv",
        )

    filtered = df_recs.copy()
    if f_cat:
        filtered = filtered[filtered["category"].isin(f_cat)]
    if f_code:
        filtered = filtered[filtered["recommendation_code"].isin(f_code)]

    st.dataframe(
        filtered[[
            "priority_rank", "sku", "product_name", "category", "recommendation_code",
            "current_stock", "on_order", "reorder_point", "weeks_of_cover",
            "recommended_action"
        ]].sort_values(["priority_rank", "sku"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "priority_rank": st.column_config.NumberColumn("Priority", format="P%d"),
            "sku": st.column_config.TextColumn("SKU", width="small"),
            "product_name": st.column_config.TextColumn("Product Name", width="medium"),
            "category": st.column_config.TextColumn("Category", width="small"),
            "recommendation_code": st.column_config.TextColumn("Action"),
            "current_stock": st.column_config.NumberColumn("Stock", format="%d"),
            "on_order": st.column_config.NumberColumn("On Order", format="%d"),
            "reorder_point": st.column_config.NumberColumn("RP", format="%d"),
            "weeks_of_cover": st.column_config.NumberColumn("Cover", format="%.1fw"),
            "recommended_action": st.column_config.TextColumn("Prescriptive Directive", width="large"),
        },
    )

st.divider()

# ---------------------------------------------------------------------------
# Deep Inspection of Specific Action
# ---------------------------------------------------------------------------
st.markdown("### 🔍 Action Inspection Drawer")

sel_sku = st.selectbox("Inspect Full Recommendation Protocol for SKU", options=sorted(df_recs["sku"].unique()))
rec_row = df_recs[df_recs["sku"] == sel_sku].iloc[0]

with st.expander(f"Protocol Details: {sel_sku} — {rec_row.get('product_name', '')}", expanded=True):
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown(f"**Recommendation:** `{rec_row.get('recommendation_code')}` — {rec_row.get('recommendation_title')}")
        st.markdown(f"**Priority Tier:** `Priority {rec_row.get('priority_rank')}`")
        st.markdown(f"**Triggering Risk:** `{rec_row.get('triggering_risk')}`")
        st.markdown(f"**Data Confidence:** `{rec_row.get('confidence_status')}`")
    with col_d2:
        st.markdown(f"**On-Hand Stock:** `{rec_row.get('current_stock'):.0f} units`")
        st.markdown(f"**Pipeline On-Order:** `{rec_row.get('on_order'):.0f} units`")
        st.markdown(f"**Supplier Lead Time:** `{rec_row.get('lead_time_days'):.0f} days`")
        st.markdown(f"**Reorder Point:** `{rec_row.get('reorder_point'):.0f} units`")

    st.markdown("#### Prescriptive Action Directive")
    st.info(rec_row.get("recommended_action", "No specific action defined."))

    st.markdown("#### Analytical Rationale")
    st.markdown(f"> {rec_row.get('rationale', 'No rationale provided.')}")

    st.markdown("#### Required Operational Follow-Up")
    st.warning(f"**Follow-up Protocol:** {rec_row.get('required_follow_up', 'Maintain standard review cadence.')}")
