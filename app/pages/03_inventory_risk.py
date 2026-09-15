"""
app/pages/03_inventory_risk.py — Multi-Horizon Inventory Risk Engine
====================================================================
Comprehensive analysis of stockout risks, safety stock breaches,
and excess inventory across the 8-week forward horizon.

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
import plotly.express as px
import plotly.graph_objects as go

try:
    from app.data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
    )
except ModuleNotFoundError:
    from data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
    )


st.set_page_config(
    page_title="Inventory Risk | FORESIGHT",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("# 🛡️ Multi-Horizon Inventory Risk Engine")
st.markdown("Multi-horizon inventory position projection, stockout probability, and coverage analysis.")

st.divider()

# Load data
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()

if df_recs.empty:
    st.error("⚠️ No recommendation data available.")
    st.stop()

# Merge risk scores with recs if needed
df_merged = df_recs.copy()
if not df_risk.empty and "SKU" in df_risk.columns:
    risk_cols = [c for c in ["stockout_score", "overstock_score", "is_stockout_risk", "is_safety_stock_breach"] if c in df_risk.columns]
    if risk_cols:
        df_merged = df_merged.merge(df_risk[["SKU"] + risk_cols].rename(columns={"SKU": "sku"}), on="sku", how="left")

# ---------------------------------------------------------------------------
# Risk Metrics Summary Row
# ---------------------------------------------------------------------------
stockout_risks = len(df_merged[df_merged["priority_rank"].isin([1, 2])])
overstock_risks = len(df_merged[df_merged["priority_rank"] == 4])
healthy_skus = len(df_merged[df_merged["priority_rank"] == 5])
avg_stockout_score = df_merged["stockout_score"].mean() if "stockout_score" in df_merged.columns else 0.0

r1, r2, r3, r4 = st.columns(4)
r1.metric("Stockout Risk Alerts", f"{stockout_risks} SKUs", "Requires Expedite/Reorder", delta_color="inverse")
r2.metric("Overstock Surplus", f"{overstock_risks} SKUs", "> 8 Weeks Cover", delta_color="off")
r3.metric("Balanced / Healthy", f"{healthy_skus} SKUs", "Optimal Inventory Range")
r4.metric("Avg Fleet Stockout Score", f"{avg_stockout_score:.1f} / 100", "Composite Risk Metric")

st.divider()

# ---------------------------------------------------------------------------
# 2D Risk Matrix: Stockout Score vs Weeks of Cover
# ---------------------------------------------------------------------------
st.markdown("### 🗺️ Enterprise Risk Matrix")
st.caption("Visualizing 50 SKUs across the supply continuum: Stockout Vulnerability vs Overstock Surplus.")

if "stockout_score" in df_merged.columns and "weeks_of_cover" in df_merged.columns:
    fig_matrix = px.scatter(
        df_merged,
        x="weeks_of_cover",
        y="stockout_score",
        color="priority_rank",
        color_continuous_scale=[
            (0.0, "#ef4444"),   # 1 Critical
            (0.25, "#f97316"),  # 2 High
            (0.5, "#eab308"),   # 3 Medium
            (0.75, "#3b82f6"),  # 4 Low
            (1.0, "#10b981"),   # 5 Healthy
        ],
        hover_data=["sku", "product_name", "category", "current_stock", "on_order", "earliest_stockout_week"],
        size="current_stock",
        size_max=24,
    )
    
    # Boundary annotations
    fig_matrix.add_vline(x=2.0, line_dash="dash", line_color="#ef4444", annotation_text="Shortage Risk (<2w)")
    fig_matrix.add_vline(x=8.0, line_dash="dash", line_color="#3b82f6", annotation_text="Ratified Overstock (>8w)")
    fig_matrix.add_hline(y=50.0, line_dash="dot", line_color="#eab308", annotation_text="Elevated Stockout Score (>50)")
    
    fig_matrix.update_layout(
        xaxis=dict(title="Weeks of Supply Cover", range=[0, min(25, df_merged["weeks_of_cover"].max() + 2)]),
        yaxis=dict(title="Stockout Risk Score (0–100)", range=[0, 105]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        coloraxis_colorbar=dict(
            title="Priority",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["P1", "P2", "P3", "P4", "P5"],
        ),
        height=450,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Breach Timeline Distribution (When will stock run out?)
# ---------------------------------------------------------------------------
st.markdown("### ⏱️ Earliest Stockout Breach Timing")
st.caption("Distribution of projected stockout weeks across the 8-week forward horizon.")

breach_df = df_merged[df_merged["earliest_stockout_week"].notna()].copy()

if not breach_df.empty:
    breach_counts = breach_df.groupby("earliest_stockout_week").size().reset_index(name="sku_count")
    breach_counts["week_label"] = breach_counts["earliest_stockout_week"].astype(int).apply(lambda w: f"Week {w}")
    
    fig_bar = px.bar(
        breach_counts,
        x="week_label",
        y="sku_count",
        color="sku_count",
        color_continuous_scale="Reds",
        text="sku_count",
    )
    fig_bar.update_layout(
        xaxis=dict(title="Projected Breach Horizon Week", showgrid=False),
        yaxis=dict(title="Number of SKUs Breaching", showgrid=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        coloraxis_showscale=False,
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("ℹ️ No imminent stockouts projected in the current 8-week horizon window.")

st.divider()

# ---------------------------------------------------------------------------
# Detailed SKU Risk Table
# ---------------------------------------------------------------------------
st.markdown("### 📋 SKU Risk Roster")

risk_cols_show = [
    "sku", "product_name", "category", "current_stock", "on_order",
    "safety_stock", "reorder_point", "weeks_of_cover", "stockout_score",
    "earliest_stockout_week", "priority_rank"
]
avail_risk_cols = [c for c in risk_cols_show if c in df_merged.columns]

st.dataframe(
    df_merged[avail_risk_cols].sort_values(["priority_rank", "stockout_score"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "sku": st.column_config.TextColumn("SKU", width="small"),
        "product_name": st.column_config.TextColumn("Product Name", width="medium"),
        "category": st.column_config.TextColumn("Category", width="small"),
        "current_stock": st.column_config.NumberColumn("Current Stock", format="%d"),
        "on_order": st.column_config.NumberColumn("On Order", format="%d"),
        "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%d"),
        "reorder_point": st.column_config.NumberColumn("Reorder Pt", format="%d"),
        "weeks_of_cover": st.column_config.NumberColumn("Cover (w)", format="%.2fw"),
        "stockout_score": st.column_config.NumberColumn("Stockout Score", format="%.1f"),
        "earliest_stockout_week": st.column_config.NumberColumn("Breach Wk", format="Wk %d"),
        "priority_rank": st.column_config.NumberColumn("Priority", format="P%d"),
    },
)
