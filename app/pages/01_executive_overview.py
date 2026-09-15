"""
app/pages/01_executive_overview.py — Executive Fleet Overview
============================================================
Provides high-level strategic visibility into enterprise inventory health,
fleet risk posture, and category-level coverage across 50 production SKUs.

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
        load_pipeline_manifest,
        load_latest_recommendations,
        load_latest_risk_scores,
    )
except ModuleNotFoundError:
    from data_loader import (
        load_pipeline_manifest,
        load_latest_recommendations,
        load_latest_risk_scores,
    )


st.set_page_config(
    page_title="Executive Overview | FORESIGHT",
    page_icon="📊",
    layout="wide",
)

st.markdown("# 📊 Executive Fleet Overview")
st.markdown("Enterprise inventory risk distribution, supply chain exposure, and category-level coverage balance.")

st.divider()

# Load data
manifest = load_pipeline_manifest()
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()

if df_recs.empty:
    st.error("⚠️ No recommendation data available. Please run the production pipeline.")
    st.stop()

# ---------------------------------------------------------------------------
# High-Level Metric Tiles
# ---------------------------------------------------------------------------
total_skus = len(df_recs)
p1_count = len(df_recs[df_recs["priority_rank"] == 1])
p2_count = len(df_recs[df_recs["priority_rank"] == 2])
p3_count = len(df_recs[df_recs["priority_rank"] == 3])
p4_count = len(df_recs[df_recs["priority_rank"] == 4])
p5_count = len(df_recs[df_recs["priority_rank"] == 5])
median_woc = df_recs["weeks_of_cover"].median() if "weeks_of_cover" in df_recs.columns else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Production Fleet", f"{total_skus} SKUs", "150 Quarantined")
c2.metric("Critical (P1)", f"{p1_count} SKUs", "Imminent Stockout" if p1_count > 0 else "None", delta_color="inverse")
c3.metric("Reorder (P2)", f"{p2_count} SKUs", "RP Breached" if p2_count > 0 else "None", delta_color="inverse")
c4.metric("Overstock (P4)", f"{p4_count} SKUs", "> 8w Coverage" if p4_count > 0 else "None", delta_color="off")
c5.metric("Median Coverage", f"{median_woc:.1f} Weeks", "Optimal: 2w–8w")

st.divider()

# ---------------------------------------------------------------------------
# Lead Time vs Weeks of Cover Scatter
# ---------------------------------------------------------------------------
col_scatter, col_cat = st.columns([3, 2])

with col_scatter:
    st.markdown("### 🎯 Lead Time vs. Weeks of Cover Distribution")
    st.caption("Dots below 2.0w indicate stockout vulnerability; dots above 8.0w indicate excess inventory.")
    
    fig_scatter = px.scatter(
        df_recs,
        x="lead_time_days",
        y="weeks_of_cover",
        color="priority_rank",
        color_continuous_scale=[
            (0.0, "#ef4444"),   # 1 Critical
            (0.25, "#f97316"),  # 2 High
            (0.5, "#eab308"),   # 3 Medium
            (0.75, "#3b82f6"),  # 4 Low
            (1.0, "#10b981"),   # 5 Healthy
        ],
        hover_data=["sku", "product_name", "category", "current_stock", "on_order"],
        size="current_stock",
        size_max=22,
    )
    # Add horizontal reference bands: 2.0w (stockout threshold) and 8.0w (overstock threshold)
    fig_scatter.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", annotation_text="Stockout Risk Threshold (2.0w)")
    fig_scatter.add_hline(y=8.0, line_dash="dash", line_color="#3b82f6", annotation_text="Ratified Overstock Threshold (8.0w)")
    fig_scatter.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Supplier Lead Time (Days)", showgrid=True),
        yaxis=dict(title="Weeks of Supply Cover", range=[0, min(25, df_recs["weeks_of_cover"].max() + 2)]),
        coloraxis_colorbar=dict(
            title="Priority",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["P1", "P2", "P3", "P4", "P5"],
        ),
        height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_cat:
    st.markdown("### 🏢 Category Fleet Summary")
    st.caption("Aggregated risk profiles across merchandise departments.")
    
    cat_summary = df_recs.groupby("category").agg(
        total_skus=("sku", "count"),
        p1_critical=("priority_rank", lambda x: (x == 1).sum()),
        p2_reorder=("priority_rank", lambda x: (x == 2).sum()),
        p4_overstock=("priority_rank", lambda x: (x == 4).sum()),
        avg_cover=("weeks_of_cover", "mean"),
    ).reset_index()
    cat_summary["avg_cover"] = cat_summary["avg_cover"].round(1)
    
    st.dataframe(
        cat_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "category": st.column_config.TextColumn("Category"),
            "total_skus": st.column_config.NumberColumn("Total SKUs"),
            "p1_critical": st.column_config.NumberColumn("P1 Stockout"),
            "p2_reorder": st.column_config.NumberColumn("P2 Reorder"),
            "p4_overstock": st.column_config.NumberColumn("P4 Overstock"),
            "avg_cover": st.column_config.NumberColumn("Avg Cover (w)", format="%.1fw"),
        },
    )

st.divider()

# ---------------------------------------------------------------------------
# Filterable Fleet Inventory Table & Export
# ---------------------------------------------------------------------------
st.markdown("### 📋 Complete Monitored Production Fleet (50 SKUs)")

f1, f2, f3 = st.columns([2, 2, 1])
with f1:
    selected_cat = st.multiselect("Filter by Category", options=sorted(df_recs["category"].dropna().unique()))
with f2:
    selected_priority = st.multiselect(
        "Filter by Priority Tier",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "P1 — Critical", 2: "P2 — High", 3: "P3 — Medium", 4: "P4 — Low", 5: "P5 — Healthy"}.get(x, str(x)),
    )
with f3:
    st.write("")
    st.write("")
    csv_bytes = df_recs.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Fleet CSV",
        data=csv_bytes,
        file_name="foresight_production_fleet_summary.csv",
        mime="text/csv",
    )

filtered_df = df_recs.copy()
if selected_cat:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_cat)]
if selected_priority:
    filtered_df = filtered_df[filtered_df["priority_rank"].isin(selected_priority)]

cols_to_show = [
    "sku", "product_name", "category", "priority_rank", "recommendation_code",
    "current_stock", "on_order", "safety_stock", "reorder_point",
    "weeks_of_cover", "earliest_stockout_week"
]
avail_cols = [c for c in cols_to_show if c in filtered_df.columns]

st.dataframe(
    filtered_df[avail_cols].sort_values(["priority_rank", "sku"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "sku": st.column_config.TextColumn("SKU", width="small"),
        "product_name": st.column_config.TextColumn("Product Name", width="medium"),
        "category": st.column_config.TextColumn("Category", width="small"),
        "priority_rank": st.column_config.NumberColumn("Priority", format="P%d"),
        "recommendation_code": st.column_config.TextColumn("Action Code"),
        "current_stock": st.column_config.NumberColumn("On-Hand", format="%d"),
        "on_order": st.column_config.NumberColumn("On-Order", format="%d"),
        "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%d"),
        "reorder_point": st.column_config.NumberColumn("Reorder Pt", format="%d"),
        "weeks_of_cover": st.column_config.NumberColumn("Cover (w)", format="%.2fw"),
        "earliest_stockout_week": st.column_config.NumberColumn("Stockout Wk", format="Wk %d"),
    },
)
