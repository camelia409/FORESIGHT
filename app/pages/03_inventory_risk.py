"""
app/pages/03_inventory_risk.py — Multi-Horizon Inventory Risk Analytics
======================================================================
Provides analytical risk scoring, coverage vulnerability assessment, and lead time exposure.

Page Responsibilities:
  - Answers: Which SKUs are at stockout risk? Which have excess inventory? How does lead time interact with coverage?
  - Uses: 2D enterprise risk matrix, lead time vs coverage scatter, stockout timing distribution, fleet risk roster.
  - Excludes: Action codes, purchase order directives, and approval workflows (delegated to Page 4 Action Center).
  - Strict UI Rules: ZERO emojis anywhere in the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
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
    from app.styles import (
        inject_custom_css,
        render_page_header,
        render_metric_card,
        get_plotly_layout,
        COLOR_CRITICAL,
        COLOR_HIGH,
        COLOR_MEDIUM,
        COLOR_LOW,
        COLOR_HEALTHY,
    )
    from app.data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
    )
except ImportError:
    from styles import (
        inject_custom_css,
        render_page_header,
        render_metric_card,
        get_plotly_layout,
        COLOR_CRITICAL,
        COLOR_HIGH,
        COLOR_MEDIUM,
        COLOR_LOW,
        COLOR_HEALTHY,
    )
    from data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
    )

try:
    st.set_page_config(page_title="Inventory Risk | FORESIGHT", layout="wide")
except Exception:
    pass

inject_custom_css()

render_page_header(
    title="Inventory Risk Analytics",
    description="Multi-horizon inventory vulnerability analysis, 2D risk matrices, and supply lead time exposure.",
    tag="Risk Analytics Layer",
)

# Load data
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()

if df_recs.empty:
    st.error("No inventory or risk data available. Verify pipeline execution.")
    st.stop()

# Merge risk scores if available
df_merged = df_recs.copy()
if not df_risk.empty and "SKU" in df_risk.columns:
    risk_cols = [c for c in ["stockout_score", "overstock_score", "is_stockout_risk", "is_safety_stock_breach"] if c in df_risk.columns]
    if risk_cols:
        df_merged = df_merged.merge(df_risk[["SKU"] + risk_cols].rename(columns={"SKU": "sku"}), on="sku", how="left")

# Fallback derivation for stockout_score if missing
if "stockout_score" not in df_merged.columns:
    df_merged["stockout_score"] = df_merged["weeks_of_cover"].apply(lambda w: max(0.0, min(100.0, (4.0 - w) * 25.0)))

# ---------------------------------------------------------------------------
# Section 1: Fleet Risk Analytics Scorecard
# ---------------------------------------------------------------------------
total_count = len(df_merged)
stockout_count = int((df_merged["priority_rank"].isin([1, 2])).sum())
stockout_pct = (stockout_count / total_count) * 100.0 if total_count > 0 else 0.0

overstock_count = int((df_merged["priority_rank"] == 4).sum())
overstock_pct = (overstock_count / total_count) * 100.0 if total_count > 0 else 0.0

balanced_count = int((df_merged["priority_rank"] == 5).sum())
balanced_pct = (balanced_count / total_count) * 100.0 if total_count > 0 else 0.0

avg_cover = float(df_merged["weeks_of_cover"].mean()) if "weeks_of_cover" in df_merged.columns else 0.0

r1, r2, r3, r4 = st.columns(4)
with r1:
    render_metric_card(
        label="Stockout Exposure Rate",
        value=f"{stockout_pct:.1f}%",
        hint=f"{stockout_count} of {total_count} SKUs at Risk",
        border_variant="critical" if stockout_pct > 25.0 else "medium",
    )
with r2:
    render_metric_card(
        label="Overstock Surplus Rate",
        value=f"{overstock_pct:.1f}%",
        hint=f"{overstock_count} of {total_count} SKUs > 8w Cover",
        border_variant="low",
    )
with r3:
    render_metric_card(
        label="Balanced Fleet Share",
        value=f"{balanced_pct:.1f}%",
        hint=f"{balanced_count} of {total_count} SKUs within Buffer",
        border_variant="healthy",
    )
with r4:
    render_metric_card(
        label="Fleet Mean Coverage",
        value=f"{avg_cover:.1f} Weeks",
        hint="Target Range: 2.0w - 8.0w",
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: 2D Enterprise Risk Matrix (Weeks of Supply vs Stockout Score)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        Enterprise Risk Matrix: Stockout Score vs. Supply Coverage
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Bi-axial mapping of 50 production SKUs against operational thresholds (Shortage &lt;2w, Ratified Overstock &gt;8w).
    </div>
    """,
    unsafe_allow_html=True,
)

fig_matrix = px.scatter(
    df_merged,
    x="weeks_of_cover",
    y="stockout_score",
    color="priority_rank",
    color_continuous_scale=[
        (0.0, COLOR_CRITICAL),
        (0.25, COLOR_HIGH),
        (0.5, COLOR_MEDIUM),
        (0.75, COLOR_LOW),
        (1.0, COLOR_HEALTHY),
    ],
    hover_data=["sku", "product_name", "category", "current_stock", "on_order", "earliest_stockout_week"],
    size="current_stock",
    size_max=22,
)

# Operational boundary guidelines
fig_matrix.add_vline(x=2.0, line_dash="dash", line_color=COLOR_CRITICAL, annotation_text="Shortage Risk (< 2.0w)", annotation_position="top left")
fig_matrix.add_vline(x=8.0, line_dash="dash", line_color=COLOR_LOW, annotation_text="Overstock Threshold (> 8.0w)", annotation_position="top right")
fig_matrix.add_hline(y=50.0, line_dash="dot", line_color=COLOR_HIGH, annotation_text="Elevated Risk Score (> 50)", annotation_position="bottom right")

fig_matrix.update_layout(
    get_plotly_layout(
        xaxis_title="Weeks of Supply Cover",
        yaxis_title="Composite Stockout Score (0 - 100)",
        height=420,
    )
)
fig_matrix.update_layout(
    coloraxis_colorbar=dict(
        title="Priority Tier",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["P1 Critical", "P2 High", "P3 Medium", "P4 Surplus", "P5 Healthy"],
        len=0.75,
    )
)
st.plotly_chart(fig_matrix, use_container_width=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3: Supply Lead Time Vulnerability & Stockout Breach Timing
# ---------------------------------------------------------------------------
col_lt, col_breach = st.columns([1, 1])

with col_lt:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Lead Time vs. Coverage Vulnerability
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Long lead times (&gt;14 days) combined with low cover (&lt;2 weeks) denote severe supply vulnerability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_lt = px.scatter(
        df_merged,
        x="lead_time_days",
        y="weeks_of_cover",
        color="priority_rank",
        color_continuous_scale=[
            (0.0, COLOR_CRITICAL),
            (0.25, COLOR_HIGH),
            (0.5, COLOR_MEDIUM),
            (0.75, COLOR_LOW),
            (1.0, COLOR_HEALTHY),
        ],
        hover_data=["sku", "product_name", "category", "current_stock", "on_order"],
        size="current_stock",
        size_max=18,
    )
    fig_lt.add_hline(y=2.0, line_dash="dash", line_color=COLOR_CRITICAL, annotation_text="Critical (< 2w)")
    fig_lt.add_vline(x=14.0, line_dash="dot", line_color=COLOR_HIGH, annotation_text="Extended Lead Time (> 14d)")
    fig_lt.update_layout(
        get_plotly_layout(
            xaxis_title="Supplier Lead Time (Days)",
            yaxis_title="Weeks of Supply Cover",
            height=320,
        ),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_lt, use_container_width=True)

with col_breach:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Projected Stockout Horizon Timing
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Number of SKUs projected to breach safety stock by forward weekly horizon.
        </div>
        """,
        unsafe_allow_html=True,
    )

    breach_skus = df_merged[df_merged["earliest_stockout_week"].notna()].copy()
    if not breach_skus.empty:
        breach_agg = breach_skus.groupby("earliest_stockout_week").size().reset_index(name="sku_count")
        breach_agg["week_label"] = breach_agg["earliest_stockout_week"].astype(int).apply(lambda w: f"Week {w}")
        
        fig_bar = go.Figure(
            go.Bar(
                x=breach_agg["week_label"],
                y=breach_agg["sku_count"],
                marker=dict(color=COLOR_CRITICAL),
                text=breach_agg["sku_count"],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Breaching SKUs: %{y}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            get_plotly_layout(
                xaxis_title="Projected Breach Horizon",
                yaxis_title="SKU Count",
                height=320,
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No safety stock breaches projected within the 8-week horizon window.")

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Fleet Risk Analytics Roster (Analytical Only)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        Fleet Risk Assessment Roster
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Analytical parameters and buffer metrics across all 50 production SKUs.
    </div>
    """,
    unsafe_allow_html=True,
)

c_f1, c_f2 = st.columns([1, 1])
with c_f1:
    cat_filter = st.multiselect("Filter Department", options=sorted(df_merged["category"].dropna().unique()), key="risk_cat_filter")
with c_f2:
    risk_filter = st.multiselect(
        "Filter Risk Severity",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "P1 Critical", 2: "P2 High", 3: "P3 Medium", 4: "P4 Surplus", 5: "P5 Healthy"}.get(x, str(x)),
        key="risk_tier_filter",
    )

filtered_risk_df = df_merged.copy()
if cat_filter:
    filtered_risk_df = filtered_risk_df[filtered_risk_df["category"].isin(cat_filter)]
if risk_filter:
    filtered_risk_df = filtered_risk_df[filtered_risk_df["priority_rank"].isin(risk_filter)]

display_cols = [
    "sku", "product_name", "category", "current_stock", "on_order",
    "lead_time_days", "safety_stock", "reorder_point", "weeks_of_cover",
    "stockout_score", "earliest_stockout_week", "priority_rank"
]
avail_cols = [c for c in display_cols if c in filtered_risk_df.columns]

st.dataframe(
    filtered_risk_df[avail_cols].sort_values(["priority_rank", "stockout_score"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "sku": st.column_config.TextColumn("SKU", width="small"),
        "product_name": st.column_config.TextColumn("Product Name", width="medium"),
        "category": st.column_config.TextColumn("Department", width="small"),
        "current_stock": st.column_config.NumberColumn("On-Hand", format="%d"),
        "on_order": st.column_config.NumberColumn("On-Order", format="%d"),
        "lead_time_days": st.column_config.NumberColumn("Lead Time", format="%d d"),
        "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%d"),
        "reorder_point": st.column_config.NumberColumn("Reorder Pt", format="%d"),
        "weeks_of_cover": st.column_config.NumberColumn("Cover (w)", format="%.2f w"),
        "stockout_score": st.column_config.NumberColumn("Risk Score", format="%.1f"),
        "earliest_stockout_week": st.column_config.NumberColumn("Breach Wk", format="Wk %d"),
        "priority_rank": st.column_config.NumberColumn("Priority", format="P%d"),
    },
)
