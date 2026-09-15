"""
app/pages/01_executive_overview.py — Executive Fleet Overview
============================================================
Provides strategic, fleet-wide situation awareness for C-suite and executive supply chain leadership.

Page Responsibilities:
  - Answers: What is the current fleet state? How many SKUs require urgent attention? Where is the risk?
  - Uses: High-level KPIs, priority distribution, category summary, and executive decision signals.
  - Excludes: Detailed SKU tables, scatter plots, and granular PO queues (delegated to Pages 3, 4, 5).
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
        load_pipeline_manifest,
        load_latest_recommendations,
        load_latest_risk_scores,
    )
except ModuleNotFoundError:
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
        load_pipeline_manifest,
        load_latest_recommendations,
        load_latest_risk_scores,
    )

try:
    st.set_page_config(page_title="Executive Overview | FORESIGHT", layout="wide")
except Exception:
    pass

inject_custom_css()

render_page_header(
    title="Executive Overview",
    description="Enterprise inventory health posture, fleet risk distribution, and strategic decision signals.",
    tag="Executive Decision Layer",
)

# Load data
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()

if df_recs.empty:
    st.error("No active recommendation data available. Verify pipeline execution.")
    st.stop()

# Metric Calculations
total_skus = len(df_recs)
p1_count = int((df_recs["priority_rank"] == 1).sum())
p2_count = int((df_recs["priority_rank"] == 2).sum())
p3_count = int((df_recs["priority_rank"] == 3).sum())
p4_count = int((df_recs["priority_rank"] == 4).sum())
p5_count = int((df_recs["priority_rank"] == 5).sum())
median_woc = float(df_recs["weeks_of_cover"].median()) if "weeks_of_cover" in df_recs.columns else 0.0

# ---------------------------------------------------------------------------
# Executive KPI Row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_metric_card(
        label="Monitored Fleet",
        value=f"{total_skus} SKUs",
        hint="150 Quarantined Orphans",
    )
with k2:
    render_metric_card(
        label="Urgent Stockouts (P1)",
        value=f"{p1_count} SKUs",
        hint="Expedite Inbound Order",
        border_variant="critical",
    )
with k3:
    render_metric_card(
        label="Reorder Required (P2)",
        value=f"{p2_count} SKUs",
        hint="Reorder Point Breached",
        border_variant="high",
    )
with k4:
    render_metric_card(
        label="Surplus / Overstock (P4)",
        value=f"{p4_count} SKUs",
        hint="> 8w Supply Coverage",
        border_variant="low",
    )
with k5:
    render_metric_card(
        label="Median Coverage",
        value=f"{median_woc:.1f} Weeks",
        hint="Target Band: 2.0w - 8.0w",
        border_variant="healthy",
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Visual Analytics: Fleet Distribution & Category Summary
# ---------------------------------------------------------------------------
col_dist, col_cat = st.columns([1, 1])

with col_dist:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Fleet Risk Tier Breakdown
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Distribution of 50 production SKUs across operational severity tiers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tier_labels = ["P1 Critical", "P2 High", "P3 Medium", "P4 Surplus", "P5 Healthy"]
    tier_counts = [p1_count, p2_count, p3_count, p4_count, p5_count]
    tier_colors = [COLOR_CRITICAL, COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW, COLOR_HEALTHY]

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=tier_labels,
                values=tier_counts,
                hole=0.55,
                marker=dict(colors=tier_colors),
                textinfo="label+value",
                texttemplate="%{label}<br>%{value} SKUs (%{percent:.0%})",
                hovertemplate="<b>%{label}</b><br>SKU Count: %{value}<br>Fleet Share: %{percent}<extra></extra>",
                textfont=dict(size=10, color="#F8FAFC"),
            )
        ]
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_cat:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Category Inventory Posture
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Risk concentration and coverage balance across merchandise departments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_df = df_recs.groupby("category").agg(
        total_skus=("sku", "count"),
        p1_critical=("priority_rank", lambda x: int((x == 1).sum())),
        p2_reorder=("priority_rank", lambda x: int((x == 2).sum())),
        p4_overstock=("priority_rank", lambda x: int((x == 4).sum())),
        avg_cover=("weeks_of_cover", "mean"),
    ).reset_index()
    cat_df["avg_cover"] = cat_df["avg_cover"].round(1)

    st.dataframe(
        cat_df.sort_values("p1_critical", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "category": st.column_config.TextColumn("Department"),
            "total_skus": st.column_config.NumberColumn("Total SKUs", format="%d"),
            "p1_critical": st.column_config.NumberColumn("P1 Urgent", format="%d"),
            "p2_reorder": st.column_config.NumberColumn("P2 Reorder", format="%d"),
            "p4_overstock": st.column_config.NumberColumn("P4 Overstock", format="%d"),
            "avg_cover": st.column_config.NumberColumn("Avg Cover", format="%.1f w"),
        },
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Strategic Executive Signals & Action Routing
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        Executive Signals &amp; Immediate Operational Priorities
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Automated synthesis of highest-priority fleet interventions.
    </div>
    """,
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        f"""
        <div class="info-callout" style="border-left: 3px solid {COLOR_CRITICAL};">
            <div style="font-weight: 600; color: #F87171; font-size: 0.85rem; margin-bottom: 0.35rem;">
                CRITICAL STOCKOUT EXPOSURE
            </div>
            <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.4;">
                <strong>{p1_count} SKUs</strong> are projected to exhaust on-hand inventory prior to scheduled delivery. Immediate supplier expediting is required to avert customer fulfillment failure.
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.72rem; color: #94A3B8;">
                Module: Navigate to <strong>Action Center</strong> to authorize expedited purchase orders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        f"""
        <div class="info-callout" style="border-left: 3px solid {COLOR_HIGH};">
            <div style="font-weight: 600; color: #FBBF24; font-size: 0.85rem; margin-bottom: 0.35rem;">
                REORDER POINT BREACHES
            </div>
            <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.4;">
                <strong>{p2_count} SKUs</strong> have breached net reorder thresholds with no pending pipeline cover. New standard replenishment orders must be placed within current lead time windows.
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.72rem; color: #94A3B8;">
                Module: Review suggested purchase quantities in <strong>Action Center</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        f"""
        <div class="info-callout" style="border-left: 3px solid {COLOR_LOW};">
            <div style="font-weight: 600; color: #A78BFA; font-size: 0.85rem; margin-bottom: 0.35rem;">
                SURPLUS &amp; CAPITAL PROTECTION
            </div>
            <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.4;">
                <strong>{p4_count} SKUs</strong> hold inventory exceeding the 8-week ratified overstock threshold. Replenishment orders are frozen under Decision #3 (Option 3C) to preserve working capital.
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.72rem; color: #94A3B8;">
                Module: Analyze holding patterns and buffer profiles in <strong>Inventory Risk</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
