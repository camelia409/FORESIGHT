"""
app/pages/05_sku_detail.py — 360-Degree SKU Intelligence Dossier
================================================================
Comprehensive single-SKU analytical profile uniting catalog metadata,
buffer parameters, forward inventory trajectory simulation, and governance audit.

Page Responsibilities:
  - Answers: What is this SKU? What is its inventory position? How will stock evolve? What is the directive?
  - Uses: SKU metadata, buffer parameters, 60-week integrated trajectory, projected inventory position, governance audit.
  - Excludes: Fleet-wide tables, multi-SKU comparisons, or generic summaries (delegated to Pages 1, 2, 3, 4).
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
        load_historical_demand,
        load_forecast_predictions,
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
        load_historical_demand,
        load_forecast_predictions,
    )

try:
    st.set_page_config(page_title="SKU Detail | FORESIGHT", layout="wide")
except Exception:
    pass

inject_custom_css()

render_page_header(
    title="SKU 360° Intelligence Dossier",
    description="Comprehensive SKU investigation: catalog identity, buffer health, forward trajectory simulation, and governance audit.",
    tag="Single-SKU Dossier Layer",
)

# Load data
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()
df_hist = load_historical_demand()
df_pred = load_forecast_predictions()

if df_recs.empty:
    st.error("No recommendation or SKU master data available. Verify pipeline execution.")
    st.stop()

# SKU Selector
sku_options = sorted(df_recs["sku"].unique().tolist())
default_sku = "SKU010" if "SKU010" in sku_options else sku_options[0]

c_sel, c_desc = st.columns([1, 2])
with c_sel:
    selected_sku = st.selectbox(
        "Select Target SKU for Investigation",
        options=sku_options,
        index=sku_options.index(default_sku),
    )

rec = df_recs[df_recs["sku"] == selected_sku].iloc[0]
prod_name = rec.get("product_name", "Unknown")
category = rec.get("category", "General")
subcategory = rec.get("subcategory", "Standard")
p_rank = int(rec.get("priority_rank", 5))

with c_desc:
    badge_class = f"badge-p{p_rank}"
    st.markdown(
        f"""
        <div style="padding-top: 1.4rem;">
            <span class="status-badge {badge_class}">Priority {p_rank} — {rec.get('recommendation_code', '')}</span>
            <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-top: 0.35rem;">
                {selected_sku} — {prod_name}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8;">
                Department: {category} | Class: {subcategory} | Production Universe: Approved Active Fleet
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: Buffer Health & Current Position Metrics
# ---------------------------------------------------------------------------
curr_stock   = float(rec.get("current_stock", 0.0))
on_order     = float(rec.get("on_order", 0.0))
lead_time    = float(rec.get("lead_time_days", 0.0))
safety_stock = float(rec.get("safety_stock", 0.0))
reorder_pt   = float(rec.get("reorder_point", 0.0))
woc          = float(rec.get("weeks_of_cover", 0.0))
avg_demand   = float(rec.get("avg_weekly_demand", 0.0))

b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    render_metric_card(
        label="On-Hand Inventory",
        value=f"{curr_stock:,.0f} Units",
        hint="Current physical warehouse stock",
        border_variant="critical" if curr_stock <= 0 else None,
    )
with b2:
    render_metric_card(
        label="Inbound On-Order",
        value=f"{on_order:,.0f} Units",
        hint=f"Policy 2A (B_LT): {lead_time:.0f}d lead time",
    )
with b3:
    render_metric_card(
        label="Safety Stock Buffer",
        value=f"{safety_stock:,.0f} Units",
        hint="Buffer reserve against variance",
    )
with b4:
    render_metric_card(
        label="Net Reorder Point",
        value=f"{reorder_pt:,.0f} Units",
        hint=f"Replenishment trigger threshold",
        border_variant="high" if curr_stock + on_order < reorder_pt else None,
    )
with b5:
    woc_variant = "critical" if woc < 2.0 else ("low" if woc > 8.0 else "healthy")
    render_metric_card(
        label="Weeks of Supply",
        value=f"{woc:.2f} Weeks",
        hint="Target Band: 2.0w - 8.0w",
        border_variant=woc_variant,
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Forward Inventory Trajectory Simulation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        8-Week Forward Inventory Position (IP) Simulation
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Simulates stock depletion against weekly demand forecasts and Policy 2A inbound purchase order arrival.
    </div>
    """,
    unsafe_allow_html=True,
)

# Extract projected inventory position across 8 horizons
ip_points = []
if not df_risk.empty and "SKU" in df_risk.columns:
    risk_match = df_risk[df_risk["SKU"] == selected_sku]
    if not risk_match.empty:
        r_row = risk_match.iloc[0]
        for h in range(1, 9):
            col_ip = f"ip_h{h}"
            if col_ip in r_row:
                ip_points.append((h, float(r_row[col_ip])))

if not ip_points:
    running_inv = curr_stock
    for h in range(1, 9):
        inbound = on_order if (h == 1 and lead_time <= 7) or (h == 2 and lead_time > 7) else 0.0
        running_inv = running_inv + inbound - avg_demand
        ip_points.append((h, running_inv))

df_sim = pd.DataFrame(ip_points, columns=["horizon", "simulated_ip"])
df_sim["week_label"] = df_sim["horizon"].apply(lambda h: f"W+{h}")

fig_sim = go.Figure()

# Plot simulated inventory trajectory
fig_sim.add_trace(
    go.Scatter(
        x=df_sim["week_label"],
        y=df_sim["simulated_ip"],
        mode="lines+markers+text",
        name="Projected Inventory Position",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#0284C7"),
        text=df_sim["simulated_ip"].round(0).astype(int).astype(str),
        textposition="top center",
        textfont=dict(size=10, color="#F8FAFC"),
        hovertemplate="<b>Horizon: %{x}</b><br>Projected Stock: %{y:.0f} units<extra></extra>",
    )
)

# Reference: Safety Stock line
fig_sim.add_hline(
    y=safety_stock,
    line_dash="dash",
    line_color=COLOR_HIGH,
    annotation_text=f"Safety Stock ({safety_stock:.0f}u)",
    annotation_position="bottom right",
)

# Reference: Zero stockout line
fig_sim.add_hline(
    y=0.0,
    line_dash="dot",
    line_color=COLOR_CRITICAL,
    annotation_text="Zero Stockout Threshold (0u)",
    annotation_position="bottom right",
)

fig_sim.update_layout(
    get_plotly_layout(
        title=f"Forward Stock Depletion Profile — {selected_sku}",
        xaxis_title="Forward Horizon Week",
        yaxis_title="Inventory Position (Units)",
        height=380,
    )
)
st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3: Prescriptive Directive & Governance Dossier
# ---------------------------------------------------------------------------
col_directive, col_gov = st.columns([1, 1])

with col_directive:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Prescriptive Operational Directive
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Deterministic recommendation derived from ratified inventory policies.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-callout" style="border-left: 3px solid {COLOR_MEDIUM};">
            <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.35rem;">
                Directive Code: {rec.get('recommendation_code', 'MAINTAIN')}
            </div>
            <div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.4; margin-bottom: 0.75rem;">
                {rec.get('recommended_action', 'Maintain standard inventory review cadence.')}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.5rem; background: rgba(15, 23, 42, 0.5); padding: 0.5rem; border-radius: 4px;">
                <strong style="color: #CBD5E1;">Root Cause Rationale:</strong><br>
                {rec.get('rationale', 'Stock parameters within acceptable operating boundaries.')}
            </div>
            <div style="font-size: 0.78rem; color: #CBD5E1;">
                <strong>Follow-Up:</strong> {rec.get('required_follow_up', 'Review at next scheduled cycle.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gov:
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
            Governance &amp; Provenance Audit
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
            Compliance verification against executive-ratified decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-callout">
            <div style="margin-bottom: 0.6rem;">
                <span class="status-badge" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3);">
                    RATIFIED: Decision #1 (Option 1D)
                </span>
                <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.25rem;">
                    <strong>Monetary Valuation Excluded:</strong> Financial fields (excess_inventory_value, capital_at_risk) strictly null. 100% unit-based operation.
                </div>
            </div>
            <div style="margin-bottom: 0.6rem;">
                <span class="status-badge" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3);">
                    RATIFIED: Decision #2 (Option 2A)
                </span>
                <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.25rem;">
                    <strong>Policy B_LT:</strong> Supplier lead time ({lead_time:.0f} days) determines on-order arrival without synthetic PO delivery dates.
                </div>
            </div>
            <div>
                <span class="status-badge" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3);">
                    RATIFIED: Decision #3 (Option 3C)
                </span>
                <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.25rem;">
                    <strong>8-Week Horizon:</strong> Overstock threshold N=8 weeks aligned with ML forecasting horizon ceiling.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
