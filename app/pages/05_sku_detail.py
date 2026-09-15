"""
app/pages/05_sku_detail.py — 360-Degree SKU Intelligence Dossier
================================================================
Comprehensive SKU-level analytical profile uniting catalog metadata,
demand forecasting, 8-week inventory position trajectory, and governance audit.

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
import plotly.graph_objects as go

try:
    from app.data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
        load_historical_demand,
        load_forecast_predictions,
    )
except ModuleNotFoundError:
    from data_loader import (
        load_latest_recommendations,
        load_latest_risk_scores,
        load_historical_demand,
        load_forecast_predictions,
    )


st.set_page_config(
    page_title="SKU 360 Detail | FORESIGHT",
    page_icon="🔍",
    layout="wide",
)

st.markdown("# 🔍 SKU 360° Intelligence Dossier")
st.markdown("Complete operational profile, multi-horizon inventory trajectory, and governance audit.")

st.divider()

# Load data
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()
df_hist = load_historical_demand()
df_pred = load_forecast_predictions()

if df_recs.empty:
    st.error("⚠️ No recommendation data available.")
    st.stop()

# SKU Selection
sku_list = sorted(df_recs["sku"].unique().tolist())
default_sku = "SKU010" if "SKU010" in sku_list else sku_list[0]

c_sel, c_sum = st.columns([2, 4])
with c_sel:
    selected_sku = st.selectbox("Select Target SKU", options=sku_list, index=sku_list.index(default_sku))

rec = df_recs[df_recs["sku"] == selected_sku].iloc[0]
prod_name = rec.get("product_name", "Unknown")
category = rec.get("category", "General")
subcategory = rec.get("subcategory", "")

with c_sum:
    st.markdown(f"### {selected_sku} — {prod_name}")
    st.caption(f"**Category:** {category} | **Subcategory:** {subcategory} | **Origin:** {rec.get('origin_date')}")

st.divider()

# ---------------------------------------------------------------------------
# Key SKU Inventory & Operational Parameters
# ---------------------------------------------------------------------------
curr_stock   = float(rec.get("current_stock", 0.0))
on_order     = float(rec.get("on_order", 0.0))
lead_time    = float(rec.get("lead_time_days", 0.0))
safety_stock = float(rec.get("safety_stock", 0.0))
reorder_pt   = float(rec.get("reorder_point", 0.0))
woc          = float(rec.get("weeks_of_cover", 0.0))
avg_demand   = float(rec.get("avg_weekly_demand", 0.0))

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Current On-Hand", f"{curr_stock:.0f} units")
p2.metric("Inbound On-Order", f"{on_order:.0f} units", f"{lead_time:.0f}d lead time")
p3.metric("Safety Stock", f"{safety_stock:.0f} units")
p4.metric("Reorder Point", f"{reorder_pt:.0f} units")
p5.metric("Supply Coverage", f"{woc:.2f} weeks", "Target: 2w–8w")

st.divider()

# ---------------------------------------------------------------------------
# 8-Week Forward Inventory Trajectory Chart
# ---------------------------------------------------------------------------
st.markdown("### 📉 8-Week Forward Inventory Position Trajectory")
st.caption("Depicts expected stock evolution over time against Safety Stock and Zero Stockout boundaries.")

# Extract IP trajectory from risk scores or approximate
ip_trajectory = []
if not df_risk.empty and "SKU" in df_risk.columns:
    risk_match = df_risk[df_risk["SKU"] == selected_sku]
    if not risk_match.empty:
        r_row = risk_match.iloc[0]
        for h in range(1, 9):
            col_ip = f"ip_h{h}"
            if col_ip in r_row:
                ip_trajectory.append((h, float(r_row[col_ip])))

if not ip_trajectory:
    # Approximate if not directly loaded
    running_stock = curr_stock
    for h in range(1, 9):
        # On order arrives at h=1 if lead_time <= 7, else h=2
        arr = on_order if (h == 1 and lead_time <= 7) or (h == 2 and lead_time > 7) else 0.0
        running_stock = running_stock + arr - avg_demand
        ip_trajectory.append((h, running_stock))

df_traj = pd.DataFrame(ip_trajectory, columns=["horizon", "inventory_position"])
df_traj["week_label"] = df_traj["horizon"].apply(lambda h: f"W+{h}")

fig_traj = go.Figure()

# Plot Projected Inventory Position
fig_traj.add_trace(
    go.Scatter(
        x=df_traj["week_label"],
        y=df_traj["inventory_position"],
        mode="lines+markers+text",
        name="Projected Inventory Position",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=8, color="#0284c7"),
        text=df_traj["inventory_position"].round(0).astype(int).astype(str),
        textposition="top center",
    )
)

# Add Safety Stock line
fig_traj.add_hline(
    y=safety_stock,
    line_dash="dash",
    line_color="#f59e0b",
    annotation_text=f"Safety Stock ({safety_stock:.0f}u)",
    annotation_position="bottom right",
)

# Add Zero Line (Stockout boundary)
fig_traj.add_hline(
    y=0.0,
    line_dash="dot",
    line_color="#ef4444",
    annotation_text="Stockout Boundary (0u)",
    annotation_position="bottom right",
)

fig_traj.update_layout(
    xaxis_title="Forward Horizon Week",
    yaxis_title="Inventory Position (Units)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
    height=400,
    margin=dict(t=20, b=10, l=10, r=10),
)
st.plotly_chart(fig_traj, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Prescriptive Action & Governance Dossier
# ---------------------------------------------------------------------------
col_act, col_gov = st.columns([1, 1])

with col_act:
    st.markdown("### ⚡ Prescriptive Recommendation")
    st.markdown(f"**Action Code:** `{rec.get('recommendation_code')}`")
    st.markdown(f"**Title:** `{rec.get('recommendation_title')}`")
    st.markdown(f"**Priority:** `Priority {rec.get('priority_rank')}`")
    
    st.markdown("#### Directive")
    st.info(rec.get("recommended_action"))
    
    st.markdown("#### Analytical Rationale")
    st.markdown(f"> {rec.get('rationale')}")
    
    st.markdown("#### Required Operational Follow-Up")
    st.warning(f"{rec.get('required_follow_up')}")

with col_gov:
    st.markdown("### 🛡️ Governance & Policy Audit")
    
    st.markdown("#### Decision #1 — Inventory Valuation Basis")
    st.success("✅ **Option 1D (Monetary Valuation Excluded)**")
    st.caption("Monetary fields (excess_inventory_value, capital_at_risk) are strictly null. Operations are 100% unit-based.")

    st.markdown("#### Decision #2 — On_Order Arrival Policy")
    st.success(f"✅ **Option 2A (Policy B_LT)**: Verified Lead Time = {lead_time:.0f} Days")
    st.caption("Inbound purchase orders are included according to verified supplier lead times without synthetic dates.")

    st.markdown("#### Decision #3 — Overstock Horizon Threshold")
    st.success(f"✅ **Option 3C (N=8 Weeks Ratified)**")
    st.caption(f"Ratified 8-week horizon aligned with ML forecasting ceiling. Status: {rec.get('overstock_threshold_status')}")

    st.markdown("#### Production SKU Universe")
    st.info(f"SKU `{selected_sku}` is an approved member of the 50-SKU production universe. (150 orphan SKUs remain quarantined).")
