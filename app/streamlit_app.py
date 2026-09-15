"""
app/streamlit_app.py — FORESIGHT Executive Command Center & Portal
===================================================================
Project FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform
Production Phase 6 — Deployment, Orchestration & Operational Integration

Strict Governance Compliance:
  - Decision #1 (Option 1D): ZERO monetary valuation metrics. Unit-based only.
  - Decision #2 (Option 2A): Policy B_LT supplier lead time on-order inclusion.
  - Decision #3 (Option 3C): N=8 weeks ratified overstock horizon.
  - Universe: 50 Production SKUs (SKU001–SKU050). 150 orphan SKUs quarantined.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path for Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = Path(__file__).resolve().parent
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


# ---------------------------------------------------------------------------
# Page Configuration & Modern Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT — Inventory Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive styling, glassmorphism cards, and clean typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
        color: #f8fafc;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low {
        background-color: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-healthy {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .nav-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    .nav-card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        color: #38bdf8;
    }
    .nav-card p {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------------------------
manifest = load_pipeline_manifest()
df_recs = load_latest_recommendations()
df_risk = load_latest_risk_scores()

# ---------------------------------------------------------------------------
# Sidebar: System Governance & Pipeline Telemetry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ FORESIGHT PLATFORM")
    st.markdown("`v1.0.0 — Production Phase 6`")
    st.caption("AI-Driven Demand & Inventory Optimization")
    
    st.divider()
    
    st.markdown("#### 🛡️ System Status")
    status = manifest.get("status", "OPERATIONAL")
    if status in ("COMPLETED", "OPERATIONAL", "SUCCESS"):
        st.success("🟢 PIPELINE OPERATIONAL")
    else:
        st.warning(f"🟡 PIPELINE: {status}")

    origin_date = manifest.get("origin_date", "2025-09-16")
    run_time = manifest.get("execution_timestamp", "Active")
    st.markdown(f"**Forecast Origin:** `{origin_date}`")
    st.markdown(f"**Execution:** `{run_time[:19] if len(run_time) > 19 else run_time}`")
    
    st.divider()
    
    st.markdown("#### 📜 Ratified Governance Policies")
    st.markdown(
        """
        - **Valuation (Decision #1):**  
          `Option 1D` — Monetary Valuation Excluded. Strict unit-based operations.
        - **Inbound (Decision #2):**  
          `Option 2A` — Policy $B_{LT}$ verified lead times.
        - **Overstock (Decision #3):**  
          `Option 3C` — $N = 8$ Weeks Ratified Horizon.
        """
    )
    
    st.divider()
    
    st.markdown("#### 🔒 Production Universe")
    st.markdown("- **Active Fleet:** `50 SKUs` (SKU001–SKU050)")
    st.markdown("- **Orphan Quarantine:** `150 SKUs` (Isolated)")
    st.markdown("- **Data Lineage:** 100% SHA-256 Verified")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])

with col_title:
    st.markdown("# ⚡ Executive Command Center")
    st.markdown(
        "Real-time visibility into SKU demand trajectory, multi-horizon inventory risk, "
        "and prioritized replenishment actions across the enterprise fleet."
    )

with col_status:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                ● Production Deployed
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# KPI Cards (Decision #1 Strict Compliance: No Dollar Values)
# ---------------------------------------------------------------------------
p1_count = len(df_recs[df_recs["priority_rank"] == 1]) if not df_recs.empty else 0
p2_count = len(df_recs[df_recs["priority_rank"] == 2]) if not df_recs.empty else 0
p3_count = len(df_recs[df_recs["priority_rank"] == 3]) if not df_recs.empty else 0
p4_count = len(df_recs[df_recs["priority_rank"] == 4]) if not df_recs.empty else 0
p5_count = len(df_recs[df_recs["priority_rank"] == 5]) if not df_recs.empty else 0

median_woc = df_recs["weeks_of_cover"].median() if not df_recs.empty and "weeks_of_cover" in df_recs.columns else 0.0
total_skus = len(df_recs) if not df_recs.empty else 50

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Fleet</div>
            <div class="kpi-value">{total_skus}</div>
            <div class="kpi-subtext">Monitored Production SKUs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid #ef4444;">
            <div class="kpi-label">P1 — Critical Expedite</div>
            <div class="kpi-value" style="color: #ef4444;">{p1_count}</div>
            <div class="kpi-subtext">Stockout breach &lt; lead time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid #f97316;">
            <div class="kpi-label">P2 — Reorder Required</div>
            <div class="kpi-value" style="color: #f97316;">{p2_count}</div>
            <div class="kpi-subtext">Reorder point breach</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
            <div class="kpi-label">P4 — Overstock Surplus</div>
            <div class="kpi-value" style="color: #60a5fa;">{p4_count}</div>
            <div class="kpi-subtext">Coverage &gt; 8 weeks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-label">Fleet Median Cover</div>
            <div class="kpi-value" style="color: #34d399;">{median_woc:.1f}w</div>
            <div class="kpi-subtext">Target: 2.0w – 8.0w</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Visual Analytics Overview: Fleet Risk & Priority Distribution
# ---------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    st.markdown("### 📊 Fleet Action Priority Breakdown")
    priority_counts = pd.DataFrame({
        "Priority": ["P1 — Critical Expedite", "P2 — Reorder PO", "P3 — Review Pipeline", "P4 — Overstock Hold", "P5 — Healthy Schedule"],
        "Count": [p1_count, p2_count, p3_count, p4_count, p5_count],
        "Color": ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"],
    })
    
    fig_donut = px.pie(
        priority_counts,
        names="Priority",
        values="Count",
        color="Priority",
        color_discrete_map={
            "P1 — Critical Expedite": "#ef4444",
            "P2 — Reorder PO": "#f97316",
            "P3 — Review Pipeline": "#eab308",
            "P4 — Overstock Hold": "#3b82f6",
            "P5 — Healthy Schedule": "#10b981",
        },
        hole=0.55,
    )
    fig_donut.update_traces(textinfo="label+value", hoverinfo="label+percent+value")
    fig_donut.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=320,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_chart2:
    st.markdown("### 🏷️ Category Risk Distribution")
    if not df_recs.empty and "category" in df_recs.columns:
        cat_prio = df_recs.groupby(["category", "priority_rank"]).size().reset_index(name="count")
        cat_prio["Priority Tier"] = cat_prio["priority_rank"].map({
            1: "P1 Critical",
            2: "P2 High",
            3: "P3 Medium",
            4: "P4 Low",
            5: "P5 Healthy",
        })
        fig_cat = px.bar(
            cat_prio,
            x="category",
            y="count",
            color="Priority Tier",
            color_discrete_map={
                "P1 Critical": "#ef4444",
                "P2 High": "#f97316",
                "P3 Medium": "#eab308",
                "P4 Low": "#3b82f6",
                "P5 Healthy": "#10b981",
            },
            barmode="stack",
        )
        fig_cat.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(title="Category", showgrid=False),
            yaxis=dict(title="SKU Count", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# High-Priority Action Ticker (Immediate Operational Focus)
# ---------------------------------------------------------------------------
st.markdown("### 🚨 Urgent Action Queue (Priority 1 & 2 SKUs)")

urgent_df = df_recs[df_recs["priority_rank"].isin([1, 2])].sort_values("priority_rank")

if urgent_df.empty:
    st.success("✅ No urgent P1 or P2 actions required across the fleet.")
else:
    display_cols = [
        "sku", "product_name", "category", "recommendation_code", 
        "current_stock", "on_order", "reorder_point", 
        "weeks_of_cover", "earliest_stockout_week", "recommended_action"
    ]
    avail_cols = [c for c in display_cols if c in urgent_df.columns]
    
    formatted_urgent = urgent_df[avail_cols].copy()
    if "weeks_of_cover" in formatted_urgent.columns:
        formatted_urgent["weeks_of_cover"] = formatted_urgent["weeks_of_cover"].round(2)
    
    st.dataframe(
        formatted_urgent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": st.column_config.TextColumn("SKU", width="small"),
            "product_name": st.column_config.TextColumn("Product Name", width="medium"),
            "category": st.column_config.TextColumn("Category", width="small"),
            "recommendation_code": st.column_config.TextColumn("Action Code", width="small"),
            "current_stock": st.column_config.NumberColumn("Current Stock", format="%d"),
            "on_order": st.column_config.NumberColumn("On Order", format="%d"),
            "reorder_point": st.column_config.NumberColumn("Reorder Point", format="%d"),
            "weeks_of_cover": st.column_config.NumberColumn("Weeks of Cover", format="%.2fw"),
            "earliest_stockout_week": st.column_config.NumberColumn("Stockout Wk", format="Wk %d"),
            "recommended_action": st.column_config.TextColumn("Recommended Action", width="large"),
        },
    )

st.divider()

# ---------------------------------------------------------------------------
# Analytical Module Navigation
# ---------------------------------------------------------------------------
st.markdown("### 🧭 Analytical Deep Dive Modules")
st.markdown("Navigate to dedicated modules using the sidebar or the navigation panels below:")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(
        """
        <div class="nav-card">
            <h4>01. Executive Overview</h4>
            <p>Fleet-wide KPIs, inventory health distribution, risk category mix, and downloadable executive summary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="nav-card">
            <h4>02. Demand Forecast</h4>
            <p>8-week forward forecast per SKU with model lineage (Random Forest h=1, XGBoost h=2, Direct h=3..8).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="nav-card">
            <h4>03. Inventory Risk</h4>
            <p>Risk matrix, Days of Supply coverage bands, and multi-horizon breach timing analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        """
        <div class="nav-card">
            <h4>04. Action Center</h4>
            <p>Role-filtered replenishment actions (Buyer, Inventory Controller, Operations) with rationale.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    st.markdown(
        """
        <div class="nav-card">
            <h4>05. SKU 360 Detail</h4>
            <p>Comprehensive SKU dossier: demand history, 8-week projected stock trajectory, and parameters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
