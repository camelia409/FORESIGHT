"""
app/pages/04_action_center.py — Operational Action Center & Decision Support
===========================================================================
Operational decision console providing prioritized, deterministic replenishment directives
and procurement workflows for supply chain buyers and inventory controllers.

Page Responsibilities:
  - Answers: What needs to be done? Which SKU first? What is recommended? Why? What is the follow-up?
  - Uses: Prioritized action queues, workflow filtering (P1 to P4), prescriptive directives, audit rationale.
  - Excludes: Risk matrix charts, historical demand plots, and fleet KPI repetitions (delegated to Pages 1, 2, 3).
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

try:
    from app.styles import (
        inject_custom_css,
        render_page_header,
        render_metric_card,
        COLOR_CRITICAL,
        COLOR_HIGH,
        COLOR_MEDIUM,
        COLOR_LOW,
        COLOR_HEALTHY,
    )
    from app.data_loader import (
        load_latest_recommendations,
    )
except ModuleNotFoundError:
    from styles import (
        inject_custom_css,
        render_page_header,
        render_metric_card,
        COLOR_CRITICAL,
        COLOR_HIGH,
        COLOR_MEDIUM,
        COLOR_LOW,
        COLOR_HEALTHY,
    )
    from data_loader import (
        load_latest_recommendations,
    )

try:
    st.set_page_config(page_title="Action Center | FORESIGHT", layout="wide")
except Exception:
    pass

inject_custom_css()

render_page_header(
    title="Operational Action Center",
    description="Deterministic replenishment directives, priority-sequenced action queues, and procurement execution protocols.",
    tag="Operational Execution Layer",
)

# Load data
df_recs = load_latest_recommendations()

if df_recs.empty:
    st.error("No active recommendation data available. Verify pipeline execution.")
    st.stop()

# Action Counts
expedite_count = int((df_recs["recommendation_code"] == "EXPEDITE_PO").sum())
place_po_count = int((df_recs["recommendation_code"] == "PLACE_PO").sum())
review_count   = int((df_recs["recommendation_code"] == "REVIEW_PIPELINE").sum())
freeze_count   = int((df_recs["recommendation_code"] == "FREEZE_REPLENISHMENT").sum())
maintain_count = int((df_recs["recommendation_code"] == "MAINTAIN_SCHEDULE").sum())

# ---------------------------------------------------------------------------
# Section 1: Operational Action Queue Cards (Workflow Stages)
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card(
        label="Expedite Inbound (P1)",
        value=f"{expedite_count} SKUs",
        hint="Immediate Supplier Contact",
        border_variant="critical",
    )
with c2:
    render_metric_card(
        label="Place New PO (P2)",
        value=f"{place_po_count} SKUs",
        hint="Reorder Point Breached",
        border_variant="high",
    )
with c3:
    render_metric_card(
        label="Review Pipeline (P3)",
        value=f"{review_count} SKUs",
        hint="Inbound Arrival Mismatch",
        border_variant="medium",
    )
with c4:
    render_metric_card(
        label="Freeze Order (P4)",
        value=f"{freeze_count} SKUs",
        hint="Surplus > 8w Coverage",
        border_variant="low",
    )
with c5:
    render_metric_card(
        label="Maintain Schedule (P5)",
        value=f"{maintain_count} SKUs",
        hint="Optimal Operational Cadence",
        border_variant="healthy",
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Priority-Sequenced Worklist Controls
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        Prioritized Operational Worklist
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Filter by operational queue, priority tier, or merchandise category.
    </div>
    """,
    unsafe_allow_html=True,
)

f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 1])

with f_col1:
    priority_filter = st.selectbox(
        "Action Workflow Queue",
        options=[
            "All Active SKUs (50)",
            "P1 — Critical Expedite Queue",
            "P2 — New Purchase Order Queue",
            "P3 — Pipeline Review Queue",
            "P4 — Overstock Freeze Queue",
            "P5 — Healthy Scheduled Queue",
        ],
        index=0,
    )

with f_col2:
    cat_options = ["All Departments"] + sorted(df_recs["category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Department Filter", options=cat_options, index=0)

with f_col3:
    action_codes = ["All Actions"] + sorted(df_recs["recommendation_code"].dropna().unique().tolist())
    selected_code = st.selectbox("Action Code Filter", options=action_codes, index=0)

with f_col4:
    st.write("")
    st.write("")
    csv_payload = df_recs.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Worklist (CSV)",
        data=csv_payload,
        file_name="foresight_operational_action_queue.csv",
        mime="text/csv",
    )

# Filter logic
worklist_df = df_recs.copy()
if "P1" in priority_filter:
    worklist_df = worklist_df[worklist_df["priority_rank"] == 1]
elif "P2" in priority_filter:
    worklist_df = worklist_df[worklist_df["priority_rank"] == 2]
elif "P3" in priority_filter:
    worklist_df = worklist_df[worklist_df["priority_rank"] == 3]
elif "P4" in priority_filter:
    worklist_df = worklist_df[worklist_df["priority_rank"] == 4]
elif "P5" in priority_filter:
    worklist_df = worklist_df[worklist_df["priority_rank"] == 5]

if selected_cat != "All Departments":
    worklist_df = worklist_df[worklist_df["category"] == selected_cat]

if selected_code != "All Actions":
    worklist_df = worklist_df[worklist_df["recommendation_code"] == selected_code]

# ---------------------------------------------------------------------------
# Section 3: Operational Worklist Table
# ---------------------------------------------------------------------------
table_cols = [
    "priority_rank", "sku", "product_name", "category", "recommendation_code",
    "current_stock", "on_order", "reorder_point", "weeks_of_cover",
    "recommended_action", "required_follow_up"
]
avail_table_cols = [c for c in table_cols if c in worklist_df.columns]

st.dataframe(
    worklist_df[avail_table_cols].sort_values(["priority_rank", "sku"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "priority_rank": st.column_config.NumberColumn("Priority", format="P%d", width="small"),
        "sku": st.column_config.TextColumn("SKU", width="small"),
        "product_name": st.column_config.TextColumn("Product Name", width="medium"),
        "category": st.column_config.TextColumn("Department", width="small"),
        "recommendation_code": st.column_config.TextColumn("Action Code", width="small"),
        "current_stock": st.column_config.NumberColumn("On-Hand", format="%d"),
        "on_order": st.column_config.NumberColumn("On-Order", format="%d"),
        "reorder_point": st.column_config.NumberColumn("Reorder Pt", format="%d"),
        "weeks_of_cover": st.column_config.NumberColumn("Cover (w)", format="%.1f w"),
        "recommended_action": st.column_config.TextColumn("Prescriptive Directive", width="large"),
        "required_follow_up": st.column_config.TextColumn("Follow-Up Protocol", width="medium"),
    },
)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Operational Action Inspector Drawer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
        Operational Action Inspector
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.75rem;">
        Inspect root-cause analytical justification and execution protocol for any monitored SKU.
    </div>
    """,
    unsafe_allow_html=True,
)

inspect_skus = sorted(worklist_df["sku"].unique().tolist()) if not worklist_df.empty else sorted(df_recs["sku"].unique().tolist())
target_sku = st.selectbox("Select Target SKU to Inspect Operational Protocol", options=inspect_skus)

sku_rec = df_recs[df_recs["sku"] == target_sku].iloc[0]

p_rank = int(sku_rec.get("priority_rank", 5))
border_color = COLOR_CRITICAL if p_rank == 1 else (COLOR_HIGH if p_rank == 2 else (COLOR_LOW if p_rank == 4 else COLOR_HEALTHY))

st.markdown(
    f"""
    <div class="info-callout" style="border-left: 3px solid {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                {target_sku} — {sku_rec.get('product_name', '')}
            </div>
            <div>
                <span class="status-badge badge-p{p_rank}">Priority {p_rank} — {sku_rec.get('recommendation_code', '')}</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 0.75rem; font-size: 0.8rem; color: #CBD5E1;">
            <div><strong>Department:</strong> {sku_rec.get('category', 'General')}</div>
            <div><strong>On-Hand Stock:</strong> {sku_rec.get('current_stock', 0):.0f} units</div>
            <div><strong>On-Order Pipeline:</strong> {sku_rec.get('on_order', 0):.0f} units</div>
            <div><strong>Supplier Lead Time:</strong> {sku_rec.get('lead_time_days', 0):.0f} days</div>
        </div>
        <div style="font-size: 0.83rem; color: #E2E8F0; margin-bottom: 0.5rem;">
            <strong>Prescriptive Action Directive:</strong><br>
            {sku_rec.get('recommended_action', 'No directive specified.')}
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem; background: rgba(15, 23, 42, 0.5); padding: 0.5rem; border-radius: 4px;">
            <strong style="color: #CBD5E1;">Root-Cause Rationale:</strong><br>
            {sku_rec.get('rationale', 'No rationale provided.')}
        </div>
        <div style="font-size: 0.8rem; color: #CBD5E1;">
            <strong>Execution &amp; Follow-Up Protocol:</strong> {sku_rec.get('required_follow_up', 'Maintain scheduled review.')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
