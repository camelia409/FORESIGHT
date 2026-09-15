"""
app/streamlit_app.py — FORESIGHT Enterprise Application & Navigation Controller
================================================================================
Project FORESIGHT — AI-Driven Demand Forecasting & Inventory Optimization Platform

Strict Design & Architecture Rules:
  - ZERO emojis anywhere in the UI.
  - Programmatic multi-page navigation across the five distinct analytical modules.
  - Enterprise telemetry and governance posture in sidebar.
  - Robust import resolution for local and Streamlit Cloud environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

try:
    from app.styles import inject_custom_css
    from app.data_loader import load_pipeline_manifest
except ImportError:
    from styles import inject_custom_css
    from data_loader import load_pipeline_manifest

# Global Streamlit page configuration (Zero emojis)
st.set_page_config(
    page_title="FORESIGHT — Demand & Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Enterprise Sidebar Branding & System Telemetry
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">FORESIGHT</div>
            <div class="sidebar-brand-sub">Demand &amp; Inventory Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    manifest = load_pipeline_manifest()
    pipeline_status = manifest.get("status", "OPERATIONAL")
    pipeline_origin = manifest.get("origin_date", "2025-09-01")
    pipeline_version = manifest.get("pipeline_version", "v1.0.0")

    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div class="sidebar-meta-item"><strong>System Status:</strong> <span style="color: #10B981;">{pipeline_status}</span></div>
            <div class="sidebar-meta-item"><strong>Data As Of:</strong> {pipeline_origin}</div>
            <div class="sidebar-meta-item"><strong>Model Lineage:</strong> {pipeline_version} Hybrid</div>
            <div class="sidebar-meta-item"><strong>Active Fleet:</strong> 50 SKUs</div>
            <div class="sidebar-meta-item"><strong>Quarantined:</strong> 150 Orphan SKUs</div>
        </div>
        <div style="padding-top: 0.75rem; border-top: 1px solid rgba(148, 163, 184, 0.15); margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.35rem;">Governance Posture</div>
            <div class="sidebar-meta-item">Valuation: Option 1D (Excl)</div>
            <div class="sidebar-meta-item">Inbound PO: Option 2A (B_LT)</div>
            <div class="sidebar-meta-item">Overstock: Option 3C (8-Week)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

PAGES_DIR = APP_DIR / "pages"
pages = [
    st.Page(str(PAGES_DIR / "01_executive_overview.py"), title="Executive Overview", default=True),
    st.Page(str(PAGES_DIR / "02_demand_forecast.py"), title="Demand Forecast"),
    st.Page(str(PAGES_DIR / "03_inventory_risk.py"), title="Inventory Risk"),
    st.Page(str(PAGES_DIR / "04_action_center.py"), title="Action Center"),
    st.Page(str(PAGES_DIR / "05_sku_detail.py"), title="SKU Detail"),
]

pg = st.navigation(pages)
pg.run()
