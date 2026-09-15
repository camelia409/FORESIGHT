"""
app/styles.py — Enterprise Design System for Project FORESIGHT
=============================================================
Provides consistent styling, typography, semantic status indicators,
and Plotly chart formatting for the FORESIGHT executive analytics platform.

Strict Design Rules:
  - ZERO emojis anywhere in the UI.
  - Semantic color palette (P1 Critical, P2 High, P3 Medium, P4 Low, P5 Healthy).
  - High information density, modern B2B SaaS aesthetics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

# ---------------------------------------------------------------------------
# Semantic Color Constants
# ---------------------------------------------------------------------------
COLOR_CRITICAL = "#EF4444"   # P1 - Red 500
COLOR_HIGH     = "#F59E0B"   # P2 - Amber 500
COLOR_MEDIUM   = "#3B82F6"   # P3 - Blue 500
COLOR_LOW      = "#8B5CF6"   # P4 - Purple 500
COLOR_HEALTHY  = "#10B981"   # P5 - Emerald 500

COLOR_NEUTRAL_DARK  = "#0F172A"  # Slate 900
COLOR_CARD_BG       = "#1E293B"  # Slate 800
COLOR_CARD_BORDER   = "rgba(148, 163, 184, 0.14)"
COLOR_TEXT_PRIMARY  = "#F8FAFC"  # Slate 50
COLOR_TEXT_MUTED    = "#94A3B8"  # Slate 400
COLOR_TEXT_FAINT    = "#64748B"  # Slate 500


def inject_custom_css() -> None:
    """Inject global enterprise stylesheet with zero emojis and clean typography."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Font & Canvas */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #F8FAFC;
        }

        code, pre, .mono-text {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Page Container Spacing */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1360px !important;
        }

        /* Enterprise Header */
        .page-header {
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        }
        .page-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            margin: 0;
            line-height: 1.2;
        }
        .page-subtitle {
            font-size: 0.875rem;
            font-weight: 400;
            color: #94A3B8;
            margin-top: 0.35rem;
            margin-bottom: 0;
        }

        /* Metric / KPI Card */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify_content: space-between;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        }
        .metric-card-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 0.25rem;
        }
        .metric-card-value {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.15;
            color: #F8FAFC;
            font-feature-settings: "tnum";
        }
        .metric-card-hint {
            font-size: 0.75rem;
            font-weight: 500;
            color: #64748B;
            margin-top: 0.4rem;
        }

        /* Semantic Accent Borders */
        .border-critical { border-left: 3px solid #EF4444 !important; }
        .border-high     { border-left: 3px solid #F59E0B !important; }
        .border-medium   { border-left: 3px solid #3B82F6 !important; }
        .border-low      { border-left: 3px solid #8B5CF6 !important; }
        .border-healthy  { border-left: 3px solid #10B981 !important; }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-p1 {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .badge-p2 {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .badge-p3 {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .badge-p4 {
            background: rgba(139, 92, 246, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(139, 92, 246, 0.35);
        }
        .badge-p5 {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        /* Detail Callout Container */
        .info-callout {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.82rem;
            color: #CBD5E1;
            margin-bottom: 1rem;
        }

        /* Clean Sidebar Branding */
        .sidebar-brand {
            padding: 0.5rem 0 1rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            margin-bottom: 1rem;
        }
        .sidebar-brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #F8FAFC;
        }
        .sidebar-brand-sub {
            font-size: 0.75rem;
            font-weight: 500;
            color: #60A5FA;
            letter-spacing: 0.02em;
        }
        .sidebar-meta-item {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .sidebar-meta-item strong {
            color: #E2E8F0;
        }

        /* Sidebar — force dark background and legible text regardless of theme */
        section[data-testid="stSidebar"] {
            background-color: #1E293B !important;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .sidebar-brand-title {
            color: #F8FAFC !important;
        }
        section[data-testid="stSidebar"] .sidebar-brand-sub {
            color: #60A5FA !important;
        }
        section[data-testid="stSidebar"] .sidebar-meta-item {
            color: #CBD5E1 !important;
        }
        section[data-testid="stSidebar"] .sidebar-meta-item strong {
            color: #F1F5F9 !important;
        }
        /* Streamlit's built-in nav links in sidebar */
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li {
            color: #CBD5E1 !important;
        }
        section[data-testid="stSidebar"] a:hover {
            color: #F8FAFC !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str, tag: Optional[str] = None) -> None:
    """Render a clean, professional enterprise top header with optional status tag."""
    tag_html = ""
    if tag:
        tag_html = f"""
        <div style="text-align: right;">
            <span class="status-badge" style="background: rgba(59, 130, 246, 0.12); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.25);">
                {tag}
            </span>
        </div>
        """

    st.markdown(
        f"""
        <div class="page-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 class="page-title">{title}</h1>
                    <p class="page-subtitle">{description}</p>
                </div>
                {tag_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(**kwargs) -> Dict[str, Any]:
    """Return standard Plotly dark-theme layout settings for high enterprise polish."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#94A3B8"),
        margin=dict(t=30, b=30, l=35, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            linecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(size=10, color="#94A3B8"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            linecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(size=10, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#94A3B8"),
        ),
    )
    layout.update(kwargs)
    return layout


def render_metric_card(
    label: str,
    value: str,
    hint: Optional[str] = None,
    border_variant: Optional[str] = None,
) -> None:
    """Render a styled enterprise KPI card with zero emojis."""
    border_class = f"border-{border_variant}" if border_variant else ""
    hint_html = f'<div class="metric-card-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card {border_class}">
            <div>
                <div class="metric-card-label">{label}</div>
                <div class="metric-card-value">{value}</div>
            </div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

