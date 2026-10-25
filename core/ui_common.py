"""
================================================================================
MODULE: core.ui_common
================================================================================
1. WHAT THIS FILE DOES:
   Provides shared user interface initialization, custom CSS injection, navigation
   banners, metric card components, and database bootstrapping for Streamlit pages.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: A multi-page research dashboard requires consistent branding,
     clear research disclosure notices, responsive layout configurations, and identical
     sidebar controls across all pages without duplicating UI boilerplate code.
   - Technical & Architectural Rationale: Centralizes Streamlit `st.set_page_config`,
     session state management, CSS styling, and first-time database auto-seeding.

3. HOW IT WORKS CONCEPTUALLY:
   - Sets the page browser title, favicon, and wide layout.
   - Automatically initializes and seeds the SQLite database on first app access if empty.
   - Injects custom CSS styling for typography, metric scorecards, and research callouts.
   - Displays persistent sidebar research notices and data provenance disclaimers.

4. HOW IT WORKS TECHNICALLY:
   - `setup_page_header()`: Invokes `st.set_page_config()`, executes `seed_if_empty()` guarded
     by `st.session_state["db_initialized"]`, injects HTML `<style>` blocks, and renders headers.
   - `render_badge()`: Helper for HTML status badges with color-coded CSS.
================================================================================
"""

from __future__ import annotations

import streamlit as st
from core.database import seed_if_empty, load_all_issues


def setup_page_header(title: str, icon: str = "♿", subtitle: str = "") -> None:
    """Render consistent page styling, banners, and title."""
    st.set_page_config(
        page_title=f"SmartWeb Access - {title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ensure database is seeded with demonstration data on first visit
    if "db_initialized" not in st.session_state:
        seed_if_empty()
        st.session_state["db_initialized"] = True

    # Modern GovTech / Academic Emerald Theme CSS
    st.markdown(
        """
        <style>
        /* Google Fonts & Global Typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        }

        /* GovTech Institutional Portal Header */
        .gov-portal-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: rgba(5, 150, 105, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 9999px;
            color: #34D399;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #F8FAFC 20%, #E2E8F0 70%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            font-weight: 400;
            line-height: 1.5;
            margin-bottom: 1.4rem;
        }

        /* GovTech Disclaimer Notice Banner */
        .synthetic-banner {
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
            border: 1px solid #334155;
            border-left: 4px solid #10B981;
            padding: 12px 18px;
            border-radius: 8px;
            color: #CBD5E1;
            font-size: 0.88rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }

        /* Enhanced Metric Scorecards (GovTech Emerald Style) */
        [data-testid="stMetric"] {
            background: #1E293B !important;
            border: 1px solid #334155 !important;
            border-top: 3px solid #059669 !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #059669 !important;
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.15) !important;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
        }

        /* Streamlit Native Button Polish */
        .stButton button {
            background-color: #059669 !important;
            color: #FFFFFF !important;
            border: 1px solid #10B981 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.15s ease-in-out !important;
        }

        .stButton button:hover {
            background-color: #047857 !important;
            border-color: #34D399 !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.35) !important;
        }

        /* Streamlit Tabs Polish */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #334155 !important;
            gap: 12px !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 8px 16px !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #34D399 !important;
            border-bottom: 2px solid #10B981 !important;
            background: rgba(5, 150, 105, 0.1) !important;
        }

        /* Expanders and Containers */
        [data-testid="stExpander"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
        }

        /* Custom Badges */
        .badge-emerald {
            background: rgba(5, 150, 105, 0.2);
            color: #34D399;
            border: 1px solid #059669;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-amber {
            background: rgba(245, 158, 11, 0.2);
            color: #FBBF24;
            border: 1px solid #D97706;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-blue {
            background: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            border: 1px solid #2563EB;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        /* Sidebar GovTech Customization */
        [data-testid="stSidebar"] {
            background-color: #0B1120 !important;
            border-right: 1px solid #1E293B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Header & Disclaimer
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0; border-bottom: 1px solid #1E293B; margin-bottom: 15px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.4rem;">🏛️</span>
                    <div>
                        <div style="font-weight:800; font-size:1.05rem; color:#F8FAFC; letter-spacing:-0.01em;">WebEase DSS</div>
                        <div style="font-size:0.75rem; color:#10B981; font-weight:600;">Academic & Civic Audit Edition</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "📌 **Research / Demo Notice**\n\n"
            "Records displayed are **synthetic demonstration data** "
            "modeled after institutional websites for algorithm benchmarking. "
            "They are not live scan results.",
            icon="ℹ️",
        )
        st.markdown("---")

    # Page Main Header with GovTech Pill
    st.markdown(
        """
        <div class="gov-portal-tag">
            <span>●</span> Official Decision Support System
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='main-title'>{icon} {title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)

    # Top Synthetic Data Disclaimer Banner
    st.markdown(
        """
        <div class="synthetic-banner">
            🛡️ <strong>Methodological Provenance Notice:</strong> Evaluation records in this workspace 
            serve as reproducible scientific benchmarks under WCAG 2.1/2.2. Prioritization rankings are 
            derived deterministically using Fuzzy TOPSIS and VIKOR multi-criteria algorithms.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_cached_issues():
    """Retrieve issues from SQLite and cache in Streamlit session state."""
    return load_all_issues()


def render_badge(text: str, variant: str = "emerald") -> str:
    """Return HTML string for a styled GovTech badge."""
    css_class = f"badge-{variant}" if variant in ("emerald", "amber", "blue") else "badge-emerald"
    return f"<span class='{css_class}'>{text}</span>"

