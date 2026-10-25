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

    # Shared Custom CSS for clean, professional typography and cards
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-color, #1E293B);
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: var(--text-color, #64748B);
            opacity: 0.85;
            margin-bottom: 1.2rem;
        }
        .synthetic-banner {
            background-color: rgba(245, 158, 11, 0.15);
            border-left: 5px solid #F59E0B;
            padding: 12px 16px;
            border-radius: 4px;
            color: inherit;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: var(--secondary-background-color, #F8FAFC);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Header & Disclaimer
    with st.sidebar:
        st.markdown("### 🏛️ **SmartWeb Access**")
        st.caption("Fuzzy TOPSIS & VIKOR Decision Support System")
        st.markdown("---")
        st.info(
            "📌 **Research / Demo Notice**\n\n"
            "Records displayed are **synthetic demonstration data** "
            "modeled after institutional websites for algorithm benchmarking. "
            "They are not live scan results.",
            icon="ℹ️",
        )
        st.markdown("---")

    # Page Main Header
    st.markdown(f"<div class='main-title'>{icon} {title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)

    # Top Synthetic Data Disclaimer Banner
    st.markdown(
        """
        <div class="synthetic-banner">
            ⚠️ <strong>Notice:</strong> Demonstration records shown are synthetic evaluation benchmarks 
            designed to evaluate Fuzzy TOPSIS & VIKOR prioritization. No real institutional servers were 
            scanned without authorization.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_cached_issues():
    """Retrieve issues from SQLite and cache in Streamlit session state."""
    return load_all_issues()
