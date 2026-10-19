"""SmartWeb Access - Main Portal and Dashboard Entry Point.

A Soft Computing Decision Support System for Prioritizing Accessibility Repairs
on Public-Service Websites using Fuzzy TOPSIS and VIKOR.
"""

from __future__ import annotations

import streamlit as st
from core.ui_common import setup_page_header
from core.database import get_stats, load_all_issues

setup_page_header(
    title="SmartWeb Access",
    icon="🏛️",
    subtitle="Prioritizing Web Accessibility Repairs with Soft Computing (Fuzzy TOPSIS & VIKOR)",
)

stats = get_stats()

# Key Metric Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Audited Websites", stats["total_websites"])
with c2:
    st.metric("Total Pages Evaluated", stats["total_pages"])
with c3:
    st.metric("Identified Barriers", stats["total_issues"])
with c4:
    st.metric("Critical Task Impact", stats["critical_task_issues"], delta="High Priority", delta_color="inverse")

st.markdown("---")

# Project Background & Research Focus
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎯 The Problem & Our Solution")
    st.markdown(
        """
        Public-service websites (such as municipal portals, university registries, transit hubs, and healthcare systems) 
        frequently contain numerous accessibility barriers. Automated tools (e.g. WAVE, Lighthouse, Axe) 
        often generate exhaustive lists of hundreds of unranked errors.

        Website remediation teams have **limited time and development capacity**. The critical dilemma is:
        > **Which accessibility issues must be fixed first to unlock essential citizen services?**

        **SmartWeb Access** transforms raw accessibility findings into an intelligent, defensible repair action plan by applying:
        1. **Fuzzy TOPSIS:** Balances qualitative expert evaluations by calculating closeness to an ideal repair profile.
        2. **VIKOR:** Identifies optimal compromise solutions that balance conflicting goals (e.g. maximizing task impact while minimizing engineering effort).
        """
    )

    st.subheader("🧭 Guided Dashboard Navigation")
    st.markdown(
        """
        Use the sidebar navigation to explore the decision-support system:
        - **1. Overview:** High-level metrics, barrier distribution, and impact breakdowns.
        - **2. Websites and Pages:** Inspection of the 5 institutional entities and evaluated pages.
        - **3. Accessibility Issues:** Detailed repository of accessibility barriers with evidence and fixes.
        - **4. Prioritization:** Interactive MCDM engine computing Fuzzy TOPSIS and VIKOR rankings.
        - **5. TOPSIS vs VIKOR:** Academic comparative analysis, rank correlation, and agreement metrics.
        - **6. Sensitivity Analysis:** Dynamic weight perturbation experiments and stability testing.
        - **7. Data and Export:** Export decision reports in CSV, JSON, and Markdown format.
        """
    )

with col_right:
    st.subheader("🔬 Decision Criteria (C1–C6)")
    st.markdown(
        """
        | Criterion | Focus | Direction |
        | :--- | :--- | :--- |
        | **C1. Severity** | Technical violation degree | Higher = Higher Priority |
        | **C2. Task Impact** | Blocks applying, paying, login | Higher = Higher Priority |
        | **C3. Pages Affected** | Scope across site | More = Higher Priority |
        | **C4. User Impact** | Affected assistive user groups | Higher = Higher Priority |
        | **C5. Repair Effort** | Engineering complexity | Lower effort increases priority |
        | **C6. Frequency** | Recurrence repeatability | Higher = Higher Priority |
        """
    )

    st.info(
        "💡 **Key Innovation:** Decoupling barrier severity from essential-task impact ensures that "
        "technically subtle barriers (e.g. an unlabeled 2FA token field) that completely block a high-stakes "
        "civic service receive urgent attention before superficial visual flaws on secondary pages."
    )

st.markdown("---")
st.caption("SmartWeb Access Research Prototype | Built with Python 3.13, Streamlit, SQLite & Plotly.")
