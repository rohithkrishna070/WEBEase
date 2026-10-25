"""
================================================================================
MODULE: pages.1_Overview
================================================================================
1. WHAT THIS FILE DOES:
   Renders the Portfolio Research Analytics & Overview dashboard, visualizing
   portfolio-wide distributions of accessibility barriers, audit provenance,
   severity classifications, and essential-task impact across institutions.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Decision makers need macro-level situational awareness
     before drilling into individual algorithm runs. Demonstrates the breakdown of
     barriers across the 5 public institutions and highlights where critical citizen
     service barriers are concentrated.
   - Technical & Architectural Rationale: Provides interactive visual data exploration
     using Plotly charts and Streamlit metrics, querying SQLite directly.

3. HOW IT WORKS CONCEPTUALLY:
   - Aggregates issues by institutional website and essential citizen journey tasks.
   - Visualizes audit provenance (automated vs manual vs synthetic benchmarks).
   - Explores the distribution of severity, affected assistive tech groups, and repair efforts.

4. HOW IT WORKS TECHNICALLY:
   - Reads data via `core.database.load_all_issues()` and `get_stats()`.
   - Uses `plotly.express` (`px.bar`, `px.pie`, `px.sunburst`) to build interactive visual charts.
   - Implements responsive Streamlit columns for scorecard metrics.
================================================================================
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from core.ui_common import setup_page_header
from core.database import load_all_issues, get_stats, get_registered_websites, get_registered_pages

setup_page_header(
    title="Portfolio Research Analytics & Overview",
    icon="📊",
    subtitle="High-level distribution of accessibility barriers, audit provenance, and citizen journey criticality",
)

# Benchmark Demonstration Disclaimer Banner
st.info(
    """
    🔬 **Research Demonstration Notice:**
    The baseline dataset in this workspace consists of **synthetic demonstration records** modeled after typical
    institutional accessibility audit patterns under WCAG 2.1/2.2 standards. These records provide a reproducible
    benchmark for multi-criteria decision algorithm evaluation (Fuzzy TOPSIS & VIKOR). They do not represent
    unverified live scans of third-party public portals.
    """
)

df = load_all_issues()
stats = get_stats()
reg_sites = get_registered_websites()
reg_pages = get_registered_pages()

# ==============================================================================
# 1. Executive Metric Scorecards
# ==============================================================================
st.subheader("🏛️ Institutional Portfolio Highlights")

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

m_col1.metric("Registered Websites", len(reg_sites) if not reg_sites.empty else stats["total_websites"])
m_col2.metric("Audited Pages", len(reg_pages) if not reg_pages.empty else stats["total_pages"])
m_col3.metric("Total Barriers", len(df))
m_col4.metric("Open / In Progress", f"{stats.get('open_issues', 0)} / {stats.get('in_progress_issues', 0)}")
m_col5.metric("Resolved Issues", stats.get("resolved_issues", 0))
m_col6.metric("Critical Journey Pages", len(reg_pages[reg_pages["is_critical_journey"] == 1]) if not reg_pages.empty else stats.get("critical_task_issues", 0))

st.markdown("---")

if df.empty:
    st.warning("⚠️ No accessibility barriers found in the database. Please register websites, record manual audit findings, or import audit files in 'Data and Export'.")
    st.stop()

# ==============================================================================
# 2. Audit Provenance: Automated Scans, Manual Reviews & Imported Records
# ==============================================================================
st.subheader("🔍 Audit Evidence Provenance & Verification Status")
st.caption("Categorizing how accessibility evidence was recorded across automated scanners, human reviews, and imported tools.")

p_col1, p_col2 = st.columns([1, 1])

with p_col1:
    st.markdown("#### Record Type & Source Ingestion")
    rec_counts = df["record_type"].value_counts().reset_index()
    rec_counts.columns = ["Record Type", "Count"]

    fig_rec = px.pie(
        rec_counts,
        names="Record Type",
        values="Count",
        hole=0.45,
        color="Record Type",
        color_discrete_map={
            "Synthetic Demonstration": "#64748B",
            "Manual Observation": "#3B82F6",
            "Imported Audit Evidence": "#10B981",
        },
    )
    fig_rec.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(fig_rec, use_container_width=True)

with p_col2:
    st.markdown("#### Verification & Triage Status")
    verif_counts = df["verification_status"].value_counts().reset_index()
    verif_counts.columns = ["Verification Status", "Count"]

    fig_verif = px.bar(
        verif_counts,
        x="Count",
        y="Verification Status",
        orientation="h",
        color="Verification Status",
        color_discrete_map={
            "Human-Verified Finding": "#2563EB",
            "Unverified Automated Detection": "#F59E0B",
            "Manual Observation": "#059669",
            "False Positive Discarded": "#DC2626",
        },
        text="Count",
    )
    fig_verif.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_verif, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 3. Issue Distribution by Barrier Type & Severity
# ==============================================================================
st.subheader("⚠️ Barrier Distributions across Categories & WCAG Severity")

col_d1, col_d2 = st.columns([3, 2])

with col_d1:
    st.markdown("#### Barrier Types Segmented by Severity")
    type_sev = df.groupby(["issue_type", "severity"]).size().reset_index(name="Count")
    fig_types = px.bar(
        type_sev,
        x="issue_type",
        y="Count",
        color="severity",
        barmode="stack",
        color_discrete_map={
            "Very Low": "#94A3B8",
            "Low": "#38A169",
            "Medium": "#EAB308",
            "High": "#EA580C",
            "Very High": "#DC2626",
        },
        labels={"issue_type": "Barrier Category", "Count": "Number of Findings", "severity": "Severity"},
        category_orders={"severity": ["Very Low", "Low", "Medium", "High", "Very High"]},
    )
    fig_types.update_layout(
        xaxis_tickangle=-30,
        margin=dict(l=10, r=10, t=20, b=80),
        height=380,
    )
    st.plotly_chart(fig_types, use_container_width=True)

with col_d2:
    st.markdown("#### Severity vs. Essential-Task Impact Heatmap")
    cross_tab = pd.crosstab(
        df["severity"],
        df["essential_task_impact"],
    ).reindex(
        index=["Very Low", "Low", "Medium", "High", "Very High"],
        columns=["Very Low", "Low", "Medium", "High", "Very High"],
        fill_value=0,
    )
    # Remove empty rows/columns if very low is absent
    cross_tab = cross_tab.loc[(cross_tab.sum(axis=1) > 0), (cross_tab.sum(axis=0) > 0)]

    fig_heat = px.imshow(
        cross_tab,
        text_auto=True,
        labels=dict(x="Essential-Task Impact (C2)", y="Barrier Severity (C1)", color="Issues"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=380)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Reveals barriers with moderate technical severity that completely block critical citizen tasks.")

st.markdown("---")

# ==============================================================================
# 4. Repair Effort & Institutional Occurrence
# ==============================================================================
st.subheader("🛠️ Remediation Engineering Effort by Institution")

e_col1, e_col2 = st.columns([1, 1])

with e_col1:
    st.markdown("#### Repair Effort (C5, Cost Criterion)")
    effort_order = ["Very Low", "Low", "Medium", "High"]
    effort_counts = df["repair_effort"].value_counts().reindex(effort_order, fill_value=0).reset_index()
    effort_counts.columns = ["Repair Effort", "Count"]

    fig_effort = px.bar(
        effort_counts,
        x="Repair Effort",
        y="Count",
        color="Repair Effort",
        color_discrete_map={
            "Very Low": "#10B981",
            "Low": "#34D399",
            "Medium": "#F59E0B",
            "High": "#EF4444",
        },
        text="Count",
    )
    fig_effort.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig_effort, use_container_width=True)
    st.caption("Very Low / Low effort repairs offer quick-win remediation opportunities under VIKOR & TOPSIS.")

with e_col2:
    st.markdown("#### Issues by Audited Public-Service Portal")
    site_counts = df["website"].value_counts().reset_index()
    site_counts.columns = ["Website", "Count"]

    fig_site = px.bar(
        site_counts,
        x="Count",
        y="Website",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        text="Count",
    )
    fig_site.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(fig_site, use_container_width=True)
