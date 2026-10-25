"""
================================================================================
MODULE: pages.6_Sensitivity_Analysis
================================================================================
1. WHAT THIS FILE DOES:
   Provides interactive visual sensitivity experiments to test decision robustness,
   tracking rank stability across preset organizational profiles, single-criterion weight
   sweeps, and scenario appearance frequencies.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Decision makers need to know whether the prioritization is
     fragile to subjective weight choices. If an organization shifts emphasis from
     "Task Impact" to "Repair Effort", which repairs remain robustly in the top echelon?
     This page provides the empirical answer through multi-scenario stress-testing.
   - Technical & Architectural Rationale: Visualizes outputs from `core.sensitivity`
     using parallel coordinate charts, rank trajectory line graphs, and stability frequency grids.

3. HOW IT WORKS CONCEPTUALLY:
   - Evaluates issues across 5 stakeholder archetypes (Balanced, Task Critical, Quick Wins, etc.).
   - Visualizes rank trajectories: shows how an issue's rank changes as a single criterion's
     weight varies continuously from 5% to 70%.
   - Computes Scenario Frequency: the number of distinct organizational scenarios in which an
     issue finishes in the Top-3 or Top-5 (a measure of decision stability, not probability).

4. HOW IT WORKS TECHNICALLY:
   - Uses `core.sensitivity.compare_profiles()`, `run_weight_sweep()`, and `calculate_scenario_stability()`.
   - Uses Plotly Express line charts to plot rank trajectory curves against weight changes.
   - Renders interactive dataframes highlighting stable vs volatile repairs.
================================================================================
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.ui_common import setup_page_header
from core.database import load_all_issues
from core.models import CriterionWeights
from core.sensitivity import (
    PRESET_WEIGHT_PROFILES,
    calculate_scenario_stability,
    compare_profiles,
    run_weight_sweep,
)

setup_page_header(
    title="Sensitivity Analysis & Decision Robustness",
    icon="🔬",
    subtitle="Evaluate decision stability under criterion weight variations, rank trajectories, and scenario frequencies",
)

# Crucial Academic Epistemological Notice
st.info(
    """
    🔬 **Methodological Interpretation Notice:**
    Scenario appearance frequencies (e.g., appearing in the Top-3 in 4 of 5 tested organizational profiles)
    are **experimental sensitivity measures** assessing decision model stability under varying stakeholder
    weight assumptions. They are **NOT probabilities** of an accessibility barrier causing user harm,
    assistive technology failure, or legal liability.
    """
)

df = load_all_issues()

# Scope filter
c_s1, c_s2 = st.columns([2, 1])
with c_s1:
    selected_site = st.selectbox(
        "Sensitivity Analysis Scope:",
        ["All Institutional Websites"] + sorted(df["website"].unique().tolist()),
    )
with c_s2:
    only_verified = st.checkbox(
        "Only confirmed WCAG violations",
        value=True,
        help="Excludes unverified candidate findings.",
    )

eval_df = df.copy()
if selected_site != "All Institutional Websites":
    eval_df = eval_df[eval_df["website"] == selected_site]
if only_verified:
    eval_df = eval_df[eval_df["wcag_verified"] == 1]
else:
    eval_df = eval_df[eval_df["verification_status"] != "False Positive Discarded"]

if eval_df.empty:
    st.warning("No accessibility issues match the active filter.")
    st.stop()

st.markdown("---")

# ==============================================================================
# Section 1: Scenario Stability & Top-3 / Top-5 Appearance Frequencies
# ==============================================================================
st.subheader("1. Multi-Scenario Ranking Stability (Top-3 & Top-5 Frequencies)")
st.markdown(
    """
    Evaluates how frequently each repair action appears in the **Top-3** or **Top-5** priorities
    across 5 diverse strategic organizational scenarios:
    1. *Balanced Focus (Default)*
    2. *Essential-Task Critical (Citizen Journeys)*
    3. *Quick Wins (Low Effort First)*
    4. *WCAG Barrier Severity First*
    5. *Widespread Occurrence First*
    """
)

stability_df = calculate_scenario_stability(eval_df)

s_tab1, s_tab2 = st.tabs(["📊 Scenario Frequency Table", "📈 Stability Distribution Chart"])

with s_tab1:
    disp_cols = [
        "issue_id",
        "page_name",
        "issue_type",
        "severity",
        "essential_task_impact",
        "repair_effort",
        "top3_count",
        "top3_frequency_pct",
        "top5_count",
        "top5_frequency_pct",
        "avg_consensus_rank",
        "rank_std_dev",
        "stability_class",
    ]
    rename_dict = {
        "issue_id": "Issue ID",
        "page_name": "Page",
        "issue_type": "Barrier Type",
        "severity": "Severity",
        "essential_task_impact": "Task Impact",
        "repair_effort": "Effort",
        "top3_count": "Top-3 Count",
        "top3_frequency_pct": "Top-3 Freq (%)",
        "top5_count": "Top-5 Count",
        "top5_frequency_pct": "Top-5 Freq (%)",
        "avg_consensus_rank": "Avg Rank",
        "rank_std_dev": "Volatility (σ)",
        "stability_class": "Stability Class",
    }
    st.dataframe(
        stability_df[disp_cols].rename(columns=rename_dict),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Repairs with high Top-5 frequencies (e.g. 80-100%) represent robust priority investments across conflicting organizational priorities.")

with s_tab2:
    top_candidates = stability_df.head(10).copy()
    fig_stab = px.bar(
        top_candidates,
        x="issue_id",
        y=["top3_frequency_pct", "top5_frequency_pct"],
        barmode="group",
        labels={
            "value": "Scenario Appearance Frequency (%)",
            "issue_id": "Issue ID",
            "variable": "Threshold",
        },
        color_discrete_map={
            "top3_frequency_pct": "#2563EB",
            "top5_frequency_pct": "#10B981",
        },
        title="Top 10 Most Stable Repair Priorities across Predefined Strategic Profiles",
    )
    fig_stab.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
    st.plotly_chart(fig_stab, use_container_width=True)

st.markdown("---")

# ==============================================================================
# Section 2: Strategic Profile Comparison Table
# ==============================================================================
st.subheader("2. Strategic Profile Priority Comparison")
st.markdown("Inspect how the #1 recommended repair and top cluster shift between strategic profiles.")

comp_df = compare_profiles(eval_df)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==============================================================================
# Section 3: Dynamic Criterion Weight Sweep & Trajectories
# ==============================================================================
st.subheader("3. Dynamic Single-Criterion Weight Sweep")
st.markdown(
    """
    Systematically sweep a single criterion's weight from a minimum to a maximum threshold.
    The remaining budget $(1.0 - w_{\\text{target}})$ is proportionally redistributed across other criteria,
    recalculating both Fuzzy TOPSIS and VIKOR at each step.
    """
)

crit_options = {
    "C1: Barrier Severity (WCAG Violation Level)": 0,
    "C2: Essential-Task Impact (Citizen Workflow Blockage)": 1,
    "C3: Affected Pages (Occurrence Breadth)": 2,
    "C4: User Impact (Assistive Technology Impairment)": 3,
    "C5: Repair Effort (Engineering Complexity / Cost)": 4,
    "C6: Issue Frequency (Repeatability across Templates)": 5,
}

sw_col1, sw_col2, sw_col3 = st.columns([3, 2, 2])

with sw_col1:
    selected_crit_label = st.selectbox(
        "Criterion to Perturb:",
        list(crit_options.keys()),
        index=1,
    )
    crit_idx = crit_options[selected_crit_label]

with sw_col2:
    w_min = st.slider("Minimum Weight:", 0.01, 0.20, 0.05, 0.01)
    w_max = st.slider("Maximum Weight:", 0.40, 0.90, 0.65, 0.05)

with sw_col3:
    steps = st.slider("Sweep Step Count:", 5, 25, 12, 1)

topsis_ranks_history, vikor_ranks_history, corr_history = run_weight_sweep(
    eval_df,
    target_criterion_idx=crit_idx,
    weight_steps=steps,
    min_w=w_min,
    max_w=w_max,
)

c_left, c_right = st.columns(2)

with c_left:
    st.markdown("#### Rank Correlation Stability vs. Baseline")
    fig_corr = px.line(
        corr_history,
        x="target_weight",
        y=["topsis_baseline_spearman", "vikor_baseline_spearman", "cross_method_spearman"],
        markers=True,
        labels={
            "target_weight": f"Weight of {selected_crit_label.split(':')[0]}",
            "value": "Spearman Rho (ρ)",
            "variable": "Correlation Type",
        },
        color_discrete_map={
            "topsis_baseline_spearman": "#2563EB",
            "vikor_baseline_spearman": "#EF4444",
            "cross_method_spearman": "#10B981",
        },
        title="Algorithmic Robustness Relative to Balanced Baseline",
    )
    fig_corr.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("A stable, horizontal trajectory indicates that rankings remain robust despite subjective weight adjustments.")

with c_right:
    st.markdown("#### Rank Trajectories of Top-5 Benchmark Issues")
    top5_ids = topsis_ranks_history.index[:5]
    top5_df = topsis_ranks_history.loc[top5_ids].reset_index().rename(columns={"index": "Issue ID"})
    top5_melted = top5_df.melt(id_vars="Issue ID", var_name="Weight Step", value_name="Rank")

    fig_traj = px.line(
        top5_melted,
        x="Weight Step",
        y="Rank",
        color="Issue ID",
        markers=True,
        title="Fuzzy TOPSIS Rank Trajectories (Rank 1 = Highest Priority)",
    )
    fig_traj.update_yaxes(autorange="reversed")  # Invert so Rank 1 is at the top
    fig_traj.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    st.plotly_chart(fig_traj, use_container_width=True)
    st.caption("Illustrates whether candidate repairs maintain their top positions or experience rank crossovers.")
