"""Page 4: Repair Prioritization Engine (Fuzzy TOPSIS & VIKOR).

Provides interactive multi-criteria decision prioritization allowing users to
select websites, specific pages, or custom alternative sets, adjust criterion
weights, view side-by-side rankings, and inspect criteria-driven rationales.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.ui_common import setup_page_header
from core.database import load_all_issues, get_registered_websites, get_registered_pages
from core.models import CriterionWeights
from core.mcdm_bridge import run_prioritization, CRITERIA_NAMES
from core.sensitivity import PRESET_WEIGHT_PROFILES

setup_page_header(
    title="Repair Prioritization Engine",
    icon="🎯",
    subtitle="Prioritizing accessibility repairs using Fuzzy TOPSIS and VIKOR Multi-Criteria Decision Making",
)

df = load_all_issues()
reg_sites = get_registered_websites()
reg_pages = get_registered_pages()

# ==============================================================================
# 1. Flexible Scope Selection (Websites, Pages, or Specific Alternatives)
# ==============================================================================
st.subheader("🎯 1. Evaluation Scope & Alternative Selection")

scope_c1, scope_c2 = st.columns([2, 1])

with scope_c1:
    selected_site = st.selectbox(
        "Audited Institution Domain:",
        ["All Institutional Websites"] + sorted(df["website"].unique().tolist()),
    )

with scope_c2:
    only_verified = st.checkbox(
        "Only confirmed WCAG violations",
        value=True,
        help="Filters to human-verified findings, excluding unreviewed candidate scans and discarded false positives.",
    )
    status_filter = st.multiselect(
        "Workflow Statuses to Include:",
        options=["Open", "In Progress", "Needs Review", "Resolved"],
        default=["Open", "In Progress"],
    )

# Filter by website, verification, and status
filtered_df = df.copy()
if selected_site != "All Institutional Websites":
    filtered_df = filtered_df[filtered_df["website"] == selected_site]

if only_verified:
    filtered_df = filtered_df[filtered_df["wcag_verified"] == 1]
else:
    filtered_df = filtered_df[filtered_df["verification_status"] != "False Positive Discarded"]

if status_filter:
    filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]

# Optional Granular Sub-Filters: Pages or Custom Alternatives
with st.expander("🔍 Filter by Specific Citizen Journey Pages or Custom Issue Subset", expanded=False):
    f_sub1, f_sub2 = st.columns(2)
    available_pages = sorted(filtered_df["page_name"].unique().tolist())
    with f_sub1:
        selected_pages = st.multiselect(
            "Filter to Specific Pages (Leave empty to evaluate all):",
            options=available_pages,
            default=[],
        )
    if selected_pages:
        filtered_df = filtered_df[filtered_df["page_name"].isin(selected_pages)]

    available_issues = sorted(filtered_df["issue_id"].unique().tolist())
    with f_sub2:
        selected_issue_subset = st.multiselect(
            "Select Specific Issues to Prioritize (Leave empty to evaluate all):",
            options=available_issues,
            default=[],
        )
    if selected_issue_subset:
        filtered_df = filtered_df[filtered_df["issue_id"].isin(selected_issue_subset)]

if filtered_df.empty:
    st.warning("⚠️ No accessibility repair alternatives match the selected scope filters.")
    st.stop()

st.info(f"📋 **Evaluation Scope Active:** `{len(filtered_df)}` accessibility repair alternatives selected for prioritization.")

st.markdown("---")

# ==============================================================================
# 2. Validated Decision Criteria Weights & Algorithm Settings
# ==============================================================================
st.subheader("⚙️ 2. Decision Criteria Weights & Algorithmic Controls")

w_col1, w_col2, w_col3 = st.columns([2, 2, 2])

with w_col1:
    selected_preset = st.selectbox(
        "Load Preset Weight Profile:",
        list(PRESET_WEIGHT_PROFILES.keys()),
        index=0,
    )
    preset_weights = PRESET_WEIGHT_PROFILES[selected_preset]

with w_col2:
    vikor_formulation = st.selectbox(
        "VIKOR Formulation:",
        ["fuzzy", "classical"],
        format_func=lambda x: "Fuzzy VIKOR (Opricovic 2011 on TFNs)" if x == "fuzzy" else "Classical VIKOR with Centroid Defuzzification (Opricovic 2004)",
    )

with w_col3:
    vikor_v = st.slider(
        "VIKOR Strategy Weight (v):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="v = 0.5 represents balanced consensus. v > 0.5 emphasizes total group utility (S); v < 0.5 emphasizes individual regret veto (R).",
    )

# Individual Weight Sliders
with st.expander("🛠️ Adjust Individual Criterion Weights (C1 - C6)", expanded=False):
    st.caption("Sliders define relative importance. Weights automatically normalize to sum to 100%.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        wc1 = st.slider("C1: Barrier Severity", 0.0, 1.0, preset_weights.c1_severity, 0.05)
        wc2 = st.slider("C2: Essential-Task Impact", 0.0, 1.0, preset_weights.c2_task_impact, 0.05)
    with sc2:
        wc3 = st.slider("C3: Affected Pages", 0.0, 1.0, preset_weights.c3_affected_pages, 0.05)
        wc4 = st.slider("C4: User Impact", 0.0, 1.0, preset_weights.c4_user_impact, 0.05)
    with sc3:
        wc5 = st.slider("C5: Repair Effort (Cost)", 0.0, 1.0, preset_weights.c5_repair_effort, 0.05)
        wc6 = st.slider("C6: Issue Frequency", 0.0, 1.0, preset_weights.c6_frequency, 0.05)

raw_weights = CriterionWeights(wc1, wc2, wc3, wc4, wc5, wc6)
norm_weights = raw_weights.normalize()

# Display active weight share
w_list = norm_weights.as_list()
st.caption(
    f"**Active Normalized Weights:** "
    f"C1 (Severity): `{w_list[0]*100:.1f}%` | "
    f"C2 (Task Impact): `{w_list[1]*100:.1f}%` | "
    f"C3 (Pages): `{w_list[2]*100:.1f}%` | "
    f"C4 (User Impact): `{w_list[3]*100:.1f}%` | "
    f"C5 (Repair Effort): `{w_list[4]*100:.1f}%` | "
    f"C6 (Frequency): `{w_list[5]*100:.1f}%`"
)

# Execution of Decision Models
prioritized_df, meta = run_prioritization(
    filtered_df,
    norm_weights,
    vikor_v=vikor_v,
    vikor_formulation=vikor_formulation,
)

st.markdown("---")

# ==============================================================================
# 3. Prioritization Results Summary & Side-by-Side Comparison Table
# ==============================================================================
st.subheader("🏆 3. Prioritized Repair Ranking (TOPSIS vs. VIKOR Side-by-Side)")

r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("Evaluated Issues", len(prioritized_df))
r_col2.metric("Spearman Correlation (ρ)", f"{meta.get('spearman_rho', 0.0):.4f}")
r_col3.metric("Kendall Tau (τ)", f"{meta.get('kendall_tau', 0.0):.4f}")
val = meta.get("vikor_validation", {})
r_col4.metric(
    "Compromise Stability",
    "Stable" if val.get("compromise_stable", False) else f"Cluster ({len(val.get('compromise_set_indices', []))})",
)

# Helper function: Deterministic Criteria-Driven Priority Rationale
def generate_criteria_rationale(row: pd.Series, weights: CriterionWeights) -> str:
    """Generate a factual, deterministic rationale derived purely from criterion values and weights."""
    reasons = []
    w = weights.as_list()

    # C2 Task Impact
    if row["essential_task_impact"] in ["High", "Very High"]:
        reasons.append(f"Essential-Task Impact is {row['essential_task_impact']} (Weight: {w[1]*100:.0f}%), directly blocking citizen transaction completion.")
    elif row["essential_task_impact"] == "Medium":
        reasons.append(f"Moderate task impairment on citizen service workflow (Weight: {w[1]*100:.0f}%).")

    # C1 Barrier Severity
    if row["severity"] in ["High", "Very High"]:
        reasons.append(f"Barrier Severity is {row['severity']} (Weight: {w[0]*100:.0f}%), presenting serious WCAG non-compliance.")

    # C5 Repair Effort (Cost)
    if row["repair_effort"] in ["Very Low", "Low"]:
        reasons.append(f"Repair Effort is {row['repair_effort']} (Weight: {w[4]*100:.0f}%), offering rapid remediation with minimal engineering regret.")
    elif row["repair_effort"] == "High":
        reasons.append(f"High remediation effort (Weight: {w[4]*100:.0f}%), penalized under VIKOR regret minimization.")

    # C3 Affected Pages
    pages_cnt = int(row.get("affected_pages", 1))
    if pages_cnt >= 3:
        reasons.append(f"Widespread occurrence across {pages_cnt} audited pages (Weight: {w[2]*100:.0f}%).")

    # C4 User Impact
    if row["user_impact"] in ["High", "Very High"]:
        reasons.append(f"Severe User Impact on assistive device users (Weight: {w[3]*100:.0f}%).")

    if not reasons:
        reasons.append("Balanced intermediate performance across criteria.")

    return " • " + "\n • ".join(reasons)

# Add criteria rationale column
prioritized_df["priority_rationale"] = [
    generate_criteria_rationale(row, norm_weights) for _, row in prioritized_df.iterrows()
]

# Side-by-side Table Columns
display_cols = [
    "consensus_rank",
    "issue_id",
    "website",
    "page_name",
    "issue_type",
    "severity",
    "essential_task_impact",
    "affected_pages",
    "user_impact",
    "repair_effort",
    "frequency",
    "topsis_score",
    "topsis_rank",
    "vikor_q",
    "vikor_rank",
    "rank_diff",
    "recommended_fix",
]

renamed_cols = {
    "consensus_rank": "Priority",
    "issue_id": "Issue ID",
    "website": "Website",
    "page_name": "Page",
    "issue_type": "Barrier Type",
    "severity": "C1: Severity",
    "essential_task_impact": "C2: Task Impact",
    "affected_pages": "C3: Pages",
    "user_impact": "C4: User Impact",
    "repair_effort": "C5: Effort",
    "frequency": "C6: Frequency",
    "topsis_score": "TOPSIS CC",
    "topsis_rank": "TOPSIS Rank",
    "vikor_q": "VIKOR Q",
    "vikor_rank": "VIKOR Rank",
    "rank_diff": "Diff (T - V)",
    "recommended_fix": "Recommended Fix",
}

st.dataframe(
    prioritized_df[display_cols].rename(columns=renamed_cols),
    use_container_width=True,
    hide_index=True,
)
st.caption("Side-by-side MCDM prioritization showing criterion ratings, TOPSIS relative closeness (CC), VIKOR compromise index (Q), and recommended repairs.")

st.markdown("---")

# ==============================================================================
# 4. Criteria-Driven Priority Rationale Deep-Dive
# ==============================================================================
st.subheader("💡 4. Criteria-Driven Priority Rationales")
st.markdown("Inspect why any specific repair received high or low priority based strictly on underlying criteria values, without fabricated AI claims.")

sel_issue_id = st.selectbox(
    "Select an Issue to Inspect Factual Mathematical Priority Rationale:",
    prioritized_df["issue_id"].tolist(),
    index=0,
)

sel_row = prioritized_df[prioritized_df["issue_id"] == sel_issue_id].iloc[0]

i_col1, i_col2 = st.columns([1, 1])

with i_col1:
    st.markdown(f"#### Repair Profile: `{sel_row['issue_id']}`")
    st.markdown(f"- **Website:** {sel_row['website']}")
    st.markdown(f"- **Page:** {sel_row['page_name']}")
    st.markdown(f"- **Barrier Type:** {sel_row['issue_type']}")
    st.markdown(f"- **Technical Description:** {sel_row['description']}")
    st.markdown(f"- **Recommended Fix:** `{sel_row['recommended_fix']}`")
    st.markdown(f"- **WCAG Reference:** `{sel_row.get('wcag_criterion', 'WCAG 2.1')}`")

with i_col2:
    st.markdown("#### Mathematical Rationale Breakdown")
    st.markdown(
        f"""
        - **Consensus Priority:** Rank `#{sel_row['consensus_rank']}` of `{len(prioritized_df)}`
        - **Fuzzy TOPSIS:** Rank `#{sel_row['topsis_rank']}` (Closeness $CC = {sel_row['topsis_score']:.4f}$)
        - **VIKOR:** Rank `#{sel_row['vikor_rank']}` (Compromise Index $Q = {sel_row['vikor_q']:.4f}$)
        """
    )
    st.markdown("**Underlying Decision Factors:**")
    st.markdown(sel_row["priority_rationale"])

st.markdown("---")

# ==============================================================================
# 5. Fuzzy TOPSIS Intermediate Calculation Matrices
# ==============================================================================
ft_res = meta.get("fuzzy_topsis_result")
if ft_res is not None:
    with st.expander("🔬 Fuzzy TOPSIS Intermediate Calculation Matrices (Chen 2000 Audit Trail)", expanded=False):
        st.markdown(
            "Review the intermediate calculation matrices, normalized fuzzy numbers, weighted values, and ideal benchmarks."
        )

        m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
            "1. Normalized Matrix (R~)",
            "2. Weighted Matrix (V~)",
            "3. Ideal Solutions (A*, A-)",
            "4. Distances & Closeness (D+, D-, CC)",
        ])

        with m_tab1:
            st.markdown("**Normalized Fuzzy Matrix $\\tilde{R} = [\\tilde{r}_{ij}]$** *(All values in [0, 1])*")
            st.dataframe(ft_res.get_normalized_df(), use_container_width=True)

        with m_tab2:
            st.markdown("**Weighted Normalized Fuzzy Matrix $\\tilde{V} = [\\tilde{v}_{ij}]$ where $\\tilde{v}_{ij} = \\tilde{r}_{ij} \\otimes w_j$**")
            st.dataframe(ft_res.get_weighted_df(), use_container_width=True)

        with m_tab3:
            st.markdown("**Fuzzy Positive-Ideal ($A^*$) and Negative-Ideal ($A^-$) Solutions**")
            crit_names = ft_res.criteria_names or [f"C{j+1}" for j in range(len(ft_res.is_benefit))]
            ideal_rows = []
            for j, c_name in enumerate(crit_names):
                fp = ft_res.fpis[j]
                fn = ft_res.fnis[j]
                ideal_rows.append({
                    "Criterion": c_name,
                    "Type": "Benefit" if ft_res.is_benefit[j] else "Cost",
                    "Weight": f"{ft_res.weights[j]:.4f}",
                    "FPIS A* (l, m, u)": f"({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f})",
                    "FNIS A- (l, m, u)": f"({fn[0]:.4f}, {fn[1]:.4f}, {fn[2]:.4f})",
                })
            st.dataframe(pd.DataFrame(ideal_rows), use_container_width=True, hide_index=True)

        with m_tab4:
            st.markdown("**Separation Distances and Relative Closeness ($CC_i = D_i^- / (D_i^+ + D_i^-)$)**")
            alt_names = ft_res.alternative_names or [f"A{i+1}" for i in range(len(ft_res.d_plus))]
            dist_df = pd.DataFrame({
                "Issue ID": alt_names,
                "Distance to FPIS (D+)": np.round(ft_res.d_plus, 4),
                "Distance to FNIS (D-)": np.round(ft_res.d_minus, 4),
                "Closeness Coefficient (CC)": np.round(ft_res.closeness_coefficients, 4),
                "Fuzzy TOPSIS Rank": ft_res.rankings,
            }).sort_values(by="Fuzzy TOPSIS Rank", ascending=True)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
