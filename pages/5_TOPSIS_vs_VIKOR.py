"""Page 5: Comparative Research - Fuzzy TOPSIS vs. VIKOR.

Rigorous comparative multi-criteria decision analysis evaluating algorithmic
concordance, rank divergences, trade-off mechanics, and compromise solutions.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.ui_common import setup_page_header
from core.database import load_all_issues
from core.models import CriterionWeights
from core.mcdm_bridge import run_prioritization, CRITERIA_NAMES
from core.sensitivity import PRESET_WEIGHT_PROFILES

setup_page_header(
    title="Comparative Analysis: Fuzzy TOPSIS vs. VIKOR",
    icon="⚖️",
    subtitle="Evaluating rank correlation, decision agreement, and trade-off mechanics between MCDM paradigms",
)

st.warning(
    """
    ⚖️ **Academic Evaluation Principle:**
    In Multi-Criteria Decision Making (MCDM), an algorithm must **never be claimed as 'accurate' or 'validated'**
    merely because its output appears plausible or aligns with uncalibrated human intuition.
    Rank concordance metrics (Spearman ρ, Kendall τ) evaluate **inter-method consistency and trade-off sensitivity**,
    not ground truth accessibility compliance.
    """
)

df = load_all_issues()

# ==============================================================================
# Configuration Sidebar / Top Bar
# ==============================================================================
st.subheader("⚙️ Analysis Scope & Method Configuration")

c_cfg1, c_cfg2, c_cfg3 = st.columns([2, 2, 2])

with c_cfg1:
    selected_site = st.selectbox(
        "Audited Institution:",
        ["All Institutional Websites"] + sorted(df["website"].unique().tolist()),
    )
    only_verified = st.checkbox(
        "Only confirmed WCAG violations",
        value=True,
        help="Excludes unverified automated detections and discarded false positives.",
    )

with c_cfg2:
    selected_preset = st.selectbox(
        "Weight Profile Preset:",
        list(PRESET_WEIGHT_PROFILES.keys()),
        index=0,
    )
    vikor_formulation = st.selectbox(
        "VIKOR Formulation:",
        ["fuzzy", "classical"],
        format_func=lambda x: "Fuzzy VIKOR (Opricovic 2011 on TFNs)" if x == "fuzzy" else "Classical VIKOR with Centroid Defuzzification (Opricovic 2004)",
        help="Choose whether VIKOR calculates regret directly on Triangular Fuzzy Numbers or applies an explicit Centroid defuzzification step.",
    )

with c_cfg3:
    vikor_v = st.slider(
        "VIKOR Compromise Weight (v):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="v > 0.5 favors majority group utility (S); v < 0.5 favors individual regret veto (R). Default v = 0.5 represents balanced consensus.",
    )

# Filter dataset based on selections
eval_df = df.copy()
if selected_site != "All Institutional Websites":
    eval_df = eval_df[eval_df["website"] == selected_site]
if only_verified:
    eval_df = eval_df[eval_df["wcag_verified"] == 1]
else:
    eval_df = eval_df[eval_df["verification_status"] != "False Positive Discarded"]

# Weights customization expander
preset_weights = PRESET_WEIGHT_PROFILES[selected_preset]
with st.expander("🛠️ Adjust Individual Criterion Weights (C1 - C6)", expanded=False):
    st.caption("Adjust sliders below. Weights automatically normalize to sum to 100%.")
    wc_cols = st.columns(3)
    with wc_cols[0]:
        wc1 = st.slider("C1: Severity", 0.0, 1.0, preset_weights.c1_severity, 0.05)
        wc2 = st.slider("C2: Task Impact", 0.0, 1.0, preset_weights.c2_task_impact, 0.05)
    with wc_cols[1]:
        wc3 = st.slider("C3: Pages", 0.0, 1.0, preset_weights.c3_affected_pages, 0.05)
        wc4 = st.slider("C4: User Impact", 0.0, 1.0, preset_weights.c4_user_impact, 0.05)
    with wc_cols[2]:
        wc5 = st.slider("C5: Repair Effort", 0.0, 1.0, preset_weights.c5_repair_effort, 0.05)
        wc6 = st.slider("C6: Frequency", 0.0, 1.0, preset_weights.c6_frequency, 0.05)

raw_weights = CriterionWeights(wc1, wc2, wc3, wc4, wc5, wc6)
norm_weights = raw_weights.normalize()

# Execute Prioritization Pipeline
results_df, meta = run_prioritization(
    eval_df,
    norm_weights,
    vikor_v=vikor_v,
    vikor_formulation=vikor_formulation,
)

if results_df.empty:
    st.warning("No accessibility issues match the active filter criteria.")
    st.stop()

st.markdown("---")

# ==============================================================================
# Statistical Correlation Metrics & Decision Agreement Scorecards
# ==============================================================================
st.subheader("📊 Statistical Rank Correlation & Agreement Metrics")

# Agreement rate (% within 2 rank spots)
close_ranks = (results_df["absolute_rank_diff"] <= 2).mean() * 100.0

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric(
    "Spearman's Rho (ρ)",
    f"{meta.get('spearman_rho', 0.0):.4f}",
    help="Measures monotonic agreement between rankings. ρ > 0.8 indicates strong alignment.",
)
m_col2.metric(
    "Kendall's Tau (τ)",
    f"{meta.get('kendall_tau', 0.0):.4f}",
    help="Pairwise concordant vs discordant pairs metric. τ > 0.6 indicates high ordering consistency.",
)
m_col3.metric(
    "Rank Proximity (≤ 2 spots)",
    f"{close_ranks:.1f}%",
    help="Percentage of repair issues where TOPSIS and VIKOR assigned ranks within 2 spots of each other.",
)
val_info = meta.get("vikor_validation", {})
m_col4.metric(
    "Compromise Stability",
    "Stable (Unique)" if val_info.get("compromise_stable") else "Compromise Set",
    help="Whether the top-ranked VIKOR alternative satisfies both Acceptable Advantage (C1) and Stability (C2).",
)

st.markdown("---")

# ==============================================================================
# Paired Visualizations: Concordance Scatter, Bar Chart & Regret Space
# ==============================================================================
st.subheader("📈 Visual Comparison: Geometric Distance vs. Compromise Regret")

chart_tabs = st.tabs([
    "1. Rank Concordance Scatter Plot",
    "2. Paired Score Comparison (Top 12)",
    "3. VIKOR Utility vs. Regret Space (S vs. R)",
])

with chart_tabs[0]:
    c_sc1, c_sc2 = st.columns([3, 2])
    with c_sc1:
        fig_scatter = px.scatter(
            results_df,
            x="topsis_rank",
            y="vikor_rank",
            hover_name="issue_id",
            hover_data=["issue_type", "severity", "essential_task_impact", "repair_effort"],
            color="absolute_rank_diff",
            color_continuous_scale="Turbo",
            labels={
                "topsis_rank": "Fuzzy TOPSIS Rank (1 = Highest Priority)",
                "vikor_rank": "VIKOR Rank (1 = Best Compromise)",
                "absolute_rank_diff": "|Rank Diff|",
            },
            title="Rank Concordance: Proximity to Diagonal (y = x) Denotes Algorithmic Agreement",
        )
        max_r = len(results_df)
        fig_scatter.add_trace(
            go.Scatter(
                x=[1, max_r],
                y=[1, max_r],
                mode="lines",
                line=dict(color="rgba(239, 68, 68, 0.8)", dash="dash", width=2),
                name="Perfect Concordance (y = x)",
            )
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=430)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_sc2:
        st.markdown("#### 💡 Understanding Ranking Disagreements")
        st.info(
            """
            **Ranking disagreement is useful evidence about trade-offs, not a mathematical failure.**

            - **Fuzzy TOPSIS** operates on a **geometric compensatory model**. An alternative that performs exceptionally well on C1 (Severity) or C2 (Task Impact) can compensate for poor performance (high cost) on C5 (Repair Effort).
            - **VIKOR** operates on a **compromise & regret minimization model**. An alternative with severe weakness on even a single high-weight criterion incurs a heavy **Individual Regret ($R_i$)** penalty, preventing it from ranking first unless its benefits overwhelmingly justify the sacrifice.
            """
        )
        st.markdown(
            f"""
            **Current Evaluation Settings:**
            - Total Evaluated Issues: **{len(results_df)}**
            - VIKOR Formulation: **{vikor_formulation.capitalize()}**
            - Strategy Weight ($v$): **{vikor_v}** ({'Emphasizing Group Utility' if vikor_v > 0.5 else ('Emphasizing Individual Regret' if vikor_v < 0.5 else 'Balanced Compromise')})
            """
        )

with chart_tabs[1]:
    top12 = results_df.head(12).copy()
    # Prepare paired metrics: TOPSIS Closeness CC (higher is better) vs VIKOR (1 - Q) (higher is better)
    top12["vikor_closeness"] = 1.0 - top12["vikor_q"]
    top12["issue_label"] = top12["issue_id"] + " (" + top12["issue_type"].str.slice(0, 16) + "...)"

    fig_paired = px.bar(
        top12,
        x="issue_id",
        y=["topsis_score", "vikor_closeness"],
        barmode="group",
        labels={
            "value": "Normalized Closeness / Desirability (Higher = Better)",
            "issue_id": "Issue ID",
            "variable": "Evaluation Metric",
        },
        color_discrete_map={
            "topsis_score": "#2563EB",
            "vikor_closeness": "#10B981",
        },
        title="Top 12 Issues: TOPSIS Relative Closeness (CC) vs. VIKOR Inverted Index (1 - Q)",
    )
    fig_paired.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    st.plotly_chart(fig_paired, use_container_width=True)
    st.caption("Both bars scaled so higher values represent superior remediation candidates.")

with chart_tabs[2]:
    st.markdown("**VIKOR Decision Plane: Group Utility ($S$) vs. Individual Regret ($R$)**")
    st.caption("The ideal compromise solution is located at the bottom-left origin (minimum S, minimum R).")
    fig_sr = px.scatter(
        results_df,
        x="vikor_s",
        y="vikor_r",
        hover_name="issue_id",
        hover_data=["issue_type", "vikor_q", "repair_effort"],
        color="vikor_q",
        color_continuous_scale="Reds_r",
        size=np.maximum(1, 15 - results_df["vikor_rank"] * 0.3),
        labels={
            "vikor_s": "Group Utility S (Total Weighted Regret - Lower is Better)",
            "vikor_r": "Individual Regret R (Worst-Case Regret - Lower is Better)",
            "vikor_q": "VIKOR Index Q",
        },
        title="Distribution in Regret Space (Bottom-Left = Optimal Compromise)",
    )
    fig_sr.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=400)
    st.plotly_chart(fig_sr, use_container_width=True)

st.markdown("---")

# ==============================================================================
# VIKOR Compromise Solution Evaluation Card (C1 & C2)
# ==============================================================================
st.subheader("🛡️ VIKOR Compromise Solution Verification (Opricovic & Tzeng, 2004)")

c_card1, c_card2 = st.columns([1, 1])

with c_card1:
    st.markdown("#### Condition Checks")
    dq = val_info.get("dq", 0.0)
    adv = val_info.get("advantage", 0.0)
    c1_ok = val_info.get("c1_acceptable_advantage", False)
    c2_ok = val_info.get("c2_acceptable_stability", False)

    st.markdown(
        f"""
        - **Advantage Threshold ($DQ = 1 / (m - 1)$):** `{dq:.4f}`
        - **Observed Advantage ($Q(A_2) - Q(A_1)$):** `{adv:.4f}`
        - **Condition 1 (Acceptable Advantage):** {'✅ Satisfied' if c1_ok else '❌ Not Satisfied (Cluster proposed)'}
        - **Condition 2 (Acceptable Stability):** {'✅ Satisfied' if c2_ok else '❌ Not Satisfied (Co-winners proposed)'}
        """
    )

with c_card2:
    st.markdown("#### Formal Compromise Recommendation")
    comp_names = val_info.get("compromise_set_names", [])
    if val_info.get("compromise_stable", False):
        st.success(
            f"**Unique Optimal Compromise:** Alternative `{comp_names[0] if comp_names else 'N/A'}` "
            f"satisfies both acceptable advantage and decision-making stability."
        )
    else:
        st.warning(
            f"**Compromise Cluster Proposed:** {len(comp_names)} alternatives qualify within the compromise set: "
            f"`{', '.join(comp_names)}`. Decision makers should consider these co-candidates."
        )
    st.caption(val_info.get("explanation", ""))

st.markdown("---")

# ==============================================================================
# Comprehensive Ranking Table
# ==============================================================================
st.subheader("📋 Comprehensive Method Comparison Table")

table_df = results_df.copy()

# Add agreement classification tag
def classify_agreement(diff: int) -> str:
    if abs(diff) <= 1:
        return "High Agreement"
    elif diff < 0:
        return "TOPSIS Favored"
    else:
        return "VIKOR Favored"

table_df["Agreement"] = table_df["rank_diff"].apply(classify_agreement)

display_cols = [
    "consensus_rank",
    "issue_id",
    "website",
    "page_name",
    "issue_type",
    "topsis_rank",
    "topsis_score",
    "vikor_rank",
    "vikor_q",
    "rank_diff",
    "Agreement",
    "severity",
    "essential_task_impact",
    "repair_effort",
]

renamed_cols = {
    "consensus_rank": "Consensus",
    "issue_id": "Issue ID",
    "website": "Website",
    "page_name": "Page",
    "issue_type": "Barrier Type",
    "topsis_rank": "TOPSIS Rank",
    "topsis_score": "TOPSIS CC",
    "vikor_rank": "VIKOR Rank",
    "vikor_q": "VIKOR Q",
    "rank_diff": "Diff (T - V)",
    "Agreement": "Alignment",
    "severity": "Severity",
    "essential_task_impact": "Task Impact",
    "repair_effort": "Effort",
}

st.dataframe(
    table_df[display_cols].rename(columns=renamed_cols),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ==============================================================================
# Deep-Dive Explanations for Selected Repair Actions
# ==============================================================================
st.subheader("🔍 Deep-Dive Repair Trade-Off Explainer")

all_issue_ids = results_df["issue_id"].tolist()
selected_issue_id = st.selectbox(
    "Select an Accessibility Issue to Inspect Method Disagreement & Decision Mechanics:",
    all_issue_ids,
    index=0,
)

selected_row = results_df[results_df["issue_id"] == selected_issue_id].iloc[0]

ex_col1, ex_col2 = st.columns([1, 1])

with ex_col1:
    st.markdown(f"#### Audit Profile: `{selected_row['issue_id']}`")
    st.markdown(f"- **Website:** {selected_row['website']}")
    st.markdown(f"- **Page:** {selected_row['page_name']}")
    st.markdown(f"- **Barrier:** {selected_row['issue_type']}")
    st.markdown(f"- **Description:** {selected_row['description']}")
    st.markdown(f"- **Recommended Fix:** `{selected_row['recommended_fix']}`")
    st.markdown("---")
    st.markdown(f"- **Severity (C1):** `{selected_row['severity']}`")
    st.markdown(f"- **Essential-Task Impact (C2):** `{selected_row['essential_task_impact']}`")
    st.markdown(f"- **Repair Effort (C5, Cost):** `{selected_row['repair_effort']}`")

with ex_col2:
    st.markdown("#### Method Disagreement Analysis")
    t_rank = int(selected_row["topsis_rank"])
    v_rank = int(selected_row["vikor_rank"])
    diff = int(selected_row["rank_diff"])

    t_score = float(selected_row["topsis_score"])
    v_q = float(selected_row["vikor_q"])
    v_s = float(selected_row["vikor_s"])
    v_r = float(selected_row["vikor_r"])

    st.markdown(f"- **Fuzzy TOPSIS Rank:** `#{t_rank}` (Closeness $CC = {t_score:.4f}$)")
    st.markdown(f"- **VIKOR Rank:** `#{v_rank}` (Index $Q = {v_q:.4f}$)")
    st.markdown(f"- **Rank Difference ($T - V$):** `{diff:+d}` spots")
    st.markdown(f"- **VIKOR Group Utility ($S$):** `{v_s:.4f}` | **Individual Regret ($R$):** `{v_r:.4f}`")

    if abs(diff) <= 2:
        st.success(
            f"**High Consensus:** Both algorithms agree on the priority of `{selected_row['issue_id']}`. "
            f"Its strong performance on essential-task impact combined with moderate effort makes it "
            f"both geometrically close to the ideal and an optimal compromise."
        )
    elif diff < -2:
        st.info(
            f"**TOPSIS Prioritized Higher (by {abs(diff)} spots):** Fuzzy TOPSIS rewarded `{selected_row['issue_id']}`'s "
            f"high severity or task impact geometrically. However, its high repair effort caused a severe "
            f"individual regret penalty ($R = {v_r:.4f}$) under VIKOR, pulling down its compromise rank."
        )
    else:
        st.info(
            f"**VIKOR Prioritized Higher (by {diff} spots):** VIKOR recognized `{selected_row['issue_id']}` as an "
            f"outstanding low-risk compromise. Its low repair effort produced minimal individual regret ($R = {v_r:.4f}$), "
            f"giving it a quick-win advantage over more difficult repairs."
        )

st.markdown("---")

# ==============================================================================
# 4. Criterion-Weight Sensitivity Summary Table
# ==============================================================================
st.subheader("📊 4. Inter-Method Correlation across Strategic Weight Profiles")
st.markdown("Evaluates how concordance between Fuzzy TOPSIS and VIKOR shifts when adopting different organizational priority perspectives.")

from core.sensitivity import compare_profiles
sens_summary_df = compare_profiles(eval_df, vikor_v=vikor_v, vikor_formulation=vikor_formulation)
st.dataframe(sens_summary_df, use_container_width=True, hide_index=True)
st.caption("Demonstrates the degree of algorithmic congruence across conflicting decision criteria weight distributions.")

st.markdown("---")

# ==============================================================================
# 5. External Reviewer Reference Ranking Comparison
# ==============================================================================
st.subheader("🧑‍🏫 5. External Reviewer Reference Ranking Comparison")
st.markdown(
    """
    Compare MCDM algorithmic outcomes against an independent **human expert reference ranking**.
    Evaluate whether algorithmic priorities reflect standard WCAG compliance auditing or expert reviewer intent.
    """
)

from scipy.stats import spearmanr, kendalltau

ref_col1, ref_col2 = st.columns([1, 1])

# Generate a sensible default reference ranking based on severity & task criticality
# (e.g. Expert Auditor prioritizing Essential-Task Impact then Severity)
expert_baseline_order = results_df.sort_values(
    by=["essential_task_impact", "severity", "affected_pages"],
    ascending=[False, False, False],
)["issue_id"].tolist()

with ref_col1:
    st.markdown("#### Reference Benchmark Mode")
    ref_mode = st.radio(
        "Select Reviewer Reference Source:",
        ["WCAG Expert Auditor Baseline", "Custom Reviewer Ranking (Ordered Top-to-Bottom)"],
        index=0,
    )

    if ref_mode == "Custom Reviewer Ranking (Ordered Top-to-Bottom)":
        custom_ref_order = st.multiselect(
            "Select Issues in your Preferred Order of Priority (1st chosen = Rank 1):",
            options=results_df["issue_id"].tolist(),
            default=results_df["issue_id"].tolist()[:min(5, len(results_df))],
        )
        if len(custom_ref_order) >= 2:
            active_ref_list = custom_ref_order + [x for x in results_df["issue_id"] if x not in custom_ref_order]
        else:
            active_ref_list = expert_baseline_order
            st.caption("Fewer than 2 issues selected; defaulting to Expert Auditor baseline.")
    else:
        active_ref_list = expert_baseline_order
        st.caption("Expert Auditor Baseline: Structured evaluation strictly ordering by essential user task impairment, technical violation severity, and page occurrence.")

# Map reference ranks
ref_rank_map = {iid: r + 1 for r, iid in enumerate(active_ref_list)}
ref_eval_df = results_df.copy()
ref_eval_df["reviewer_ref_rank"] = ref_eval_df["issue_id"].map(ref_rank_map).fillna(len(ref_eval_df))

# Compute correlation with reference
rho_t_ref, _ = spearmanr(ref_eval_df["topsis_rank"], ref_eval_df["reviewer_ref_rank"])
tau_t_ref, _ = kendalltau(ref_eval_df["topsis_rank"], ref_eval_df["reviewer_ref_rank"])

rho_v_ref, _ = spearmanr(ref_eval_df["vikor_rank"], ref_eval_df["reviewer_ref_rank"])
tau_v_ref, _ = kendalltau(ref_eval_df["vikor_rank"], ref_eval_df["reviewer_ref_rank"])

rho_c_ref, _ = spearmanr(ref_eval_df["consensus_rank"], ref_eval_df["reviewer_ref_rank"])
tau_c_ref, _ = kendalltau(ref_eval_df["consensus_rank"], ref_eval_df["reviewer_ref_rank"])

with ref_col2:
    st.markdown("#### Correlation with Reviewer Reference")
    st.markdown(
        f"""
        | Decision Method | Spearman Rho (ρ) | Kendall Tau (τ) | Agreement Degree |
        | :--- | :---: | :---: | :--- |
        | **Fuzzy TOPSIS** | `{rho_t_ref:.4f}` | `{tau_t_ref:.4f}` | {'High' if rho_t_ref >= 0.7 else ('Moderate' if rho_t_ref >= 0.4 else 'Low')} |
        | **VIKOR** | `{rho_v_ref:.4f}` | `{tau_v_ref:.4f}` | {'High' if rho_v_ref >= 0.7 else ('Moderate' if rho_v_ref >= 0.4 else 'Low')} |
        | **Composite Consensus** | `{rho_c_ref:.4f}` | `{tau_c_ref:.4f}` | {'High' if rho_c_ref >= 0.7 else ('Moderate' if rho_c_ref >= 0.4 else 'Low')} |
        """
    )

ref_eval_df["delta_topsis_ref"] = ref_eval_df["topsis_rank"] - ref_eval_df["reviewer_ref_rank"]
ref_eval_df["delta_vikor_ref"] = ref_eval_df["vikor_rank"] - ref_eval_df["reviewer_ref_rank"]

st.markdown("#### Comparison with Reviewer Benchmark Table")
comp_ref_cols = [
    "issue_id",
    "page_name",
    "issue_type",
    "reviewer_ref_rank",
    "topsis_rank",
    "delta_topsis_ref",
    "vikor_rank",
    "delta_vikor_ref",
    "consensus_rank",
    "severity",
    "essential_task_impact",
    "repair_effort",
]
st.dataframe(
    ref_eval_df[comp_ref_cols].rename(
        columns={
            "issue_id": "Issue ID",
            "page_name": "Page",
            "issue_type": "Barrier Type",
            "reviewer_ref_rank": "Reviewer Ref Rank",
            "topsis_rank": "TOPSIS Rank",
            "delta_topsis_ref": "Δ (TOPSIS - Ref)",
            "vikor_rank": "VIKOR Rank",
            "delta_vikor_ref": "Δ (VIKOR - Ref)",
            "consensus_rank": "Consensus Rank",
            "severity": "Severity",
            "essential_task_impact": "Task Impact",
            "repair_effort": "Effort",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ==============================================================================
# VIKOR Intermediate Matrices & Audit Trail
# ==============================================================================
vk_res = meta.get("vikor_result")
if vk_res is not None:
    with st.expander("🔬 VIKOR Intermediate Calculation Matrices & Regret Audit Trail", expanded=False):
        st.markdown(
            """
            This panel provides a transparent mathematical audit trail for **VIKOR compromise calculations**.
            Inspect the ideal benchmarks ($f^*, f^-$), the normalized regret distances, and the weighted regret matrix.
            """
        )
        v_tab1, v_tab2, v_tab3 = st.tabs([
            "1. Best & Worst Criteria Bounds (f*, f⁻)",
            "2. Weighted Regret Matrix (w_j * d_ij)",
            "3. Summary Utility & Regret Measures (S, R, Q)",
        ])

        with v_tab1:
            st.markdown("**Criteria Extremes ($f_j^*$ and $f_j^-$)**")
            crit_names = vk_res.criteria_names or [f"C{j+1}" for j in range(len(vk_res.is_benefit))]
            bounds_rows = []
            for j, c_name in enumerate(crit_names):
                b_val = vk_res.best_f[j]
                w_val = vk_res.worst_f[j]
                if isinstance(b_val, (tuple, list)):
                    b_str = f"({b_val[0]:.3f}, {b_val[1]:.3f}, {b_val[2]:.3f})"
                    w_str = f"({w_val[0]:.3f}, {w_val[1]:.3f}, {w_val[2]:.3f})"
                else:
                    b_str = f"{float(b_val):.4f}"
                    w_str = f"{float(w_val):.4f}"

                bounds_rows.append({
                    "Criterion": c_name,
                    "Type": "Benefit" if vk_res.is_benefit[j] else "Cost",
                    "Weight": f"{vk_res.weights[j]:.4f}",
                    "Best f*": b_str,
                    "Worst f-": w_str,
                })
            st.dataframe(pd.DataFrame(bounds_rows), use_container_width=True, hide_index=True)

        with v_tab2:
            st.markdown(r"**Weighted Regret Matrix $[w_j \cdot d_{ij}]$**")
            st.caption("Each cell represents the alternative's weighted distance from the ideal value on that criterion.")
            st.dataframe(vk_res.get_weighted_regret_df(), use_container_width=True)

        with v_tab3:
            st.markdown("**Alternative Utility ($S_i$), Regret ($R_i$), and VIKOR Index ($Q_i$)**")
            st.dataframe(vk_res.get_summary_df(), use_container_width=True, hide_index=True)
