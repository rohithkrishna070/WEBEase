"""Sensitivity Analysis and Decision Robustness Engine for SmartWeb Access.

Analyzes how variations in criterion weights impact Fuzzy TOPSIS and VIKOR
prioritization outcomes, measuring rank stability, scenario frequencies, and
rank inversion risks across different organizational priority profiles.

References
----------
1. Triantaphyllou, E., & Sánchez, A. (1997). A sensitivity analysis approach for
   some deterministic multi-criteria decision-making methods. Decision Sciences,
   28(1), 151-194.
2. Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods:
   A comparative analysis of VIKOR and TOPSIS. European Journal of Operational
   Research, 156(2), 445-455.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from core.models import CriterionWeights
from core.mcdm_bridge import run_prioritization, CRITERIA_NAMES


PRESET_WEIGHT_PROFILES: Dict[str, CriterionWeights] = {
    "Balanced Focus (Default)": CriterionWeights(
        c1_severity=0.25,
        c2_task_impact=0.30,
        c3_affected_pages=0.10,
        c4_user_impact=0.15,
        c5_repair_effort=0.10,
        c6_frequency=0.10,
    ),
    "Essential-Task Critical": CriterionWeights(
        c1_severity=0.15,
        c2_task_impact=0.50,
        c3_affected_pages=0.05,
        c4_user_impact=0.15,
        c5_repair_effort=0.05,
        c6_frequency=0.10,
    ),
    "Quick Wins (Low Effort First)": CriterionWeights(
        c1_severity=0.15,
        c2_task_impact=0.20,
        c3_affected_pages=0.10,
        c4_user_impact=0.15,
        c5_repair_effort=0.30,
        c6_frequency=0.10,
    ),
    "WCAG Barrier Severity First": CriterionWeights(
        c1_severity=0.45,
        c2_task_impact=0.15,
        c3_affected_pages=0.10,
        c4_user_impact=0.15,
        c5_repair_effort=0.05,
        c6_frequency=0.10,
    ),
    "Widespread Occurrence First": CriterionWeights(
        c1_severity=0.15,
        c2_task_impact=0.15,
        c3_affected_pages=0.35,
        c4_user_impact=0.15,
        c5_repair_effort=0.05,
        c6_frequency=0.15,
    ),
}


def redistribute_weights_for_sweep(
    base_weights: Sequence[float],
    target_idx: int,
    target_value: float,
) -> List[float]:
    """Redistribute the remaining weight budget (1.0 - target_value) proportionally.

    Parameters
    ----------
    base_weights : Sequence[float]
        Normalized base weights of length n.
    target_idx : int
        Index of the criterion being swept.
    target_value : float
        Target weight to assign to target_idx (must be in [0.0, 1.0]).

    Returns
    -------
    List[float]
        New normalized weights summing strictly to 1.0.
    """
    n = len(base_weights)
    target_clamped = max(0.0, min(1.0, float(target_value)))
    remaining_budget = max(0.0, 1.0 - target_clamped)

    other_indices = [k for k in range(n) if k != target_idx]
    other_sum = sum(base_weights[k] for k in other_indices)

    new_weights = [0.0] * n
    new_weights[target_idx] = target_clamped

    if other_sum > 1e-12:
        scale = remaining_budget / other_sum
        for k in other_indices:
            new_weights[k] = base_weights[k] * scale
    else:
        # If other base weights were all zero, split equally
        split = remaining_budget / len(other_indices) if other_indices else 0.0
        for k in other_indices:
            new_weights[k] = split

    # Ensure precision cleanup
    tot = sum(new_weights)
    if tot > 0:
        new_weights = [w / tot for w in new_weights]
    return new_weights


def run_weight_sweep(
    df: pd.DataFrame,
    target_criterion_idx: int = 1,
    weight_steps: int = 10,
    min_w: float = 0.05,
    max_w: float = 0.60,
    base_weights: CriterionWeights | None = None,
    vikor_v: float = 0.5,
    vikor_formulation: str = "fuzzy",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sweep a single criterion's weight across a range and observe rank changes.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - topsis_ranks_history: DataFrame with issues as index and TOPSIS ranks across steps
        - vikor_ranks_history: DataFrame with issues as index and VIKOR ranks across steps
        - correlation_history: DataFrame tracking correlation metrics across steps
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_obj = (base_weights or CriterionWeights()).normalize()
    base_list = base_obj.as_list()
    sweep_values = np.linspace(min_w, max_w, max(2, int(weight_steps)))

    # Baseline run
    base_res, _ = run_prioritization(
        df, base_obj, vikor_v=vikor_v, vikor_formulation=vikor_formulation
    )
    base_topsis_rank = dict(zip(base_res["issue_id"], base_res["topsis_rank"]))
    base_vikor_rank = dict(zip(base_res["issue_id"], base_res["vikor_rank"]))

    records_topsis: Dict[str, List[int]] = {str(row["issue_id"]): [] for _, row in df.iterrows()}
    records_vikor: Dict[str, List[int]] = {str(row["issue_id"]): [] for _, row in df.iterrows()}
    corr_records = []

    for target_w in sweep_values:
        new_w = redistribute_weights_for_sweep(base_list, target_criterion_idx, float(target_w))
        w_obj = CriterionWeights(*new_w)

        step_res, _ = run_prioritization(
            df, w_obj, vikor_v=vikor_v, vikor_formulation=vikor_formulation
        )

        step_topsis_map = dict(zip(step_res["issue_id"], step_res["topsis_rank"]))
        step_vikor_map = dict(zip(step_res["issue_id"], step_res["vikor_rank"]))

        for issue_id in records_topsis:
            records_topsis[issue_id].append(step_topsis_map.get(issue_id, 0))
            records_vikor[issue_id].append(step_vikor_map.get(issue_id, 0))

        # Correlation between current step and baseline
        step_order_t = [step_topsis_map[iid] for iid in base_res["issue_id"]]
        base_order_t = [base_topsis_rank[iid] for iid in base_res["issue_id"]]
        rho_t, _ = spearmanr(step_order_t, base_order_t)

        step_order_v = [step_vikor_map[iid] for iid in base_res["issue_id"]]
        base_order_v = [base_vikor_rank[iid] for iid in base_res["issue_id"]]
        rho_v, _ = spearmanr(step_order_v, base_order_v)

        # Cross-method correlation at this step
        step_t_ranks = [step_topsis_map[iid] for iid in step_res["issue_id"]]
        step_v_ranks = [step_vikor_map[iid] for iid in step_res["issue_id"]]
        rho_cross, _ = spearmanr(step_t_ranks, step_v_ranks)

        corr_records.append({
            "target_weight": round(float(target_w), 3),
            "topsis_baseline_spearman": round(float(rho_t) if not np.isnan(rho_t) else 1.0, 4),
            "vikor_baseline_spearman": round(float(rho_v) if not np.isnan(rho_v) else 1.0, 4),
            "cross_method_spearman": round(float(rho_cross) if not np.isnan(rho_cross) else 1.0, 4),
        })

    cols = [f"w_{round(val, 2)}" for val in sweep_values]
    topsis_ranks_df = pd.DataFrame.from_dict(records_topsis, orient="index", columns=cols)
    vikor_ranks_df = pd.DataFrame.from_dict(records_vikor, orient="index", columns=cols)
    corr_df = pd.DataFrame(corr_records)

    return topsis_ranks_df, vikor_ranks_df, corr_df


def calculate_scenario_stability(
    df: pd.DataFrame,
    scenarios: Optional[Dict[str, CriterionWeights]] = None,
    vikor_v: float = 0.5,
    vikor_formulation: str = "fuzzy",
) -> pd.DataFrame:
    """Calculate ranking stability and appearance frequencies in Top-3 and Top-5.

    IMPORTANT:
    Scenario frequencies are experimental sensitivity measures reflecting decision
    stability across varying priority weight assumptions, NOT empirical probabilities
    of an issue causing real-world harm.

    Parameters
    ----------
    df : pd.DataFrame
        Accessibility issues DataFrame.
    scenarios : Optional[Dict[str, CriterionWeights]]
        Dictionary of named weight profiles. Defaults to PRESET_WEIGHT_PROFILES.

    Returns
    -------
    pd.DataFrame
        DataFrame with issue details, Top-3 count/percentage, Top-5 count/percentage,
        average rank, and stability classification.
    """
    if df.empty:
        return pd.DataFrame()

    profiles = scenarios or PRESET_WEIGHT_PROFILES
    num_scenarios = len(profiles)

    # Track ranks across profiles for each issue
    topsis_rank_records: Dict[str, List[int]] = {str(row["issue_id"]): [] for _, row in df.iterrows()}
    vikor_rank_records: Dict[str, List[int]] = {str(row["issue_id"]): [] for _, row in df.iterrows()}
    consensus_rank_records: Dict[str, List[int]] = {str(row["issue_id"]): [] for _, row in df.iterrows()}

    for profile_name, weights in profiles.items():
        res, _ = run_prioritization(
            df, weights, vikor_v=vikor_v, vikor_formulation=vikor_formulation
        )
        t_map = dict(zip(res["issue_id"], res["topsis_rank"]))
        v_map = dict(zip(res["issue_id"], res["vikor_rank"]))
        c_map = dict(zip(res["issue_id"], res["consensus_rank"]))

        for iid in topsis_rank_records:
            topsis_rank_records[iid].append(t_map.get(iid, 999))
            vikor_rank_records[iid].append(v_map.get(iid, 999))
            consensus_rank_records[iid].append(c_map.get(iid, 999))

    rows = []
    meta_df = df.drop_duplicates(subset=["issue_id"]).set_index("issue_id")

    for iid in topsis_rank_records:
        c_ranks = consensus_rank_records[iid]
        t_ranks = topsis_rank_records[iid]
        v_ranks = vikor_rank_records[iid]

        top3_count = sum(1 for r in c_ranks if r <= 3)
        top5_count = sum(1 for r in c_ranks if r <= 5)
        top3_pct = (top3_count / num_scenarios) * 100.0
        top5_pct = (top5_count / num_scenarios) * 100.0

        avg_rank = float(np.mean(c_ranks))
        std_rank = float(np.std(c_ranks))

        # Stability classification
        if top5_count >= (num_scenarios * 0.8):
            stability_label = "Robust High Priority"
        elif top5_count >= (num_scenarios * 0.4):
            stability_label = "Scenario-Dependent Priority"
        elif avg_rank <= (len(df) * 0.5):
            stability_label = "Moderate General Priority"
        else:
            stability_label = "Low Sensitivity Priority"

        issue_row = meta_df.loc[iid] if iid in meta_df.index else {}
        rows.append({
            "issue_id": iid,
            "website": issue_row.get("website", ""),
            "page_name": issue_row.get("page_name", ""),
            "issue_type": issue_row.get("issue_type", ""),
            "severity": issue_row.get("severity", ""),
            "essential_task_impact": issue_row.get("essential_task_impact", ""),
            "repair_effort": issue_row.get("repair_effort", ""),
            "top3_count": top3_count,
            "top3_frequency_pct": round(top3_pct, 1),
            "top5_count": top5_count,
            "top5_frequency_pct": round(top5_pct, 1),
            "avg_consensus_rank": round(avg_rank, 2),
            "rank_std_dev": round(std_rank, 2),
            "stability_class": stability_label,
            "scenarios_evaluated": num_scenarios,
        })

    stability_df = pd.DataFrame(rows)
    # Sort primarily by Top-5 frequency descending, then average rank ascending
    stability_df.sort_values(
        by=["top5_count", "top3_count", "avg_consensus_rank"],
        ascending=[False, False, True],
        inplace=True,
    )
    stability_df.reset_index(drop=True, inplace=True)
    return stability_df


def compare_profiles(
    df: pd.DataFrame,
    scenarios: Optional[Dict[str, CriterionWeights]] = None,
    vikor_v: float = 0.5,
    vikor_formulation: str = "fuzzy",
) -> pd.DataFrame:
    """Compare top-ranked items and correlation statistics across predefined criteria profiles."""
    if df.empty:
        return pd.DataFrame()

    profiles = scenarios or PRESET_WEIGHT_PROFILES
    comparison_data = []

    for profile_name, weights in profiles.items():
        res, meta = run_prioritization(
            df, weights, vikor_v=vikor_v, vikor_formulation=vikor_formulation
        )
        top1 = res.iloc[0]
        top3_ids = ", ".join(res.iloc[:3]["issue_id"].tolist())

        val = meta.get("vikor_validation", {})
        comp_status = "Unique Stable" if val.get("compromise_stable") else f"Cluster ({len(val.get('compromise_set_indices', []))})"

        comparison_data.append({
            "Profile": profile_name,
            "Top Priority Repair": f"{top1['issue_id']} ({top1['issue_type']})",
            "Top 3 Issues": top3_ids,
            "Spearman Rho (T vs V)": round(meta.get("spearman_rho", 1.0), 4),
            "Kendall Tau (T vs V)": round(meta.get("kendall_tau", 1.0), 4),
            "VIKOR Compromise Status": comp_status,
        })

    return pd.DataFrame(comparison_data)
