"""MCDM Orchestration Bridge for SmartWeb Access.

Maps qualitative and quantitative accessibility audit fields to Fuzzy Numbers
and Crisp decision matrices, coordinates execution of Fuzzy TOPSIS and VIKOR,
and computes comparative rank correlation statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from core.fuzzy_topsis import fuzzy_topsis, TFN
from core.vikor import vikor
from core.models import (
    CriterionWeights,
    LINGUISTIC_BENEFIT_SCALE,
    CRISP_DEF_SCALE,
)

# Criteria direction: C1, C2, C3, C4, C5, C6
# C5 (Repair Effort) is Cost (False), others are Benefit (True)
CRITERIA_BENEFIT_FLAGS: List[bool] = [True, True, True, True, False, True]
CRITERIA_NAMES: List[str] = [
    "C1: Barrier Severity",
    "C2: Essential-Task Impact",
    "C3: Affected Pages",
    "C4: User Impact",
    "C5: Repair Effort",
    "C6: Issue Frequency",
]


def _map_to_tfn(val: str, default: TFN = (0.3, 0.5, 0.7)) -> TFN:
    """Safely map a linguistic term to a Triangular Fuzzy Number."""
    return LINGUISTIC_BENEFIT_SCALE.get(str(val).strip(), default)


def _map_to_crisp(val: str, default: float = 3.0) -> float:
    """Safely map a linguistic term to a crisp defuzzified 1-5 scale."""
    return CRISP_DEF_SCALE.get(str(val).strip(), default)


def _frequency_to_tfn(freq_str: str) -> TFN:
    """Map frequency terms (Systematic, Frequent, Occasional, Rare) to TFN."""
    freq_map: Dict[str, TFN] = {
        "Rare": (0.0, 0.1, 0.3),
        "Occasional": (0.1, 0.3, 0.5),
        "Frequent": (0.5, 0.7, 0.9),
        "Systematic": (0.7, 0.9, 1.0),
        # Also support standard 5-point scale terms
        "Very Low": (0.0, 0.1, 0.3),
        "Low": (0.1, 0.3, 0.5),
        "Medium": (0.3, 0.5, 0.7),
        "High": (0.5, 0.7, 0.9),
        "Very High": (0.7, 0.9, 1.0),
    }
    return freq_map.get(str(freq_str).strip(), (0.3, 0.5, 0.7))


def _frequency_to_crisp(freq_str: str) -> float:
    """Map frequency terms to crisp 1-5 value."""
    freq_map: Dict[str, float] = {
        "Rare": 1.0,
        "Occasional": 2.5,
        "Frequent": 4.0,
        "Systematic": 5.0,
        "Very Low": 1.0,
        "Low": 2.0,
        "Medium": 3.0,
        "High": 4.0,
        "Very High": 5.0,
    }
    return freq_map.get(str(freq_str).strip(), 3.0)


def build_decision_matrices(
    df: pd.DataFrame,
) -> Tuple[List[List[TFN]], np.ndarray]:
    """Construct Fuzzy TOPSIS (TFN) and VIKOR (crisp) decision matrices from issues DataFrame.

    Returns
    -------
    Tuple[List[List[TFN]], np.ndarray]
        - fuzzy_matrix: m x 6 list of TFNs
        - crisp_matrix: m x 6 numpy array
    """
    m = len(df)
    fuzzy_matrix: List[List[TFN]] = []
    crisp_matrix = np.zeros((m, 6), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        # C1: Barrier Severity
        c1_tfn = _map_to_tfn(row["severity"])
        c1_crisp = _map_to_crisp(row["severity"])

        # C2: Essential-Task Impact
        c2_tfn = _map_to_tfn(row["essential_task_impact"])
        c2_crisp = _map_to_crisp(row["essential_task_impact"])

        # C3: Number of Affected Pages (numerical)
        pages_cnt = float(row.get("affected_pages", 1))
        # Formulate a narrow TFN around the known count
        c3_tfn = (pages_cnt * 0.9, pages_cnt, pages_cnt * 1.1)
        c3_crisp = pages_cnt

        # C4: User Impact
        c4_tfn = _map_to_tfn(row["user_impact"])
        c4_crisp = _map_to_crisp(row["user_impact"])

        # C5: Repair Effort (Cost criterion)
        c5_tfn = _map_to_tfn(row["repair_effort"])
        c5_crisp = _map_to_crisp(row["repair_effort"])

        # C6: Issue Frequency
        c6_tfn = _frequency_to_tfn(row["frequency"])
        c6_crisp = _frequency_to_crisp(row["frequency"])

        fuzzy_matrix.append([c1_tfn, c2_tfn, c3_tfn, c4_tfn, c5_tfn, c6_tfn])
        crisp_matrix[i, :] = [c1_crisp, c2_crisp, c3_crisp, c4_crisp, c5_crisp, c6_crisp]

    return fuzzy_matrix, crisp_matrix


def run_prioritization(
    df: pd.DataFrame,
    weights: Optional[CriterionWeights] = None,
    vikor_v: float = 0.5,
    vikor_formulation: str = "fuzzy",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete prioritization pipeline running both Fuzzy TOPSIS and VIKOR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing accessibility issues.
    weights : Optional[CriterionWeights]
        Custom weights; defaults to balanced preset if None.
    vikor_v : float, default=0.5
        VIKOR consensus parameter (strategy of maximum group utility).
    vikor_formulation : str, default='fuzzy'
        VIKOR formulation: 'fuzzy' (Opricovic 2011 on TFNs) or
        'classical' (Opricovic 2004 with Centroid defuzzification).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - results_df: DataFrame with original issue data augmented with TOPSIS/VIKOR scores & ranks.
        - metadata: Dictionary containing correlation metrics, parameters, and VIKOR validation.
    """
    if df.empty:
        return pd.DataFrame(), {}

    w_obj = (weights or CriterionWeights()).normalize()
    w_list = w_obj.as_list()

    fuzzy_mat, crisp_mat = build_decision_matrices(df)

    # 1. Fuzzy TOPSIS Execution
    alt_ids = df["issue_id"].astype(str).tolist() if "issue_id" in df else [f"A{i+1}" for i in range(len(df))]
    topsis_res = fuzzy_topsis(
        fuzzy_mat,
        w_list,
        CRITERIA_BENEFIT_FLAGS,
        criteria_names=CRITERIA_NAMES,
        alternative_names=alt_ids,
    )
    topsis_cc = topsis_res.closeness_coefficients
    topsis_ranks = topsis_res.rankings
    d_plus = topsis_res.d_plus
    d_minus = topsis_res.d_minus

    # 2. VIKOR Execution (Fuzzy or Classical with Defuzzification)
    vikor_mat = fuzzy_mat if vikor_formulation == "fuzzy" else crisp_mat
    vikor_res = vikor(
        vikor_mat,
        w_list,
        CRITERIA_BENEFIT_FLAGS,
        v=vikor_v,
        formulation=vikor_formulation,
        criteria_names=CRITERIA_NAMES,
        alternative_names=alt_ids,
    )
    vikor_q = vikor_res.q_values
    vikor_ranks = vikor_res.rankings
    vikor_s = vikor_res.s_values
    vikor_r = vikor_res.r_values
    vikor_val = vikor_res.validation

    # 3. Assemble results DataFrame
    res = df.copy()
    res["topsis_score"] = np.round(topsis_cc, 4)
    res["topsis_rank"] = topsis_ranks
    res["vikor_s"] = np.round(vikor_s, 4)
    res["vikor_r"] = np.round(vikor_r, 4)
    res["vikor_q"] = np.round(vikor_q, 4)
    res["vikor_rank"] = vikor_ranks
    res["rank_diff"] = res["topsis_rank"] - res["vikor_rank"]
    res["absolute_rank_diff"] = np.abs(res["rank_diff"])

    # Combined composite consensus rank (average rank of both methods)
    res["consensus_score"] = np.round((res["topsis_score"] + (1.0 - res["vikor_q"])) / 2.0, 4)
    consensus_order = np.argsort(-res["consensus_score"])
    res["consensus_rank"] = np.empty(len(res), dtype=int)
    res.loc[res.index[consensus_order], "consensus_rank"] = np.arange(1, len(res) + 1)

    # Sort results primarily by TOPSIS rank or Consensus rank
    res.sort_values(by="consensus_rank", ascending=True, inplace=True)
    res.reset_index(drop=True, inplace=True)

    # 4. Statistical Rank Correlation
    if len(res) >= 2:
        rho, rho_pvalue = spearmanr(res["topsis_rank"], res["vikor_rank"])
        tau, tau_pvalue = kendalltau(res["topsis_rank"], res["vikor_rank"])
    else:
        rho, rho_pvalue, tau, tau_pvalue = 1.0, 0.0, 1.0, 0.0

    metadata = {
        "spearman_rho": float(rho) if not np.isnan(rho) else 1.0,
        "spearman_pvalue": float(rho_pvalue) if not np.isnan(rho_pvalue) else 0.0,
        "kendall_tau": float(tau) if not np.isnan(tau) else 1.0,
        "kendall_pvalue": float(tau_pvalue) if not np.isnan(tau_pvalue) else 0.0,
        "weights": w_obj.as_list(),
        "vikor_v": vikor_v,
        "vikor_formulation": vikor_formulation,
        "vikor_validation": vikor_val,
        "vikor_result": vikor_res,
        "vikor_details": vikor_res.to_dict(),
        "total_evaluated": len(res),
        "fuzzy_topsis_result": topsis_res,
        "fuzzy_topsis_details": topsis_res.to_dict(),
    }

    return res, metadata
