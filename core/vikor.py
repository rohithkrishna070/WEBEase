"""
================================================================================
MODULE: core.vikor
================================================================================
1. WHAT THIS FILE DOES:
   Implements the VIKOR (ViseKriterijumska Optimizacija I Kompromisno Resenje)
   multi-criteria compromise ranking engine in two rigorous formulations:
   - Classical VIKOR (Opricovic & Tzeng, 2004) with explicit centroid defuzzification.
   - Fuzzy VIKOR (Opricovic, 2011) directly computing on Triangular Fuzzy Numbers (TFNs).

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: In accessibility remediation, criteria inherently conflict:
     maximum barrier severity and essential-task impact often demand high engineering effort.
     While TOPSIS selects solutions closest to an ideal point (which might have unacceptable
     worst-case individual scores), VIKOR focuses on finding a *compromise solution* that
     balances maximum "group utility" (majority benefit) with minimum "individual regret"
     (preventing unacceptable extreme flaws).
   - Technical & Architectural Rationale: Provides a pure mathematical implementation
     with comprehensive verification of VIKOR's formal compromise conditions:
     Condition 1 (Acceptable Advantage: Q(A2) - Q(A1) >= DQ) and
     Condition 2 (Acceptable Stability in Decision Making: A1 best in S and/or R).

3. HOW IT WORKS CONCEPTUALLY:
   - Determines the best f_j* and worst f_j^- values for each criterion j.
   - Computes:
     * S_i (Group Utility / Manhattan distance): sum_j [w_j * (f_j* - f_ij) / (f_j* - f_j^-)]
     * R_i (Individual Regret / Chebyshev distance): max_j [w_j * (f_j* - f_ij) / (f_j* - f_j^-)]
     * Q_i (Compromise Index): v * (S_i - S*) / (S^- - S*) + (1 - v) * (R_i - R*) / (R^- - R*)
       where v is the weight of strategy of maximum group utility (typically v=0.5).
   - Ranks alternatives by ascending Q_i (lower Q = higher repair priority).
   - Validates whether top candidate(s) meet Acceptable Advantage and Stability.

4. HOW IT WORKS TECHNICALLY:
   - Implements both `vikor_crisp()` and `fuzzy_vikor()` algorithms.
   - Structured results returned in typed `VIKORResult` dataclass.
   - Computes DQ threshold = 1 / (J - 1) for alternative set size J.
   - Handles boundary edge cases (all identical scores, zero span, single alternative).

References:
- Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis.
- Opricovic, S. (2011). Fuzzy VIKOR with an application to water resources planning.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from core.fuzzy_topsis import TFN, tfn_defuzzify_centroid, tfn_defuzzify_gmir, tfn_distance, validate_tfn


# ==============================================================================
# VIKOR Result Data Structure
# ==============================================================================

@dataclass
class VikorResult:
    """Encapsulates complete VIKOR evaluation output including compromise validation.

    Provides transparent auditability for MCDM research, classroom verification,
    and administrative reporting.
    """

    q_values: np.ndarray
    rankings: np.ndarray
    s_values: np.ndarray
    r_values: np.ndarray
    validation: Dict[str, Any]
    best_f: List[Any]
    worst_f: List[Any]
    normalized_distances: np.ndarray
    weighted_distances: np.ndarray
    weights: List[float]
    is_benefit: List[bool]
    v: float = 0.5
    formulation: str = "classical"
    defuzzification_applied: bool = False
    criteria_names: Optional[List[str]] = field(default=None)
    alternative_names: Optional[List[str]] = field(default=None)

    def __iter__(self) -> Iterator[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Enable backwards-compatible 5-tuple unpacking:
        q, ranks, s, r, val = vikor(...)
        """
        yield self.q_values
        yield self.rankings
        yield self.s_values
        yield self.r_values
        yield self.validation

    def __getitem__(self, index: int) -> Any:
        return [self.q_values, self.rankings, self.s_values, self.r_values, self.validation][index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "q_values": self.q_values.tolist(),
            "rankings": self.rankings.tolist(),
            "s_values": self.s_values.tolist(),
            "r_values": self.r_values.tolist(),
            "validation": self.validation,
            "weights": self.weights,
            "is_benefit": self.is_benefit,
            "v": self.v,
            "formulation": self.formulation,
            "defuzzification_applied": self.defuzzification_applied,
        }

    def get_summary_df(self) -> pd.DataFrame:
        """Return formatted summary DataFrame of VIKOR evaluation."""
        alts = self.alternative_names or [f"A{i+1}" for i in range(len(self.q_values))]
        df = pd.DataFrame({
            "Alternative": alts,
            "Group Utility (S)": np.round(self.s_values, 4),
            "Individual Regret (R)": np.round(self.r_values, 4),
            "VIKOR Index (Q)": np.round(self.q_values, 4),
            "VIKOR Rank": self.rankings,
        })
        return df.sort_values(by="VIKOR Rank", ascending=True).reset_index(drop=True)

    def get_weighted_regret_df(self) -> pd.DataFrame:
        """Return alternative-by-criterion weighted regret distance matrix."""
        cols = self.criteria_names or [f"C{j+1}" for j in range(len(self.is_benefit))]
        alts = self.alternative_names or [f"A{i+1}" for i in range(len(self.q_values))]
        return pd.DataFrame(np.round(self.weighted_distances, 4), index=alts, columns=cols)


# ==============================================================================
# Compromise Solution Validation (Opricovic & Tzeng, 2004)
# ==============================================================================

def evaluate_compromise_conditions(
    q: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    alternative_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate VIKOR compromise solution acceptance conditions C1 and C2.

    Condition 1: Acceptable Advantage
    ---------------------------------
        Q(A^(2)) - Q(A^(1)) >= DQ
        where DQ = 1 / (m - 1) for m > 1.
        (A^(1) is first-ranked by Q, A^(2) is second-ranked by Q).

    Condition 2: Acceptable Stability in Decision Making
    ----------------------------------------------------
        Alternative A^(1) must also be best-ranked by S (minimum S) and/or R (minimum R).

    Compromise Solution Determination
    ---------------------------------
    - If both C1 and C2 are satisfied: A^(1) is the unique compromise solution.
    - If C1 is violated: A set of compromise solutions {A^(1), A^(2), ..., A^(k)}
      is proposed where A^(k) is determined by Q(A^(k)) - Q(A^(1)) < DQ for maximum k.
    - If C2 is violated: The compromise set {A^(1), A^(2)} is proposed.
    """
    m = len(q)
    if m < 2:
        names = list(alternative_names) if alternative_names else (["A1"] if m == 1 else [])
        return {
            "dq": 1.0,
            "advantage": 0.0,
            "c1_acceptable_advantage": True,
            "c2_acceptable_stability": True,
            "compromise_stable": True,
            "compromise_set_indices": [0] if m == 1 else [],
            "compromise_set_names": names,
            "explanation": "Single alternative evaluated; trivially optimal.",
        }

    q_order = np.argsort(q)
    best_idx = int(q_order[0])
    second_best_idx = int(q_order[1])

    dq = 1.0 / (m - 1)
    adv = float(q[second_best_idx] - q[best_idx])
    c1_satisfied = bool(adv >= (dq - 1e-9))

    s_best_idx = int(np.argmin(s))
    r_best_idx = int(np.argmin(r))
    c2_satisfied = bool(best_idx == s_best_idx or best_idx == r_best_idx)

    # Determine compromise solution set
    compromise_indices: List[int] = [best_idx]
    if c1_satisfied and c2_satisfied:
        explanation = (
            f"Unique compromise solution: Alternative {best_idx + 1} achieves acceptable advantage "
            f"(Adv={adv:.4f} >= DQ={dq:.4f}) and decision stability."
        )
    elif not c1_satisfied and c2_satisfied:
        # Include all alternatives within DQ of the best
        for idx in q_order[1:]:
            if float(q[idx] - q[best_idx]) < (dq + 1e-9):
                compromise_indices.append(int(idx))
            else:
                break
        explanation = (
            f"Compromise set under Acceptable Advantage failure (Adv={adv:.4f} < DQ={dq:.4f}): "
            f"Alternatives {', '.join(str(i+1) for i in compromise_indices)} form the close compromise cluster."
        )
    elif c1_satisfied and not c2_satisfied:
        compromise_indices = [best_idx, second_best_idx]
        explanation = (
            f"Compromise set under Stability failure: Alternative {best_idx + 1} has sufficient advantage, "
            f"but Alternative {s_best_idx + 1} (or {r_best_idx + 1}) achieved better utility/regret."
        )
    else:
        # Both failed
        for idx in q_order[1:]:
            if float(q[idx] - q[best_idx]) < (dq + 1e-9):
                compromise_indices.append(int(idx))
            else:
                break
        if second_best_idx not in compromise_indices:
            compromise_indices.append(second_best_idx)
        explanation = (
            f"Compromise set under dual failure: Neither advantage nor stability was satisfied. "
            f"A compromise cluster of {len(compromise_indices)} alternatives is proposed."
        )

    alt_labels = list(alternative_names) if alternative_names else [f"A{i+1}" for i in range(m)]
    compromise_names = [alt_labels[idx] for idx in compromise_indices if idx < len(alt_labels)]

    return {
        "dq": float(dq),
        "advantage": float(adv),
        "c1_acceptable_advantage": c1_satisfied,
        "c2_acceptable_stability": c2_satisfied,
        "compromise_stable": bool(c1_satisfied and c2_satisfied),
        "compromise_set_indices": compromise_indices,
        "compromise_set_names": compromise_names,
        "explanation": explanation,
        "best_by_s_index": s_best_idx,
        "best_by_r_index": r_best_idx,
    }


# ==============================================================================
# Input Validation & Weight Normalization
# ==============================================================================

def validate_vikor_inputs(
    weights: Sequence[Union[float, TFN]],
    is_benefit: Sequence[bool],
    num_criteria: int,
    v: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Validate weights, criterion directions, and compromise parameter v."""
    if len(weights) != num_criteria or len(is_benefit) != num_criteria:
        raise ValueError(
            f"Dimension mismatch: expected {num_criteria} criteria, but received "
            f"{len(weights)} weights and {len(is_benefit)} criterion direction flags."
        )

    if not (0.0 <= v <= 1.0):
        raise ValueError(f"Compromise parameter v must be in [0.0, 1.0], got {v}.")

    crisp_weights: List[float] = []
    for idx, w in enumerate(weights):
        if isinstance(w, (tuple, list)) and len(w) == 3:
            crisp_weights.append(tfn_defuzzify_centroid(validate_tfn(w)))
        else:
            w_val = float(w)
            if math.isnan(w_val) or math.isinf(w_val):
                raise ValueError(f"Weight at index {idx} must be a finite number, got {w_val}.")
            if w_val < -1e-12:
                raise ValueError(f"Criterion weights must be non-negative: weight[{idx}] = {w_val} < 0.")
            crisp_weights.append(max(0.0, w_val))

    total = sum(crisp_weights)
    if total <= 1e-12:
        raise ValueError("Total criterion weight must be strictly positive (sum of weights > 0).")

    norm_w = np.array([w / total for w in crisp_weights], dtype=float)
    return norm_w, float(v)


# ==============================================================================
# Classical VIKOR Engine (Opricovic & Tzeng, 2004)
# ==============================================================================

def classical_vikor(
    matrix: np.ndarray | Sequence[Sequence[float]],
    weights: Sequence[float],
    is_benefit: Sequence[bool],
    v: float = 0.5,
    criteria_names: Optional[Sequence[str]] = None,
    alternative_names: Optional[Sequence[str]] = None,
) -> VikorResult:
    """Execute Classical VIKOR algorithm on a crisp decision matrix.

    Formulation:
    ------------
    1. Determine best f_j* and worst f_j-:
       For benefit: f_j* = max_i(x_ij), f_j- = min_i(x_ij)
       For cost:    f_j* = min_i(x_ij), f_j- = max_i(x_ij)

    2. Compute Group Utility S_i and Individual Regret R_i:
       d_ij = |f_j* - x_ij| / |f_j* - f_j-|
       S_i = sum_j (w_j * d_ij)
       R_i = max_j (w_j * d_ij)

    3. Compute VIKOR Index Q_i:
       Q_i = v * (S_i - S*) / (S- - S*) + (1 - v) * (R_i - R*) / (R- - R*)
       where S* = min(S), S- = max(S), R* = min(R), R- = max(R).

    4. Rank by Q_i ascending (lowest Q is Rank 1).
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Decision matrix must be 2-dimensional (m x n), got shape {X.shape}.")

    m, n = X.shape
    if m == 0:
        empty = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return VikorResult(
            q_values=empty,
            rankings=empty_int,
            s_values=empty,
            r_values=empty,
            validation={},
            best_f=[],
            worst_f=[],
            normalized_distances=np.empty((0, n)),
            weighted_distances=np.empty((0, n)),
            weights=[],
            is_benefit=list(is_benefit),
            v=v,
            formulation="classical",
            defuzzification_applied=False,
        )

    norm_w, v_val = validate_vikor_inputs(weights, is_benefit, n, v)

    # 1. Best f_star and Worst f_minus values
    f_star = np.zeros(n, dtype=float)
    f_minus = np.zeros(n, dtype=float)

    for j in range(n):
        col = X[:, j]
        if is_benefit[j]:
            f_star[j] = float(np.max(col))
            f_minus[j] = float(np.min(col))
        else:
            f_star[j] = float(np.min(col))
            f_minus[j] = float(np.max(col))

    # 2. Normalized Regret Distances & Weighted Regret Matrix
    norm_dist = np.zeros((m, n), dtype=float)
    weighted_dist = np.zeros((m, n), dtype=float)
    S = np.zeros(m, dtype=float)
    R = np.zeros(m, dtype=float)

    for i in range(m):
        for j in range(n):
            denom = abs(f_star[j] - f_minus[j])
            if denom > 1e-12:
                d_ij = abs(f_star[j] - X[i, j]) / denom
            else:
                # All alternatives have identical rating for criterion j
                d_ij = 0.0
            norm_dist[i, j] = d_ij
            weighted_dist[i, j] = norm_w[j] * d_ij

        S[i] = float(np.sum(weighted_dist[i, :]))
        R[i] = float(np.max(weighted_dist[i, :]))

    # 3. Compute VIKOR Index Q_i
    s_star = float(np.min(S))
    s_minus = float(np.max(S))
    r_star = float(np.min(R))
    r_minus = float(np.max(R))

    s_denom = s_minus - s_star
    r_denom = r_minus - r_star

    Q = np.zeros(m, dtype=float)
    for i in range(m):
        term_s = (S[i] - s_star) / s_denom if s_denom > 1e-12 else 0.0
        term_r = (R[i] - r_star) / r_denom if r_denom > 1e-12 else 0.0
        Q[i] = v_val * term_s + (1.0 - v_val) * term_r

    # 4. Rank by Q ascending (lowest Q is Rank 1). Deterministic tie-breaking.
    sort_keys = [(Q[i], S[i], R[i], i) for i in range(m)]
    order = [idx for _, _, _, idx in sorted(sort_keys)]

    rankings = np.empty(m, dtype=int)
    for rank_pos, alt_idx in enumerate(order, start=1):
        rankings[alt_idx] = rank_pos

    # 5. Compromise Acceptance Conditions (C1 & C2)
    c_names = list(criteria_names) if criteria_names else None
    a_names = list(alternative_names) if alternative_names else None
    validation = evaluate_compromise_conditions(Q, S, R, alternative_names=a_names)

    return VikorResult(
        q_values=Q,
        rankings=rankings,
        s_values=S,
        r_values=R,
        validation=validation,
        best_f=f_star.tolist(),
        worst_f=f_minus.tolist(),
        normalized_distances=norm_dist,
        weighted_distances=weighted_dist,
        weights=norm_w.tolist(),
        is_benefit=list(is_benefit),
        v=v_val,
        formulation="classical",
        defuzzification_applied=False,
        criteria_names=c_names,
        alternative_names=a_names,
    )


# ==============================================================================
# Fuzzy VIKOR Engine (Opricovic, 2011)
# ==============================================================================

def fuzzy_vikor(
    matrix: Sequence[Sequence[Union[TFN, Sequence[float]]]],
    weights: Sequence[Union[float, TFN]],
    is_benefit: Sequence[bool],
    v: float = 0.5,
    criteria_names: Optional[Sequence[str]] = None,
    alternative_names: Optional[Sequence[str]] = None,
) -> VikorResult:
    """Execute Fuzzy VIKOR algorithm directly on Triangular Fuzzy Numbers (TFNs).

    Formulation (Opricovic, 2011):
    -----------------------------
    1. Determine best f_j* and worst f_j- TFNs:
       For benefit: f_j* = (max l, max m, max u), f_j- = (min l, min m, min u)
       For cost:    f_j* = (min l, min m, min u), f_j- = (max l, max m, max u)

    2. Compute Normalized Fuzzy Separation Distance:
       Using the vertex distance metric:
       d_ij = d(f_j*, x_ij) / d(f_j*, f_j-)
       where d(A, B) = sqrt( 1/3 * [ (l_a - l_b)^2 + (m_a - m_b)^2 + (u_a - u_b)^2 ] ).

    3. Group Utility S_i and Individual Regret R_i:
       S_i = sum_j (w_j * d_ij)
       R_i = max_j (w_j * d_ij)

    4. VIKOR Index Q_i:
       Q_i = v * (S_i - S*) / (S- - S*) + (1 - v) * (R_i - R*) / (R- - R*)

    5. Compromise Acceptance Checks (C1 and C2) applied directly to S_i, R_i, Q_i.
    """
    m = len(matrix)
    if m == 0:
        empty = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return VikorResult(
            q_values=empty,
            rankings=empty_int,
            s_values=empty,
            r_values=empty,
            validation={},
            best_f=[],
            worst_f=[],
            normalized_distances=np.empty((0, 0)),
            weighted_distances=np.empty((0, 0)),
            weights=[],
            is_benefit=list(is_benefit),
            v=v,
            formulation="fuzzy",
            defuzzification_applied=False,
        )

    n = len(matrix[0])
    norm_w, v_val = validate_vikor_inputs(weights, is_benefit, n, v)

    # Validate all TFN elements
    validated_matrix: List[List[TFN]] = []
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(f"Row {i} dimension mismatch: expected {n} criteria, got {len(row)}.")
        v_row = [validate_tfn(cell, name=f"Cell[{i},{j}]") for j, cell in enumerate(row)]
        validated_matrix.append(v_row)

    # 1. Determine best f_j* and worst f_j- TFNs
    f_star_tfn: List[TFN] = []
    f_minus_tfn: List[TFN] = []

    for j in range(n):
        l_vals = [validated_matrix[i][j][0] for i in range(m)]
        m_vals = [validated_matrix[i][j][1] for i in range(m)]
        u_vals = [validated_matrix[i][j][2] for i in range(m)]

        if is_benefit[j]:
            f_star_tfn.append((max(l_vals), max(m_vals), max(u_vals)))
            f_minus_tfn.append((min(l_vals), min(m_vals), min(u_vals)))
        else:
            f_star_tfn.append((min(l_vals), min(m_vals), min(u_vals)))
            f_minus_tfn.append((max(l_vals), max(m_vals), max(u_vals)))

    # 2. Compute Normalized Vertex Distance & Weighted Regret Matrix
    norm_dist = np.zeros((m, n), dtype=float)
    weighted_dist = np.zeros((m, n), dtype=float)
    S = np.zeros(m, dtype=float)
    R = np.zeros(m, dtype=float)

    for i in range(m):
        for j in range(n):
            dist_to_star = tfn_distance(f_star_tfn[j], validated_matrix[i][j])
            dist_span = tfn_distance(f_star_tfn[j], f_minus_tfn[j])
            if dist_span > 1e-12:
                d_ij = dist_to_star / dist_span
            else:
                d_ij = 0.0

            norm_dist[i, j] = d_ij
            weighted_dist[i, j] = norm_w[j] * d_ij

        S[i] = float(np.sum(weighted_dist[i, :]))
        R[i] = float(np.max(weighted_dist[i, :]))

    # 3. Compute VIKOR Index Q_i
    s_star = float(np.min(S))
    s_minus = float(np.max(S))
    r_star = float(np.min(R))
    r_minus = float(np.max(R))

    s_denom = s_minus - s_star
    r_denom = r_minus - r_star

    Q = np.zeros(m, dtype=float)
    for i in range(m):
        term_s = (S[i] - s_star) / s_denom if s_denom > 1e-12 else 0.0
        term_r = (R[i] - r_star) / r_denom if r_denom > 1e-12 else 0.0
        Q[i] = v_val * term_s + (1.0 - v_val) * term_r

    # 4. Rank by Q ascending
    sort_keys = [(Q[i], S[i], R[i], i) for i in range(m)]
    order = [idx for _, _, _, idx in sorted(sort_keys)]

    rankings = np.empty(m, dtype=int)
    for rank_pos, alt_idx in enumerate(order, start=1):
        rankings[alt_idx] = rank_pos

    # 5. Compromise Acceptance Conditions
    c_names = list(criteria_names) if criteria_names else None
    a_names = list(alternative_names) if alternative_names else None
    validation = evaluate_compromise_conditions(Q, S, R, alternative_names=a_names)

    return VikorResult(
        q_values=Q,
        rankings=rankings,
        s_values=S,
        r_values=R,
        validation=validation,
        best_f=f_star_tfn,
        worst_f=f_minus_tfn,
        normalized_distances=norm_dist,
        weighted_distances=weighted_dist,
        weights=norm_w.tolist(),
        is_benefit=list(is_benefit),
        v=v_val,
        formulation="fuzzy",
        defuzzification_applied=False,
        criteria_names=c_names,
        alternative_names=a_names,
    )


# ==============================================================================
# Unified VIKOR Gateway Function
# ==============================================================================

def vikor(
    matrix: Union[np.ndarray, Sequence[Sequence[Union[float, TFN, Sequence[float]]]]],
    weights: Sequence[Union[float, TFN]],
    is_benefit: Sequence[bool],
    v: float = 0.5,
    formulation: str = "auto",
    defuzzification_method: str = "centroid",
    criteria_names: Optional[Sequence[str]] = None,
    alternative_names: Optional[Sequence[str]] = None,
) -> VikorResult:
    """Execute VIKOR Multi-Criteria Compromise Ranking.

    Parameters
    ----------
    matrix : Union[np.ndarray, Sequence[Sequence[Any]]]
        Decision matrix of shape (m, n). May contain crisp numbers or Triangular
        Fuzzy Numbers (l, m, u).
    weights : Sequence[Union[float, TFN]]
        Criterion weights of length n. Automatically validated and normalized to sum to 1.
    is_benefit : Sequence[bool]
        Boolean flags indicating True for benefit criteria and False for cost criteria.
    v : float, default=0.5
        Weight of the strategy of maximum group utility (v in [0, 1]).
        v > 0.5 favors majority consensus; v < 0.5 favors individual regret veto.
    formulation : str, default='auto'
        - 'auto': Inspects matrix elements. If elements are TFNs, uses 'fuzzy';
                  if crisp floats, uses 'classical'.
        - 'fuzzy': Executes Fuzzy VIKOR (Opricovic, 2011) directly on TFNs.
        - 'classical': Executes Classical VIKOR (Opricovic & Tzeng, 2004). If input
                       contains TFNs, an explicit defuzzification step is performed.
    defuzzification_method : str, default='centroid'
        Defuzzification method when executing Classical VIKOR on fuzzy data:
        - 'centroid': Center of Area (l + m + u) / 3
        - 'gmir': Graded Mean Integration Representation (l + 4m + u) / 6
    criteria_names : Optional[Sequence[str]]
        Descriptive names for criteria.
    alternative_names : Optional[Sequence[str]]
        Identifiers or labels for alternatives.

    Returns
    -------
    VikorResult
        Dataclass containing q_values, rankings, s_values, r_values, validation,
        distances, and compromise solution sets.
        Supports 5-tuple unpacking: `q, ranks, s, r, val = vikor(...)`.
    """
    m = len(matrix)
    if m == 0:
        return classical_vikor(
            np.empty((0, len(weights))), weights, is_benefit, v=v,
            criteria_names=criteria_names, alternative_names=alternative_names,
        )

    # Detect whether elements are fuzzy TFNs
    first_cell = matrix[0][0]
    is_input_fuzzy = isinstance(first_cell, (tuple, list)) and len(first_cell) == 3

    if formulation == "auto":
        target_formulation = "fuzzy" if is_input_fuzzy else "classical"
    else:
        target_formulation = formulation.lower()

    if target_formulation == "fuzzy":
        if not is_input_fuzzy:
            # Wrap crisp numbers into degenerate TFNs (x, x, x)
            fuzzy_mat = [[(float(cell), float(cell), float(cell)) for cell in row] for row in matrix]
        else:
            fuzzy_mat = matrix  # type: ignore
        return fuzzy_vikor(
            fuzzy_mat, weights, is_benefit, v=v,
            criteria_names=criteria_names, alternative_names=alternative_names,
        )

    # Target formulation is Classical VIKOR
    defuzz_applied = False
    if is_input_fuzzy:
        defuzz_applied = True
        crisp_mat = np.zeros((m, len(matrix[0])), dtype=float)
        defuzz_func = tfn_defuzzify_gmir if defuzzification_method == "gmir" else tfn_defuzzify_centroid
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                tfn_val = validate_tfn(cell)  # type: ignore
                crisp_mat[i, j] = defuzz_func(tfn_val)
    else:
        crisp_mat = np.asarray(matrix, dtype=float)

    result = classical_vikor(
        crisp_mat, weights, is_benefit, v=v,
        criteria_names=criteria_names, alternative_names=alternative_names,
    )
    result.defuzzification_applied = defuzz_applied
    return result
