"""
================================================================================
MODULE: core.fuzzy_topsis
================================================================================
1. WHAT THIS FILE DOES:
   Implements the full mathematical engine for Chen (2000) Fuzzy TOPSIS (Technique
   for Order Preference by Similarity to Ideal Solution) using Triangular Fuzzy Numbers (TFNs).

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Human evaluations of web accessibility (e.g. "severe barrier",
     "moderate effort") possess inherent vagueness and cognitive ambiguity. Crisp numbers
     oversimplify real-world auditor judgments. Fuzzy TOPSIS models this uncertainty using
     triangular membership functions, measuring which accessibility repair is closest to
     the Fuzzy Positive-Ideal Solution (FPIS) and furthest from the Fuzzy Negative-Ideal Solution (FNIS).
   - Technical & Architectural Rationale: Pure, self-contained mathematical library
     independent of databases or web frameworks. Can be tested rigorously, verified with
     numerical edge cases (single alternative, zero-variance, identical scores), and reused.

3. HOW IT WORKS CONCEPTUALLY:
   - Takes alternatives (accessibility issues) evaluated on m criteria (C1-C6).
   - Constructs a fuzzy decision matrix of TFNs: (a, b, c).
   - Normalizes matrix:
     * Benefit criteria: r_ij = (a/c_max*, b/c_max*, c/c_max*)
     * Cost criteria (e.g. Repair Effort): r_ij = (a_min-/c, a_min-/b, a_min-/a)
   - Multiplies by criterion weights: v_ij = r_ij (*) w_j
   - Determines FPIS (A*) and FNIS (A-).
   - Calculates vertex Euclidean distances d*(A_i, A*) and d-(A_i, A-).
   - Computes Closeness Coefficient CC_i = d- / (d* + d-).
   - Ranks alternatives in descending order of CC_i (higher CC = higher repair priority).

4. HOW IT WORKS TECHNICALLY:
   - Encapsulates state and results in `FuzzyTOPSISResult` dataclass.
   - Uses `numpy` for efficient vectorized array processing.
   - Implements `vertex_distance_tfn()` using the Chen (2000) formula:
     d(A, B) = sqrt(1/3 * [(a1-b1)^2 + (a2-b2)^2 + (a3-b3)^2]).
   - Handles edge cases (zero denominator, identical distances, single alternative).

References:
- Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment.
  Fuzzy Sets and Systems, 114(1), 1-9.
- Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

# Triangular Fuzzy Number: (lower, modal, upper) with 0 <= l <= m <= u
TFN = Tuple[float, float, float]


# ==============================================================================
# Triangular Fuzzy Number Operations & Validation
# ==============================================================================

def validate_tfn(tfn: Union[TFN, Sequence[float]], name: str = "TFN") -> TFN:
    """Validate that a 3-tuple is a mathematically valid Triangular Fuzzy Number.

    Requirements:
    - Must have length 3 (l, m, u).
    - All elements must be finite non-negative real numbers: 0 <= l <= m <= u.

    Raises
    ------
    ValueError or TypeError if invalid.
    """
    if not isinstance(tfn, (tuple, list)) or len(tfn) != 3:
        raise TypeError(f"{name} must be a 3-element sequence of floats (l, m, u), got {tfn!r}.")

    l, m, u = float(tfn[0]), float(tfn[1]), float(tfn[2])

    if math.isnan(l) or math.isnan(m) or math.isnan(u) or math.isinf(l) or math.isinf(m) or math.isinf(u):
        raise ValueError(f"{name} values must be finite real numbers, got ({l}, {m}, {u}).")

    if l < -1e-12:
        raise ValueError(f"{name} lower bound must be non-negative: l={l} < 0.")

    if l > m + 1e-12:
        raise ValueError(f"{name} lower bound cannot exceed modal value: l={l} > m={m}.")

    if m > u + 1e-12:
        raise ValueError(f"{name} modal value cannot exceed upper bound: m={m} > u={u}.")

    # Clean small float precision artifacts
    l_clean = max(0.0, l)
    m_clean = max(l_clean, m)
    u_clean = max(m_clean, u)
    return (l_clean, m_clean, u_clean)


def tfn_add(a: TFN, b: TFN) -> TFN:
    """Fuzzy addition: (l1, m1, u1) (+) (l2, m2, u2) = (l1 + l2, m1 + m2, u1 + u2)."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tfn_scale(a: TFN, scalar: float) -> TFN:
    """Fuzzy scalar multiplication: k (*) (l, m, u) = (k*l, k*m, k*u) for k >= 0."""
    if scalar < -1e-12:
        raise ValueError(f"Fuzzy scalar multiplication requires non-negative scalar, got {scalar}.")
    k = max(0.0, scalar)
    return (a[0] * k, a[1] * k, a[2] * k)


def tfn_mul(a: TFN, b: TFN) -> TFN:
    """Fuzzy multiplication for non-negative TFNs: (l1*l2, m1*m2, u1*u2)."""
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def tfn_distance(a: TFN, b: TFN) -> float:
    """Calculate the Euclidean vertex distance between two Triangular Fuzzy Numbers.

    Following Chen (2000), Eq. (7):
        d(A, B) = sqrt( (1/3) * [ (l_a - l_b)^2 + (m_a - m_b)^2 + (u_a - u_b)^2 ] )
    """
    diff_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return math.sqrt(diff_sq / 3.0)


def tfn_defuzzify_centroid(a: TFN) -> float:
    """Defuzzify a TFN using the center of area / centroid method: (l + m + u) / 3."""
    return (a[0] + a[1] + a[2]) / 3.0


def tfn_defuzzify_gmir(a: TFN) -> float:
    """Graded Mean Integration Representation: (l + 4*m + u) / 6."""
    return (a[0] + 4.0 * a[1] + a[2]) / 6.0


# ==============================================================================
# Weight Normalization & Input Validation
# ==============================================================================

def validate_weights(
    weights: Sequence[Union[float, TFN]],
    num_criteria: int,
) -> Tuple[List[float], bool]:
    """Validate and normalize criterion weights.

    Parameters
    ----------
    weights : Sequence[Union[float, TFN]]
        Sequence of crisp non-negative weights or TFN weights.
    num_criteria : int
        Expected number of criteria.

    Returns
    -------
    Tuple[List[float], bool]
        - Normalized crisp weights summing strictly to 1.0.
        - Boolean indicating whether original input was fuzzy weights.
    """
    if len(weights) != num_criteria:
        raise ValueError(
            f"Weight dimension mismatch: expected {num_criteria} weights, got {len(weights)}."
        )

    is_fuzzy = False
    crisp_weights: List[float] = []

    for idx, w in enumerate(weights):
        if isinstance(w, (tuple, list)) and len(w) == 3:
            is_fuzzy = True
            validated = validate_tfn(w, name=f"Weight[{idx}]")
            # Use centroid defuzzification to obtain crisp operational weight
            crisp_weights.append(tfn_defuzzify_centroid(validated))
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

    # Normalize so sum(weights) == 1.0
    normalized_weights = [w / total for w in crisp_weights]
    return normalized_weights, is_fuzzy


# ==============================================================================
# Fuzzy TOPSIS Execution & Intermediate Results Data Structure
# ==============================================================================

@dataclass
class FuzzyTopsisResult:
    """Encapsulates complete Fuzzy TOPSIS evaluation including intermediate matrices.

    Provides transparent auditability for MCDM research, classroom verification,
    and administrative reporting.
    """

    closeness_coefficients: np.ndarray
    rankings: np.ndarray
    d_plus: np.ndarray
    d_minus: np.ndarray
    decision_matrix: List[List[TFN]]
    normalized_matrix: List[List[TFN]]
    weighted_matrix: List[List[TFN]]
    fpis: List[TFN]
    fnis: List[TFN]
    weights: List[float]
    is_benefit: List[bool]
    normalization_method: str = "chen"
    ideal_solution_type: str = "relative"
    criteria_names: Optional[List[str]] = field(default=None)
    alternative_names: Optional[List[str]] = field(default=None)

    def __iter__(self) -> Iterator[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Enable backwards-compatible 4-tuple unpacking:
        cc, ranks, d_plus, d_minus = fuzzy_topsis(...)
        """
        yield self.closeness_coefficients
        yield self.rankings
        yield self.d_plus
        yield self.d_minus

    def __getitem__(self, index: int) -> np.ndarray:
        return [self.closeness_coefficients, self.rankings, self.d_plus, self.d_minus][index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert intermediate matrices and summary metrics to a serializable dictionary."""
        return {
            "closeness_coefficients": self.closeness_coefficients.tolist(),
            "rankings": self.rankings.tolist(),
            "d_plus": self.d_plus.tolist(),
            "d_minus": self.d_minus.tolist(),
            "weights": self.weights,
            "is_benefit": self.is_benefit,
            "fpis": self.fpis,
            "fnis": self.fnis,
            "normalization_method": self.normalization_method,
            "ideal_solution_type": self.ideal_solution_type,
        }

    def get_normalized_df(self) -> pd.DataFrame:
        """Return formatted normalized decision matrix as pandas DataFrame."""
        rows = []
        cols = self.criteria_names or [f"C{j+1}" for j in range(len(self.is_benefit))]
        alts = self.alternative_names or [f"A{i+1}" for i in range(len(self.normalized_matrix))]
        for i, row in enumerate(self.normalized_matrix):
            rows.append({cols[j]: f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})" for j, r in enumerate(row)})
        return pd.DataFrame(rows, index=alts)

    def get_weighted_df(self) -> pd.DataFrame:
        """Return formatted weighted decision matrix as pandas DataFrame."""
        rows = []
        cols = self.criteria_names or [f"C{j+1}" for j in range(len(self.is_benefit))]
        alts = self.alternative_names or [f"A{i+1}" for i in range(len(self.weighted_matrix))]
        for i, row in enumerate(self.weighted_matrix):
            rows.append({cols[j]: f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})" for j, r in enumerate(row)})
        return pd.DataFrame(rows, index=alts)


def fuzzy_topsis(
    matrix: Sequence[Sequence[Union[TFN, Sequence[float]]]],
    weights: Sequence[Union[float, TFN]],
    is_benefit: Sequence[bool],
    normalization_method: str = "chen",
    ideal_solution_type: str = "relative",
    criteria_names: Optional[Sequence[str]] = None,
    alternative_names: Optional[Sequence[str]] = None,
) -> FuzzyTopsisResult:
    """Execute mathematically sound Fuzzy TOPSIS on Triangular Fuzzy Numbers.

    Parameters
    ----------
    matrix : Sequence[Sequence[TFN]]
        m x n decision matrix where each cell is a TFN (lower, modal, upper).
    weights : Sequence[Union[float, TFN]]
        Criterion weights of length n (crisp floats or TFNs). Automatically validated
        and normalized to sum strictly to 1.0.
    is_benefit : Sequence[bool]
        Boolean flags of length n (True for benefit criteria, False for cost criteria).
    normalization_method : str, default='chen'
        - 'chen': Linear scale transformation (Chen, 2000). For benefit criteria:
                  r_ij = (a_ij / c_j*, b_ij / c_j*, c_ij / c_j*) with c_j* = max_i(c_ij).
                  For cost criteria:
                  r_ij = (a_j- / c_ij, a_j- / b_ij, a_j- / a_ij) with a_j- = min_i(a_ij).
                  (With positive floor/shift defense against division by zero).
        - 'linear': Linear max-min scaling (Jahanshahloo et al., 2006).
    ideal_solution_type : str, default='relative'
        - 'relative': Data-driven ideal bounds:
                      FPIS A* = (max a, max b, max c) across alternatives.
                      FNIS A- = (min a, min b, min c) across alternatives.
        - 'absolute': Theoretical absolute ideal bounds (Chen, 2000):
                      FPIS A* = (w_j, w_j, w_j), FNIS A- = (0, 0, 0).
    criteria_names : Optional[Sequence[str]]
        Descriptive names for criteria (for audit reporting).
    alternative_names : Optional[Sequence[str]]
        Identifiers or labels for alternatives.

    Returns
    -------
    FuzzyTopsisResult
        Dataclass containing:
        - closeness_coefficients: 1D array of CC_i in [0, 1]
        - rankings: 1D array of 1-based ranks (1 = highest priority)
        - d_plus: Euclidean distances to Fuzzy Positive Ideal Solution
        - d_minus: Euclidean distances to Fuzzy Negative Ideal Solution
        - decision_matrix, normalized_matrix, weighted_matrix
        - fpis, fnis
        Also supports tuple unpacking: `cc, ranks, d_plus, d_minus = fuzzy_topsis(...)`.
    """
    m = len(matrix)
    if m == 0:
        empty = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return FuzzyTopsisResult(
            closeness_coefficients=empty,
            rankings=empty_int,
            d_plus=empty,
            d_minus=empty,
            decision_matrix=[],
            normalized_matrix=[],
            weighted_matrix=[],
            fpis=[],
            fnis=[],
            weights=[],
            is_benefit=list(is_benefit),
            normalization_method=normalization_method,
            ideal_solution_type=ideal_solution_type,
        )

    n = len(matrix[0])
    if len(weights) != n or len(is_benefit) != n:
        raise ValueError(
            f"Dimension mismatch: decision matrix has {n} criteria, but received "
            f"{len(weights)} weights and {len(is_benefit)} benefit direction flags."
        )

    # 1. Validate Decision Matrix & Weights
    validated_matrix: List[List[TFN]] = []
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"Row {i} dimension mismatch: expected {n} criteria, got {len(row)}."
            )
        v_row = [validate_tfn(cell, name=f"Cell[{i},{j}]") for j, cell in enumerate(row)]
        validated_matrix.append(v_row)

    norm_weights, _ = validate_weights(weights, n)

    # 2. Normalization
    normalized: List[List[TFN]] = [[(0.0, 0.0, 0.0) for _ in range(n)] for _ in range(m)]

    for j in range(n):
        if is_benefit[j]:
            c_star = max(validated_matrix[i][j][2] for i in range(m))
            # Guard against all-zero criteria column
            c_star_safe = max(c_star, 1e-12)
            for i in range(m):
                a, b, c = validated_matrix[i][j]
                normalized[i][j] = (a / c_star_safe, b / c_star_safe, c / c_star_safe)
        else:
            # Cost Criterion Normalization
            min_a = min(validated_matrix[i][j][0] for i in range(m))
            max_c = max(validated_matrix[i][j][2] for i in range(m))

            if normalization_method == "linear":
                # Linear Max-Min normalization (Jahanshahloo et al., 2006)
                denom = max_c - min_a
                denom_safe = denom if denom > 1e-12 else 1.0
                for i in range(m):
                    a, b, c = validated_matrix[i][j]
                    if denom <= 1e-12:
                        # All alternatives have identical cost
                        normalized[i][j] = (1.0, 1.0, 1.0)
                    else:
                        normalized[i][j] = (
                            max(0.0, (max_c - c) / denom_safe),
                            max(0.0, (max_c - b) / denom_safe),
                            max(0.0, (max_c - a) / denom_safe),
                        )
            else:
                # Chen (2000) Linear Scale Transformation for Cost Criteria:
                # r_ij = (a_j- / c_ij, a_j- / b_ij, a_j- / a_ij) where a_j- = min_i(a_ij)
                # When raw lower bounds can be zero, Chen's reciprocal requires positive scaling
                # to prevent division by zero and collapse to zero.
                if min_a <= 1e-12:
                    # Apply minimal positive baseline shift (preserving exact relative distances)
                    shift = 0.1
                else:
                    shift = 0.0

                shifted_min_a = min_a + shift
                for i in range(m):
                    a, b, c = validated_matrix[i][j]
                    a_sh = a + shift
                    b_sh = b + shift
                    c_sh = c + shift

                    # Bounds check: a_sh <= b_sh <= c_sh guarantees 1/c_sh <= 1/b_sh <= 1/a_sh
                    r_l = shifted_min_a / c_sh
                    r_m = shifted_min_a / b_sh
                    r_u = shifted_min_a / a_sh

                    # Clamp cleanly to [0.0, 1.0]
                    normalized[i][j] = (
                        max(0.0, min(1.0, r_l)),
                        max(0.0, min(1.0, r_m)),
                        max(0.0, min(1.0, r_u)),
                    )

    # 3. Apply Criterion Weights
    # v_ij = r_ij (*) w_j
    weighted: List[List[TFN]] = [[(0.0, 0.0, 0.0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            w = norm_weights[j]
            a, b, c = normalized[i][j]
            weighted[i][j] = (a * w, b * w, c * w)

    # 4. Determine Fuzzy Positive-Ideal (FPIS) and Negative-Ideal (FNIS) Solutions
    fpis: List[TFN] = []
    fnis: List[TFN] = []

    if ideal_solution_type == "absolute":
        # Chen (2000) absolute theoretical ideal solution
        for j in range(n):
            w = norm_weights[j]
            fpis.append((w, w, w))
            fnis.append((0.0, 0.0, 0.0))
    else:
        # Relative data-driven ideal solution across evaluated alternatives
        for j in range(n):
            max_a = max(weighted[i][j][0] for i in range(m))
            max_b = max(weighted[i][j][1] for i in range(m))
            max_c = max(weighted[i][j][2] for i in range(m))
            min_a = min(weighted[i][j][0] for i in range(m))
            min_b = min(weighted[i][j][1] for i in range(m))
            min_c = min(weighted[i][j][2] for i in range(m))

            fpis.append((max_a, max_b, max_c))
            fnis.append((min_a, min_b, min_c))

    # 5. Calculate Euclidean Vertex Distances to FPIS and FNIS
    d_plus = np.zeros(m, dtype=float)
    d_minus = np.zeros(m, dtype=float)

    for i in range(m):
        dist_p = 0.0
        dist_m = 0.0
        for j in range(n):
            dist_p += tfn_distance(weighted[i][j], fpis[j])
            dist_m += tfn_distance(weighted[i][j], fnis[j])
        d_plus[i] = dist_p
        d_minus[i] = dist_m

    # 6. Calculate Relative Closeness Coefficient CC_i = D_i- / (D_i+ + D_i-)
    closeness = np.zeros(m, dtype=float)
    for i in range(m):
        denom = d_plus[i] + d_minus[i]
        if denom > 1e-12:
            closeness[i] = d_minus[i] / denom
        else:
            # Degenerate case: all alternatives are identical, resulting in d_plus = d_minus = 0.
            # Documented neutral closeness assignment: 0.5
            closeness[i] = 0.5

    # 7. Rank: Higher Closeness Coefficient CC_i -> Closer to FPIS -> Rank 1
    # Deterministic ranking with tie-breaking preserving index stability
    # Use negative closeness for descending sort
    sort_keys = [(-closeness[i], i) for i in range(m)]
    order = [idx for _, idx in sorted(sort_keys)]

    rankings = np.empty(m, dtype=int)
    for rank_pos, alt_idx in enumerate(order, start=1):
        rankings[alt_idx] = rank_pos

    c_names = list(criteria_names) if criteria_names else None
    a_names = list(alternative_names) if alternative_names else None

    return FuzzyTopsisResult(
        closeness_coefficients=closeness,
        rankings=rankings,
        d_plus=d_plus,
        d_minus=d_minus,
        decision_matrix=validated_matrix,
        normalized_matrix=normalized,
        weighted_matrix=weighted,
        fpis=fpis,
        fnis=fnis,
        weights=norm_weights,
        is_benefit=list(is_benefit),
        normalization_method=normalization_method,
        ideal_solution_type=ideal_solution_type,
        criteria_names=c_names,
        alternative_names=a_names,
    )
