"""Comprehensive unit, mathematical, and edge-case tests for the Fuzzy TOPSIS engine.

Verifies:
1. Triangular Fuzzy Number (TFN) operations (add, scale, mul, distance, defuzzify).
2. Input and weight validation (bounds, negativity, dimension mismatches, zero sums).
3. Hand-calculated step-by-step benchmark verification against analytical derivation.
4. Degenerate and edge cases (equal alternatives, zero lower bounds, zero column elements).
5. Cost criteria behavior (reducing cost improves priority).
6. Monotonicity and sensitivity (input rating increases improve closeness and ranking).
7. Intermediate matrix accessibility and backwards-compatible tuple unpacking.
"""

import math
import numpy as np
import pytest

from core.fuzzy_topsis import (
    FuzzyTopsisResult,
    fuzzy_topsis,
    tfn_add,
    tfn_defuzzify_centroid,
    tfn_defuzzify_gmir,
    tfn_distance,
    tfn_mul,
    tfn_scale,
    validate_tfn,
    validate_weights,
)


# ==============================================================================
# 1. TFN Elementary Operations & Vertex Distance Tests
# ==============================================================================

def test_validate_tfn_valid():
    assert validate_tfn((0.1, 0.3, 0.5)) == (0.1, 0.3, 0.5)
    assert validate_tfn((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert validate_tfn([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)


def test_validate_tfn_invalid():
    with pytest.raises(ValueError, match="lower bound must be non-negative"):
        validate_tfn((-0.1, 0.2, 0.5))

    with pytest.raises(ValueError, match="lower bound cannot exceed modal value"):
        validate_tfn((0.5, 0.3, 0.6))

    with pytest.raises(ValueError, match="modal value cannot exceed upper bound"):
        validate_tfn((0.2, 0.7, 0.5))

    with pytest.raises(TypeError, match="must be a 3-element sequence"):
        validate_tfn((0.2, 0.5))  # Only 2 elements


def test_tfn_arithmetic():
    a = (1.0, 2.0, 3.0)
    b = (2.0, 3.0, 4.0)

    # Addition
    added = tfn_add(a, b)
    assert added == (3.0, 5.0, 7.0)

    # Scaling
    scaled = tfn_scale(a, 2.0)
    assert scaled == (2.0, 4.0, 6.0)

    # Multiplication
    multiplied = tfn_mul(a, b)
    assert multiplied == (2.0, 6.0, 12.0)

    # Defuzzification
    assert pytest.approx(tfn_defuzzify_centroid(a), 1e-4) == 2.0
    assert pytest.approx(tfn_defuzzify_gmir(a), 1e-4) == 2.0


def test_tfn_distance():
    # Distance between identical TFNs is zero
    a = (1.0, 2.0, 3.0)
    assert tfn_distance(a, a) == 0.0

    # Distance to a shifted TFN: (2, 3, 4) shifted by 1 along all vertices
    # diff = sqrt( (1^2 + 1^2 + 1^2) / 3 ) = 1.0
    b = (2.0, 3.0, 4.0)
    assert pytest.approx(tfn_distance(a, b), 1e-6) == 1.0


# ==============================================================================
# 2. Weights & Matrix Dimension Validation
# ==============================================================================

def test_validate_weights_normalization():
    # Crisp weights normalized to sum=1.0
    weights = [2.0, 3.0, 5.0]
    norm, is_fuzzy = validate_weights(weights, 3)
    assert is_fuzzy is False
    assert pytest.approx(sum(norm), 1e-6) == 1.0
    assert pytest.approx(norm[0], 1e-6) == 0.2
    assert pytest.approx(norm[1], 1e-6) == 0.3
    assert pytest.approx(norm[2], 1e-6) == 0.5


def test_validate_weights_fuzzy_tfn():
    # Fuzzy TFN weights: Low (0.1, 0.3, 0.5) and High (0.5, 0.7, 0.9)
    # Centroids: 0.3 and 0.7. Sum = 1.0
    weights = [(0.1, 0.3, 0.5), (0.5, 0.7, 0.9)]
    norm, is_fuzzy = validate_weights(weights, 2)
    assert is_fuzzy is True
    assert pytest.approx(norm[0], 1e-4) == 0.3
    assert pytest.approx(norm[1], 1e-4) == 0.7


def test_validate_weights_invalid():
    # Negative weight
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_weights([0.5, -0.2], 2)

    # Sum of weights is zero
    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_weights([0.0, 0.0], 2)

    # Dimension mismatch
    with pytest.raises(ValueError, match="Weight dimension mismatch"):
        validate_weights([0.5, 0.5], 3)


# ==============================================================================
# 3. Hand-Calculated Step-by-Step Numerical Verification
# ==============================================================================

def test_fuzzy_topsis_hand_calculated_benchmark():
    """Verify Fuzzy TOPSIS against exact analytically calculated steps:

    Problem Specification:
    ----------------------
    2 Alternatives: A1, A2
    2 Criteria: C1 (Benefit, weight=0.6), C2 (Cost, weight=0.4)

    Alternative Ratings:
    - A1: C1 = (0.2, 0.4, 0.6), C2 = (0.1, 0.2, 0.3) [Lower cost]
    - A2: C1 = (0.6, 0.8, 1.0), C2 = (0.4, 0.6, 0.8) [Higher cost]

    Analytical Calculations:
    1. Normalization:
       - C1 (Benefit): max_u = 1.0
         A1: (0.2, 0.4, 0.6)
         A2: (0.6, 0.8, 1.0)
       - C2 (Cost): min_l = 0.1
         A1: (0.1/0.3, 0.1/0.2, 0.1/0.1) = (1/3, 1/2, 1) = (0.3333, 0.5, 1.0)
         A2: (0.1/0.8, 0.1/0.6, 0.1/0.4) = (1/8, 1/6, 1/4) = (0.125, 0.1667, 0.25)

    2. Weighted Normalization (w1=0.6, w2=0.4):
       - A1: C1 = (0.12, 0.24, 0.36), C2 = (0.1333, 0.20, 0.40)
       - A2: C1 = (0.36, 0.48, 0.60), C2 = (0.05, 0.0667, 0.10)

    3. Relative Ideal Solutions:
       - FPIS (A*): C1 = (0.36, 0.48, 0.60), C2 = (0.1333, 0.20, 0.40)
       - FNIS (A-): C1 = (0.12, 0.24, 0.36), C2 = (0.05, 0.0667, 0.10)

    4. Euclidean Vertex Distances:
       - d(A1, A*): d(C1) = 0.24, d(C2) = 0.0 -> D1+ = 0.24
       - d(A1, A-): d(C1) = 0.0, d(C2) ~= 0.19555 -> D1- ~= 0.19555
       - d(A2, A*): d(C1) = 0.0, d(C2) ~= 0.19555 -> D2+ ~= 0.19555
       - d(A2, A-): d(C1) = 0.24, d(C2) = 0.0 -> D2- = 0.24

    5. Closeness Coefficients:
       - CC1 = 0.19555 / (0.24 + 0.19555) ~= 0.4490
       - CC2 = 0.24 / (0.19555 + 0.24) ~= 0.5510

    6. Ranking:
       - A2 has higher CC (0.5510) -> Rank 1
       - A1 has lower CC (0.4490) -> Rank 2
    """
    matrix = [
        [(0.2, 0.4, 0.6), (0.1, 0.2, 0.3)],  # A1
        [(0.6, 0.8, 1.0), (0.4, 0.6, 0.8)],  # A2
    ]
    weights = [0.6, 0.4]
    is_benefit = [True, False]

    res = fuzzy_topsis(matrix, weights, is_benefit, normalization_method="chen")

    # Verify Normalized Matrix
    # A1, C2: (1/3, 1/2, 1)
    assert pytest.approx(res.normalized_matrix[0][1][0], 1e-3) == 1 / 3
    assert pytest.approx(res.normalized_matrix[0][1][1], 1e-3) == 0.5
    assert pytest.approx(res.normalized_matrix[0][1][2], 1e-3) == 1.0
    # A2, C2: (0.125, 0.1667, 0.25)
    assert pytest.approx(res.normalized_matrix[1][1][0], 1e-3) == 0.125
    assert pytest.approx(res.normalized_matrix[1][1][1], 1e-3) == 1 / 6
    assert pytest.approx(res.normalized_matrix[1][1][2], 1e-3) == 0.25

    # Verify Weighted Matrix
    assert pytest.approx(res.weighted_matrix[0][0][1], 1e-3) == 0.24
    assert pytest.approx(res.weighted_matrix[1][0][1], 1e-3) == 0.48

    # Verify Distances
    expected_d2_plus = math.sqrt(
        ((0.05 - 0.4 / 3) ** 2 + (0.4 / 6 - 0.20) ** 2 + (0.10 - 0.40) ** 2) / 3.0
    )
    assert pytest.approx(res.d_plus[0], 1e-4) == 0.24
    assert pytest.approx(res.d_plus[1], 1e-4) == expected_d2_plus
    assert pytest.approx(res.d_minus[0], 1e-4) == expected_d2_plus
    assert pytest.approx(res.d_minus[1], 1e-4) == 0.24

    # Verify Closeness Coefficients
    assert pytest.approx(res.closeness_coefficients[0], 1e-3) == 0.4490
    assert pytest.approx(res.closeness_coefficients[1], 1e-3) == 0.5510

    # Verify Rankings
    assert res.rankings[1] == 1  # A2 is Rank 1
    assert res.rankings[0] == 2  # A1 is Rank 2


# ==============================================================================
# 4. Cost Criteria Behavior & Trade-Offs
# ==============================================================================

def test_cost_criterion_low_cost_wins():
    """Verify that an alternative with lower repair effort achieves higher priority."""
    # Both alternatives have identical high severity on C1 (Benefit)
    # A1 has High Repair Effort (Bad)
    # A2 has Very Low Repair Effort (Good)
    matrix = [
        [(0.7, 0.9, 1.0), (0.7, 0.9, 1.0)],  # A1: High benefit, High cost
        [(0.7, 0.9, 1.0), (0.1, 0.2, 0.3)],  # A2: High benefit, Low cost
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = fuzzy_topsis(matrix, weights, is_benefit)
    assert res.rankings[1] == 1
    assert res.rankings[0] == 2
    assert res.closeness_coefficients[1] > res.closeness_coefficients[0]


def test_linear_maxmin_cost_normalization():
    """Verify alternative linear max-min cost normalization (Jahanshahloo et al., 2006)."""
    matrix = [
        [(0.5, 0.7, 0.9), (0.5, 0.7, 0.9)],
        [(0.5, 0.7, 0.9), (0.1, 0.2, 0.3)],
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = fuzzy_topsis(matrix, weights, is_benefit, normalization_method="linear")
    assert res.rankings[1] == 1
    assert res.closeness_coefficients[1] > res.closeness_coefficients[0]


# ==============================================================================
# 5. Degenerate Cases, Edge Cases & Division by Zero
# ==============================================================================

def test_equal_alternatives_degenerate_case():
    """When all alternatives are identical, D+ = D- = 0, CC should equal 0.5."""
    matrix = [
        [(0.3, 0.5, 0.7), (0.2, 0.4, 0.6)],
        [(0.3, 0.5, 0.7), (0.2, 0.4, 0.6)],
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, True]

    res = fuzzy_topsis(matrix, weights, is_benefit)
    assert len(res.closeness_coefficients) == 2
    # Both alternatives must receive documented neutral closeness 0.5
    assert res.closeness_coefficients[0] == 0.5
    assert res.closeness_coefficients[1] == 0.5
    assert res.d_plus[0] == 0.0 and res.d_minus[0] == 0.0


def test_zero_lower_bounds_safe_shift():
    """Test when raw ratings have lower bound l = 0.0."""
    matrix = [
        [(0.0, 0.1, 0.3), (0.0, 0.1, 0.3)],  # Very Low rating starting at 0
        [(0.5, 0.7, 0.9), (0.5, 0.7, 0.9)],  # High rating
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, False]  # Criterion 2 is cost

    # Must complete safely without ZeroDivisionError or NaN
    res = fuzzy_topsis(matrix, weights, is_benefit)
    assert not np.isnan(res.closeness_coefficients).any()
    assert np.all(res.closeness_coefficients >= 0.0)
    assert np.all(res.closeness_coefficients <= 1.0)


def test_empty_decision_matrix():
    """Empty matrix returns cleanly without errors."""
    res = fuzzy_topsis([], [], [])
    assert len(res.closeness_coefficients) == 0
    assert len(res.rankings) == 0


def test_single_alternative():
    """Single alternative evaluates to neutral CC=0.5 and rank 1."""
    matrix = [[(0.3, 0.5, 0.7), (0.5, 0.7, 0.9)]]
    weights = [0.5, 0.5]
    is_benefit = [True, True]

    res = fuzzy_topsis(matrix, weights, is_benefit)
    assert res.rankings[0] == 1
    assert res.closeness_coefficients[0] == 0.5


# ==============================================================================
# 6. Monotonicity & Sensitivity Tests
# ==============================================================================

def test_monotonicity_improving_rating_improves_rank():
    """Increasing benefit rating of an alternative must increase its CC score."""
    # Base setup: A1 and A2 are initially identical
    matrix_base = [
        [(0.3, 0.5, 0.7), (0.3, 0.5, 0.7)],
        [(0.3, 0.5, 0.7), (0.3, 0.5, 0.7)],
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, True]

    res_base = fuzzy_topsis(matrix_base, weights, is_benefit)
    cc_base = res_base.closeness_coefficients[0]

    # Improve A1's severity to High (0.7, 0.9, 1.0)
    matrix_improved = [
        [(0.7, 0.9, 1.0), (0.3, 0.5, 0.7)],
        [(0.3, 0.5, 0.7), (0.3, 0.5, 0.7)],
    ]
    res_improved = fuzzy_topsis(matrix_improved, weights, is_benefit)

    assert res_improved.closeness_coefficients[0] > cc_base
    assert res_improved.rankings[0] == 1  # A1 must be Rank 1


def test_weight_sensitivity_changes_rankings():
    """Shifting weight to a criterion where an alternative excels promotes that alternative."""
    # A1 has superior severity (C1) but inferior task impact (C2)
    # A2 has inferior severity (C1) but superior task impact (C2)
    matrix = [
        [(0.7, 0.9, 1.0), (0.1, 0.3, 0.5)],  # A1: high C1, low C2
        [(0.1, 0.3, 0.5), (0.7, 0.9, 1.0)],  # A2: low C1, high C2
    ]
    is_benefit = [True, True]

    # Case 1: Heavy weight on C1 (0.8, 0.2) -> A1 should win
    res_c1_heavy = fuzzy_topsis(matrix, [0.8, 0.2], is_benefit)
    assert res_c1_heavy.rankings[0] == 1
    assert res_c1_heavy.rankings[1] == 2

    # Case 2: Heavy weight on C2 (0.2, 0.8) -> A2 should win
    res_c2_heavy = fuzzy_topsis(matrix, [0.2, 0.8], is_benefit)
    assert res_c2_heavy.rankings[1] == 1
    assert res_c2_heavy.rankings[0] == 2


# ==============================================================================
# 7. Intermediate Values & Backward-Compatible Unpacking
# ==============================================================================

def test_result_tuple_unpacking():
    """Ensure `cc, ranks, d_plus, d_minus = fuzzy_topsis(...)` works seamlessly."""
    matrix = [
        [(0.7, 0.9, 1.0), (0.7, 0.9, 1.0)],
        [(0.3, 0.5, 0.7), (0.3, 0.5, 0.7)],
    ]
    cc, ranks, d_plus, d_minus = fuzzy_topsis(matrix, [0.5, 0.5], [True, True])

    assert isinstance(cc, np.ndarray)
    assert isinstance(ranks, np.ndarray)
    assert len(cc) == 2
    assert ranks[0] == 1
    assert ranks[1] == 2


def test_result_intermediate_dataframes():
    """Verify helper methods for inspection and reporting."""
    matrix = [
        [(0.7, 0.9, 1.0), (0.1, 0.3, 0.5)],
        [(0.3, 0.5, 0.7), (0.5, 0.7, 0.9)],
    ]
    res = fuzzy_topsis(
        matrix,
        [0.6, 0.4],
        [True, False],
        criteria_names=["Severity", "Repair Effort"],
        alternative_names=["ISS-01", "ISS-02"],
    )

    norm_df = res.get_normalized_df()
    assert norm_df.shape == (2, 2)
    assert "Severity" in norm_df.columns
    assert "Repair Effort" in norm_df.columns
    assert "ISS-01" in norm_df.index

    d = res.to_dict()
    assert "closeness_coefficients" in d
    assert "fpis" in d
    assert "fnis" in d
