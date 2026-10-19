"""Comprehensive tests for Classical VIKOR and Fuzzy VIKOR compromise ranking engines.

Verifies:
1. Classical VIKOR against hand-calculated analytical benchmarks.
2. Fuzzy VIKOR (Opricovic, 2011) directly operating on Triangular Fuzzy Numbers.
3. Benefit vs. cost criteria mechanics (lower cost improves compromise ranking).
4. Compromise acceptance conditions (C1 Acceptable Advantage, C2 Acceptable Stability).
5. Compromise parameter v sensitivity (majority utility vs. individual regret veto).
6. Input validation (dimensions, negative weights, invalid v).
7. Degenerate cases (tied alternatives, single alternative, identical columns).
8. Regression reproducibility tests.
9. Backwards-compatible 5-tuple unpacking.
"""

import math
import numpy as np
import pytest

from core.vikor import (
    VikorResult,
    classical_vikor,
    evaluate_compromise_conditions,
    fuzzy_vikor,
    validate_vikor_inputs,
    vikor,
)


# ==============================================================================
# 1. Hand-Calculated Step-by-Step Analytical Benchmarks
# ==============================================================================

def test_classical_vikor_hand_calculated_benchmark():
    """Verify Classical VIKOR against exact step-by-step hand calculation.

    Problem Specification:
    ----------------------
    2 Alternatives: A1, A2
    2 Criteria: C1 (Benefit, weight=0.6), C2 (Cost, weight=0.4)
    Compromise Parameter: v = 0.5

    Matrix:
    - A1: C1 = 3.0, C2 = 2.0 (Low cost)
    - A2: C1 = 5.0, C2 = 8.0 (High cost)

    Analytical Derivation:
    1. Extremes:
       - C1 (Benefit): f1* = 5.0, f1- = 3.0 (Span = 2.0)
       - C2 (Cost):    f2* = 2.0, f2- = 8.0 (Span = 6.0)

    2. Normalized Weighted Regrets:
       - A1: C1 = 0.6 * (|5 - 3| / 2) = 0.6 * 1.0 = 0.6
             C2 = 0.4 * (|2 - 2| / 6) = 0.4 * 0.0 = 0.0
             S1 = 0.6 + 0.0 = 0.6
             R1 = max(0.6, 0.0) = 0.6
       - A2: C1 = 0.6 * (|5 - 5| / 2) = 0.6 * 0.0 = 0.0
             C2 = 0.4 * (|2 - 8| / 6) = 0.4 * 1.0 = 0.4
             S2 = 0.0 + 0.4 = 0.4
             R2 = max(0.0, 0.4) = 0.4

    3. Extreme Utility and Regret:
       - S* = min(0.6, 0.4) = 0.4, S- = max(0.6, 0.4) = 0.6 (Span = 0.2)
       - R* = min(0.6, 0.4) = 0.4, R- = max(0.6, 0.4) = 0.6 (Span = 0.2)

    4. VIKOR Index Q (v=0.5):
       - Q1 = 0.5 * ((0.6 - 0.4) / 0.2) + 0.5 * ((0.6 - 0.4) / 0.2) = 0.5 * 1 + 0.5 * 1 = 1.0
       - Q2 = 0.5 * ((0.4 - 0.4) / 0.2) + 0.5 * ((0.4 - 0.4) / 0.2) = 0.0

    5. Ranking:
       - A2: Q2 = 0.0 -> Rank 1 (best compromise)
       - A1: Q1 = 1.0 -> Rank 2

    6. Compromise Solution Conditions:
       - DQ = 1 / (2 - 1) = 1.0
       - Advantage: Q(A1) - Q(A2) = 1.0 - 0.0 = 1.0 >= DQ -> C1 Satisfied.
       - Stability: A2 is minimal in S (0.4) and R (0.4) -> C2 Satisfied.
       - Unique Stable Compromise Solution: A2.
    """
    matrix = np.array([
        [3.0, 2.0],  # A1
        [5.0, 8.0],  # A2
    ])
    weights = [0.6, 0.4]
    is_benefit = [True, False]

    res = classical_vikor(matrix, weights, is_benefit, v=0.5)

    assert pytest.approx(res.s_values[0], 1e-4) == 0.6
    assert pytest.approx(res.s_values[1], 1e-4) == 0.4
    assert pytest.approx(res.r_values[0], 1e-4) == 0.6
    assert pytest.approx(res.r_values[1], 1e-4) == 0.4

    assert pytest.approx(res.q_values[0], 1e-4) == 1.0
    assert pytest.approx(res.q_values[1], 1e-4) == 0.0

    assert res.rankings[1] == 1  # A2 is Rank 1
    assert res.rankings[0] == 2  # A1 is Rank 2

    val = res.validation
    assert pytest.approx(val["dq"], 1e-4) == 1.0
    assert pytest.approx(val["advantage"], 1e-4) == 1.0
    assert val["c1_acceptable_advantage"] is True
    assert val["c2_acceptable_stability"] is True
    assert val["compromise_stable"] is True
    assert val["compromise_set_indices"] == [1]


def test_fuzzy_vikor_hand_calculated_benchmark():
    """Verify Fuzzy VIKOR directly operating on Triangular Fuzzy Numbers."""
    matrix = [
        [(0.2, 0.4, 0.6), (0.1, 0.2, 0.3)],  # A1
        [(0.6, 0.8, 1.0), (0.4, 0.6, 0.8)],  # A2
    ]
    weights = [0.6, 0.4]
    is_benefit = [True, False]

    res = fuzzy_vikor(matrix, weights, is_benefit, v=0.5)

    assert pytest.approx(res.s_values[0], 1e-4) == 0.6
    assert pytest.approx(res.s_values[1], 1e-4) == 0.4
    assert pytest.approx(res.r_values[0], 1e-4) == 0.6
    assert pytest.approx(res.r_values[1], 1e-4) == 0.4

    assert pytest.approx(res.q_values[0], 1e-4) == 1.0
    assert pytest.approx(res.q_values[1], 1e-4) == 0.0

    assert res.rankings[1] == 1  # A2 is Rank 1
    assert res.rankings[0] == 2  # A1 is Rank 2


# ==============================================================================
# 2. Benefit and Cost Criteria Behavior
# ==============================================================================

def test_cost_criterion_lower_cost_improves_vikor_rank():
    """Verify that an alternative with lower repair effort achieves superior compromise rank."""
    # A1: High Benefit, High Cost (Effort = 8.0)
    # A2: High Benefit, Low Cost (Effort = 1.0)
    matrix = np.array([
        [5.0, 8.0],  # A1
        [5.0, 1.0],  # A2: superior due to low cost
    ])
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = classical_vikor(matrix, weights, is_benefit, v=0.5)
    assert res.rankings[1] == 1
    assert res.rankings[0] == 2
    assert res.q_values[1] < res.q_values[0]


def test_unified_vikor_with_defuzzification():
    """Verify unified vikor() with explicit Centroid defuzzification."""
    fuzzy_matrix = [
        [(0.5, 0.7, 0.9), (0.5, 0.7, 0.9)],
        [(0.5, 0.7, 0.9), (0.1, 0.2, 0.3)],
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = vikor(fuzzy_matrix, weights, is_benefit, v=0.5, formulation="classical")
    assert res.defuzzification_applied is True
    assert res.rankings[1] == 1
    assert res.q_values[1] < res.q_values[0]


# ==============================================================================
# 3. Compromise Solution Acceptance Conditions (C1 & C2)
# ==============================================================================

def test_compromise_cluster_when_advantage_fails():
    """When the best alternative does not have sufficient gap (Adv < DQ), a cluster is formed."""
    # 4 alternatives -> DQ = 1 / (4 - 1) = 0.3333
    # Alternatives with very close scores
    q = np.array([0.0, 0.1, 0.2, 0.9])
    s = np.array([0.2, 0.3, 0.4, 0.8])
    r = np.array([0.1, 0.15, 0.2, 0.5])
    names = ["A1", "A2", "A3", "A4"]

    val = evaluate_compromise_conditions(q, s, r, alternative_names=names)
    assert pytest.approx(val["dq"], 1e-3) == 1 / 3
    assert pytest.approx(val["advantage"], 1e-3) == 0.1
    assert val["c1_acceptable_advantage"] is False
    assert val["c2_acceptable_stability"] is True
    assert val["compromise_stable"] is False
    # Compromise cluster should include A1, A2, and A3 because Q - Q(A1) < DQ
    assert val["compromise_set_names"] == ["A1", "A2", "A3"]


def test_compromise_set_when_stability_fails():
    """When best by Q is NOT best by S or R, both A1 and A2 are proposed."""
    # 3 alternatives -> DQ = 1 / (3 - 1) = 0.5
    # A1 has best Q (0.0), A2 has Q=0.6. Advantage = 0.6 >= DQ (C1 is True)
    # But A2 has lowest S and lowest R (C2 is False)
    q = np.array([0.0, 0.6, 0.9])
    s = np.array([0.7, 0.2, 0.8])  # A2 is best by S
    r = np.array([0.6, 0.3, 0.7])  # A2 is best by R
    names = ["A1", "A2", "A3"]

    val = evaluate_compromise_conditions(q, s, r, alternative_names=names)
    assert pytest.approx(val["dq"], 1e-4) == 0.5
    assert pytest.approx(val["advantage"], 1e-4) == 0.6
    assert val["c1_acceptable_advantage"] is True
    assert val["c2_acceptable_stability"] is False
    assert val["compromise_stable"] is False
    assert val["compromise_set_names"] == ["A1", "A2"]


# ==============================================================================
# 4. Compromise Parameter v Sensitivity
# ==============================================================================

def test_v_parameter_sensitivity():
    """Testing shift from majority consensus (v=1.0) to individual regret veto (v=0.0).

    A1: Better group utility S (low sum of regrets), but higher worst-case regret R.
    A2: Worse group utility S, but lower worst-case regret R (balanced across all criteria).
    A3: Poor baseline alternative to provide span.
    """
    matrix = np.array([
        [10.0, 10.0, 2.0],  # A1: high C1 & C2, poor C3 (spiked regret on C3)
        [6.0, 6.0, 6.0],    # A2: balanced across all criteria (low max regret)
        [0.0, 0.0, 0.0],    # A3: worst baseline
    ])
    weights = [1 / 3, 1 / 3, 1 / 3]
    is_benefit = [True, True, True]

    # When v = 1.0 (pure group utility S), A1 should win because sum of regrets is lower
    res_majority = classical_vikor(matrix, weights, is_benefit, v=1.0)
    assert res_majority.rankings[0] == 1  # A1 is Rank 1
    assert res_majority.rankings[1] == 2  # A2 is Rank 2

    # When v = 0.0 (pure individual regret R), A2 should win because max regret is lower
    res_veto = classical_vikor(matrix, weights, is_benefit, v=0.0)
    assert res_veto.rankings[1] == 1  # A2 is Rank 1
    assert res_veto.rankings[0] == 2  # A1 is Rank 2

    # S and R values must be strictly identical regardless of v
    assert np.allclose(res_majority.s_values, res_veto.s_values)
    assert np.allclose(res_majority.r_values, res_veto.r_values)

    # But Q values must shift
    assert not np.allclose(res_majority.q_values, res_veto.q_values)


# ==============================================================================
# 5. Input Validation & Edge Cases
# ==============================================================================

def test_validate_vikor_invalid_inputs():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_vikor_inputs([0.5], [True, True], 2)

    with pytest.raises(ValueError, match="v must be in"):
        validate_vikor_inputs([0.5, 0.5], [True, True], 2, v=1.5)

    with pytest.raises(ValueError, match="must be non-negative"):
        validate_vikor_inputs([-0.1, 0.5], [True, True], 2)

    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_vikor_inputs([0.0, 0.0], [True, True], 2)


def test_vikor_tied_alternatives():
    """Identical alternatives must produce identical Q scores without division-by-zero."""
    matrix = np.array([
        [4.0, 4.0],
        [4.0, 4.0],
    ])
    weights = [0.5, 0.5]
    is_benefit = [True, True]

    res = classical_vikor(matrix, weights, is_benefit)
    assert res.q_values[0] == 0.0
    assert res.q_values[1] == 0.0
    assert not np.isnan(res.q_values).any()


def test_vikor_single_alternative():
    """Single alternative evaluates cleanly."""
    matrix = np.array([[3.0, 4.0]])
    weights = [0.5, 0.5]
    is_benefit = [True, True]

    res = classical_vikor(matrix, weights, is_benefit)
    assert len(res.rankings) == 1
    assert res.rankings[0] == 1


# ==============================================================================
# 6. Regression & Reproducibility Tests
# ==============================================================================

def test_vikor_reproducibility():
    """Verify that multiple consecutive runs with identical inputs produce identical arrays."""
    matrix = np.array([
        [4.2, 1.8, 9.0],
        [2.1, 4.5, 3.2],
        [5.0, 2.0, 8.5],
    ])
    weights = [0.4, 0.3, 0.3]
    is_benefit = [True, False, True]

    res1 = classical_vikor(matrix, weights, is_benefit, v=0.5)
    res2 = classical_vikor(matrix, weights, is_benefit, v=0.5)

    assert np.array_equal(res1.q_values, res2.q_values)
    assert np.array_equal(res1.rankings, res2.rankings)
    assert np.array_equal(res1.s_values, res2.s_values)
    assert np.array_equal(res1.r_values, res2.r_values)


def test_vikor_tuple_unpacking():
    """Verify backwards-compatible 5-tuple unpacking."""
    matrix = np.array([
        [5.0, 5.0],
        [3.0, 3.0],
    ])
    q, ranks, s, r, val = vikor(matrix, [0.5, 0.5], [True, True])
    assert isinstance(q, np.ndarray)
    assert isinstance(ranks, np.ndarray)
    assert isinstance(s, np.ndarray)
    assert isinstance(r, np.ndarray)
    assert isinstance(val, dict)
    assert ranks[0] == 1
    assert ranks[1] == 2
