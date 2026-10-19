"""Tests for Sensitivity Analysis, Weight Redistribution, and Multi-Scenario Robustness.

Verifies:
1. Proportional weight redistribution during single-criterion sweeps.
2. Dynamic weight sweep execution and correlation tracking against baseline.
3. Multi-scenario ranking stability calculations (Top-3 and Top-5 appearance frequencies).
4. Strategic organizational profile comparisons.
5. Mathematical reproducibility across repeated runs.
"""

import numpy as np
import pandas as pd
import pytest

from core.models import CriterionWeights
from core.sensitivity import (
    PRESET_WEIGHT_PROFILES,
    calculate_scenario_stability,
    compare_profiles,
    redistribute_weights_for_sweep,
    run_weight_sweep,
)


@pytest.fixture
def sample_test_df():
    """Create a minimal representative set of accessibility issues for sensitivity testing."""
    return pd.DataFrame([
        {
            "issue_id": "ISS-001",
            "website": "Portal A",
            "page_name": "Login",
            "page_url": "https://portal-a.gov/login",
            "issue_type": "Missing accessible names or form labels",
            "description": "Form input missing label",
            "severity": "Very High",
            "essential_task_impact": "Very High",
            "affected_pages": 4,
            "user_impact": "High",
            "repair_effort": "Very Low",
            "frequency": "Systematic",
            "evidence": "<input id='u'>",
            "recommended_fix": "Add label",
            "status": "Open",
            "record_type": "Synthetic Demonstration",
            "audit_method": "Synthetic Demonstration Benchmark",
            "verification_status": "Human-Verified Finding",
            "tool_name": "Manual Inspection",
            "wcag_criterion": "WCAG 4.1.2",
            "wcag_verified": 1,
            "essential_task": "Sign in",
        },
        {
            "issue_id": "ISS-002",
            "website": "Portal A",
            "page_name": "Checkout",
            "page_url": "https://portal-a.gov/pay",
            "issue_type": "Keyboard navigation barriers",
            "description": "Trapped focus in modal",
            "severity": "High",
            "essential_task_impact": "High",
            "affected_pages": 1,
            "user_impact": "Very High",
            "repair_effort": "Medium",
            "frequency": "Frequent",
            "evidence": "Modal trap",
            "recommended_fix": "Fix key listener",
            "status": "Open",
            "record_type": "Synthetic Demonstration",
            "audit_method": "Synthetic Demonstration Benchmark",
            "verification_status": "Human-Verified Finding",
            "tool_name": "Manual Inspection",
            "wcag_criterion": "WCAG 2.1.2",
            "wcag_verified": 1,
            "essential_task": "Pay fees",
        },
        {
            "issue_id": "ISS-003",
            "website": "Portal B",
            "page_name": "Home",
            "page_url": "https://portal-b.gov/",
            "issue_type": "Low color contrast",
            "description": "Footer link contrast 2.8:1",
            "severity": "Low",
            "essential_task_impact": "Low",
            "affected_pages": 10,
            "user_impact": "Low",
            "repair_effort": "Very Low",
            "frequency": "Rare",
            "evidence": "Color #888888",
            "recommended_fix": "Use #595959",
            "status": "Open",
            "record_type": "Synthetic Demonstration",
            "audit_method": "Synthetic Demonstration Benchmark",
            "verification_status": "Human-Verified Finding",
            "tool_name": "Manual Inspection",
            "wcag_criterion": "WCAG 1.4.3",
            "wcag_verified": 1,
            "essential_task": "Footer navigation",
        },
        {
            "issue_id": "ISS-004",
            "website": "Portal B",
            "page_name": "Search",
            "page_url": "https://portal-b.gov/search",
            "issue_type": "Heading hierarchy problems",
            "description": "Skipped h2 to h4",
            "severity": "Medium",
            "essential_task_impact": "Medium",
            "affected_pages": 2,
            "user_impact": "Medium",
            "repair_effort": "High",
            "frequency": "Occasional",
            "evidence": "Heading skip",
            "recommended_fix": "Fix hierarchy",
            "status": "Open",
            "record_type": "Synthetic Demonstration",
            "audit_method": "Synthetic Demonstration Benchmark",
            "verification_status": "Human-Verified Finding",
            "tool_name": "Manual Inspection",
            "wcag_criterion": "WCAG 1.3.1",
            "wcag_verified": 1,
            "essential_task": "Site search",
        },
    ])


# ==============================================================================
# 1. Weight Budget Redistribution Tests
# ==============================================================================

def test_redistribute_weights_sum_to_one():
    base = [0.25, 0.30, 0.10, 0.15, 0.10, 0.10]
    # Sweep C2 (index 1) to 0.50
    new_w = redistribute_weights_for_sweep(base, target_idx=1, target_value=0.50)

    assert len(new_w) == 6
    assert pytest.approx(sum(new_w), 1e-6) == 1.0
    assert pytest.approx(new_w[1], 1e-6) == 0.50
    assert all(w >= 0.0 for w in new_w)


def test_redistribute_weights_boundary_values():
    base = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

    # Extreme: 0.0 target weight
    w_zero = redistribute_weights_for_sweep(base, target_idx=0, target_value=0.0)
    assert pytest.approx(sum(w_zero), 1e-6) == 1.0
    assert w_zero[0] == 0.0

    # Extreme: 1.0 target weight
    w_one = redistribute_weights_for_sweep(base, target_idx=0, target_value=1.0)
    assert pytest.approx(sum(w_one), 1e-6) == 1.0
    assert pytest.approx(w_one[0], 1e-6) == 1.0
    assert all(pytest.approx(w_one[k], 1e-6) == 0.0 for k in range(1, 6))


# ==============================================================================
# 2. Dynamic Weight Sweep Tests
# ==============================================================================

def test_run_weight_sweep(sample_test_df):
    t_hist, v_hist, corr = run_weight_sweep(
        sample_test_df,
        target_criterion_idx=1,  # C2: Task Impact
        weight_steps=5,
        min_w=0.10,
        max_w=0.50,
    )

    assert not t_hist.empty
    assert not v_hist.empty
    assert not corr.empty
    assert len(t_hist) == 4  # 4 issues
    assert t_hist.shape[1] == 5  # 5 steps
    assert len(corr) == 5

    # Check column names
    assert "topsis_baseline_spearman" in corr.columns
    assert "vikor_baseline_spearman" in corr.columns
    assert "cross_method_spearman" in corr.columns


# ==============================================================================
# 3. Scenario Stability & Frequency Tests
# ==============================================================================

def test_calculate_scenario_stability(sample_test_df):
    stability_df = calculate_scenario_stability(sample_test_df)

    assert not stability_df.empty
    assert len(stability_df) == 4
    assert "top3_count" in stability_df.columns
    assert "top3_frequency_pct" in stability_df.columns
    assert "top5_count" in stability_df.columns
    assert "top5_frequency_pct" in stability_df.columns
    assert "avg_consensus_rank" in stability_df.columns
    assert "stability_class" in stability_df.columns

    # ISS-001 has Very High severity, Very High task impact, Very Low effort
    # It must have strong Top-3 appearance across the 5 preset scenarios
    iss1 = stability_df[stability_df["issue_id"] == "ISS-001"].iloc[0]
    assert iss1["top3_count"] >= 3
    assert iss1["top5_count"] == 5
    assert iss1["top5_frequency_pct"] == 100.0


# ==============================================================================
# 4. Strategic Profile Comparison Tests
# ==============================================================================

def test_compare_profiles(sample_test_df):
    comp_df = compare_profiles(sample_test_df)

    assert not comp_df.empty
    assert len(comp_df) == len(PRESET_WEIGHT_PROFILES)
    assert "Profile" in comp_df.columns
    assert "Top Priority Repair" in comp_df.columns
    assert "Top 3 Issues" in comp_df.columns
    assert "Spearman Rho (T vs V)" in comp_df.columns


# ==============================================================================
# 5. Reproducibility Tests
# ==============================================================================

def test_sensitivity_reproducibility(sample_test_df):
    """Ensure identical runs produce identical stability values."""
    s1 = calculate_scenario_stability(sample_test_df)
    s2 = calculate_scenario_stability(sample_test_df)

    assert np.array_equal(s1["top3_count"].values, s2["top3_count"].values)
    assert np.array_equal(s1["top5_count"].values, s2["top5_count"].values)
    assert np.array_equal(s1["avg_consensus_rank"].values, s2["avg_consensus_rank"].values)
