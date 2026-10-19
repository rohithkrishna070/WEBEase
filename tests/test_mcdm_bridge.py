"""Integration tests for the MCDM orchestration bridge."""

import pandas as pd
import pytest
from core.database import load_all_issues, seed_if_empty
from core.mcdm_bridge import run_prioritization
from core.models import CriterionWeights


@pytest.fixture
def sample_df():
    seed_if_empty()
    df = load_all_issues()
    return df


def test_prioritization_pipeline(sample_df):
    assert len(sample_df) >= 30

    res, meta = run_prioritization(sample_df)

    assert not res.empty
    assert "topsis_rank" in res.columns
    assert "vikor_rank" in res.columns
    assert "consensus_rank" in res.columns
    assert "topsis_score" in res.columns
    assert "vikor_q" in res.columns

    # Check rank boundaries
    assert res["topsis_rank"].min() == 1
    assert res["topsis_rank"].max() == len(sample_df)
    assert res["vikor_rank"].min() == 1
    assert res["vikor_rank"].max() == len(sample_df)

    # Check correlation metrics
    assert "spearman_rho" in meta
    assert -1.0 <= meta["spearman_rho"] <= 1.0
    assert "kendall_tau" in meta


def test_custom_weights(sample_df):
    weights = CriterionWeights(
        c1_severity=0.1,
        c2_task_impact=0.6,
        c3_affected_pages=0.1,
        c4_user_impact=0.1,
        c5_repair_effort=0.05,
        c6_frequency=0.05,
    )
    res, meta = run_prioritization(sample_df, weights=weights)
    assert not res.empty
    assert meta["weights"][1] == pytest.approx(0.6, abs=1e-3)
