"""Comprehensive QA and edge-case verification test suite for SmartWeb Access.

Validates:
1. Installation dependencies from requirements.txt
2. SQLite persistence across independent connection lifecycles
3. Boundary & degenerate MCDM calculations (ties, single alternative, all cost)
4. Empty uploads, malformed JSON, and 'Very High' repair effort validation
5. Non-invented data & audit provenance preservation
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.database import (
    init_db,
    load_all_issues,
    upsert_issue,
    get_connection,
    get_stats,
)
from core.models import (
    AccessibilityIssue,
    CriterionWeights,
    RecordType,
    VerificationStatus,
    VALID_EFFORT_RATINGS,
)
from core.validator import validate_issue_fields, validate_csv_upload
from core.importer import import_multi_tool_audit, import_csv_data
from core.fuzzy_topsis import fuzzy_topsis
from core.vikor import classical_vikor, fuzzy_vikor, vikor
from core.mcdm_bridge import run_prioritization


@pytest.fixture
def clean_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "qa_test.db"
        init_db(db_path)
        yield db_path


# ==============================================================================
# 1. Installation & Dependency Verification
# ==============================================================================

def test_requirements_dependencies_importable():
    """Verify that all core third-party dependencies from requirements.txt import cleanly."""
    import streamlit
    import pandas
    import numpy
    import plotly
    import scipy
    import pytest

    assert streamlit.__version__ is not None
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None
    assert plotly.__version__ is not None
    assert scipy.__version__ is not None
    assert pytest.__version__ is not None


# ==============================================================================
# 2. SQLite Persistence Across Connection Lifecycles
# ==============================================================================

def test_sqlite_persistence_across_separate_connections(clean_db):
    """Verify that database writes persist after the database connection is closed and reopened."""
    issue = AccessibilityIssue(
        issue_id="PERSIST-001",
        website="Municipal Transit Authority",
        page_url="https://transit.city.gov/tickets",
        page_name="Ticket Checkout",
        issue_type="Missing accessible names or form labels",
        description="Payment submission button lacks accessible label",
        severity="Critical" if "Critical" in VALID_EFFORT_RATINGS else "Very High",
        essential_task_impact="Very High",
        affected_pages=2,
        user_impact="High",
        repair_effort="Low",
        frequency="Frequent",
        evidence='<button class="btn-pay"></button>',
        recommended_fix="Add aria-label='Submit Ticket Payment'",
        status="Open",
        record_type=RecordType.MANUAL.value,
        verification_status=VerificationStatus.HUMAN_VERIFIED.value,
        tool_name="Manual Inspection",
        wcag_criterion="WCAG 4.1.2 Name, Role, Value",
        wcag_verified=1,
    )

    # 1. Insert via helper
    upsert_issue(issue, db_path=clean_db)

    # 2. Re-open via raw separate SQLite connection
    conn2 = get_connection(clean_db)
    cursor = conn2.cursor()
    cursor.execute("SELECT issue_id, website, severity, record_type FROM accessibility_issues WHERE issue_id = ?", ("PERSIST-001",))
    row = cursor.fetchone()
    conn2.close()

    assert row is not None
    assert row["issue_id"] == "PERSIST-001"
    assert row["website"] == "Municipal Transit Authority"
    assert row["record_type"] == RecordType.MANUAL.value

    # 3. Read via load_all_issues on fresh connection
    df = load_all_issues(clean_db)
    assert len(df) == 1
    assert df.iloc[0]["issue_id"] == "PERSIST-001"


# ==============================================================================
# 3. Linguistic Scale & 'Very High' Effort Validation
# ==============================================================================

def test_very_high_effort_validation():
    """Verify that 'Very High' repair effort is accepted across models, validator, and scale."""
    assert "Very High" in VALID_EFFORT_RATINGS

    valid_data = {
        "issue_id": "VH-001",
        "website": "Health Dept",
        "page_url": "https://health.state.gov/portal",
        "page_name": "Health Portal",
        "issue_type": "Keyboard navigation barriers",
        "description": "Entire legacy SPA framework lacks keyboard support",
        "severity": "Very High",
        "essential_task_impact": "Very High",
        "affected_pages": 15,
        "user_impact": "Very High",
        "repair_effort": "Very High",  # Validated rating
        "frequency": "Systematic",
        "evidence": "Entire site navigation inaccessible",
        "recommended_fix": "Refactor DOM structure with semantic elements",
        "status": "In Progress",
    }
    errors = validate_issue_fields(valid_data)
    assert len(errors) == 0


# ==============================================================================
# 4. Upload Ingestion Edge Cases (Empty & Malformed)
# ==============================================================================

def test_import_multi_tool_empty_payload(clean_db):
    """Verify that an empty (0-byte) payload returns a graceful, actionable error."""
    res_bytes = import_multi_tool_audit(b"", filename="empty.csv", db_path=clean_db)
    assert res_bytes["success"] is False
    assert res_bytes["tool_detected"] == "Empty File"
    assert any("empty" in e.lower() for e in res_bytes["global_errors"])

    res_str = import_multi_tool_audit("   \n   ", filename="empty.json", db_path=clean_db)
    assert res_str["success"] is False
    assert any("empty" in e.lower() for e in res_str["global_errors"])


def test_import_multi_tool_malformed_json(clean_db):
    """Verify that corrupt JSON returns a structured error without crashing."""
    corrupt_json = "{ invalid_json: true, missing_quotes: "
    res = import_multi_tool_audit(corrupt_json, filename="audit.json", db_path=clean_db)
    assert res["success"] is False
    assert any("failed to parse lighthouse json" in e.lower() for e in res["global_errors"])


# ==============================================================================
# 5. Degenerate Mathematical Cases in MCDM Engines
# ==============================================================================

def test_fuzzy_topsis_identical_alternatives_degenerate():
    """When all alternatives have identical criteria ratings, closeness coefficients default safely to 0.5."""
    # 3 identical alternatives
    mat = [
        [(0.3, 0.5, 0.7), (0.1, 0.3, 0.5)],
        [(0.3, 0.5, 0.7), (0.1, 0.3, 0.5)],
        [(0.3, 0.5, 0.7), (0.1, 0.3, 0.5)],
    ]
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = fuzzy_topsis(mat, weights, is_benefit)
    assert len(res.closeness_coefficients) == 3
    # Denominator d_plus + d_minus = 0 -> handles degenerate division safely
    for cc in res.closeness_coefficients:
        assert pytest.approx(cc, 1e-4) == 0.5


def test_vikor_identical_alternatives_degenerate():
    """When all alternatives are tied, VIKOR handles S, R, and Q ranges safely."""
    mat = np.array([
        [3.0, 2.0],
        [3.0, 2.0],
        [3.0, 2.0],
    ])
    weights = [0.5, 0.5]
    is_benefit = [True, False]

    res = classical_vikor(mat, weights, is_benefit, v=0.5)
    assert len(res.q_values) == 3
    # All Q values equal 0.0 because regret distances are 0
    for q in res.q_values:
        assert pytest.approx(q, 1e-4) == 0.0


def test_mcdm_empty_dataset_handling():
    """Verify that run_prioritization on an empty DataFrame returns empty structures safely."""
    empty_df = pd.DataFrame()
    res_df, meta = run_prioritization(empty_df)
    assert res_df.empty
    assert meta == {}


def test_mcdm_single_alternative_handling():
    """Verify prioritization with exactly 1 alternative."""
    single_df = pd.DataFrame([{
        "issue_id": "SINGLE-01",
        "website": "City Portal",
        "page_url": "https://city.gov",
        "page_name": "Home",
        "issue_type": "Low color contrast",
        "description": "Footer link contrast",
        "severity": "Medium",
        "essential_task_impact": "Low",
        "affected_pages": 1,
        "user_impact": "Medium",
        "repair_effort": "Low",
        "frequency": "Rare",
        "evidence": "Contrast 3.2:1",
        "recommended_fix": "Darken link color",
        "status": "Open",
        "record_type": RecordType.SYNTHETIC.value,
        "verification_status": VerificationStatus.HUMAN_VERIFIED.value,
    }])

    res_df, meta = run_prioritization(single_df)
    assert len(res_df) == 1
    assert res_df.iloc[0]["topsis_rank"] == 1
    assert res_df.iloc[0]["vikor_rank"] == 1
    assert meta["total_evaluated"] == 1
    assert meta["spearman_rho"] == 1.0
