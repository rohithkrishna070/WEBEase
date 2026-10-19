"""Tests for core data models and criterion weights."""

import pytest
from core.models import (
    AccessibilityIssue,
    CriterionWeights,
    LINGUISTIC_BENEFIT_SCALE,
    IssueType,
    IssueStatus,
    RecordType,
)


def test_accessibility_issue_serialization():
    issue = AccessibilityIssue(
        issue_id="TEST-001",
        website="Test University",
        page_url="https://test.edu/apply",
        page_name="Application Form",
        issue_type=IssueType.MISSING_FORM_LABELS.value,
        description="Missing input label",
        severity="High",
        essential_task_impact="Very High",
        affected_pages=2,
        user_impact="High",
        repair_effort="Low",
        frequency="Frequent",
        evidence="<input id='name'> without label",
        recommended_fix="Add <label for='name'>Name</label>",
        status=IssueStatus.OPEN.value,
        record_type=RecordType.MANUAL.value,
        wcag_criterion="WCAG 4.1.2",
        essential_task="Submitting freshman application",
        notes="Urgent fix needed",
    )
    d = issue.to_dict()
    assert d["issue_id"] == "TEST-001"
    assert d["affected_pages"] == 2
    assert d["status"] == "Open"
    assert d["record_type"] == RecordType.MANUAL.value
    assert d["wcag_criterion"] == "WCAG 4.1.2"
    assert d["is_synthetic"] == 0

    # Test from_dict roundtrip
    rebuilt = AccessibilityIssue.from_dict(d)
    assert rebuilt.issue_id == issue.issue_id
    assert rebuilt.wcag_criterion == "WCAG 4.1.2"
    assert rebuilt.notes == "Urgent fix needed"


def test_statuses_and_legacy_compatibility():
    statuses = [s.value for s in IssueStatus]
    assert "Open" in statuses
    assert "In Progress" in statuses
    assert "Resolved" in statuses
    assert "Needs Review" in statuses

    # Legacy status "Verified" maps to "Resolved"
    legacy = AccessibilityIssue.from_dict({
        "issue_id": "LEG-1",
        "website": "Site",
        "page_url": "https://site.gov",
        "page_name": "Home",
        "issue_type": "Low color contrast",
        "description": "Desc",
        "severity": "Low",
        "essential_task_impact": "Low",
        "affected_pages": 1,
        "user_impact": "Low",
        "repair_effort": "Low",
        "frequency": "Rare",
        "evidence": "Evidence",
        "recommended_fix": "Fix",
        "status": "Verified",
    })
    assert legacy.status == IssueStatus.RESOLVED.value


def test_criterion_weights_normalization():
    w = CriterionWeights(0.2, 0.2, 0.2, 0.2, 0.1, 0.1)
    assert w.validate() is True

    # Test normalization when unnormalized
    w_unnorm = CriterionWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert w_unnorm.validate() is False
    normalized = w_unnorm.normalize()
    assert normalized.validate() is True
    assert pytest.approx(sum(normalized.as_list()), 1e-4) == 1.0


def test_linguistic_scale_integrity():
    for label, tfn in LINGUISTIC_BENEFIT_SCALE.items():
        assert len(tfn) == 3
        assert tfn[0] <= tfn[1] <= tfn[2]
