"""Tests for validation routines in core/validator.py."""

import pytest
from core.validator import (
    validate_url,
    validate_issue_fields,
    validate_csv_upload,
)


def test_validate_url_valid():
    assert validate_url("https://example.gov") is True
    assert validate_url("http://transit.metrocity.gov/pass-application") is True
    assert validate_url("https://sub.domain.edu:8080/path?query=1") is True


def test_validate_url_invalid():
    assert validate_url("ftp://example.com") is False
    assert validate_url("htp://broken.com") is False
    assert validate_url("not a url") is False
    assert validate_url("") is False
    assert validate_url("https://") is False
    assert validate_url("http:// spaced url.com") is False


def test_validate_issue_missing_fields():
    data = {
        "issue_id": "TEST-01",
        # missing website, page_url, etc.
    }
    errors = validate_issue_fields(data)
    assert len(errors) > 0
    assert any("website" in e for e in errors)
    assert any("page_url" in e for e in errors)


def test_validate_issue_invalid_ratings():
    data = {
        "issue_id": "TEST-01",
        "website": "City Portal",
        "page_url": "https://city.gov/portal",
        "page_name": "Portal Home",
        "issue_type": "Low color contrast",
        "description": "Low contrast on header text",
        "severity": "Ultra High",  # Invalid
        "essential_task_impact": "Severe",  # Invalid
        "affected_pages": -2,  # Invalid
        "user_impact": "Extreme",  # Invalid
        "repair_effort": "Crazy High",  # Invalid
        "frequency": "All The Time",  # Invalid
        "evidence": "<span>contrast 1:1</span>",
        "recommended_fix": "Increase contrast",
        "status": "Archived",  # Invalid
    }
    errors = validate_issue_fields(data)
    assert any("severity" in e for e in errors)
    assert any("essential_task_impact" in e for e in errors)
    assert any("affected_pages" in e for e in errors)
    assert any("repair_effort" in e for e in errors)
    assert any("status" in e for e in errors)


def test_validate_issue_duplicate_id():
    data = {
        "issue_id": "DUP-001",
        "website": "City Portal",
        "page_url": "https://city.gov/portal",
        "page_name": "Portal Home",
        "issue_type": "Low color contrast",
        "description": "Low contrast on header text",
        "severity": "Medium",
        "essential_task_impact": "High",
        "affected_pages": 1,
        "user_impact": "Medium",
        "repair_effort": "Low",
        "frequency": "Frequent",
        "evidence": "<span>contrast 1:1</span>",
        "recommended_fix": "Increase contrast",
        "status": "Open",
    }
    existing = {"DUP-001", "ISS-001"}
    # As new issue: fails duplicate check
    errs_new = validate_issue_fields(data, existing_ids=existing, is_edit=False)
    assert any("Duplicate Issue ID" in e for e in errs_new)

    # In edit mode: duplicate check is skipped for same ID
    errs_edit = validate_issue_fields(data, existing_ids=existing, is_edit=True)
    assert not any("Duplicate Issue ID" in e for e in errs_edit)


def test_validate_csv_upload_empty():
    valid, invalid, global_errs = validate_csv_upload(b"")
    assert len(global_errs) == 1
    assert "Empty upload" in global_errs[0]

    valid, invalid, global_errs = validate_csv_upload("   \n\n  ")
    assert len(global_errs) == 1
    assert "Empty upload" in global_errs[0]


def test_validate_csv_upload_missing_columns():
    csv_str = "issue_id,website\nISS-1,Example Site"
    valid, invalid, global_errs = validate_csv_upload(csv_str)
    assert len(global_errs) == 1
    assert "Missing required CSV columns" in global_errs[0]
