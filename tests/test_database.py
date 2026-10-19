"""Tests for SQLite database operations, schema migrations, and filtering."""

import tempfile
from pathlib import Path
import pytest
from core.database import (
    init_db,
    seed_if_empty,
    load_all_issues,
    upsert_issue,
    delete_issue,
    get_stats,
    get_existing_ids,
    filter_issues,
    get_registered_websites,
    upsert_registered_website,
    delete_registered_website,
    get_registered_pages,
    upsert_registered_page,
    delete_registered_page,
)
from core.models import (
    AccessibilityIssue,
    RegisteredWebsite,
    RegisteredPage,
    RecordType,
    IssueStatus,
    VerificationStatus,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


def test_init_and_seed_db(temp_db):
    init_db(temp_db)
    seeded = seed_if_empty(temp_db)
    assert seeded > 0

    df = load_all_issues(temp_db)
    assert len(df) == seeded
    assert "issue_id" in df.columns
    assert "website" in df.columns
    assert "record_type" in df.columns
    assert "verification_status" in df.columns
    assert "wcag_criterion" in df.columns
    assert "essential_task" in df.columns
    assert "notes" in df.columns

    stats = get_stats(temp_db)
    assert stats["total_issues"] == seeded
    assert stats["total_websites"] >= 5

    ids = get_existing_ids(temp_db)
    assert len(ids) == seeded
    assert "ISS-001" in ids


def test_registered_websites_and_pages_crud(temp_db):
    init_db(temp_db)

    # 1. Register Website
    site = RegisteredWebsite(
        website_id="WEB-99",
        name="State Parks & Recreation",
        base_url="https://parks.state.gov",
        institution_type="State Civic Agency",
        description="Public recreation facilities",
        contact_email="access@parks.state.gov",
    )
    upsert_registered_website(site, temp_db)

    sites_df = get_registered_websites(temp_db)
    assert len(sites_df) == 1
    assert sites_df.iloc[0]["name"] == "State Parks & Recreation"

    # 2. Register Page
    page = RegisteredPage(
        page_id="PG-99",
        website_name="State Parks & Recreation",
        page_name="Campsite Reservation Portal",
        page_url="https://parks.state.gov/reserve",
        essential_task="Reserving accessible campsite stall",
        is_critical_journey=True,
        notes="High summer traffic",
    )
    upsert_registered_page(page, temp_db)

    pages_df = get_registered_pages(website_name="State Parks & Recreation", db_path=temp_db)
    assert len(pages_df) == 1
    assert pages_df.iloc[0]["page_name"] == "Campsite Reservation Portal"
    assert pages_df.iloc[0]["is_critical_journey"] == 1

    # 3. Delete Page and Website
    assert delete_registered_page("PG-99", temp_db) is True
    assert len(get_registered_pages(website_name="State Parks & Recreation", db_path=temp_db)) == 0

    assert delete_registered_website("WEB-99", temp_db) is True
    assert len(get_registered_websites(temp_db)) == 0


def test_upsert_and_delete_issue(temp_db):
    init_db(temp_db)
    issue = AccessibilityIssue(
        issue_id="TEMP-999",
        website="State Library",
        page_url="https://state.lib/home",
        page_name="Library Homepage",
        issue_type="Low color contrast",
        description="Contrast is low",
        severity="Medium",
        essential_task_impact="Low",
        affected_pages=1,
        user_impact="Medium",
        repair_effort="Very Low",
        frequency="Rare",
        evidence="#header contrast 2.1:1",
        recommended_fix="Update color",
        status="Open",
        record_type=RecordType.MANUAL.value,
        verification_status=VerificationStatus.HUMAN_VERIFIED.value,
        tool_name="Manual Keyboard Audit",
        wcag_criterion="WCAG 1.4.3 Contrast (Minimum) (Level AA)",
        wcag_verified=1,
        essential_task="Reading homepage announcements",
        notes="Audit finding verified by analyst",
    )
    upsert_issue(issue, temp_db)
    df = load_all_issues(temp_db)
    assert len(df) == 1
    assert df.iloc[0]["issue_id"] == "TEMP-999"
    assert df.iloc[0]["record_type"] == RecordType.MANUAL.value
    assert df.iloc[0]["verification_status"] == VerificationStatus.HUMAN_VERIFIED.value
    assert df.iloc[0]["essential_task"] == "Reading homepage announcements"

    # Update status to Needs Review
    issue.status = IssueStatus.NEEDS_REVIEW.value
    issue.notes = "Needs team leader review"
    upsert_issue(issue, temp_db)
    df_updated = load_all_issues(temp_db)
    assert df_updated.iloc[0]["status"] == IssueStatus.NEEDS_REVIEW.value
    assert df_updated.iloc[0]["notes"] == "Needs team leader review"

    # Delete
    deleted = delete_issue("TEMP-999", temp_db)
    assert deleted is True
    df_empty = load_all_issues(temp_db)
    assert len(df_empty) == 0


def test_filter_issues(temp_db):
    init_db(temp_db)
    seed_if_empty(temp_db)
    df = load_all_issues(temp_db)

    # Filter by website
    transit_df = filter_issues(df, websites=["Metro Transit Authority"])
    assert len(transit_df) > 0
    assert (transit_df["website"] == "Metro Transit Authority").all()

    # Filter by severity and status
    high_open = filter_issues(df, severities=["Very High"], statuses=["Open"])
    assert len(high_open) > 0
    assert (high_open["severity"] == "Very High").all()
    assert (high_open["status"] == "Open").all()

    # Search query
    searched = filter_issues(df, search_query="modal")
    assert len(searched) > 0
    assert any("modal" in row["evidence"].lower() or "modal" in row["description"].lower() for _, row in searched.iterrows())
