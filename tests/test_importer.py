"""Tests for CSV import and export logic."""

import tempfile
from pathlib import Path
import pytest

from core.database import init_db, load_all_issues
from core.importer import import_csv_data
from core.exporter import export_to_csv
from core.models import RecordType


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_import.db"
        init_db(db_path)
        yield db_path


def test_import_valid_csv(temp_db):
    csv_content = """issue_id,website,page_url,page_name,issue_type,description,severity,essential_task_impact,affected_pages,user_impact,repair_effort,frequency,evidence,recommended_fix,status
AUDIT-001,Civic Library,https://lib.gov/search,Search Page,Missing accessible names or form labels,Search field has no label,High,High,1,High,Low,Frequent,<input type="search">,Add accessible label,Open
AUDIT-002,Civic Library,https://lib.gov/login,Login Page,Keyboard navigation barriers,Focus trap in login modal,Very High,Very High,1,Very High,Medium,Frequent,<div class="modal">,Fix tab order and escape,In Progress
"""
    result = import_csv_data(csv_content, db_path=str(temp_db), record_type_override=RecordType.IMPORTED.value)
    assert result["success"] is True
    assert result["imported_count"] == 2
    assert result["invalid_count"] == 0

    df = load_all_issues(temp_db)
    assert len(df) == 2
    assert df.iloc[0]["record_type"] == RecordType.IMPORTED.value


def test_import_duplicate_handling(temp_db):
    csv_content = """issue_id,website,page_url,page_name,issue_type,description,severity,essential_task_impact,affected_pages,user_impact,repair_effort,frequency,evidence,recommended_fix,status
DUP-101,Civic Library,https://lib.gov/search,Search Page,Low color contrast,Contrast below 4.5:1,Medium,Medium,1,Medium,Low,Frequent,color #ccc,Use #333,Open
"""
    # First import
    res1 = import_csv_data(csv_content, db_path=str(temp_db))
    assert res1["imported_count"] == 1

    # Second import without overwrite -> should skip
    res2 = import_csv_data(csv_content, overwrite_existing=False, db_path=str(temp_db))
    assert res2["imported_count"] == 0
    assert res2["skipped_count"] == 1

    # Third import with overwrite -> should update
    res3 = import_csv_data(csv_content, overwrite_existing=True, db_path=str(temp_db))
    assert res3["imported_count"] == 1
    assert res3["skipped_count"] == 0


def test_export_to_csv_roundtrip(temp_db):
    csv_content = """issue_id,website,page_url,page_name,issue_type,description,severity,essential_task_impact,affected_pages,user_impact,repair_effort,frequency,evidence,recommended_fix,status
EXP-001,Civic Transit,https://transit.gov/bus,Bus Schedule,Unclear links,Read more link text,Low,Low,2,Low,Very Low,Systematic,<a href="#">Read more</a>,Provide descriptive link,Resolved
"""
    import_csv_data(csv_content, db_path=str(temp_db))
    df = load_all_issues(temp_db)
    exported = export_to_csv(df)

    assert "EXP-001" in exported
    assert "Civic Transit" in exported
    assert "Resolved" in exported
