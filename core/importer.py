"""
================================================================================
MODULE: core.importer
================================================================================
1. WHAT THIS FILE DOES:
   Orchestrates batch importing of accessibility audit evidence into the SQLite database,
   supporting standard SmartWeb CSVs, Lighthouse JSON, Axe CSV, and WAVE CSV files.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Accessibility teams need to ingest audit files from multiple
     auditors and automated tools into a single central repository without manual retyping.
   - Technical & Architectural Rationale: Coordinates between `core.validator` (for schema
     checks), `core.tool_adapters` (for tool-specific parsing), and `core.database` (for
     persisting valid rows with transaction integrity).

3. HOW IT WORKS CONCEPTUALLY:
   - Takes file bytes or streams from Streamlit file upload widgets.
   - Identifies the audit tool and parses the file into standardized issue objects.
   - Evaluates whether issues should overwrite existing records or be skipped if duplicate IDs exist.
   - Records metadata (e.g. tool name, audit method, verification status).
   - Returns a structured status summary report (total rows, imported, skipped, errors).

4. HOW IT WORKS TECHNICALLY:
   - `import_csv_data()`: Handles standard SmartWeb schema CSVs using `validate_csv_upload()`.
   - `import_multi_tool_audit()`: Auto-detects format, runs the appropriate adapter,
     and persists records via `upsert_issue()`.
================================================================================
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from core.database import get_existing_ids, upsert_issue
from core.models import (
    AccessibilityIssue,
    RecordType,
    VerificationStatus,
)
from core.validator import validate_csv_upload
from core.tool_adapters import (
    detect_tool_format,
    parse_axe_csv,
    parse_wave_csv,
    parse_lighthouse_json,
)


def import_csv_data(
    file_bytes_or_str: bytes | str | io.BytesIO | io.StringIO,
    overwrite_existing: bool = False,
    record_type_override: Optional[str] = None,
    verification_status_override: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and ingest standard CSV accessibility issues into the database."""
    existing_ids = get_existing_ids(db_path)

    valid_rows, invalid_rows, global_errors = validate_csv_upload(
        file_bytes_or_str, existing_ids=None
    )

    if global_errors:
        return {
            "success": False,
            "tool_detected": "SMARTWEB_STANDARD",
            "total_rows_read": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "invalid_count": 0,
            "global_errors": global_errors,
            "invalid_rows": [],
        }

    imported_count = 0
    skipped_count = 0

    for row_data in valid_rows:
        issue_id = str(row_data["issue_id"]).strip()

        if issue_id in existing_ids and not overwrite_existing:
            skipped_count += 1
            continue

        if record_type_override:
            row_data["record_type"] = record_type_override
        if verification_status_override:
            row_data["verification_status"] = verification_status_override

        issue = AccessibilityIssue.from_dict(row_data)
        upsert_issue(issue, db_path=db_path)
        imported_count += 1

    return {
        "success": True,
        "tool_detected": "SMARTWEB_STANDARD",
        "total_rows_read": len(valid_rows) + len(invalid_rows),
        "imported_count": imported_count,
        "skipped_count": skipped_count,
        "invalid_count": len(invalid_rows),
        "global_errors": [],
        "invalid_rows": invalid_rows,
    }


def import_multi_tool_audit(
    file_bytes_or_str: bytes | str,
    filename: str = "",
    website_name: str = "Institutional Target",
    target_page_url: str = "https://example.gov",
    overwrite_existing: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect and import audit findings from Lighthouse, Axe, WAVE, or generic CSVs."""
    existing_ids = get_existing_ids(db_path)

    # 0. Check for empty payload
    if not file_bytes_or_str or (isinstance(file_bytes_or_str, (str, bytes, bytearray)) and len(file_bytes_or_str.strip() if isinstance(file_bytes_or_str, str) else file_bytes_or_str) == 0):
        return {
            "success": False,
            "tool_detected": "Empty File",
            "total_rows_read": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "invalid_count": 0,
            "global_errors": ["The uploaded file is empty (0 bytes). Please provide a valid CSV or JSON audit report."],
            "invalid_rows": [],
        }

    # 1. Check if JSON (Lighthouse JSON)
    is_json = filename.lower().endswith(".json") or (
        isinstance(file_bytes_or_str, (str, bytes)) and str(file_bytes_or_str).strip().startswith("{")
    )

    if is_json:
        try:
            issues = parse_lighthouse_json(file_bytes_or_str, website_name=website_name)
            imported = 0
            skipped = 0
            for iss in issues:
                if iss.issue_id in existing_ids and not overwrite_existing:
                    skipped += 1
                    continue
                upsert_issue(iss, db_path=db_path)
                imported += 1

            return {
                "success": True,
                "tool_detected": "Google Lighthouse (JSON)",
                "total_rows_read": len(issues),
                "imported_count": imported,
                "skipped_count": skipped,
                "invalid_count": 0,
                "global_errors": [],
                "invalid_rows": [],
            }
        except Exception as e:
            return {
                "success": False,
                "tool_detected": "Lighthouse JSON",
                "total_rows_read": 0,
                "imported_count": 0,
                "skipped_count": 0,
                "invalid_count": 0,
                "global_errors": [f"Failed to parse Lighthouse JSON report: {e}"],
                "invalid_rows": [],
            }

    # 2. Parse as CSV
    try:
        if isinstance(file_bytes_or_str, (bytes, bytearray)):
            stream = io.BytesIO(file_bytes_or_str)
        else:
            stream = io.StringIO(file_bytes_or_str)

        df = pd.read_csv(stream)
    except Exception as e:
        return {
            "success": False,
            "tool_detected": "Unknown",
            "total_rows_read": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "invalid_count": 0,
            "global_errors": [f"CSV parsing error: {e}"],
            "invalid_rows": [],
        }

    if df.empty:
        return {
            "success": False,
            "tool_detected": "Empty CSV",
            "total_rows_read": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "invalid_count": 0,
            "global_errors": ["The uploaded file contains headers but no data rows."],
            "invalid_rows": [],
        }

    tool_format = detect_tool_format(list(df.columns))

    if tool_format == "AXE_CORE":
        issues = parse_axe_csv(df, website_name=website_name, target_page_url=target_page_url)
        imported = 0
        skipped = 0
        for iss in issues:
            if iss.issue_id in existing_ids and not overwrite_existing:
                skipped += 1
                continue
            upsert_issue(iss, db_path=db_path)
            imported += 1

        return {
            "success": True,
            "tool_detected": "Axe-core / Axe DevTools (CSV)",
            "total_rows_read": len(issues),
            "imported_count": imported,
            "skipped_count": skipped,
            "invalid_count": 0,
            "global_errors": [],
            "invalid_rows": [],
        }

    elif tool_format == "WAVE":
        issues = parse_wave_csv(df, website_name=website_name, target_page_url=target_page_url)
        imported = 0
        skipped = 0
        for iss in issues:
            if iss.issue_id in existing_ids and not overwrite_existing:
                skipped += 1
                continue
            upsert_issue(iss, db_path=db_path)
            imported += 1

        return {
            "success": True,
            "tool_detected": "WAVE Evaluation Tool (CSV)",
            "total_rows_read": len(issues),
            "imported_count": imported,
            "skipped_count": skipped,
            "invalid_count": 0,
            "global_errors": [],
            "invalid_rows": [],
        }

    else:
        # Standard SmartWeb Access CSV
        res = import_csv_data(
            file_bytes_or_str,
            overwrite_existing=overwrite_existing,
            record_type_override=RecordType.IMPORTED.value,
            verification_status_override=VerificationStatus.HUMAN_VERIFIED.value,
            db_path=db_path,
        )
        res["tool_detected"] = "SmartWeb Access Standard CSV"
        return res
