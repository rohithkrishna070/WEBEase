"""
================================================================================
MODULE: core.validator
================================================================================
1. WHAT THIS FILE DOES:
   Provides comprehensive validation gates for user inputs, URL formats,
   linguistic criteria ratings, duplicate ID checks, and uploaded CSV file schemas.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: A Multi-Criteria Decision Support System is highly sensitive
     to corrupt or incomplete input data. Invalid or missing ratings (e.g. non-existent
     linguistic terms or negative values) would break matrix normalization in TOPSIS/VIKOR.
   - Technical & Architectural Rationale: Centralizes data hygiene and defensive validation.
     Provides clear, user-friendly error messages when invalid data is entered in UI forms
     or uploaded in CSV batch files.

3. HOW IT WORKS CONCEPTUALLY:
   - Validates that mandatory audit fields (issue ID, website, page, criteria ratings) are populated.
   - Enforces that linguistic terms match recognized scales ("Very Low", "Low", "Medium", "High", "Very High").
   - Ensures URLs adhere to RFC web schemes (http/https).
   - Validates CSV file uploads row-by-row, segregating valid rows from invalid rows with error annotations.

4. HOW IT WORKS TECHNICALLY:
   - `validate_url()`: Regex and `urllib.parse.urlparse` domain validation.
   - `validate_issue_fields()`: Checks required keys, rating enum membership, and ID formats.
   - `validate_csv_upload()`: Reads CSV buffers via pandas, identifies missing columns, empty rows,
     and validates records against database collision sets.
================================================================================
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import pandas as pd

from core.models import (
    VALID_SEVERITY_RATINGS,
    VALID_TASK_IMPACT_RATINGS,
    VALID_USER_IMPACT_RATINGS,
    VALID_EFFORT_RATINGS,
    VALID_FREQUENCY_RATINGS,
    VALID_STATUSES,
    VALID_RECORD_TYPES,
)

REQUIRED_FIELDS = [
    "issue_id",
    "website",
    "page_url",
    "page_name",
    "issue_type",
    "description",
    "severity",
    "essential_task_impact",
    "affected_pages",
    "user_impact",
    "repair_effort",
    "frequency",
    "evidence",
    "recommended_fix",
]

URL_PATTERN = re.compile(
    r"^https?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\.?|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def validate_url(url: str) -> bool:
    """Validate that a URL starts with http(s) and possesses a valid host/domain."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc and URL_PATTERN.match(url))
    except Exception:
        return False


def validate_issue_fields(
    data: Dict[str, Any],
    existing_ids: Optional[Set[str]] = None,
    is_edit: bool = False,
) -> List[str]:
    """Validate an individual accessibility issue dictionary.

    Returns a list of validation error descriptions. Empty list means data is valid.
    """
    errors: List[str] = []
    existing = existing_ids or set()

    # 1. Check Missing Required Fields
    for field_name in REQUIRED_FIELDS:
        val = data.get(field_name)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing required field: '{field_name}'.")

    # 2. Check Issue ID Format & Duplication
    issue_id = str(data.get("issue_id", "")).strip()
    if not issue_id:
        errors.append("Issue ID cannot be empty.")
    elif not re.match(r"^[A-Za-z0-9_-]+$", issue_id):
        errors.append(
            f"Invalid Issue ID '{issue_id}': must contain only alphanumeric characters, dashes, or underscores."
        )
    elif not is_edit and issue_id in existing:
        errors.append(
            f"Duplicate Issue ID '{issue_id}': an issue with this identifier already exists."
        )

    # 3. Validate Page URL
    page_url = str(data.get("page_url", "")).strip()
    if page_url and not validate_url(page_url):
        errors.append(
            f"Malformed URL '{page_url}': must begin with http:// or https:// and include a valid domain."
        )

    # 4. Validate Affected Pages count
    try:
        pages_val = int(data.get("affected_pages", 0))
        if pages_val < 1:
            errors.append(f"Invalid affected_pages: must be an integer >= 1 (got {pages_val}).")
    except (ValueError, TypeError):
        errors.append(
            f"Invalid affected_pages: '{data.get('affected_pages')}'. Must be an integer."
        )

    # 5. Validate Linguistic Ratings
    severity = str(data.get("severity", "")).strip()
    if severity and severity not in VALID_SEVERITY_RATINGS:
        errors.append(
            f"Invalid severity '{severity}'. Allowed: {', '.join(VALID_SEVERITY_RATINGS)}."
        )

    task_impact = str(data.get("essential_task_impact", "")).strip()
    if task_impact and task_impact not in VALID_TASK_IMPACT_RATINGS:
        errors.append(
            f"Invalid essential_task_impact '{task_impact}'. Allowed: {', '.join(VALID_TASK_IMPACT_RATINGS)}."
        )

    user_impact = str(data.get("user_impact", "")).strip()
    if user_impact and user_impact not in VALID_USER_IMPACT_RATINGS:
        errors.append(
            f"Invalid user_impact '{user_impact}'. Allowed: {', '.join(VALID_USER_IMPACT_RATINGS)}."
        )

    effort = str(data.get("repair_effort", "")).strip()
    if effort and effort not in VALID_EFFORT_RATINGS:
        errors.append(
            f"Invalid repair_effort '{effort}'. Allowed: {', '.join(VALID_EFFORT_RATINGS)}."
        )

    freq = str(data.get("frequency", "")).strip()
    if freq and freq not in VALID_FREQUENCY_RATINGS:
        errors.append(
            f"Invalid frequency '{freq}'. Allowed: {', '.join(VALID_FREQUENCY_RATINGS)}."
        )

    status = str(data.get("status", "Open")).strip()
    if status and status not in VALID_STATUSES and status != "Verified":
        errors.append(
            f"Invalid status '{status}'. Allowed: {', '.join(VALID_STATUSES)}."
        )

    rec_type = str(data.get("record_type", "")).strip()
    if rec_type and rec_type not in VALID_RECORD_TYPES:
        errors.append(
            f"Invalid record_type '{rec_type}'. Allowed: {', '.join(VALID_RECORD_TYPES)}."
        )

    return errors


def validate_csv_upload(
    file_bytes_or_str: bytes | str | io.StringIO | io.BytesIO,
    existing_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Validate an uploaded CSV file.

    Returns
    -------
    Tuple[List[Dict], List[Dict], List[str]]
        - valid_rows: list of validated row dicts ready for database insertion
        - invalid_rows: list of row dicts with errors
        - global_errors: list of structural errors (empty file, missing columns)
    """
    global_errors: List[str] = []
    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    existing = set(existing_ids or set())

    # 1. Empty Upload Check
    if not file_bytes_or_str:
        return [], [], ["Empty upload: The provided file is empty."]

    try:
        if isinstance(file_bytes_or_str, (bytes, bytearray)):
            if len(file_bytes_or_str.strip()) == 0:
                return [], [], ["Empty upload: The provided file contains no data."]
            stream = io.BytesIO(file_bytes_or_str)
            df = pd.read_csv(stream)
        elif isinstance(file_bytes_or_str, str):
            if not file_bytes_or_str.strip():
                return [], [], ["Empty upload: The provided CSV string is empty."]
            stream = io.StringIO(file_bytes_or_str)
            df = pd.read_csv(stream)
        else:
            df = pd.read_csv(file_bytes_or_str)
    except Exception as e:
        return [], [], [f"CSV Parsing Error: Could not read CSV file. {e}"]

    if df.empty:
        return [], [], ["Empty upload: The CSV file contains headers but zero data rows."]

    # 2. Check Required Column Headers
    missing_cols = [col for col in REQUIRED_FIELDS if col not in df.columns]
    if missing_cols:
        return (
            [],
            [],
            [
                f"Missing required CSV columns: {', '.join(missing_cols)}. "
                f"Expected all of: {', '.join(REQUIRED_FIELDS)}."
            ],
        )

    # 3. Validate Each Row
    seen_in_batch: Set[str] = set()

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-based index including header row
        row_dict = row.to_dict()

        # Check in-batch duplicate
        row_id = str(row_dict.get("issue_id", "")).strip()
        row_errors: List[str] = []

        if row_id in seen_in_batch:
            row_errors.append(f"Duplicate Issue ID '{row_id}' found multiple times within this CSV.")
        else:
            seen_in_batch.add(row_id)

        # Field validation
        field_errs = validate_issue_fields(row_dict, existing_ids=existing, is_edit=False)
        row_errors.extend(field_errs)

        if row_errors:
            row_dict["_row_number"] = row_num
            row_dict["_errors"] = row_errors
            invalid_rows.append(row_dict)
        else:
            valid_rows.append(row_dict)

    return valid_rows, invalid_rows, global_errors
