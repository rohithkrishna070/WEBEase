"""
================================================================================
MODULE: core.database
================================================================================
1. WHAT THIS FILE DOES:
   Manages all SQLite database persistence, table schema creation, migrations,
   seed data ingestion, and CRUD operations for websites, pages, and accessibility issues.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: A decision support system requires reliable, persistent
     storage of audit records, metadata, verification states, and institutional properties
     so decisions can be tracked and updated over time without relying on ephemeral in-memory state.
   - Technical & Architectural Rationale: Provides an embedded, zero-configuration SQL
     store using Python's standard `sqlite3` module. Abstracts SQL queries away from UI
     pages, returning clean Pandas DataFrames or typed dataclass objects.

3. HOW IT WORKS CONCEPTUALLY:
   - Maintains relational entities:
     * `websites`: Institutional properties (universities, transit, civic portals).
     * `pages`: Individual evaluated URLs, tagging whether each is an essential citizen journey.
     * `accessibility_issues`: Specific WCAG violations, rating criteria (C1-C6), audit provenance,
       verification status, and fix recommendations.
   - Automatically seeds the database from `sample_issues.csv` if empty on first startup.

4. HOW IT WORKS TECHNICALLY:
   - Uses parameterized SQL statements (`?` placeholders) to prevent SQL injection.
   - Implements schema migrations (`init_db`) ensuring backwards compatibility.
   - Uses `sqlite3.Row` row factories for dictionary-like column access.
   - Exposes clean high-level helpers: `load_all_issues()`, `upsert_issue()`, `get_stats()`,
     `get_registered_websites()`, and `get_registered_pages()`.
================================================================================
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from core.models import (
    AccessibilityIssue,
    RegisteredWebsite,
    RegisteredPage,
    RecordType,
    AuditMethod,
    IssueStatus,
    VerificationStatus,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "smartweb_access.db"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_issues.csv"

# WCAG mapping helper for enriching demonstration records
WCAG_MAP: Dict[str, str] = {
    "Low color contrast": "WCAG 1.4.3 Contrast (Minimum) (Level AA)",
    "Missing accessible names or form labels": "WCAG 4.1.2 Name, Role, Value (Level A)",
    "Keyboard navigation barriers": "WCAG 2.1.1 Keyboard (Level A)",
    "Missing alternative text": "WCAG 1.1.1 Non-text Content (Level A)",
    "Heading hierarchy problems": "WCAG 1.3.1 Info and Relationships (Level A)",
    "Unclear links": "WCAG 2.4.4 Link Purpose (In Context) (Level A)",
    "Form validation and error-message problems": "WCAG 3.3.1 Error Identification (Level A)",
    "Focus visibility problems": "WCAG 2.4.7 Focus Visible (Level AA)",
}

ESSENTIAL_TASKS_MAP: Dict[str, str] = {
    "Reduced-Fare Pass Application": "Submitting transit disability fare pass application",
    "Card Reload and Payment": "Adding transit funds via credit card checkout",
    "Trip Planner and Accessible Stops": "Finding wheelchair accessible bus and rail routes",
    "Route Schedules and Maps": "Reviewing accessible transit timetables",
    "Service Alerts and Notices": "Reviewing elevator and detour disruption notices",
    "Vital Records Ordering": "Ordering certified copy of birth/death certificate",
    "Immunization Scheduler": "Scheduling mandatory child vaccination appointment",
    "Healthcare Benefit Enrollment": "Submitting Medicaid/CHIP income eligibility renewal",
    "Community Health Clinics Locator": "Finding nearest subsidized community medical clinic",
    "Restaurant Inspection Database": "Looking up public health sanitation compliance",
    "Digital Catalog Search": "Searching for braille & large-print library resources",
    "Library Card Online Registration": "Applying for institutional library borrowing card",
    "Assistive Tech Lab and Services": "Reserving assistive screen magnifier workstation",
    "Meeting Room Reservation": "Booking community civic study room",
    "Events Calendar": "Browsing public library community programs",
    "Undergraduate Application Portal": "Completing and submitting freshman admissions application",
    "Tuition Payment and Financial Aid": "Paying semester tuition and accepting financial aid",
    "Course Schedule and Registration": "Registering for required degree course schedule",
    "Disability Resource Center": "Requesting academic accommodation plan",
    "Student Account Login": "Accessing student portal via two-factor authentication",
    "Campus Accessibility Map": "Navigating wheelchair-accessible building entrances",
    "Driver License Online Renewal": "Renewing driver license and updating residential address",
    "Real ID Checklist and Upload": "Uploading identity verification documents for Real ID",
    "Driving Test Appointment Scheduler": "Scheduling behind-the-wheel road driving test",
    "Voter Registration and Address Change": "Updating state civic voter registration address",
    "Office Locations and Wait Times": "Checking DMV branch operating hours and wait times",
}


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Return an active SQLite connection with row factory enabled."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path | str] = None) -> None:
    """Initialize the SQLite schema and perform safe column migrations."""
    conn = get_connection(db_path)
    with conn:
        # 1. Registered Websites Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_websites (
                website_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                base_url TEXT NOT NULL,
                institution_type TEXT NOT NULL DEFAULT 'Public Service / Civic Entity',
                description TEXT DEFAULT '',
                contact_email TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 2. Registered Pages Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_pages (
                page_id TEXT PRIMARY KEY,
                website_name TEXT NOT NULL,
                page_name TEXT NOT NULL,
                page_url TEXT NOT NULL,
                essential_task TEXT DEFAULT '',
                is_critical_journey INTEGER NOT NULL DEFAULT 1,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(website_name, page_url)
            )
            """
        )

        # 3. Accessibility Issues Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accessibility_issues (
                issue_id TEXT PRIMARY KEY,
                website TEXT NOT NULL,
                page_url TEXT NOT NULL,
                page_name TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                essential_task_impact TEXT NOT NULL,
                affected_pages INTEGER NOT NULL,
                user_impact TEXT NOT NULL,
                repair_effort TEXT NOT NULL,
                frequency TEXT NOT NULL,
                evidence TEXT NOT NULL,
                recommended_fix TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                record_type TEXT NOT NULL DEFAULT 'Synthetic Demonstration',
                audit_method TEXT NOT NULL DEFAULT 'Synthetic Demonstration Benchmark',
                verification_status TEXT NOT NULL DEFAULT 'Human-Verified Finding',
                tool_name TEXT NOT NULL DEFAULT 'Manual Inspection',
                wcag_criterion TEXT DEFAULT '',
                wcag_verified INTEGER NOT NULL DEFAULT 1,
                essential_task TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                reviewer_notes TEXT DEFAULT '',
                is_synthetic INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Safe column migrations if table already existed from earlier schema
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(accessibility_issues)")
        existing_cols = {col[1] for col in cursor.fetchall()}

        new_columns = [
            ("record_type", "TEXT NOT NULL DEFAULT 'Synthetic Demonstration'"),
            ("audit_method", "TEXT NOT NULL DEFAULT 'Synthetic Demonstration Benchmark'"),
            ("verification_status", "TEXT NOT NULL DEFAULT 'Human-Verified Finding'"),
            ("tool_name", "TEXT NOT NULL DEFAULT 'Manual Inspection'"),
            ("wcag_criterion", "TEXT DEFAULT ''"),
            ("wcag_verified", "INTEGER NOT NULL DEFAULT 1"),
            ("essential_task", "TEXT DEFAULT ''"),
            ("notes", "TEXT DEFAULT ''"),
            ("reviewer_notes", "TEXT DEFAULT ''"),
        ]

        for col_name, col_def in new_columns:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE accessibility_issues ADD COLUMN {col_name} {col_def}")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_issues_website ON accessibility_issues(website)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_issues_type ON accessibility_issues(issue_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_issues_status ON accessibility_issues(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_issues_verif ON accessibility_issues(verification_status)"
        )

    conn.close()


# ==============================================================================
# Registered Websites & Pages CRUD
# ==============================================================================

def get_registered_websites(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Fetch all registered institutional websites."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM registered_websites ORDER BY name ASC", conn)
    conn.close()
    return df


def upsert_registered_website(site: RegisteredWebsite, db_path: Optional[Path | str] = None) -> None:
    """Register or update an institutional website."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO registered_websites (
                website_id, name, base_url, institution_type, description, contact_email
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                base_url = excluded.base_url,
                institution_type = excluded.institution_type,
                description = excluded.description,
                contact_email = excluded.contact_email
            """,
            (site.website_id, site.name, site.base_url, site.institution_type, site.description, site.contact_email),
        )
    conn.close()


def delete_registered_website(website_id: str, db_path: Optional[Path | str] = None) -> bool:
    """Delete a registered website by website_id."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM registered_websites WHERE website_id = ?", (website_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_registered_pages(website_name: Optional[str] = None, db_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Fetch registered pages, optionally filtered by website."""
    init_db(db_path)
    conn = get_connection(db_path)
    if website_name:
        df = pd.read_sql_query(
            "SELECT * FROM registered_pages WHERE website_name = ? ORDER BY page_name ASC",
            conn,
            params=(website_name,),
        )
    else:
        df = pd.read_sql_query("SELECT * FROM registered_pages ORDER BY website_name, page_name ASC", conn)
    conn.close()
    return df


def upsert_registered_page(page: RegisteredPage, db_path: Optional[Path | str] = None) -> None:
    """Register or update a page under a website."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO registered_pages (
                page_id, website_name, page_name, page_url, essential_task, is_critical_journey, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(website_name, page_url) DO UPDATE SET
                page_name = excluded.page_name,
                essential_task = excluded.essential_task,
                is_critical_journey = excluded.is_critical_journey,
                notes = excluded.notes
            """,
            (
                page.page_id,
                page.website_name,
                page.page_name,
                page.page_url,
                page.essential_task,
                1 if page.is_critical_journey else 0,
                page.notes,
            ),
        )
    conn.close()


def delete_registered_page(page_id: str, db_path: Optional[Path | str] = None) -> bool:
    """Delete a registered page by page_id."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM registered_pages WHERE page_id = ?", (page_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ==============================================================================
# Seeding Logic
# ==============================================================================

def seed_if_empty(
    db_path: Optional[Path | str] = None,
    csv_path: Optional[Path | str] = None,
    force: bool = False,
) -> int:
    """Seed SQLite database with demonstration CSV data and websites if empty or forced."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM accessibility_issues")
        count = cursor.fetchone()[0]

        if count > 0 and not force:
            return count

        source_csv = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        if not source_csv.exists():
            return 0

        df = pd.read_csv(source_csv)

        with conn:
            if force:
                cursor.execute("DELETE FROM accessibility_issues")
                cursor.execute("DELETE FROM registered_websites")
                cursor.execute("DELETE FROM registered_pages")

            # Seed registered websites
            websites_data = [
                ("WEB-001", "Metro Transit Authority", "https://transit.metrocity.gov", "Public Transit / Municipal", "City transit provider for rail and bus services.", "access@transit.metrocity.gov"),
                ("WEB-002", "State Department of Public Health", "https://health.state.gov", "State Healthcare / Regulatory", "Statewide vital records and healthcare access.", "webmaster@health.state.gov"),
                ("WEB-003", "City Central Public Library", "https://library.city.org", "Public Library System", "Public educational and community catalog resources.", "ada@library.city.org"),
                ("WEB-004", "State University Portal", "https://admissions.stateuniv.edu", "Higher Education", "Public university admissions and enrollment systems.", "drc@stateuniv.edu"),
                ("WEB-005", "Department of Motor Vehicles", "https://dmv.state.gov", "State Licensing / Civic Services", "Driver license and state identification records.", "support@dmv.state.gov"),
            ]
            cursor.executemany(
                """
                INSERT OR IGNORE INTO registered_websites (
                    website_id, name, base_url, institution_type, description, contact_email
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                websites_data,
            )

            # Seed registered pages
            seen_pages = set()
            page_records = []
            page_counter = 1
            for _, row in df.iterrows():
                site = row["website"]
                purl = row["page_url"]
                pname = row["page_name"]
                key = (site, purl)
                if key not in seen_pages:
                    seen_pages.add(key)
                    task = ESSENTIAL_TASKS_MAP.get(pname, f"Accessing {pname}")
                    page_records.append((
                        f"PG-{page_counter:03d}",
                        site,
                        pname,
                        purl,
                        task,
                        1,
                        "Benchmark public service page",
                    ))
                    page_counter += 1

            cursor.executemany(
                """
                INSERT OR IGNORE INTO registered_pages (
                    page_id, website_name, page_name, page_url, essential_task, is_critical_journey, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                page_records,
            )

            # Seed issues
            insert_sql = """
                INSERT INTO accessibility_issues (
                    issue_id, website, page_url, page_name, issue_type,
                    description, severity, essential_task_impact, affected_pages,
                    user_impact, repair_effort, frequency, evidence,
                    recommended_fix, status, record_type, audit_method,
                    verification_status, tool_name, wcag_criterion, wcag_verified,
                    essential_task, notes, reviewer_notes, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """

            records = []
            for _, row in df.iterrows():
                itype = row["issue_type"]
                pname = row["page_name"]
                wcag = row.get("wcag_criterion") or WCAG_MAP.get(itype, "WCAG 2.1 Principle")
                task = row.get("essential_task") or ESSENTIAL_TASKS_MAP.get(pname, f"Browsing {pname}")
                method = row.get("audit_method") or AuditMethod.SYNTHETIC_BENCHMARK.value
                rec_type = row.get("record_type") or RecordType.SYNTHETIC.value
                status = row.get("status", IssueStatus.OPEN.value)
                if status == "Verified":
                    status = IssueStatus.RESOLVED.value

                records.append((
                    row["issue_id"],
                    row["website"],
                    row["page_url"],
                    row["page_name"],
                    row["issue_type"],
                    row["description"],
                    row["severity"],
                    row["essential_task_impact"],
                    int(row["affected_pages"]),
                    row["user_impact"],
                    row["repair_effort"],
                    row["frequency"],
                    row["evidence"],
                    row["recommended_fix"],
                    status,
                    rec_type,
                    method,
                    VerificationStatus.HUMAN_VERIFIED.value,
                    "Synthetic Benchmark",
                    wcag,
                    1,
                    task,
                    str(row.get("notes", "") or ""),
                    "Benchmark dataset record for MCDM evaluation.",
                ))

            cursor.executemany(insert_sql, records)
            seeded_count = len(records)
        return seeded_count
    finally:
        conn.close()


def load_all_issues(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Fetch all accessibility issues as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = """
        SELECT issue_id, website, page_url, page_name, issue_type,
               description, severity, essential_task_impact, affected_pages,
               user_impact, repair_effort, frequency, evidence,
               recommended_fix, status, record_type, audit_method,
               verification_status, tool_name, wcag_criterion, wcag_verified,
               essential_task, notes, reviewer_notes, is_synthetic
        FROM accessibility_issues
        ORDER BY issue_id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_existing_ids(db_path: Optional[Path | str] = None) -> Set[str]:
    """Return a set of all current Issue IDs in the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT issue_id FROM accessibility_issues")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids


def upsert_issue(issue: AccessibilityIssue, db_path: Optional[Path | str] = None) -> None:
    """Insert or update an accessibility issue in the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO accessibility_issues (
                issue_id, website, page_url, page_name, issue_type,
                description, severity, essential_task_impact, affected_pages,
                user_impact, repair_effort, frequency, evidence,
                recommended_fix, status, record_type, audit_method,
                verification_status, tool_name, wcag_criterion, wcag_verified,
                essential_task, notes, reviewer_notes, is_synthetic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(issue_id) DO UPDATE SET
                website = excluded.website,
                page_url = excluded.page_url,
                page_name = excluded.page_name,
                issue_type = excluded.issue_type,
                description = excluded.description,
                severity = excluded.severity,
                essential_task_impact = excluded.essential_task_impact,
                affected_pages = excluded.affected_pages,
                user_impact = excluded.user_impact,
                repair_effort = excluded.repair_effort,
                frequency = excluded.frequency,
                evidence = excluded.evidence,
                recommended_fix = excluded.recommended_fix,
                status = excluded.status,
                record_type = excluded.record_type,
                audit_method = excluded.audit_method,
                verification_status = excluded.verification_status,
                tool_name = excluded.tool_name,
                wcag_criterion = excluded.wcag_criterion,
                wcag_verified = excluded.wcag_verified,
                essential_task = excluded.essential_task,
                notes = excluded.notes,
                reviewer_notes = excluded.reviewer_notes,
                is_synthetic = excluded.is_synthetic
            """,
            (
                issue.issue_id,
                issue.website,
                issue.page_url,
                issue.page_name,
                issue.issue_type,
                issue.description,
                issue.severity,
                issue.essential_task_impact,
                issue.affected_pages,
                issue.user_impact,
                issue.repair_effort,
                issue.frequency,
                issue.evidence,
                issue.recommended_fix,
                issue.status,
                issue.record_type,
                issue.audit_method,
                issue.verification_status,
                issue.tool_name,
                issue.wcag_criterion,
                issue.wcag_verified,
                issue.essential_task,
                issue.notes,
                issue.reviewer_notes,
                issue.is_synthetic,
            ),
        )
    conn.close()


def delete_issue(issue_id: str, db_path: Optional[Path | str] = None) -> bool:
    """Delete an issue by ID. Returns True if a record was removed."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM accessibility_issues WHERE issue_id = ?", (issue_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def filter_issues(
    df: pd.DataFrame,
    websites: Optional[List[str]] = None,
    pages: Optional[List[str]] = None,
    issue_types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None,
    task_impacts: Optional[List[str]] = None,
    user_impacts: Optional[List[str]] = None,
    statuses: Optional[List[str]] = None,
    record_types: Optional[List[str]] = None,
    verification_statuses: Optional[List[str]] = None,
    search_query: str = "",
) -> pd.DataFrame:
    """Filter an issues DataFrame across multiple dimensional criteria."""
    filtered = df.copy()

    if websites:
        filtered = filtered[filtered["website"].isin(websites)]
    if pages:
        filtered = filtered[filtered["page_name"].isin(pages)]
    if issue_types:
        filtered = filtered[filtered["issue_type"].isin(issue_types)]
    if severities:
        filtered = filtered[filtered["severity"].isin(severities)]
    if task_impacts:
        filtered = filtered[filtered["essential_task_impact"].isin(task_impacts)]
    if user_impacts:
        filtered = filtered[filtered["user_impact"].isin(user_impacts)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if record_types:
        filtered = filtered[filtered["record_type"].isin(record_types)]
    if verification_statuses:
        filtered = filtered[filtered["verification_status"].isin(verification_statuses)]

    if search_query:
        q = search_query.strip().lower()
        search_cols = [
            "issue_id", "website", "page_name", "issue_type", "description",
            "evidence", "recommended_fix", "wcag_criterion", "essential_task",
            "notes", "reviewer_notes", "tool_name"
        ]
        mask = pd.Series(False, index=filtered.index)
        for col in search_cols:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    return filtered


def get_stats(db_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Calculate summary statistics for overview reporting."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues")
    total_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT website) FROM accessibility_issues")
    total_websites = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT page_url) FROM accessibility_issues")
    total_pages = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM accessibility_issues WHERE essential_task_impact IN ('High', 'Very High')"
    )
    critical_task_issues = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM accessibility_issues WHERE severity IN ('High', 'Very High')"
    )
    high_severity_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE status = 'Open'")
    open_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE status = 'In Progress'")
    in_progress_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE status = 'Resolved'")
    resolved_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE status = 'Needs Review'")
    needs_review_issues = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE verification_status = 'Unverified Automated Detection'")
    unverified_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM accessibility_issues WHERE verification_status = 'Human-Verified Finding'")
    verified_findings_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total_issues": total_issues,
        "total_websites": total_websites,
        "total_pages": total_pages,
        "critical_task_issues": critical_task_issues,
        "high_severity_issues": high_severity_issues,
        "open_issues": open_issues,
        "in_progress_issues": in_progress_issues,
        "resolved_issues": resolved_issues,
        "needs_review_issues": needs_review_issues,
        "unverified_count": unverified_count,
        "verified_findings_count": verified_findings_count,
    }
