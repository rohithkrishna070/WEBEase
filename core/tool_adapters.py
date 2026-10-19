"""Multi-Tool Audit Adapters for SmartWeb Access.

Normalizes diverse accessibility audit exports from Google Lighthouse, Axe-core,
WAVE, and generic CSV tools into a standardized model.
Automated detections are strictly categorized as 'Unverified Automated Detection'
with unverified WCAG status (wcag_verified = 0) pending human review.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from core.models import (
    AccessibilityIssue,
    IssueStatus,
    RecordType,
    AuditMethod,
    VerificationStatus,
)
from core.database import WCAG_MAP


def detect_tool_format(columns_or_keys: List[str]) -> str:
    """Detect audit tool provenance based on exported table headers."""
    cols_lower = [c.strip().lower() for c in columns_or_keys]

    # SmartWeb Access Standard Schema
    if "essential_task_impact" in cols_lower and "issue_type" in cols_lower:
        return "SMARTWEB_STANDARD"

    # Axe-core / Axe DevTools format
    if ("rule id" in cols_lower or "rule_id" in cols_lower) and ("impact" in cols_lower or "help" in cols_lower):
        return "AXE_CORE"

    # WAVE Accessibility Tool format
    if ("issue" in cols_lower or "code" in cols_lower) and ("type" in cols_lower or "selector" in cols_lower):
        return "WAVE"

    # Lighthouse CSV export
    if ("audit_id" in cols_lower or "audit id" in cols_lower or "score" in cols_lower) and "title" in cols_lower:
        return "LIGHTHOUSE"

    return "GENERIC_COMPATIBLE"


def map_axe_impact_to_severity(impact: str) -> Tuple[str, str]:
    """Map Axe impact (critical, serious, moderate, minor) to SmartWeb Severity & User Impact."""
    imp = str(impact).strip().lower()
    if imp == "critical":
        return "Very High", "Very High"
    if imp == "serious":
        return "High", "High"
    if imp == "moderate":
        return "Medium", "Medium"
    return "Low", "Low"


def parse_axe_csv(
    df: pd.DataFrame,
    website_name: str = "Institutional Target",
    target_page_url: str = "https://example.gov",
) -> List[AccessibilityIssue]:
    """Normalize Axe-core CSV exports into candidate AccessibilityIssue records."""
    issues: List[AccessibilityIssue] = []

    # Case-insensitive column lookup
    col_map = {c.strip().lower(): c for c in df.columns}
    
    rule_col = col_map.get("rule id") or col_map.get("rule_id") or col_map.get("id")
    impact_col = col_map.get("impact")
    desc_col = col_map.get("description")
    help_col = col_map.get("help")
    target_col = col_map.get("target") or col_map.get("selector") or col_map.get("html") or col_map.get("element")
    url_col = col_map.get("url")

    for idx, row in df.iterrows():
        rule_id = str(row[rule_col]).strip() if rule_col else f"axe-rule-{idx+1}"
        impact = str(row[impact_col]).strip() if impact_col and pd.notna(row[impact_col]) else "moderate"
        severity, user_impact = map_axe_impact_to_severity(impact)

        desc = str(row[desc_col]).strip() if desc_col and pd.notna(row[desc_col]) else "Axe-detected accessibility barrier."
        help_text = str(row[help_col]).strip() if help_col and pd.notna(row[help_col]) else "Remediate per WCAG guidelines."
        evidence = str(row[target_col]).strip() if target_col and pd.notna(row[target_col]) else f"Rule: {rule_id}"
        page_url = str(row[url_col]).strip() if url_col and pd.notna(row[url_col]) else target_page_url

        # Infer barrier category from rule
        issue_type = "Missing accessible names or form labels"
        r_lower = rule_id.lower()
        if "contrast" in r_lower:
            issue_type = "Low color contrast"
        elif "keyboard" in r_lower or "tabindex" in r_lower or "focus" in r_lower:
            issue_type = "Keyboard navigation barriers"
        elif "image-alt" in r_lower or "alt" in r_lower:
            issue_type = "Missing alternative text"
        elif "heading" in r_lower or "landmark" in r_lower:
            issue_type = "Heading hierarchy problems"
        elif "link" in r_lower:
            issue_type = "Unclear links"

        issues.append(
            AccessibilityIssue(
                issue_id=f"AXE-{idx+1:03d}",
                website=website_name,
                page_url=page_url,
                page_name="Audited Page",
                issue_type=issue_type,
                description=desc,
                severity=severity,
                essential_task_impact="Medium",  # Provisional: awaiting human task impact evaluation
                affected_pages=1,
                user_impact=user_impact,
                repair_effort="Medium",  # Provisional
                frequency="Frequent",
                evidence=evidence[:400],
                recommended_fix=help_text,
                status=IssueStatus.NEEDS_REVIEW.value,
                record_type=RecordType.IMPORTED.value,
                audit_method=AuditMethod.AUTOMATED.value,
                verification_status=VerificationStatus.UNVERIFIED_AUTOMATED.value,
                tool_name="Axe-core (Automated)",
                wcag_criterion=WCAG_MAP.get(issue_type, "WCAG 2.1 Principle"),
                wcag_verified=0,  # Unverified candidate finding
                essential_task="Provisional task - verify in triage queue",
                notes=f"Auto-imported from Axe-core. Rule: {rule_id}.",
                is_synthetic=0,
            )
        )

    return issues


def parse_wave_csv(
    df: pd.DataFrame,
    website_name: str = "Institutional Target",
    target_page_url: str = "https://example.gov",
) -> List[AccessibilityIssue]:
    """Normalize WAVE CSV exports into candidate AccessibilityIssue records."""
    issues: List[AccessibilityIssue] = []
    col_map = {c.strip().lower(): c for c in df.columns}

    issue_col = col_map.get("issue") or col_map.get("code") or col_map.get("name")
    type_col = col_map.get("type")
    desc_col = col_map.get("description")
    sel_col = col_map.get("selector") or col_map.get("element") or col_map.get("xpath")
    url_col = col_map.get("url")

    for idx, row in df.iterrows():
        issue_name = str(row[issue_col]).strip() if issue_col else f"WAVE Issue {idx+1}"
        wave_type = str(row[type_col]).strip().lower() if type_col and pd.notna(row[type_col]) else "error"
        desc = str(row[desc_col]).strip() if desc_col and pd.notna(row[desc_col]) else issue_name
        evidence = str(row[sel_col]).strip() if sel_col and pd.notna(row[sel_col]) else f"WAVE: {issue_name}"
        page_url = str(row[url_col]).strip() if url_col and pd.notna(row[url_col]) else target_page_url

        severity = "High" if "error" in wave_type or "contrast" in wave_type else "Medium"
        user_impact = "High"

        # Categorize
        name_lower = issue_name.lower()
        issue_type = "Form validation and error-message problems"
        if "contrast" in name_lower:
            issue_type = "Low color contrast"
        elif "alt" in name_lower or "image" in name_lower:
            issue_type = "Missing alternative text"
        elif "label" in name_lower or "name" in name_lower:
            issue_type = "Missing accessible names or form labels"
        elif "heading" in name_lower:
            issue_type = "Heading hierarchy problems"
        elif "link" in name_lower:
            issue_type = "Unclear links"
        elif "focus" in name_lower:
            issue_type = "Focus visibility problems"

        issues.append(
            AccessibilityIssue(
                issue_id=f"WAVE-{idx+1:03d}",
                website=website_name,
                page_url=page_url,
                page_name="Audited Page",
                issue_type=issue_type,
                description=desc,
                severity=severity,
                essential_task_impact="Medium",
                affected_pages=1,
                user_impact=user_impact,
                repair_effort="Medium",
                frequency="Frequent",
                evidence=evidence[:400],
                recommended_fix=f"Review WAVE guideline for '{issue_name}' and remediate element.",
                status=IssueStatus.NEEDS_REVIEW.value,
                record_type=RecordType.IMPORTED.value,
                audit_method=AuditMethod.AUTOMATED.value,
                verification_status=VerificationStatus.UNVERIFIED_AUTOMATED.value,
                tool_name="WAVE Evaluation Tool",
                wcag_criterion=WCAG_MAP.get(issue_type, "WCAG 2.1 Principle"),
                wcag_verified=0,
                essential_task="Provisional task - verify in triage queue",
                notes=f"Auto-imported from WAVE. Type: {wave_type}.",
                is_synthetic=0,
            )
        )

    return issues


def parse_lighthouse_json(
    json_bytes_or_str: bytes | str,
    website_name: str = "Institutional Target",
) -> List[AccessibilityIssue]:
    """Parse Google Lighthouse JSON audit output."""
    if isinstance(json_bytes_or_str, bytes):
        raw = json.loads(json_bytes_or_str.decode("utf-8", errors="replace"))
    else:
        raw = json.loads(json_bytes_or_str)

    page_url = raw.get("finalUrl") or raw.get("requestedUrl") or "https://example.gov"
    audits = raw.get("audits", {})
    issues: List[AccessibilityIssue] = []

    count = 1
    for audit_id, data in audits.items():
        # Score is 0 or False for failed accessibility audits
        score = data.get("score")
        if score is not None and score == 0:
            title = data.get("title", audit_id)
            desc = data.get("description", "")
            
            # Extract DOM snippets from details if available
            details = data.get("details", {})
            items = details.get("items", [])
            snippets = []
            for item in items[:3]:
                node = item.get("node", {})
                snippet = node.get("snippet") or node.get("selector")
                if snippet:
                    snippets.append(snippet)

            evidence_str = " | ".join(snippets) if snippets else f"Audit ID: {audit_id}"

            # Type mapping
            aid = audit_id.lower()
            issue_type = "Form validation and error-message problems"
            if "color-contrast" in aid:
                issue_type = "Low color contrast"
            elif "image-alt" in aid or "alt" in aid:
                issue_type = "Missing alternative text"
            elif "label" in aid or "button-name" in aid:
                issue_type = "Missing accessible names or form labels"
            elif "heading" in aid:
                issue_type = "Heading hierarchy problems"
            elif "link-name" in aid:
                issue_type = "Unclear links"
            elif "focus" in aid:
                issue_type = "Focus visibility problems"

            issues.append(
                AccessibilityIssue(
                    issue_id=f"LH-{count:03d}",
                    website=website_name,
                    page_url=page_url,
                    page_name="Audited Page",
                    issue_type=issue_type,
                    description=title,
                    severity="High",
                    essential_task_impact="Medium",
                    affected_pages=1,
                    user_impact="High",
                    repair_effort="Medium",
                    frequency="Frequent",
                    evidence=evidence_str[:400],
                    recommended_fix=desc[:300] or "Remediate per Lighthouse documentation.",
                    status=IssueStatus.NEEDS_REVIEW.value,
                    record_type=RecordType.IMPORTED.value,
                    audit_method=AuditMethod.AUTOMATED.value,
                    verification_status=VerificationStatus.UNVERIFIED_AUTOMATED.value,
                    tool_name="Google Lighthouse (Automated)",
                    wcag_criterion=WCAG_MAP.get(issue_type, "WCAG 2.1 Principle"),
                    wcag_verified=0,
                    essential_task="Provisional task - verify in triage queue",
                    notes=f"Auto-imported from Google Lighthouse. Rule: {audit_id}.",
                    is_synthetic=0,
                )
            )
            count += 1

    return issues
