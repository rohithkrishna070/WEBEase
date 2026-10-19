"""Tests for multi-tool audit adapters (Lighthouse, Axe, WAVE)."""

import json
import pandas as pd
import pytest
from core.tool_adapters import (
    detect_tool_format,
    parse_axe_csv,
    parse_wave_csv,
    parse_lighthouse_json,
)
from core.models import VerificationStatus


def test_detect_tool_format():
    axe_headers = ["Rule ID", "Impact", "Description", "Help", "Target"]
    assert detect_tool_format(axe_headers) == "AXE_CORE"

    wave_headers = ["URL", "Issue", "Type", "Selector", "Description"]
    assert detect_tool_format(wave_headers) == "WAVE"

    lh_headers = ["audit_id", "title", "score", "description"]
    assert detect_tool_format(lh_headers) == "LIGHTHOUSE"

    std_headers = [
        "issue_id", "website", "page_url", "page_name", "issue_type",
        "description", "severity", "essential_task_impact", "evidence", "recommended_fix"
    ]
    assert detect_tool_format(std_headers) == "SMARTWEB_STANDARD"


def test_parse_axe_csv():
    df = pd.DataFrame([
        {
            "Rule ID": "color-contrast",
            "Impact": "serious",
            "Description": "Elements must have sufficient color contrast",
            "Help": "Ensure contrast ratio of at least 4.5:1",
            "Target": "#header-title",
            "URL": "https://service.gov/home",
        },
        {
            "Rule ID": "button-name",
            "Impact": "critical",
            "Description": "Buttons must have discernible text",
            "Help": "Add inner text or aria-label",
            "Target": "button.submit-btn",
            "URL": "https://service.gov/home",
        },
    ])

    issues = parse_axe_csv(df, website_name="Civic Portal")
    assert len(issues) == 2
    assert issues[0].issue_type == "Low color contrast"
    assert issues[0].severity == "High"
    assert issues[0].verification_status == VerificationStatus.UNVERIFIED_AUTOMATED.value
    assert issues[0].wcag_verified == 0

    assert issues[1].issue_type == "Missing accessible names or form labels"
    assert issues[1].severity == "Very High"


def test_parse_wave_csv():
    df = pd.DataFrame([
        {
            "URL": "https://transit.gov",
            "Issue": "Missing Alternative Text",
            "Type": "error",
            "Selector": "img.logo",
            "Description": "An image does not have alternative text",
        }
    ])

    issues = parse_wave_csv(df, website_name="Transit Authority")
    assert len(issues) == 1
    assert issues[0].issue_type == "Missing alternative text"
    assert issues[0].severity == "High"
    assert issues[0].verification_status == VerificationStatus.UNVERIFIED_AUTOMATED.value
    assert issues[0].wcag_verified == 0


def test_parse_lighthouse_json():
    lh_dict = {
        "requestedUrl": "https://library.org",
        "audits": {
            "image-alt": {
                "id": "image-alt",
                "title": "Image elements have [alt] attributes",
                "description": "Informative elements should have descriptive alt text.",
                "score": 0,
                "details": {
                    "items": [
                        {"node": {"snippet": '<img src="book.jpg">', "selector": "div > img"}}
                    ]
                },
            },
            "document-title": {
                "id": "document-title",
                "title": "Document has a <title> element",
                "description": "The title gives screen reader users an overview of the page.",
                "score": 1,  # Passed audit, should NOT be converted to an issue
            },
        },
    }

    issues = parse_lighthouse_json(json.dumps(lh_dict), website_name="Public Library")
    assert len(issues) == 1
    assert issues[0].issue_type == "Missing alternative text"
    assert '<img src="book.jpg">' in issues[0].evidence
    assert issues[0].verification_status == VerificationStatus.UNVERIFIED_AUTOMATED.value
    assert issues[0].wcag_verified == 0
