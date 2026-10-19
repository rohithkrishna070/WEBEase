"""Unit and security tests for the safe public URL accessibility scanner."""

import pytest
from core.safe_scanner import (
    is_safe_public_url,
    scan_public_url,
    AccessibilityDOMParser,
    ScanSecurityError,
    RobotsDisallowedError,
    ScanNetworkError,
)


def test_url_validation_safe_public():
    safe, msg = is_safe_public_url("https://w3.org/WAI")
    assert safe is True
    assert msg == ""

    safe, msg = is_safe_public_url("http://example.com/test")
    assert safe is True


def test_url_validation_rejects_localhost_and_private():
    # Localhost
    safe, msg = is_safe_public_url("http://localhost:8000")
    assert safe is False
    assert "localhost" in msg.lower()

    safe, msg = is_safe_public_url("http://127.0.0.1/admin")
    assert safe is False

    safe, msg = is_safe_public_url("http://[::1]:8080")
    assert safe is False

    # Private RFC 1918 subnets
    safe, msg = is_safe_public_url("http://192.168.1.10/router")
    assert safe is False
    assert "private" in msg.lower() or "prohibited" in msg.lower()

    safe, msg = is_safe_public_url("http://10.0.0.1/secret")
    assert safe is False

    safe, msg = is_safe_public_url("http://172.16.0.5/api")
    assert safe is False

    # Cloud metadata link-local
    safe, msg = is_safe_public_url("http://169.254.169.254/latest/meta-data/")
    assert safe is False

    # Non-HTTP/HTTPS schemes
    safe, msg = is_safe_public_url("file:///etc/passwd")
    assert safe is False
    assert "disallowed protocol" in msg.lower()

    safe, msg = is_safe_public_url("ftp://ftp.is.co.za")
    assert safe is False


def test_scan_public_url_security_exception():
    with pytest.raises(ScanSecurityError):
        scan_public_url("http://127.0.0.1:8080/test")

    with pytest.raises(ScanSecurityError):
        scan_public_url("http://192.168.0.1/dashboard")


def test_dom_parser_accessibility_checks():
    html_sample = """
    <!DOCTYPE html>
    <html>
      <head>
        <!-- Missing title -->
        <meta name="viewport" content="width=device-width, user-scalable=no">
      </head>
      <body>
        <!-- Missing H1, directly starts with H3 -->
        <h3>Section Title</h3>
        
        <!-- Image without alt -->
        <img src="/logo.png">
        
        <!-- Form control without label -->
        <input type="text" id="username" name="user">
        
        <!-- Unclear link -->
        <a href="/faq">Click here</a>
      </body>
    </html>
    """

    parser = AccessibilityDOMParser(base_url="https://example.gov")
    parser.feed(html_sample)

    assert parser.html_has_lang is False
    assert parser.meta_zoom_disabled is True
    assert len(parser.images_without_alt) == 1
    assert len(parser.unclear_links) == 1
    assert len(parser.input_ids) == 1
    assert parser.headings[0] == (3, "Section Title")
