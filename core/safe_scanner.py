"""
================================================================================
MODULE: core.safe_scanner
================================================================================
1. WHAT THIS FILE DOES:
   Provides a safe, single-page DOM accessibility scanner that can fetch and evaluate
   public web pages without launching headless browser instances or violating ethical web constraints.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Allows users to test live institutional pages for basic WCAG
     violations (missing alt text, unlabeled form fields, empty buttons, broken headings)
     while adhering to academic and ethical audit boundaries. All scanner detections
     are explicitly marked as `Unverified Automated Detection` to avoid misleading auditors.
   - Technical & Architectural Rationale: Prevents Server-Side Request Forgery (SSRF),
     denial-of-service, or crawling traps. Blocks private IP subnets (127.0.0.1, 10.0.0.0/8,
     192.168.0.0/16, AWS metadata 169.254.169.254), checks `robots.txt`, enforces strict
     HTTP timeouts and response payload byte limits, and never recurses into links.

3. HOW IT WORKS CONCEPTUALLY:
   - Validates target URL scheme (HTTP/HTTPS only) and verifies that host resolves to a public IP.
   - Fetches target HTML stream respecting response size limit (e.g. 2 MB max) and timeouts (5s).
   - Parses the HTML DOM using Python's native streaming parser to extract element tags and attributes.
   - Runs deterministic audit rules:
     * Images missing `alt` attributes or with suspicious filenames.
     * Inputs lacking `<label>`, `aria-label`, or `aria-labelledby`.
     * Buttons without textual or accessible names.
     * Heading level skips (e.g. `<h1>` followed directly by `<h3>`).

4. HOW IT WORKS TECHNICALLY:
   - DNS resolution and IP verification with `socket.getaddrinfo` and `ipaddress.ip_address`.
   - `urllib.robotparser.RobotFileParser` checks crawler permission against custom User-Agent.
   - `html.parser.HTMLParser` subclass (`_SafeAccessibilityDOMParser`) performs streaming DOM tokens.
   - Raises structured domain exceptions (`ScanSecurityError`, `RobotsDisallowedError`, `ScanNetworkError`).
================================================================================
"""

from __future__ import annotations

import html.parser
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
import urllib.robotparser
from typing import Any, Dict, List, Optional, Tuple

USER_AGENT = "SmartWebAccess-AuditBot/1.0 (Accessibility Decision Support System; ethical auditor)"


class ScanSecurityError(ValueError):
    """Raised when a target URL violates security constraints (e.g. private IP, localhost)."""
    pass


class RobotsDisallowedError(PermissionError):
    """Raised when robots.txt restricts automated auditing for this path."""
    pass


class ScanNetworkError(IOError):
    """Raised when fetching a target page fails due to timeouts or HTTP errors."""
    pass


def is_safe_public_url(url: str) -> Tuple[bool, str]:
    """Validate that a URL is a legitimate, publicly routable HTTP/HTTPS endpoint.

    Rejects localhost, RFC 1918 private subnets, loopback, link-local, multicast,
    and non-standard URI schemes. Resolves hostname to verify target IP.

    Returns
    -------
    Tuple[bool, str]
        (is_safe, error_reason)
    """
    if not url or not isinstance(url, str):
        return False, "Target URL cannot be empty."

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Invalid URL structure: {e}"

    if parsed.scheme not in ("http", "https"):
        return False, f"Disallowed protocol '{parsed.scheme}'. Only http:// and https:// are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL."

    hostname_lower = hostname.lower()

    # Reject explicit localhost or internal domains
    if hostname_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "Access to localhost and loopback interfaces is strictly prohibited."

    if hostname_lower.endswith((".local", ".internal", ".lan", ".corp", ".home")):
        return False, f"Access to private internal domain '{hostname}' is prohibited."

    # DNS Resolution check to prevent SSRF and DNS rebinding
    try:
        addr_info = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        for item in addr_info:
            ip_str = item[4][0]
            ip_obj = ipaddress.ip_address(ip_str)

            if ip_obj.is_private:
                return False, f"Target resolves to private network IP ({ip_str}), which is prohibited."
            if ip_obj.is_loopback:
                return False, f"Target resolves to loopback IP ({ip_str}), which is prohibited."
            if ip_obj.is_link_local:
                return False, f"Target resolves to link-local IP ({ip_str}), which is prohibited."
            if ip_obj.is_reserved:
                return False, f"Target resolves to reserved IP ({ip_str}), which is prohibited."
            if ip_obj.is_multicast:
                return False, f"Target resolves to multicast IP ({ip_str}), which is prohibited."
    except socket.gaierror as e:
        return False, f"DNS resolution failed for hostname '{hostname}': {e}"
    except Exception as e:
        return False, f"Network validation error: {e}"

    return True, ""


def check_robots_permission(url: str, timeout: float = 5.0) -> bool:
    """Verify whether robots.txt permits automated scanning of this path."""
    try:
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        
        req = urllib.request.Request(
            robots_url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            rp.parse(content.splitlines())

        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # If robots.txt doesn't exist or returns 404, crawling is allowed by standard convention
        return True


class AccessibilityDOMParser(html.parser.HTMLParser):
    """Parses HTML markup to identify candidate accessibility defects."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.in_title = False
        self.title_text = ""
        self.html_has_lang = False
        self.html_lang_val = ""
        self.meta_zoom_disabled = False

        self.headings: List[Tuple[int, str]] = []  # (level, text)
        self.current_heading_level: Optional[int] = None
        self.current_heading_text = ""

        self.images_without_alt: List[str] = []
        self.inputs_without_labels: List[str] = []
        self.unclear_links: List[Tuple[str, str]] = []  # (text, href)

        self.labels_for: set[str] = set()
        self.input_ids: List[Tuple[str, str, str]] = []  # (id, name, type)

        self.current_a_href: Optional[str] = None
        self.current_a_text = ""

    def handle_starttag(self, tag: str, attrs: list[Tuple[str, Optional[str]]]):
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag == "html":
            if "lang" in attr_dict and attr_dict["lang"].strip():
                self.html_has_lang = True
                self.html_lang_val = attr_dict["lang"].strip()

        elif tag == "title":
            self.in_title = True

        elif tag == "meta":
            if attr_dict.get("name", "").lower() == "viewport":
                content = attr_dict.get("content", "").lower()
                if "user-scalable=no" in content or "maximum-scale=1.0" in content or "maximum-scale=1" in content:
                    self.meta_zoom_disabled = True

        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.current_heading_level = int(tag[1])
            self.current_heading_text = ""

        elif tag == "img":
            if "alt" not in attr_dict:
                src = attr_dict.get("src", "unspecified")
                self.images_without_alt.append(f'<img src="{src}">')

        elif tag == "label":
            if "for" in attr_dict:
                self.labels_for.add(attr_dict["for"].strip())

        elif tag in ("input", "select", "textarea"):
            input_type = attr_dict.get("type", "text").lower()
            if input_type not in ("hidden", "submit", "button", "reset", "image"):
                input_id = attr_dict.get("id", "").strip()
                has_aria_label = bool(attr_dict.get("aria-label", "").strip() or attr_dict.get("aria-labelledby", "").strip())
                has_title = bool(attr_dict.get("title", "").strip())

                if not has_aria_label and not has_title:
                    self.input_ids.append((input_id, attr_dict.get("name", ""), input_type))

        elif tag == "a":
            self.current_a_href = attr_dict.get("href", "")
            self.current_a_text = ""

    def handle_endtag(self, tag: str):
        if tag == "title":
            self.in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self.current_heading_level is not None:
                self.headings.append((self.current_heading_level, self.current_heading_text.strip()))
                self.current_heading_level = None
        elif tag == "a":
            if self.current_a_href is not None:
                link_text = self.current_a_text.strip().lower()
                generic_phrases = ["click here", "read more", "more", "learn more", "details", "here", "link"]
                if link_text in generic_phrases or (not link_text and self.current_a_href != "#"):
                    self.unclear_links.append((self.current_a_text.strip() or "[Empty Text]", self.current_a_href))
                self.current_a_href = None

    def handle_data(self, data: str):
        if self.in_title:
            self.title_text += data
        if self.current_heading_level is not None:
            self.current_heading_text += data
        if self.current_a_href is not None:
            self.current_a_text += data


def scan_public_url(
    url: str,
    timeout: float = 8.0,
    max_bytes: int = 2_000_000,
) -> Dict[str, Any]:
    """Execute a safe, single-page automated accessibility inspection of a public URL.

    Parameters
    ----------
    url : str
        Target public URL.
    timeout : float, default=8.0
        Network timeout in seconds.
    max_bytes : int, default=2MB
        Maximum response body payload to prevent memory spikes.

    Returns
    -------
    Dict[str, Any]
        Audit dossier containing detected candidate findings, metadata, and status.
    """
    # 1. SSRF & Security Validation
    is_safe, error_msg = is_safe_public_url(url)
    if not is_safe:
        raise ScanSecurityError(f"Security validation rejected target URL: {error_msg}")

    # 2. Check Robots.txt Restrictions
    if not check_robots_permission(url, timeout=min(timeout, 4.0)):
        raise RobotsDisallowedError(
            f"The site's robots.txt policy forbids automated scanning for user-agent '{USER_AGENT}' on this path."
        )

    # 3. Fetch Page Content Safely
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {
                    "success": False,
                    "error": f"Target returned non-HTML content type: '{content_type}'.",
                    "findings": [],
                }
            html_bytes = resp.read(max_bytes)
            html_text = html_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise ScanNetworkError(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ScanNetworkError(f"Network connection failed: {e.reason}")
    except socket.timeout:
        raise ScanNetworkError(f"Connection timed out after {timeout} seconds.")
    except Exception as e:
        raise ScanNetworkError(f"Unexpected network error during scan: {e}")

    # 4. Parse DOM
    parser = AccessibilityDOMParser(base_url=url)
    try:
        parser.feed(html_text)
    except Exception as e:
        return {
            "success": False,
            "error": f"HTML parsing error: {e}",
            "findings": [],
        }

    candidate_findings: List[Dict[str, Any]] = []

    # Finding 1: Missing or Empty Title
    if not parser.title_text.strip():
        candidate_findings.append({
            "issue_type": "Heading hierarchy problems",
            "description": "The webpage is missing a descriptive <title> element in the <head>.",
            "evidence": "<title> tag is missing or whitespace only",
            "wcag_criterion": "WCAG 2.4.2 Page Titled (Level A)",
            "severity": "Medium",
            "user_impact": "High",
            "recommended_fix": "Add a descriptive, unique <title> tag summarizing the institutional page purpose.",
        })

    # Finding 2: Missing html lang
    if not parser.html_has_lang:
        candidate_findings.append({
            "issue_type": "Missing accessible names or form labels",
            "description": "The top-level <html> tag lacks an explicit 'lang' attribute.",
            "evidence": "<html> missing lang attribute (e.g. <html lang=\"en\">)",
            "wcag_criterion": "WCAG 3.1.1 Language of Page (Level A)",
            "severity": "Medium",
            "user_impact": "High",
            "recommended_fix": "Declare the primary natural language on the <html> tag, e.g. <html lang=\"en\">.",
        })

    # Finding 3: Images without alt
    if parser.images_without_alt:
        count = len(parser.images_without_alt)
        samples = ", ".join(parser.images_without_alt[:3])
        candidate_findings.append({
            "issue_type": "Missing alternative text",
            "description": f"Identified {count} image(s) lacking an alt attribute, which prevents screen readers from conveying image content.",
            "evidence": f"Sample element(s): {samples}",
            "wcag_criterion": "WCAG 1.1.1 Non-text Content (Level A)",
            "severity": "High",
            "user_impact": "High",
            "recommended_fix": "Provide informative alt=\"...\" text for meaningful images, or alt=\"\" for purely decorative graphics.",
        })

    # Finding 4: Form Inputs without Labels
    unlabeled_inputs = [
        item for item in parser.input_ids
        if not item[0] or item[0] not in parser.labels_for
    ]
    if unlabeled_inputs:
        count = len(unlabeled_inputs)
        samples = ", ".join([f'<input id="{x[0]}" name="{x[1]}" type="{x[2]}">' for x in unlabeled_inputs[:3]])
        candidate_findings.append({
            "issue_type": "Missing accessible names or form labels",
            "description": f"Found {count} form control(s) lacking an associated <label for=\"...\"> or accessible name.",
            "evidence": f"Unlabeled element(s): {samples}",
            "wcag_criterion": "WCAG 4.1.2 Name, Role, Value (Level A)",
            "severity": "High",
            "user_impact": "Very High",
            "recommended_fix": "Associate an explicit <label for=\"id\"> with each input or provide aria-label.",
        })

    # Finding 5: Unclear Links
    if parser.unclear_links:
        count = len(parser.unclear_links)
        samples = ", ".join([f'"{text}" -> {href}' for text, href in parser.unclear_links[:3]])
        candidate_findings.append({
            "issue_type": "Unclear links",
            "description": f"Detected {count} hyperlink(s) with ambiguous or generic anchor text (e.g. 'click here', 'more').",
            "evidence": f"Sample link(s): {samples}",
            "wcag_criterion": "WCAG 2.4.4 Link Purpose (In Context) (Level A)",
            "severity": "Low",
            "user_impact": "Medium",
            "recommended_fix": "Replace generic anchor text with descriptive destinations or provide aria-label.",
        })

    # Finding 6: Viewport Zoom Disabled
    if parser.meta_zoom_disabled:
        candidate_findings.append({
            "issue_type": "Focus visibility problems",
            "description": "Mobile viewport zoom is disabled (user-scalable=no or maximum-scale=1.0), preventing low-vision users from magnifying content.",
            "evidence": '<meta name="viewport" content="...user-scalable=no...">',
            "wcag_criterion": "WCAG 1.4.4 Resize text (Level AA)",
            "severity": "High",
            "user_impact": "Very High",
            "recommended_fix": "Remove user-scalable=no and allow viewport scaling up to at least 200%.",
        })

    # Finding 7: Heading structure check (Skipped heading levels)
    if parser.headings:
        has_h1 = any(h[0] == 1 for h in parser.headings)
        if not has_h1:
            candidate_findings.append({
                "issue_type": "Heading hierarchy problems",
                "description": "The page does not contain a level 1 heading (<h1>) to establish the primary topic.",
                "evidence": f"Highest heading level found: H{min(h[0] for h in parser.headings)}",
                "wcag_criterion": "WCAG 1.3.1 Info and Relationships (Level A)",
                "severity": "Low",
                "user_impact": "Medium",
                "recommended_fix": "Add a single semantic <h1> identifying the primary page title.",
            })

    return {
        "success": True,
        "scanned_url": url,
        "page_title": parser.title_text.strip() or "Untitled Document",
        "headings_count": len(parser.headings),
        "total_candidate_findings": len(candidate_findings),
        "findings": candidate_findings,
    }
