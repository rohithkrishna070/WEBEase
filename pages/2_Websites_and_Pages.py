"""
================================================================================
MODULE: pages.2_Websites_and_Pages
================================================================================
1. WHAT THIS FILE DOES:
   Provides institutional entity management, essential-task page registration,
   and safe single-page public URL accessibility scanning.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Accessibility is context-dependent: a contrast error on a
     homepage marketing banner is vastly different from an unreadable form field on a
     disability bus pass application page. Registering websites and tagging critical
     journey pages establishes the operational context needed for Criterion C2 (Task Impact).
   - Technical & Architectural Rationale: Provides full CRUD interfaces for the `websites`
     and `pages` tables and wires the safe single-page URL scanner into the UI.

3. HOW IT WORKS CONCEPTUALLY:
   - Allows users to add, update, or delete institutional web portals.
   - Allows registering individual URLs and specifying whether each page is a critical citizen service.
   - Integrates the ethical public URL inspector: scans any public page and stages findings
     as `Unverified Automated Detection` in the database.

4. HOW IT WORKS TECHNICALLY:
   - Uses `st.tabs()` to organize Directory, Register Website, Register Page, and Live Inspector.
   - Form submission handlers interact with `core.database.upsert_registered_website()`,
     `upsert_registered_page()`, and `core.safe_scanner.scan_public_url()`.
   - Validates URL inputs via `core.validator.validate_url()`.
================================================================================
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from core.ui_common import setup_page_header
from core.database import (
    load_all_issues,
    get_registered_websites,
    upsert_registered_website,
    delete_registered_website,
    get_registered_pages,
    upsert_registered_page,
    delete_registered_page,
    upsert_issue,
    get_existing_ids,
)
from core.models import (
    RegisteredWebsite,
    RegisteredPage,
    AccessibilityIssue,
    RecordType,
    AuditMethod,
    VerificationStatus,
    IssueStatus,
)
from core.validator import validate_url
from core.safe_scanner import (
    scan_public_url,
    ScanSecurityError,
    RobotsDisallowedError,
    ScanNetworkError,
)

setup_page_header(
    title="Websites & Pages Management",
    icon="🌐",
    subtitle="Register institutional websites, document critical citizen service pages, and conduct safe URL scans",
)

websites_df = get_registered_websites()
pages_df = get_registered_pages()
issues_df = load_all_issues()

tab_dir, tab_reg_site, tab_reg_page, tab_scan = st.tabs([
    "🏛️ Registered Websites & Pages Directory",
    "➕ Register Website",
    "📄 Register Important Page",
    "🔍 Safe Public URL Page Inspector",
])

# ==============================================================================
# TAB 1: Directory & Drilldown
# ==============================================================================
with tab_dir:
    st.markdown("### Institutional Platforms & Audited Page Catalog")
    st.caption("Review registered public-service platforms, essential citizen journeys, and detected barriers.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Websites", len(websites_df))
    col2.metric("Registered Service Pages", len(pages_df))
    col3.metric("Total Documented Barriers", len(issues_df))

    st.markdown("#### Registered Institutional Entities")
    if websites_df.empty:
        st.info("No websites registered yet. Register a website in the 'Register Website' tab.")
    else:
        st.dataframe(
            websites_df[["name", "base_url", "institution_type", "description", "contact_email"]].rename(
                columns={
                    "name": "Institution Name",
                    "base_url": "Base URL",
                    "institution_type": "Sector / Type",
                    "description": "Description",
                    "contact_email": "Point of Contact",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### Inspect Registered Pages & Barriers by Institution")
    available_sites = sorted(websites_df["name"].tolist()) if not websites_df.empty else sorted(issues_df["website"].unique().tolist())
    
    if available_sites:
        selected_site = st.selectbox("Select institutional website to inspect:", available_sites, key="sel_dir_site")
        
        site_pages_df = get_registered_pages(website_name=selected_site)
        site_issues_df = issues_df[issues_df["website"] == selected_site]

        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown(f"##### Registered Pages ({len(site_pages_df)})")
            if site_pages_df.empty:
                st.info("No specific pages registered for this website yet.")
            else:
                for _, prow in site_pages_df.iterrows():
                    critical_badge = "⭐ **[Critical Journey]**" if prow["is_critical_journey"] else ""
                    st.markdown(
                        f"- **{prow['page_name']}** {critical_badge}  \n"
                        f"  URL: [{prow['page_url']}]({prow['page_url']})  \n"
                        f"  *Task:* {prow['essential_task']}"
                    )

        with d_col2:
            st.markdown(f"##### Documented Accessibility Barriers ({len(site_issues_df)})")
            if site_issues_df.empty:
                st.info("No accessibility barriers currently logged for this website.")
            else:
                disp_cols = ["issue_id", "page_name", "issue_type", "severity", "essential_task_impact", "status", "verification_status"]
                st.dataframe(
                    site_issues_df[disp_cols].rename(
                        columns={
                            "issue_id": "ID",
                            "page_name": "Page",
                            "issue_type": "Barrier Type",
                            "severity": "Severity",
                            "essential_task_impact": "Task Impact",
                            "status": "Status",
                            "verification_status": "Verification",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

# ==============================================================================
# TAB 2: Register Website
# ==============================================================================
with tab_reg_site:
    st.markdown("### Register New Institutional Website")
    st.caption("Add a public-service website to track its accessibility audit coverage.")

    with st.form("form_register_website"):
        w_id = f"WEB-{len(websites_df) + 1:03d}"
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            site_name = st.text_input("Institution Name *:", placeholder="e.g. State Department of Transportation")
        with w_col2:
            site_base_url = st.text_input("Base URL *:", placeholder="https://dot.state.gov")

        w_col3, w_col4 = st.columns(2)
        with w_col3:
            site_type = st.selectbox(
                "Institution Sector / Category *:",
                [
                    "Higher Education / University",
                    "Municipal / City Government",
                    "State Agency / Civic Services",
                    "Public Transit Authority",
                    "Public Library System",
                    "Public Healthcare / Vital Records",
                    "Judicial / Court Administration",
                ],
                index=2,
            )
        with w_col4:
            site_contact = st.text_input("Contact Email:", placeholder="webmaster@dot.state.gov")

        site_desc = st.text_area("Description / Service Scope:", placeholder="Overview of citizen services provided by this entity...")

        submit_site = st.form_submit_button("🏛️ Register Website", type="primary")

    if submit_site:
        if not site_name.strip():
            st.error("Institution name cannot be empty.")
        elif not validate_url(site_base_url):
            st.error("Invalid Base URL. Must begin with http:// or https:// and include a valid domain.")
        else:
            site_obj = RegisteredWebsite(
                website_id=w_id,
                name=site_name.strip(),
                base_url=site_base_url.strip(),
                institution_type=site_type,
                description=site_desc.strip(),
                contact_email=site_contact.strip(),
            )
            upsert_registered_website(site_obj)
            st.success(f"✅ Website **{site_name}** successfully registered in SQLite database!")
            st.rerun()

# ==============================================================================
# TAB 3: Register Important Page
# ==============================================================================
with tab_reg_page:
    st.markdown("### Register Important Service Page")
    st.caption("Define key transaction pages where accessibility barriers impede citizen task completion.")

    if websites_df.empty:
        st.warning("Please register at least one institutional website before adding service pages.")
    else:
        with st.form("form_register_page"):
            p_site = st.selectbox("Parent Institutional Website *:", sorted(websites_df["name"].tolist()))
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                p_name = st.text_input("Page Name *:", placeholder="e.g. Driver License Renewal Portal")
            with p_col2:
                p_url = st.text_input("Page URL *:", placeholder="https://dmv.state.gov/license-renewal")

            p_task = st.text_input(
                "Essential Citizen Task *:",
                placeholder="e.g. Submitting license renewal payment and address confirmation",
                help="Describe the critical user goal on this page.",
            )

            p_crit = st.checkbox("Mark as High-Stakes Critical Citizen Journey", value=True)
            p_notes = st.text_area("Audit Notes:", placeholder="Authentication requirements, dynamic widgets, date pickers...")

            submit_page = st.form_submit_button("📄 Register Page", type="primary")

        if submit_page:
            if not p_name.strip():
                st.error("Page name cannot be empty.")
            elif not validate_url(p_url):
                st.error("Invalid Page URL. Must begin with http:// or https:// and include a valid domain.")
            elif not p_task.strip():
                st.error("Please specify the essential citizen task for this page.")
            else:
                page_id = f"PG-{len(pages_df) + 1:03d}"
                page_obj = RegisteredPage(
                    page_id=page_id,
                    website_name=p_site,
                    page_name=p_name.strip(),
                    page_url=p_url.strip(),
                    essential_task=p_task.strip(),
                    is_critical_journey=p_crit,
                    notes=p_notes.strip(),
                )
                upsert_registered_page(page_obj)
                st.success(f"✅ Page **{p_name}** registered successfully under **{p_site}**!")
                st.rerun()

# ==============================================================================
# TAB 4: Safe Public URL Inspector
# ==============================================================================
with tab_scan:
    st.markdown("### Safe Public URL Accessibility Inspector")
    st.caption("Safely inspect a public webpage for automated candidate accessibility defects.")

    st.info(
        "🛡️ **Safe Scanning Protocol:**\n\n"
        "- **SSRF & Private Network Defense:** Localhost, loopback, and private RFC 1918 subnets are strictly blocked.\n"
        "- **Robots.txt Respected:** Scans check robots.txt policies and do not bypass access controls or CAPTCHAs.\n"
        "- **Single-Page Restriction:** Audits strictly the requested URL; no recursive crawling.\n"
        "- **Triage Requirement:** Automated detections are strictly marked **'Unverified Automated Detection'** "
        "and require manual human review before confirmation as genuine WCAG violations.",
        icon="ℹ️",
    )

    s_col1, s_col2 = st.columns([2, 1])
    with s_col1:
        scan_target_url = st.text_input(
            "Target Public URL to Inspect:",
            placeholder="https://example.gov/admissions",
            help="Enter a publicly reachable HTTP/HTTPS URL.",
        )
    with s_col2:
        associate_site = st.selectbox(
            "Associate with Institutional Website:",
            options=["[None / Standalone Scan]"] + (sorted(websites_df["name"].tolist()) if not websites_df.empty else []),
        )

    btn_scan = st.button("🔍 Run Safe Single-Page Inspection", type="primary")

    if btn_scan:
        if not scan_target_url.strip():
            st.error("Please enter a target URL.")
        else:
            with st.spinner("Inspecting public DOM and evaluating accessibility checks..."):
                try:
                    scan_results = scan_public_url(scan_target_url.strip(), timeout=8.0)
                    
                    if not scan_results.get("success"):
                        st.error(f"Scan failed: {scan_results.get('error')}")
                    else:
                        st.session_state["last_scan_results"] = scan_results
                        st.session_state["scan_site_name"] = (
                            associate_site if associate_site != "[None / Standalone Scan]" else "Public Web Target"
                        )
                except ScanSecurityError as e:
                    st.error(f"🚨 **Security Blocked:** {e}")
                except RobotsDisallowedError as e:
                    st.warning(f"🚫 **Robots.txt Policy Disallowed:** {e}")
                except ScanNetworkError as e:
                    st.error(f"🌐 **Network / Target Error:** {e}")
                except Exception as e:
                    st.error(f"Unexpected scan error: {e}")

    # Display Scan Results if present
    if "last_scan_results" in st.session_state:
        res = st.session_state["last_scan_results"]
        site_name_ctx = st.session_state.get("scan_site_name", "Public Web Target")
        findings = res.get("findings", [])

        st.markdown("---")
        st.markdown(f"#### Scan Results for: `{res['scanned_url']}`")
        st.markdown(f"**Document Title:** *{res.get('page_title')}* | **Headings Count:** `{res.get('headings_count')}`")
        st.metric("Automated Candidate Findings", len(findings))

        if not findings:
            st.success("🎉 No obvious automated DOM barriers detected by the safe parser! (Remember: keyboard traps and dynamic widgets still require manual testing).")
        else:
            st.warning(
                f"⚠️ Detected **{len(findings)} candidate accessibility defects**. "
                "These findings are provisional detections and have NOT been verified as genuine WCAG barriers."
            )
            
            findings_df = pd.DataFrame(findings)
            st.dataframe(findings_df[["issue_type", "severity", "wcag_criterion", "evidence", "recommended_fix"]], use_container_width=True, hide_index=True)

            # Ingest button to push candidate findings to verification queue
            if st.button("📥 Ingest Candidate Findings into Triage & Verification Queue", type="primary"):
                existing_ids = get_existing_ids()
                ingested = 0
                for idx, f in enumerate(findings):
                    cand_id = f"CAND-{len(existing_ids) + idx + 1:03d}"
                    issue = AccessibilityIssue(
                        issue_id=cand_id,
                        website=site_name_ctx,
                        page_url=res["scanned_url"],
                        page_name=res.get("page_title", "Scanned Page")[:50],
                        issue_type=f["issue_type"],
                        description=f["description"],
                        severity=f["severity"],
                        essential_task_impact="Medium",  # Provisional: awaiting human task evaluation
                        affected_pages=1,
                        user_impact=f.get("user_impact", "Medium"),
                        repair_effort="Medium",  # Provisional
                        frequency="Frequent",
                        evidence=f["evidence"][:400],
                        recommended_fix=f["recommended_fix"],
                        status=IssueStatus.NEEDS_REVIEW.value,
                        record_type=RecordType.IMPORTED.value,
                        audit_method=AuditMethod.SAFE_DOM_SCANNER.value,
                        verification_status=VerificationStatus.UNVERIFIED_AUTOMATED.value,
                        tool_name="Safe DOM Inspector (Automated)",
                        wcag_criterion=f["wcag_criterion"],
                        wcag_verified=0,  # Unverified candidate
                        essential_task="Provisional task: awaiting reviewer assignment",
                        notes=f"Automated scan candidate finding from {res['scanned_url']}.",
                        is_synthetic=0,
                    )
                    upsert_issue(issue)
                    ingested += 1

                st.success(f"✅ Ingested {ingested} candidate findings into the Verification Queue in '3. Accessibility Issues'!")
                del st.session_state["last_scan_results"]
                st.rerun()
