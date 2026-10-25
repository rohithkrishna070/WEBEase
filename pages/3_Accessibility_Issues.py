"""
================================================================================
MODULE: pages.3_Accessibility_Issues
================================================================================
1. WHAT THIS FILE DOES:
   Provides the core accessibility issue repository, search and filter engine,
   manual audit recording form, multi-tool audit importer, and a human-verification triage queue.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Raw automated accessibility scans produce high rates of false
     positives and lack qualitative context (e.g. how badly a broken button impairs a screen
     reader user). A dedicated triage workstation allows accessibility specialists to review
     evidence, assign qualitative ratings, discard false positives, and mark issues as Human-Verified.
   - Technical & Architectural Rationale: Centralizes all CRUD interactions for `accessibility_issues`,
     integrates `core.importer`, and provides faceted search across severity, WCAG criteria, and status.

3. HOW IT WORKS CONCEPTUALLY:
   - Repository Tab: Interactive data grid with search, filtering, and detail drawers showing evidence and code fixes.
   - Record Manual Issue Tab: Form allowing accessibility auditors to record manual keyboard or screen reader findings.
   - Import Audits Tab: Drag-and-drop file uploader for Axe CSV, Lighthouse JSON, and WAVE exports.
   - Verification Triage Tab: Dedicated workflow to convert unreviewed automated scans into human-verified violations.

4. HOW IT WORKS TECHNICALLY:
   - Queries `core.database.load_all_issues()` and `filter_issues()`.
   - Uses `st.form()` blocks to isolate input state and prevent premature page re-runs.
   - Integrates `core.importer.import_multi_tool_audit()` and `core.validator.validate_issue_fields()`.
================================================================================
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from core.ui_common import setup_page_header
from core.database import (
    load_all_issues,
    get_existing_ids,
    upsert_issue,
    delete_issue,
    filter_issues,
    get_connection,
    get_registered_websites,
    get_registered_pages,
    WCAG_MAP,
)
from core.models import (
    AccessibilityIssue,
    IssueType,
    IssueStatus,
    RecordType,
    AuditMethod,
    VerificationStatus,
    VALID_SEVERITY_RATINGS,
    VALID_TASK_IMPACT_RATINGS,
    VALID_USER_IMPACT_RATINGS,
    VALID_EFFORT_RATINGS,
    VALID_FREQUENCY_RATINGS,
    VALID_STATUSES,
    VALID_RECORD_TYPES,
    VALID_VERIFICATION_STATUSES,
)
from core.validator import validate_issue_fields
from core.importer import import_multi_tool_audit, import_csv_data
from core.exporter import export_to_csv

setup_page_header(
    title="Accessibility Issue Management & Audit Triage",
    icon="⚠️",
    subtitle="Audit public-service pages, record evidence, triage automated findings, and maintain WCAG compliance",
)

df = load_all_issues()
all_existing_ids = get_existing_ids()
websites_reg_df = get_registered_websites()
pages_reg_df = get_registered_pages()

# Status and Record-Type Summary Badges
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
total_count = len(df)
open_count = len(df[df["status"] == IssueStatus.OPEN.value])
in_prog_count = len(df[df["status"] == IssueStatus.IN_PROGRESS.value])
resolved_count = len(df[df["status"] == IssueStatus.RESOLVED.value])
unverified_count = len(df[df["verification_status"] == VerificationStatus.UNVERIFIED_AUTOMATED.value])
verified_findings = len(df[df["verification_status"] == VerificationStatus.HUMAN_VERIFIED.value])

col_m1.metric("Total Barriers", total_count)
col_m2.metric("Open Active", open_count)
col_m3.metric("In Progress", in_prog_count)
col_m4.metric("Resolved", resolved_count)
col_m5.metric("Human-Verified", verified_findings)
col_m6.metric(
    "Unverified Candidate Queue",
    unverified_count,
    delta="Review Required" if unverified_count > 0 else "Clear",
    delta_color="inverse" if unverified_count > 0 else "normal",
)

st.markdown("---")

# Main Multi-Tab Workflow
tab_explore, tab_detail, tab_manual, tab_triage, tab_import, tab_edit = st.tabs([
    "📋 Search & Filter Issues",
    "🔍 Detailed Issue Dossier",
    "✍️ Manual Audit Form",
    "⚖️ Audit Triage & Review Queue",
    "📥 Multi-Tool Audit Import & Export",
    "✏️ Edit & Delete Issue",
])

# ==============================================================================
# TAB 1: Search & Filter Issues
# ==============================================================================
with tab_explore:
    st.markdown("### Search & Dimensional Filters")
    st.caption("Filter by institutional website, page, barrier type, WCAG ratings, verification status, and audit provenance.")

    search_q = st.text_input(
        "Search issues (matches ID, page, description, evidence, WCAG, notes, tool):",
        value="",
        placeholder="e.g. color contrast, modal trap, cert_num, Lighthouse, Axe...",
        key="search_bar_query",
        label_visibility="visible",
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sel_websites = st.multiselect(
            "Filter by Website:",
            options=sorted(df["website"].dropna().unique().tolist()),
            default=[],
            placeholder="All Websites",
            key="filter_websites",
        )
    with fc2:
        available_pages = (
            df[df["website"].isin(sel_websites)]["page_name"].unique().tolist()
            if sel_websites
            else df["page_name"].unique().tolist()
        )
        sel_pages = st.multiselect(
            "Filter by Page:",
            options=sorted(available_pages),
            default=[],
            placeholder="All Pages",
            key="filter_pages",
        )
    with fc3:
        sel_types = st.multiselect(
            "Filter by Barrier Type:",
            options=sorted(df["issue_type"].dropna().unique().tolist()),
            default=[],
            placeholder="All Barrier Types",
            key="filter_types",
        )
    with fc4:
        sel_statuses = st.multiselect(
            "Filter by Remediation Status:",
            options=VALID_STATUSES,
            default=[],
            placeholder="All Statuses",
            key="filter_statuses",
        )

    fc5, fc6, fc7, fc8 = st.columns(4)
    with fc5:
        sel_severities = st.multiselect(
            "Filter by Barrier Severity:",
            options=VALID_SEVERITY_RATINGS,
            default=[],
            placeholder="All Severities",
            key="filter_severities",
        )
    with fc6:
        sel_tasks = st.multiselect(
            "Filter by Essential-Task Impact:",
            options=VALID_TASK_IMPACT_RATINGS,
            default=[],
            placeholder="All Task Impacts",
            key="filter_task_impacts",
        )
    with fc7:
        sel_verif = st.multiselect(
            "Filter by Verification Status:",
            options=VALID_VERIFICATION_STATUSES,
            default=[],
            placeholder="All Verification Statuses",
            key="filter_verification_statuses",
        )
    with fc8:
        sel_record_types = st.multiselect(
            "Filter by Record Provenance:",
            options=VALID_RECORD_TYPES,
            default=[],
            placeholder="All Record Types",
            key="filter_record_types",
        )

    filtered_df = filter_issues(
        df,
        websites=sel_websites or None,
        pages=sel_pages or None,
        issue_types=sel_types or None,
        severities=sel_severities or None,
        task_impacts=sel_tasks or None,
        statuses=sel_statuses or None,
        record_types=sel_record_types or None,
        verification_statuses=sel_verif or None,
        search_query=search_q,
    )

    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} accessibility barriers**")
    with t_col2:
        filtered_csv = export_to_csv(filtered_df)
        st.download_button(
            label="📥 Export Filtered (CSV)",
            data=filtered_csv,
            file_name="filtered_accessibility_issues.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_filtered",
        )

    disp_cols = [
        "issue_id",
        "website",
        "page_name",
        "issue_type",
        "severity",
        "essential_task_impact",
        "verification_status",
        "tool_name",
        "status",
    ]
    st.dataframe(
        filtered_df[disp_cols].rename(
            columns={
                "issue_id": "Issue ID",
                "website": "Website",
                "page_name": "Page",
                "issue_type": "Barrier Type",
                "severity": "Severity",
                "essential_task_impact": "Task Impact",
                "verification_status": "Verification",
                "tool_name": "Tool / Method",
                "status": "Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ==============================================================================
# TAB 2: Detailed Issue Dossier
# ==============================================================================
with tab_detail:
    st.markdown("### Detailed Barrier Dossier & Evidence View")
    st.caption("Inspect technical code evidence, audit provenance, verified WCAG criteria, and citizen impact.")

    inspect_options = filtered_df["issue_id"].tolist() if not filtered_df.empty else df["issue_id"].tolist()
    if not inspect_options:
        st.warning("No issues available to inspect.")
    else:
        selected_insp_id = st.selectbox(
            "Select an Issue ID to inspect full dossier:",
            options=inspect_options,
            key="insp_select_box",
            label_visibility="visible",
        )

        issue_rec = df[df["issue_id"] == selected_insp_id].iloc[0]

        # Provenance and Verification Badges
        v_status = issue_rec.get("verification_status", VerificationStatus.HUMAN_VERIFIED.value)
        is_wcag_conf = bool(int(issue_rec.get("wcag_verified", 1)))

        p_c1, p_c2 = st.columns([2, 1])
        with p_c1:
            if v_status == VerificationStatus.UNVERIFIED_AUTOMATED.value:
                st.warning("⚠️ **Unverified Automated Detection:** Requires manual human verification before ranking.", icon="🤖")
            elif v_status == VerificationStatus.HUMAN_VERIFIED.value:
                st.success("✅ **Human-Verified Finding:** Verified by accessibility reviewer with supporting evidence.", icon="👤")
            elif v_status == VerificationStatus.FALSE_POSITIVE.value:
                st.info("🚫 **False Positive Discarded:** Marked as non-barrier or benign element.", icon="🗑️")
            else:
                st.info("✍️ **Manual Observation:** Directly logged by auditor.", icon="📝")

        with p_c2:
            st.markdown(f"**Audit Tool:** `{issue_rec.get('tool_name', 'Manual Inspection')}`")
            st.markdown(f"**WCAG Confirmed:** `{'Yes' if is_wcag_conf else 'Pending Verification'}`")

        d_col1, d_col2 = st.columns([3, 2])

        with d_col1:
            st.markdown(f"#### `{issue_rec['issue_id']}`: {issue_rec['issue_type']}")
            st.markdown(f"**Website:** {issue_rec['website']}")
            st.markdown(f"**Audited Page URL:** [{issue_rec['page_name']} ({issue_rec['page_url']})]({issue_rec['page_url']})")
            
            st.markdown("##### Description of Barrier")
            st.markdown(issue_rec["description"])

            st.markdown("##### Technical Evidence / DOM Selector")
            st.code(issue_rec["evidence"], language="html")

            st.markdown("##### Audit Method & Tooling")
            st.markdown(f"- **Method:** `{issue_rec.get('audit_method', AuditMethod.MANUAL_INSPECTION.value)}`")
            st.markdown(f"- **Audit Source:** `{issue_rec.get('tool_name', 'Manual')}`")

            st.markdown("##### Applicable WCAG Success Criterion")
            wcag_val = issue_rec.get("wcag_criterion") or WCAG_MAP.get(issue_rec["issue_type"], "WCAG 2.1 Criterion")
            st.info(f"📘 **{wcag_val}**")

        with d_col2:
            st.markdown("#### Citizen Impact & Remediation Dossier")
            st.markdown(f"**Essential Citizen Task Obstructed:**  \n🎯 *{issue_rec.get('essential_task', 'Completing web workflow')}*")
            st.markdown(f"**Affected User Population:**  \n♿ *{issue_rec.get('user_impact', 'Medium')} severity on assistive tech users*")

            st.markdown("##### Soft Computing Decision Ratings")
            st.markdown(
                f"""
                - **C1. Barrier Severity:** `{issue_rec['severity']}`
                - **C2. Essential-Task Impact:** `{issue_rec['essential_task_impact']}`
                - **C3. Affected Pages:** `{issue_rec['affected_pages']}` page(s)
                - **C4. User Impact:** `{issue_rec['user_impact']}`
                - **C5. Repair Effort:** `{issue_rec['repair_effort']}`
                - **C6. Issue Frequency:** `{issue_rec['frequency']}`
                """
            )

            st.markdown("##### Recommended Code Remediation")
            st.success(f"🛠️ **Remediation Action:**  \n{issue_rec['recommended_fix']}")

            st.markdown("---")
            st.markdown("##### Update Status & Remediator Notes")
            curr_status = issue_rec.get("status", IssueStatus.OPEN.value)
            status_idx = VALID_STATUSES.index(curr_status) if curr_status in VALID_STATUSES else 0
            
            updated_status = st.selectbox(
                "Remediation Status:",
                options=VALID_STATUSES,
                index=status_idx,
                key=f"status_select_quick_{selected_insp_id}",
            )

            updated_notes = st.text_area(
                "Audit / Remediation Notes:",
                value=str(issue_rec.get("notes", "") or ""),
                placeholder="Add developer notes, remediation PR link, or audit verification comments...",
                key=f"notes_text_quick_{selected_insp_id}",
            )

            if st.button("💾 Save Status & Notes", key=f"btn_save_notes_{selected_insp_id}", type="primary"):
                conn = get_connection()
                with conn:
                    conn.execute(
                        "UPDATE accessibility_issues SET status = ?, notes = ? WHERE issue_id = ?",
                        (updated_status, updated_notes, selected_insp_id),
                    )
                conn.close()
                st.success(f"Saved status and notes for {selected_insp_id}!")
                st.rerun()

# ==============================================================================
# TAB 3: Manual Audit Form
# ==============================================================================
with tab_manual:
    st.markdown("### Manual Accessibility Audit Entry")
    st.caption("Record human-verified findings from manual keyboard audits, screen-reader sessions, and expert DOM inspections.")

    with st.form("form_manual_audit", clear_on_submit=False):
        st.markdown("#### 1. Identification & Registered Target")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            suggested_id = f"MAN-{len(df) + 1:03d}"
            m_id = st.text_input("Issue ID *:", value=suggested_id, help="Unique identifier for this manual finding.")
        with m_col2:
            site_options = sorted(websites_reg_df["name"].tolist()) if not websites_reg_df.empty else sorted(df["website"].unique().tolist())
            m_website = st.selectbox("Target Institution *:", options=site_options)
        with m_col3:
            registered_pages_for_site = (
                get_registered_pages(website_name=m_website)["page_name"].tolist()
                if not websites_reg_df.empty else []
            )
            m_page_name = st.selectbox(
                "Target Service Page *:",
                options=registered_pages_for_site + ["[Custom / Unlisted Page]"],
            )

        m_col4, m_col5, m_col6 = st.columns(3)
        with m_col4:
            m_custom_page_name = st.text_input(
                "Custom Page Name (if unlisted):",
                value="" if m_page_name != "[Custom / Unlisted Page]" else "Service Portal",
            )
        with m_col5:
            m_page_url = st.text_input("Page URL *:", placeholder="https://agency.gov/service")
        with m_col6:
            m_method = st.selectbox(
                "Manual Audit Method *:",
                [
                    AuditMethod.MANUAL_KEYBOARD.value,
                    AuditMethod.SCREEN_READER.value,
                    AuditMethod.MANUAL_INSPECTION.value,
                ],
                index=0,
            )

        st.markdown("#### 2. Barrier Classification & WCAG Criteria")
        m_col7, m_col8, m_col9 = st.columns(3)
        with m_col7:
            m_issue_type = st.selectbox("Barrier Type *:", options=[t.value for t in IssueType])
        with m_col8:
            default_wcag = WCAG_MAP.get(m_issue_type, "WCAG 2.1 Principle")
            m_wcag = st.text_input("Verified WCAG Criterion *:", value=default_wcag)
        with m_col9:
            m_essential_task = st.text_input("Essential Citizen Task Blocked *:", placeholder="e.g. Submitting medical clinic appointment")

        st.markdown("#### 3. Soft Computing Decision Ratings (C1–C6)")
        m_r1, m_r2, m_r3 = st.columns(3)
        with m_r1:
            m_severity = st.selectbox("C1. Barrier Severity *:", options=VALID_SEVERITY_RATINGS, index=3)
        with m_r2:
            m_task_impact = st.selectbox("C2. Essential-Task Impact *:", options=VALID_TASK_IMPACT_RATINGS, index=3)
        with m_r3:
            m_pages_cnt = st.number_input("C3. Affected Pages Count *:", min_value=1, max_value=100, value=1, step=1)

        m_r4, m_r5, m_r6 = st.columns(3)
        with m_r4:
            m_user_impact = st.selectbox("C4. User Impact *:", options=VALID_USER_IMPACT_RATINGS, index=3)
        with m_r5:
            m_effort = st.selectbox("C5. Repair Effort *:", options=VALID_EFFORT_RATINGS, index=1)
        with m_r6:
            m_freq = st.selectbox("C6. Issue Frequency *:", options=VALID_FREQUENCY_RATINGS, index=2)

        st.markdown("#### 4. Technical Observation & Fix Action")
        m_desc = st.text_area("Detailed Description *:", placeholder="Describe the barrier and user journey disruption observed...")
        m_evidence = st.text_input("Evidence / Selector / Keypress Trace *:", placeholder="e.g. Tab trap inside #upload-dialog; Escape suppressed")
        m_fix = st.text_area("Recommended Fix Action *:", placeholder="Actionable guidance to remediate the barrier...")

        m_status = st.selectbox("Initial Remediation Status *:", options=VALID_STATUSES, index=0)
        m_notes = st.text_input("Auditor Notes:", placeholder="Assessor initials, testing environment, assistive tech tested...")

        btn_submit_manual = st.form_submit_button("✍️ Save Manual Audit Finding", type="primary")

    if btn_submit_manual:
        eff_page_name = m_custom_page_name.strip() if m_page_name == "[Custom / Unlisted Page]" else m_page_name
        candidate = {
            "issue_id": m_id.strip(),
            "website": m_website,
            "page_url": m_page_url.strip(),
            "page_name": eff_page_name,
            "issue_type": m_issue_type,
            "description": m_desc.strip(),
            "severity": m_severity,
            "essential_task_impact": m_task_impact,
            "affected_pages": m_pages_cnt,
            "user_impact": m_user_impact,
            "repair_effort": m_effort,
            "frequency": m_freq,
            "evidence": m_evidence.strip(),
            "recommended_fix": m_fix.strip(),
            "status": m_status,
            "record_type": RecordType.MANUAL.value,
            "audit_method": m_method,
            "verification_status": VerificationStatus.HUMAN_VERIFIED.value,
            "tool_name": f"Manual Audit ({m_method.split(' ')[0]})",
            "wcag_criterion": m_wcag.strip(),
            "wcag_verified": 1,
            "essential_task": m_essential_task.strip(),
            "notes": m_notes.strip(),
            "reviewer_notes": "Audited and human-verified manually.",
        }

        errs = validate_issue_fields(candidate, existing_ids=all_existing_ids, is_edit=False)
        if errs:
            st.error("⚠️ **Validation Errors:**")
            for e in errs:
                st.markdown(f"- {e}")
        else:
            issue_obj = AccessibilityIssue.from_dict(candidate)
            upsert_issue(issue_obj)
            st.success(f"✅ Manual finding **{issue_obj.issue_id}** recorded and saved to SQLite!")
            st.rerun()

# ==============================================================================
# TAB 4: Audit Triage & Review Queue
# ==============================================================================
with tab_triage:
    st.markdown("### Automated Audit Triage & Manual Review Queue")
    st.caption("Review candidate findings from Lighthouse, Axe, WAVE, or DOM scans before confirming WCAG violations.")

    st.info(
        "🧠 **Why Manual Review is Essential:**\n\n"
        "Automated tools flag syntactic patterns but cannot determine:\n"
        "1. Whether an issue actually halts an **essential citizen task** (e.g. paying tuition vs reading an archive footer).\n"
        "2. Whether an element is decorative or a genuine **false positive**.\n"
        "Review each automated candidate finding below, confirm the WCAG criterion, and assign task impact before prioritizing.",
        icon="ℹ️",
    )

    unverified_df = df[df["verification_status"] == VerificationStatus.UNVERIFIED_AUTOMATED.value]

    if unverified_df.empty:
        st.success("🎉 **Triage Queue Clear:** All automated findings in the database have been human-reviewed and verified!")
    else:
        st.warning(f"There are **{len(unverified_df)} candidate finding(s)** awaiting human review.")

        triage_options = unverified_df["issue_id"].tolist()
        sel_triage_id = st.selectbox("Select Candidate Issue to Triage:", triage_options)

        t_row = unverified_df[unverified_df["issue_id"] == sel_triage_id].iloc[0]

        t_c1, t_c2 = st.columns([1, 1])
        with t_c1:
            st.markdown(f"#### `{t_row['issue_id']}`: {t_row['issue_type']}")
            st.markdown(f"**Target Website:** {t_row['website']}")
            st.markdown(f"**Target URL:** [{t_row['page_name']}]({t_row['page_url']})")
            st.markdown(f"**Automated Source Tool:** `{t_row.get('tool_name', 'Automated Tool')}`")
            st.markdown(f"**Description:**  \n{t_row['description']}")
            st.markdown("**Automated Code Evidence:**")
            st.code(t_row["evidence"], language="html")

        with t_c2:
            st.markdown("#### Human Review & WCAG Verification Form")
            with st.form(f"form_triage_{sel_triage_id}"):
                conf_wcag = st.text_input(
                    "Applicable WCAG Success Criterion *:",
                    value=str(t_row.get("wcag_criterion") or WCAG_MAP.get(t_row["issue_type"], "WCAG 2.1 Criterion")),
                )
                conf_task = st.text_input(
                    "Essential Citizen Task Blocked *:",
                    placeholder="Describe how this blocks citizen completion of key task...",
                )
                conf_task_impact = st.selectbox(
                    "C2. Essential-Task Impact *:",
                    options=VALID_TASK_IMPACT_RATINGS,
                    index=2,
                    help="Rate how severely this obstructs citizen workflow.",
                )
                conf_effort = st.selectbox(
                    "C5. Estimated Repair Effort *:",
                    options=VALID_EFFORT_RATINGS,
                    index=1,
                )
                conf_notes = st.text_area(
                    "Reviewer Verification Notes:",
                    placeholder="Confirm manual testing result, screen reader behavior, or reason...",
                )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    btn_confirm = st.form_submit_button("✅ Confirm as Verified Barrier", type="primary")
                with b_col2:
                    btn_discard = st.form_submit_button("🗑️ Discard as False Positive")

            if btn_confirm:
                if not conf_task.strip():
                    st.error("Please specify the essential citizen task impacted before confirming.")
                else:
                    conn = get_connection()
                    with conn:
                        conn.execute(
                            """
                            UPDATE accessibility_issues
                            SET verification_status = ?,
                                wcag_verified = 1,
                                status = 'Open',
                                wcag_criterion = ?,
                                essential_task = ?,
                                essential_task_impact = ?,
                                repair_effort = ?,
                                reviewer_notes = ?
                            WHERE issue_id = ?
                            """,
                            (
                                VerificationStatus.HUMAN_VERIFIED.value,
                                conf_wcag.strip(),
                                conf_task.strip(),
                                conf_task_impact,
                                conf_effort,
                                conf_notes.strip() or "Verified by human auditor.",
                                sel_triage_id,
                            ),
                        )
                    conn.close()
                    st.success(f"Issue {sel_triage_id} confirmed as verified WCAG violation and promoted to active prioritization!")
                    st.rerun()

            if btn_discard:
                conn = get_connection()
                with conn:
                    conn.execute(
                        """
                        UPDATE accessibility_issues
                        SET verification_status = ?,
                            status = 'Resolved',
                            notes = ?
                        WHERE issue_id = ?
                        """,
                        (
                            VerificationStatus.FALSE_POSITIVE.value,
                            f"Discarded as false positive: {conf_notes.strip() or 'No genuine accessibility barrier present.'}",
                            sel_triage_id,
                        ),
                    )
                conn.close()
                st.info(f"Candidate {sel_triage_id} marked as False Positive and discarded from active prioritization.")
                st.rerun()

# ==============================================================================
# TAB 5: Multi-Tool Audit Import & Export
# ==============================================================================
with tab_import:
    st.markdown("### Multi-Tool Audit Import & Export")
    st.caption("Import findings from Google Lighthouse, Axe-core, WAVE, or SmartWeb Access standard CSVs.")

    st.markdown(
        """
        #### 📖 Supported Tool Schema Workflows
        | Audit Tool | Supported Format | Workflow & Automated Mapping |
        | :--- | :--- | :--- |
        | **Google Lighthouse** | JSON export | Ingests audits where `score == 0`, parses failed selector snippets, flags as unverified candidate findings. |
        | **Axe-core / Axe DevTools** | CSV export | Maps `Impact` (`critical`, `serious`) to Severity, extracts target selectors, flags as unverified candidate findings. |
        | **WAVE Accessibility Tool** | CSV export | Normalizes WAVE error codes, maps selectors, flags as unverified candidate findings. |
        | **SmartWeb Access** | Standard CSV | Ingests all 18 fields with full linguistic MCDM rating validation. |
        """
    )

    imp_col1, imp_col2 = st.columns([1, 1])

    with imp_col1:
        st.markdown("#### 📥 Ingest Audit File")
        
        target_site_for_imp = st.selectbox(
            "Associate Imported Issues with Website:",
            options=sorted(websites_reg_df["name"].tolist()) if not websites_reg_df.empty else ["Public Service Website"],
            key="imp_site_select",
        )
        
        target_url_for_imp = st.text_input(
            "Target Page URL (for tools that omit full URL):",
            value="https://example.gov",
            key="imp_url_input",
        )

        overwrite_imp = st.checkbox(
            "Overwrite existing records if Issue ID matches",
            value=False,
            key="imp_overwrite_box",
        )

        uploaded_audit = st.file_uploader(
            "Upload Audit File (CSV or Lighthouse JSON):",
            type=["csv", "json"],
            key="audit_file_uploader",
        )

        if uploaded_audit is not None:
            raw_bytes = uploaded_audit.getvalue()
            fname = uploaded_audit.name

            report = import_multi_tool_audit(
                raw_bytes,
                filename=fname,
                website_name=target_site_for_imp,
                target_page_url=target_url_for_imp,
                overwrite_existing=overwrite_imp,
            )

            if not report["success"]:
                st.error(f"❌ **Import Error:** {report['global_errors'][0]}")
            else:
                st.success(
                    f"🎉 **Ingestion Complete:** Detected format **{report['tool_detected']}**!  \n"
                    f"- Ingested: `{report['imported_count']}`  \n"
                    f"- Skipped duplicates: `{report['skipped_count']}`"
                )
                if report["tool_detected"] != "SmartWeb Access Standard CSV":
                    st.info("👉 Imported findings have been routed to the **Audit Triage & Review Queue** for human verification.")

                if st.button("🔄 Refresh Data View", key="btn_refresh_after_tool_imp"):
                    st.rerun()

    with imp_col2:
        st.markdown("#### 📤 Export Complete Issue Database")
        st.markdown("Download all stored barriers including verification statuses, tool sources, and WCAG criteria.")
        full_csv = export_to_csv(df)
        st.download_button(
            label="📥 Download Full Database (CSV)",
            data=full_csv,
            file_name="smartweb_access_all_issues.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_all_issues_p3",
        )

# ==============================================================================
# TAB 6: Edit & Delete Issue
# ==============================================================================
with tab_edit:
    st.markdown("### Edit or Remove Accessibility Barrier")
    st.caption("Update barrier parameters or permanently delete records with explicit safety confirmation.")

    edit_target_id = st.selectbox(
        "Select Issue ID to Edit or Delete:",
        options=sorted(df["issue_id"].tolist()),
        key="edit_target_select_p3",
    )

    if edit_target_id:
        target_row = df[df["issue_id"] == edit_target_id].iloc[0]

        with st.form("form_edit_issue_p3"):
            st.markdown(f"#### Editing Issue: `{edit_target_id}`")
            
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                e_website = st.text_input("Website *:", value=str(target_row["website"]))
            with e_col2:
                e_page_name = st.text_input("Page Name *:", value=str(target_row["page_name"]))
            with e_col3:
                e_page_url = st.text_input("Page URL *:", value=str(target_row["page_url"]))

            e_col4, e_col5, e_col6 = st.columns(3)
            with e_col4:
                curr_type = target_row["issue_type"]
                type_idx = [t.value for t in IssueType].index(curr_type) if curr_type in [t.value for t in IssueType] else 0
                e_issue_type = st.selectbox("Issue Type *:", options=[t.value for t in IssueType], index=type_idx)
            with e_col5:
                e_wcag = st.text_input("WCAG Criterion:", value=str(target_row.get("wcag_criterion", "") or ""))
            with e_col6:
                e_essential_task = st.text_input("Essential Citizen Task:", value=str(target_row.get("essential_task", "") or ""))

            e_col7, e_col8, e_col9 = st.columns(3)
            with e_col7:
                s_idx = VALID_SEVERITY_RATINGS.index(target_row["severity"]) if target_row["severity"] in VALID_SEVERITY_RATINGS else 2
                e_severity = st.selectbox("C1. Severity *:", options=VALID_SEVERITY_RATINGS, index=s_idx)
            with e_col8:
                t_idx = VALID_TASK_IMPACT_RATINGS.index(target_row["essential_task_impact"]) if target_row["essential_task_impact"] in VALID_TASK_IMPACT_RATINGS else 2
                e_task_impact = st.selectbox("C2. Task Impact *:", options=VALID_TASK_IMPACT_RATINGS, index=t_idx)
            with e_col9:
                e_pages_cnt = st.number_input("C3. Affected Pages Count *:", min_value=1, max_value=100, value=int(target_row.get("affected_pages", 1)))

            e_col10, e_col11, e_col12 = st.columns(3)
            with e_col10:
                u_idx = VALID_USER_IMPACT_RATINGS.index(target_row["user_impact"]) if target_row["user_impact"] in VALID_USER_IMPACT_RATINGS else 2
                e_user_impact = st.selectbox("C4. User Impact *:", options=VALID_USER_IMPACT_RATINGS, index=u_idx)
            with e_col11:
                ef_idx = VALID_EFFORT_RATINGS.index(target_row["repair_effort"]) if target_row["repair_effort"] in VALID_EFFORT_RATINGS else 1
                e_effort = st.selectbox("C5. Repair Effort *:", options=VALID_EFFORT_RATINGS, index=ef_idx)
            with e_col12:
                f_idx = VALID_FREQUENCY_RATINGS.index(target_row["frequency"]) if target_row["frequency"] in VALID_FREQUENCY_RATINGS else 2
                e_frequency = st.selectbox("C6. Frequency *:", options=VALID_FREQUENCY_RATINGS, index=f_idx)

            e_desc = st.text_area("Description *:", value=str(target_row["description"]))
            e_evidence = st.text_input("Evidence / Code *:", value=str(target_row["evidence"]))
            e_fix = st.text_area("Recommended Fix *:", value=str(target_row["recommended_fix"]))

            e_col13, e_col14, e_col15 = st.columns(3)
            with e_col13:
                st_idx = VALID_STATUSES.index(target_row["status"]) if target_row["status"] in VALID_STATUSES else 0
                e_status = st.selectbox("Status *:", options=VALID_STATUSES, index=st_idx)
            with e_col14:
                rec_idx = VALID_RECORD_TYPES.index(target_row.get("record_type", RecordType.SYNTHETIC.value)) if target_row.get("record_type") in VALID_RECORD_TYPES else 0
                e_rec_type = st.selectbox("Record Provenance *:", options=VALID_RECORD_TYPES, index=rec_idx)
            with e_col15:
                v_idx = VALID_VERIFICATION_STATUSES.index(target_row.get("verification_status", VerificationStatus.HUMAN_VERIFIED.value)) if target_row.get("verification_status") in VALID_VERIFICATION_STATUSES else 0
                e_verif = st.selectbox("Verification Status *:", options=VALID_VERIFICATION_STATUSES, index=v_idx)

            e_notes = st.text_input("Remediator Notes:", value=str(target_row.get("notes", "") or ""))

            btn_update = st.form_submit_button("💾 Save Changes", type="primary")

        if btn_update:
            edit_data = {
                "issue_id": edit_target_id,
                "website": e_website,
                "page_url": e_page_url,
                "page_name": e_page_name,
                "issue_type": e_issue_type,
                "description": e_desc,
                "severity": e_severity,
                "essential_task_impact": e_task_impact,
                "affected_pages": e_pages_cnt,
                "user_impact": e_user_impact,
                "repair_effort": e_effort,
                "frequency": e_frequency,
                "evidence": e_evidence,
                "recommended_fix": e_fix,
                "status": e_status,
                "record_type": e_rec_type,
                "audit_method": str(target_row.get("audit_method", AuditMethod.MANUAL_INSPECTION.value)),
                "verification_status": e_verif,
                "tool_name": str(target_row.get("tool_name", "Manual Inspection")),
                "wcag_criterion": e_wcag,
                "wcag_verified": 1 if e_verif != VerificationStatus.UNVERIFIED_AUTOMATED.value else 0,
                "essential_task": e_essential_task,
                "notes": e_notes,
            }
            edit_errs = validate_issue_fields(edit_data, existing_ids=all_existing_ids, is_edit=True)
            if edit_errs:
                st.error("⚠️ **Validation Errors:**")
                for err in edit_errs:
                    st.markdown(f"- {err}")
            else:
                updated_issue = AccessibilityIssue.from_dict(edit_data)
                upsert_issue(updated_issue)
                st.success(f"✅ Issue **{edit_target_id}** updated successfully!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🗑️ Danger Zone: Delete Barrier")
        confirm_del = st.checkbox(
            f"I confirm that I want to permanently delete issue **{edit_target_id}**.",
            key=f"confirm_del_{edit_target_id}",
        )
        if st.button("🚨 Delete Issue Permanently", disabled=not confirm_del, key=f"btn_del_{edit_target_id}"):
            if delete_issue(edit_target_id):
                st.success(f"Issue **{edit_target_id}** deleted.")
                st.rerun()
            else:
                st.error("Failed to delete issue.")
