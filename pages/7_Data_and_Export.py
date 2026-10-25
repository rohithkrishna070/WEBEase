"""
================================================================================
MODULE: pages.7_Data_and_Export
================================================================================
1. WHAT THIS FILE DOES:
   Provides the research data management, report generation, and export hub, allowing
   stakeholders to download prioritized repair outputs in CSV, JSON, and comprehensive
   Markdown research reports for compliance audits and reproducibility verification.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: A decision support tool must not be a black box. Scientific
     integrity and public administration demand transparent provenance. Exporting full
     reproducibility dossiers enables external auditors and academic reviewers to independently
     verify rankings using the exact mathematical parameters recorded during the session.
   - Technical & Architectural Rationale: Connects Streamlit download buttons (`st.download_button`)
     directly to `core.exporter` serialization functions and provides database maintenance controls.

3. HOW IT WORKS CONCEPTUALLY:
   - Formats prioritized issues with ranking metrics into tabular CSV files.
   - Archives raw issues, weights vector, and correlation stats into structured JSON.
   - Generates publication-ready Markdown research reports summarizing methodology and findings.
   - Displays database reset and re-seeding controls.

4. HOW IT WORKS TECHNICALLY:
   - Calls `core.exporter.export_to_csv()`, `export_to_json()`, and `generate_research_reproducibility_report()`.
   - Uses `core.database.seed_if_empty()` and SQLite connection operations.
================================================================================
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from core.ui_common import setup_page_header
from core.database import load_all_issues, seed_if_empty
from core.mcdm_bridge import run_prioritization
from core.exporter import export_to_csv, export_to_json, generate_research_reproducibility_report

setup_page_header(
    title="Data Management, Export & Research Reproducibility",
    icon="💾",
    subtitle="Export prioritization findings, download scientific research reports, and inspect reproducibility parameters",
)

df = load_all_issues()
results_df, meta = run_prioritization(df)

# ==============================================================================
# 1. Export Decision Support Outputs
# ==============================================================================
st.subheader("1. Export Decision Support Outputs & Research Reports")
st.markdown("Download ranked repair priorities and statistical outputs for stakeholder reporting and academic archiving.")

e_col1, e_col2, e_col3 = st.columns(3)

with e_col1:
    csv_data = export_to_csv(results_df)
    st.download_button(
        label="📥 Download Rankings (CSV)",
        data=csv_data,
        file_name="smartweb_access_prioritization.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Complete tabular dataset with original ratings, TOPSIS scores, VIKOR indices, and recommended repairs.")

with e_col2:
    json_data = export_to_json(results_df, meta)
    st.download_button(
        label="📥 Download Archive (JSON)",
        data=json_data,
        file_name="smartweb_access_prioritization.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption("Machine-readable JSON archive containing full MCDM parameters, correlation metrics, and records.")

with e_col3:
    md_report = generate_research_reproducibility_report(results_df, meta)
    st.download_button(
        label="📥 Download Research Report (Markdown)",
        data=md_report,
        file_name="smartweb_access_research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.caption("Publication-ready Markdown report recording exact weights, rankings, and statistical analyses.")

st.markdown("---")

# ==============================================================================
# 2. Research Reproducibility Audit Parameters
# ==============================================================================
st.subheader("📋 2. Active Research Reproducibility Audit Vector")
st.markdown("This audit record documents all active mathematical settings, weighting vectors, and scope parameters for experimental reproducibility.")

rep_col1, rep_col2 = st.columns(2)

with rep_col1:
    st.markdown("#### Mathematical Engine Specifications")
    st.markdown(
        f"""
        - **Fuzzy TOPSIS Variant:** Chen (2000) Linear Scale Transformation with Euclidean Vertex Distance
        - **VIKOR Variant:** `{meta.get('vikor_formulation', 'Fuzzy VIKOR').capitalize()}`
        - **Compromise Strategy Weight ($v$):** `{meta.get('vikor_v', 0.5)}`
        - **Evaluated Alternatives Count:** `{meta.get('total_evaluated', len(results_df))}`
        - **Spearman Correlation ($\rho$):** `{meta.get('spearman_rho', 0.0):.4f}`
        - **Kendall Correlation ($\\tau$):** `{meta.get('kendall_tau', 0.0):.4f}`
        """
    )

with rep_col2:
    st.markdown("#### Criteria Weight Allocation Vector")
    w = meta.get("weights", [1 / 6] * 6)
    w_df = pd.DataFrame([
        {"Criterion": "C1: Barrier Severity", "Type": "Benefit", "Weight": f"{w[0]:.4f}", "Share": f"{w[0]*100:.1f}%"},
        {"Criterion": "C2: Essential-Task Impact", "Type": "Benefit", "Weight": f"{w[1]:.4f}", "Share": f"{w[1]*100:.1f}%"},
        {"Criterion": "C3: Affected Pages", "Type": "Benefit", "Weight": f"{w[2]:.4f}", "Share": f"{w[2]*100:.1f}%"},
        {"Criterion": "C4: User Impact", "Type": "Benefit", "Weight": f"{w[3]:.4f}", "Share": f"{w[3]*100:.1f}%"},
        {"Criterion": "C5: Repair Effort", "Type": "Cost", "Weight": f"{w[4]:.4f}", "Share": f"{w[4]*100:.1f}%"},
        {"Criterion": "C6: Issue Frequency", "Type": "Benefit", "Weight": f"{w[5]:.4f}", "Share": f"{w[5]*100:.1f}%"},
    ])
    st.dataframe(w_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==============================================================================
# 3. Import Multi-Tool Audit Data
# ==============================================================================
st.subheader("3. Ingest External Audit Data (CSV / JSON)")
st.markdown(
    """
    Ingest audit results from **Lighthouse (JSON)**, **Axe-core (CSV)**, **WAVE (CSV)**, or
    **SmartWeb standard CSV**. Ingested candidate findings are tagged as imported evidence and queued for human triage.
    """
)

from core.importer import import_multi_tool_audit

uploaded_file = st.file_uploader(
    "Upload Accessibility Audit Data (CSV or JSON):",
    type=["csv", "json"],
    key="uploader_data_export_page",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    fname = uploaded_file.name
    overwrite = st.checkbox("Overwrite existing records with matching IDs", value=True, key="chk_overwrite_p7")
    report = import_multi_tool_audit(
        file_bytes,
        filename=fname,
        overwrite_existing=overwrite,
    )

    if not report["success"]:
        err_msg = report["global_errors"][0] if report.get("global_errors") else "Unknown upload error."
        st.error(f"❌ **Upload Failed:** {err_msg}")
    else:
        st.info(
            f"📊 **Ingestion Summary:** Read: `{report['total_rows_read']}` | "
            f"Imported: `{report['imported_count']}` | Skipped: `{report['skipped_count']}` | "
            f"Tool Detected: `{report.get('tool_detected', 'Audit File')}`"
        )
        if report["imported_count"] > 0:
            st.success(f"Successfully ingested {report['imported_count']} records! Review candidates in the Audit Triage Queue.")
            if st.button("🔄 Refresh Application Data"):
                st.rerun()

st.markdown("---")

# ==============================================================================
# 4. Benchmark Maintenance
# ==============================================================================
st.subheader("4. Benchmark Maintenance & Reset")
st.markdown("Restore the SQLite database to the original 38 synthetic demonstration records if needed.")

if st.button("🔄 Reset Database to Demonstration Benchmark"):
    seed_if_empty(force=True)
    st.success("Database reset to the initial benchmark dataset.")
    st.rerun()
