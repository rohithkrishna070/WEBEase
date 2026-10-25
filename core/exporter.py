"""
================================================================================
MODULE: core.exporter
================================================================================
1. WHAT THIS FILE DOES:
   Provides export, reporting, and scientific reproducibility serialization utilities,
   converting decision matrices, MCDM outputs, and metadata into CSV, JSON, and
   publication-grade Markdown research reports.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Academic research and enterprise governance require full
     auditability. Decision outcomes must be accompanied by exact weight vectors,
     algorithm formulations, timestamped parameters, and correlation statistics so third
     parties can independently reproduce the prioritization rankings.
   - Technical & Architectural Rationale: Centralizes data formatting and document
     synthesis. Keeps presentation and file generation logic isolated from computation.

3. HOW IT WORKS CONCEPTUALLY:
   - Formats prioritized issues into standard CSV for data analysis tools (Excel, R, Python).
   - Generates JSON archives packaging raw issues alongside complete algorithmic metadata.
   - Generates publication-ready Markdown reports documenting:
     * Problem statement and dataset notices
     * Criteria weighting percentages
     * Inter-method concordance (Spearman rho, Kendall tau)
     * VIKOR compromise condition validation results
     * Top-ranked repair action plans with WCAG criteria and remediation guidance.

4. HOW IT WORKS TECHNICALLY:
   - `export_to_csv()`: Pandas DataFrame string serialization.
   - `export_to_json()`: Serializes records and metadata dictionaries using `json.dumps`.
   - `generate_research_reproducibility_report()`: Dynamic markdown template rendering
     with Markdown tables and formatted parameter sections.
================================================================================
"""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, Optional
import pandas as pd


def export_to_csv(df: pd.DataFrame) -> str:
    """Serialize DataFrame to a CSV formatted string."""
    return df.to_csv(index=False)


def export_to_json(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize DataFrame and MCDM metadata to a formatted JSON string."""
    payload = {
        "generated_at": datetime.now().isoformat(),
        "metadata": metadata or {},
        "records_count": len(df),
        "issues": df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, default=str)


def generate_research_reproducibility_report(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    report_title: str = "SmartWeb Access: Accessibility Repair Prioritization Research Report",
) -> str:
    """Generate a comprehensive, publication-grade Markdown research summary.

    Includes:
    - Formal research provenance and dataset notice
    - Exact algorithm configurations (TOPSIS and VIKOR formulations)
    - Normalized criteria weights vector with percentage allocations
    - Summary of evaluated alternatives, websites, and pages
    - Top prioritized repairs with criteria rationale and recommended fixes
    - Inter-method concordance metrics (Spearman rho, Kendall tau)
    - VIKOR compromise solution validation (Conditions C1 and C2)
    - Complete reproducibility parameters for verification
    """
    meta = metadata or {}
    total_evaluated = len(df)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rho = meta.get("spearman_rho", "N/A")
    rho_p = meta.get("spearman_pvalue", "N/A")
    tau = meta.get("kendall_tau", "N/A")
    tau_p = meta.get("kendall_pvalue", "N/A")

    weights = meta.get("weights", [1 / 6] * 6)
    vikor_v = meta.get("vikor_v", 0.5)
    vikor_formulation = meta.get("vikor_formulation", "Fuzzy VIKOR")
    val = meta.get("vikor_validation", {})

    md_lines = [
        f"# {report_title}",
        "",
        f"> **Generated:** `{now_str}` | **System:** SmartWeb Access Decision Support Platform (P35)",
        "",
        "---",
        "",
        "## 1. Research Background & Methodological Notice",
        "",
        "This report documents a multi-criteria decision-support evaluation for prioritizing web accessibility",
        "repairs on public-service web portals. Rankings combine **Fuzzy TOPSIS** (Chen, 2000) and **VIKOR** (Opricovic, 2011/2004)",
        "to address cognitive uncertainty and conflicting remediation priorities.",
        "",
        "> [!NOTE]",
        "> **Benchmark Integrity Notice:** All baseline benchmark records evaluated in this report represent synthetic",
        "> demonstration data modeled under WCAG 2.1/2.2 AA success criteria for reproducible algorithmic testing.",
        "> Inter-method rank concordance evaluates mathematical trade-offs, not absolute empirical infallibility.",
        "",
        "---",
        "",
        "## 2. Decision Model Parameters & Reproducibility Vector",
        "",
        f"- **Evaluated Alternatives Count:** `{total_evaluated}`",
        f"- **Fuzzy TOPSIS Variant:** Linear Scale Transformation with Euclidean Vertex Distance (Chen, 2000)",
        f"- **VIKOR Variant:** `{vikor_formulation.capitalize()}` with compromise consensus weight `v = {vikor_v}`",
        "",
        "### Criteria Weights Allocation",
        "",
        "| Criterion Code | Criterion Description | Direction | Normalized Weight | Percentage |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| **C1** | Barrier Severity (WCAG violation degree) | Benefit | `{weights[0]:.4f}` | {weights[0]*100:.1f}% |",
        f"| **C2** | Essential-Task Impact (Citizen workflow blockage) | Benefit | `{weights[1]:.4f}` | {weights[1]*100:.1f}% |",
        f"| **C3** | Affected Pages (Occurrence breadth) | Benefit | `{weights[2]:.4f}` | {weights[2]*100:.1f}% |",
        f"| **C4** | User Impact (Assistive technology impairment) | Benefit | `{weights[3]:.4f}` | {weights[3]*100:.1f}% |",
        f"| **C5** | Repair Effort (Engineering complexity / cost) | **Cost** | `{weights[4]:.4f}` | {weights[4]*100:.1f}% |",
        f"| **C6** | Issue Frequency (Repeatability across templates) | Benefit | `{weights[5]:.4f}` | {weights[5]*100:.1f}% |",
        "",
        "---",
        "",
        "## 3. Top Prioritized Accessibility Repairs (Consensus Priority)",
        "",
        "| Priority | Issue ID | Website | Page | Barrier Type | TOPSIS CC | VIKOR Q | Effort | Recommended Repair Fix |",
        "| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    top_n = min(15, len(df))
    for _, row in df.head(top_n).iterrows():
        p_rank = row.get("consensus_rank", "-")
        iid = row.get("issue_id", "-")
        site = row.get("website", "-")
        page = row.get("page_name", "-")
        itype = row.get("issue_type", "-")
        t_cc = row.get("topsis_score", "-")
        v_q = row.get("vikor_q", "-")
        eff = row.get("repair_effort", "-")
        fix = str(row.get("recommended_fix", "-")).replace("|", "\\|")

        md_lines.append(
            f"| {p_rank} | `{iid}` | {site} | {page} | {itype} | {t_cc} | {v_q} | {eff} | {fix} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Algorithmic Agreement & Statistical Evaluation",
        "",
        f"- **Spearman Rank Correlation (ρ):** `{rho}` (p-value: `{rho_p}`)",
        f"- **Kendall Rank Correlation (τ):** `{tau}` (p-value: `{tau_p}`)",
        f"- **Advantage Threshold (DQ):** `{val.get('dq', 0.0):.4f}`",
        f"- **Observed Advantage:** `{val.get('advantage', 0.0):.4f}`",
        f"- **Condition 1 (Acceptable Advantage):** `{'Satisfied' if val.get('c1_acceptable_advantage') else 'Not Satisfied'}`",
        f"- **Condition 2 (Acceptable Stability):** `{'Satisfied' if val.get('c2_acceptable_stability') else 'Not Satisfied'}`",
        f"- **Compromise Set:** `{', '.join(val.get('compromise_set_names', []))}`",
        "",
        "### Interpretation of Disagreement",
        "Disagreement between Fuzzy TOPSIS and VIKOR reflects structural trade-offs between compensatory geometric distance",
        "and opponent regret minimization. Issues with high technical severity and high task impact are favored by TOPSIS,",
        "whereas VIKOR prioritizes low-effort repairs that eliminate disproportionate single-criterion regret.",
        "",
        "---",
        "",
        "## 5. References",
        "",
        "1. Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. *Fuzzy Sets and Systems*, 114(1), 1–9.",
        "2. Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445–455.",
        "3. Opricovic, S. (2011). Fuzzy VIKOR with an application to water resources planning. *Water Resources Management*, 25(5), 1297–1315.",
        "4. World Wide Web Consortium (W3C). Web Content Accessibility Guidelines (WCAG) 2.1 / 2.2 Recommendation.",
        "",
        "---",
        "*Report compiled by SmartWeb Access Research Analytics Suite.*",
    ])

    return "\n".join(md_lines)


# Backwards compatibility alias
def generate_markdown_report(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Backwards-compatible wrapper for generating markdown reports."""
    return generate_research_reproducibility_report(df, metadata)
