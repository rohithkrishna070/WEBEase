# SmartWeb Access: Decision Support System for Web Accessibility Repairs

> **Proposed Title:** A Fuzzy TOPSIS and VIKOR-Based Decision Support System for Prioritizing Accessibility Repairs on Public-Service Websites  
> **Project Code:** P35 | **Domain:** Soft Computing / Multi-Criteria Decision Making (MCDM) / Web Accessibility (WCAG 2.1/2.2)

---

## 📌 Project Overview

Public-service websites (such as university portals, civic service registries, public transit authorities, and municipal platforms) often contain numerous accessibility barriers. These barriers prevent citizens—especially people with visual, motor, auditory, or cognitive disabilities—from completing essential public services (such as submitting applications, paying fees, or scheduling appointments).

Automated testing tools (such as Lighthouse, WAVE, or Axe) frequently generate exhaustive, unranked lists of hundreds of errors. Website administrators and development teams operate under finite time and engineering resources. **SmartWeb Access** systematically prioritizes accessibility repairs using **Fuzzy TOPSIS** and **VIKOR**, separating raw technical severity from essential-task impediment to ensure high-impact user journeys are restored first.

> [!NOTE]
> The included sample dataset consists of **synthetic evaluation records** modeled after typical institutional audit patterns for research and algorithm benchmarking. The system does not simulate or claim unverified scans.

---

## 🏗️ Folder Structure

```
WebEase/
├── app.py                      # Streamlit application portal & navigation hub
├── pages/                      # Multi-page dashboard modules
│   ├── 1_Overview.py           # Executive portfolio statistics & issue distributions
│   ├── 2_Websites_and_Pages.py # Audited institutional websites & page catalog
│   ├── 3_Accessibility_Issues.py # Issue management: filter, inspect, create, edit, delete, CSV import
│   ├── 4_Prioritization.py     # Interactive MCDM engine (Fuzzy TOPSIS & VIKOR)
│   ├── 5_TOPSIS_vs_VIKOR.py    # Comparative correlation (Spearman rho, Kendall tau)
│   ├── 6_Sensitivity_Analysis.py # Weight perturbation sweeps & rank stability
│   └── 7_Data_and_Export.py    # CSV/JSON/Markdown exports & audit data ingestion
├── core/                       # Decoupled mathematical & data access layers
│   ├── __init__.py
│   ├── models.py               # Dataclasses, schemas, TFN linguistic scales, RecordType, IssueStatus
│   ├── database.py             # SQLite persistence, CRUD, migrations, and seed ingestion
│   ├── validator.py            # Field, URL, rating, duplicate, and CSV structure validation
│   ├── importer.py             # CSV audit importer with duplicate resolution & validation
│   ├── fuzzy_topsis.py         # Chen (2000) Fuzzy TOPSIS implementation
│   ├── vikor.py                # Opricovic (2004) VIKOR compromise ranking
│   ├── mcdm_bridge.py          # Decision matrix construction & statistical correlation
│   ├── sensitivity.py          # Criterion weight sweeps & profile benchmarks
│   ├── exporter.py             # CSV, JSON, and executive Markdown generators
│   └── ui_common.py            # Shared navigation, styling, and synthetic notices
├── data/                       # Demonstration benchmark data & SQLite storage
│   ├── sample_issues.csv       # 38 realistic synthetic issues across 5 websites
│   └── smartweb_access.db      # SQLite database (auto-generated on launch)
├── tests/                      # Comprehensive pytest unit, integration & QA test suite (73 tests)
│   ├── __init__.py
│   ├── test_models.py          # Data model and weight normalization tests
│   ├── test_database.py        # SQLite transactions, upsert, delete, migrations & filter tests
│   ├── test_validator.py       # URL, rating scale, missing field & CSV upload validation tests
│   ├── test_importer.py        # CSV import, duplicate resolution & export tests
│   ├── test_safe_scanner.py    # SSRF defense, URL validation & DOM accessibility checks
│   ├── test_tool_adapters.py   # Multi-tool format adapters (Lighthouse, Axe, WAVE)
│   ├── test_fuzzy_topsis.py    # Fuzzy TOPSIS TFN operations, benchmark & ranking tests
│   ├── test_vikor.py           # VIKOR utility, regret, and compromise tests
│   ├── test_mcdm_bridge.py     # End-to-end MCDM pipeline integration tests
│   ├── test_sensitivity.py     # Weight sweeps, budget renormalization & scenario stability tests
│   └── test_qa_suite.py        # Installation, SQLite persistence, 'Very High' effort & degenerate math tests
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation, mathematical background & setup guides
├── .gitignore                  # Git ignore rules for Python, SQLite, and IDE caches
└── PROJECT_PROPOSAL.md         # Original academic proposal specification
```

---

## 🧮 Mathematical Methodology: Fuzzy TOPSIS Engine

SmartWeb Access ranks accessibility repairs by combining qualitative human evaluations, technical audit metrics, and multi-criteria decision analysis under uncertainty.

### 1. Decision Criteria Model

Each candidate repair action $A_i$ ($i = 1, \dots, m$) is evaluated across six criteria $C_j$ ($j = 1, \dots, 6$):

| Criterion | What It Measures | Optimization Direction | Mathematical Representation |
| :--- | :--- | :--- | :--- |
| **C1. Barrier Severity** | Technical violation degree under WCAG (A, AA, AAA) | **Benefit** (Higher severity = Higher priority) | Triangular Fuzzy Number (TFN) |
| **C2. Essential-Task Impact** | Impediment to core citizen workflows (e.g. apply, pay) | **Benefit** (Greater blockage = Higher priority) | Triangular Fuzzy Number (TFN) |
| **C3. Affected Pages** | Breadth of barrier occurrence across audited URLs | **Benefit** (More pages = Higher priority) | Numerical count / Narrow TFN |
| **C4. User Impact** | Severity of impairment to assistive tech users | **Benefit** (Greater impact = Higher priority) | Triangular Fuzzy Number (TFN) |
| **C5. Repair Effort** | Engineering time & architectural remediation effort | **Cost** (Lower effort increases priority / quick wins) | Triangular Fuzzy Number (TFN) |
| **C6. Issue Frequency** | Recurrence repeatability across page templates | **Benefit** (More frequent = Higher priority) | Scaled Fuzzy / Numerical |

---

### 2. Triangular Fuzzy Numbers & Linguistic Scales

Uncertain qualitative assessments are modeled as **Triangular Fuzzy Numbers (TFNs)** $\tilde{A} = (l, m, u)$, where $0 \le l \le m \le u$. The membership function $\mu_{\tilde{A}}(x)$ is defined as:

$$\mu_{\tilde{A}}(x) = \begin{cases} 
0, & x < l \\
\frac{x - l}{m - l}, & l \le x \le m \\
\frac{u - x}{u - m}, & m \le x \le u \\
0, & x > u 
\end{cases}$$

The standard 5-point linguistic scale used in SmartWeb Access provides smooth overlap across grades:

| Linguistic Term | Abbreviation | Triangular Fuzzy Number $(l, m, u)$ | Crisp Centroid |
| :--- | :---: | :---: | :---: |
| **Very Low** | VL | $(0.05, 0.10, 0.30)$ | $0.150$ |
| **Low** | L | $(0.10, 0.30, 0.50)$ | $0.300$ |
| **Medium** | M | $(0.30, 0.50, 0.70)$ | $0.500$ |
| **High** | H | $(0.50, 0.70, 0.90)$ | $0.700$ |
| **Very High** | VH | $(0.70, 0.90, 1.00)$ | $0.867$ |

---

### 3. Step-by-Step Fuzzy TOPSIS Formulation (Chen, 2000)

The algorithm executes through the following mathematical steps:

#### Step 1: Decision Matrix Construction & Weight Normalization
For $m$ alternatives and $n = 6$ criteria, form the fuzzy decision matrix $\tilde{X} = [\tilde{x}_{ij}]_{m \times n}$, where $\tilde{x}_{ij} = (l_{ij}, m_{ij}, u_{ij})$.
Given user-defined weights $W = [w_1, \dots, w_n]$ with $w_j \ge 0$ and $\sum w_j > 0$, normalize to sum to 1:
$$w_j^* = \frac{w_j}{\sum_{k=1}^n w_k}$$

#### Step 2: Normalization of Benefit and Cost Criteria
Following Chen (2000), criteria scales are normalized into comparable ranges $[0, 1]$ to form $\tilde{R} = [\tilde{r}_{ij}]_{m \times n}$:
- **For Benefit Criteria ($j \in B$):**
  $$\tilde{r}_{ij} = \left( \frac{l_{ij}}{u_j^*}, \frac{m_{ij}}{u_j^*}, \frac{u_{ij}}{u_j^*} \right), \quad \text{where } u_j^* = \max_i u_{ij}$$
- **For Cost Criteria ($j \in C$, e.g. C5: Repair Effort):**
  $$\tilde{r}_{ij} = \left( \frac{l_j^-}{u_{ij}}, \frac{l_j^-}{m_{ij}}, \frac{l_j^-}{l_{ij}} \right), \quad \text{where } l_j^- = \min_i l_{ij}$$
  *(With positive floor shift when $l_j^- \le 0$ to guarantee strictly positive denominators).*

#### Step 3: Weighted Normalized Matrix Calculation
$$\tilde{V} = [\tilde{v}_{ij}]_{m \times n}, \quad \tilde{v}_{ij} = \tilde{r}_{ij} \otimes w_j^* = (l_{r,ij} \cdot w_j^*, \; m_{r,ij} \cdot w_j^*, \; u_{r,ij} \cdot w_j^*)$$

#### Step 4: Fuzzy Positive-Ideal (FPIS) and Negative-Ideal (FNIS) Solutions
- **FPIS ($A^*$):** Represents maximum priority across criteria:
  $$A^* = (\tilde{v}_1^*, \tilde{v}_2^*, \dots, \tilde{v}_n^*), \quad \tilde{v}_j^* = (\max_i v_{ij1}, \max_i v_{ij2}, \max_i v_{ij3})$$
- **FNIS ($A^-$):** Represents minimum priority across criteria:
  $$A^- = (\tilde{v}_1^-, \tilde{v}_2^-, \dots, \tilde{v}_n^-), \quad \tilde{v}_j^- = (\min_i v_{ij1}, \min_i v_{ij2}, \min_i v_{ij3})$$

#### Step 5: Euclidean Vertex Distances
Calculate separation measures using the Chen (2000) vertex metric:
$$d(\tilde{a}, \tilde{b}) = \sqrt{\frac{1}{3} \left[ (l_a - l_b)^2 + (m_a - m_b)^2 + (u_a - u_b)^2 \right]}$$
$$D_i^+ = \sum_{j=1}^n d(\tilde{v}_{ij}, \tilde{v}_j^*), \qquad D_i^- = \sum_{j=1}^n d(\tilde{v}_{ij}, \tilde{v}_j^-)$$

#### Step 6: Relative Closeness Coefficient & Ranking
$$CC_i = \frac{D_i^-}{D_i^+ + D_i^-}, \quad CC_i \in [0, 1]$$
- Alternatives are ranked in descending order of $CC_i$ ($CC_i \to 1.0$ receives **Rank 1**, highest priority for remediation).
- **Degenerate handling:** If $D_i^+ + D_i^- = 0$ (all alternatives identical), $CC_i = 0.5$.

---

## ⚖️ Mathematical Methodology: VIKOR & Fuzzy VIKOR Compromise Engine

SmartWeb Access integrates **VIKOR** (*ViseKriterijumska Optimizacija I Kompromisno Resenje*) as an independent decision-making engine to evaluate compromise solutions under conflicting criteria.

### 1. Distinction: Classical VIKOR vs. Fuzzy VIKOR

The system explicitly distinguishes two formulations:

1. **Classical VIKOR (Opricovic & Tzeng, 2004):**
   - Evaluates crisp numbers $x_{ij}$.
   - When linguistic fuzzy ratings are evaluated, an explicit **Centroid Defuzzification** step ($x_{ij} = \frac{l_{ij} + m_{ij} + u_{ij}}{3}$) or Graded Mean Integration Representation is applied and documented prior to calculation.
2. **Fuzzy VIKOR (Opricovic, 2011):**
   - Directly operates on Triangular Fuzzy Numbers $\tilde{x}_{ij} = (l_{ij}, m_{ij}, u_{ij})$ using vertex separation distances without premature scalarization.

---

### 2. Step-by-Step VIKOR Formulation

#### Step 1: Best ($f_j^*$) and Worst ($f_j^-$) Criteria Bounds
- **For Benefit Criteria ($j \in B$):**
  - Classical: $f_j^* = \max_i x_{ij}, \quad f_j^- = \min_i x_{ij}$
  - Fuzzy: $\tilde{f}_j^* = (\max_i l_{ij}, \max_i m_{ij}, \max_i u_{ij}), \quad \tilde{f}_j^- = (\min_i l_{ij}, \min_i m_{ij}, \min_i u_{ij})$
- **For Cost Criteria ($j \in C$, e.g. C5: Repair Effort):**
  - Classical: $f_j^* = \min_i x_{ij}, \quad f_j^- = \max_i x_{ij}$
  - Fuzzy: $\tilde{f}_j^* = (\min_i l_{ij}, \min_i m_{ij}, \min_i u_{ij}), \quad \tilde{f}_j^- = (\max_i l_{ij}, \max_i m_{ij}, \max_i u_{ij})$

#### Step 2: Normalized Regret Distances & Measures
- **Classical normalized distance:**
  $$d_{ij} = \frac{|f_j^* - x_{ij}|}{|f_j^* - f_j^-|}$$
- **Fuzzy normalized distance:**
  $$d_{ij} = \frac{d(\tilde{f}_j^*, \tilde{x}_{ij})}{d(\tilde{f}_j^*, \tilde{f}_j^-)}$$
  *(where $d(\tilde{a}, \tilde{b})$ is the Euclidean vertex distance).*

#### Step 3: Group Utility ($S_i$) and Individual Regret ($R_i$)
- **Group Utility ($S_i$):** Represents total weighted regret (majority rule):
  $$S_i = \sum_{j=1}^n w_j^* \cdot d_{ij}$$
- **Individual Regret ($R_i$):** Represents worst-case single-criterion regret (opponent veto):
  $$R_i = \max_{j} \left( w_j^* \cdot d_{ij} \right)$$

#### Step 4: VIKOR Compromise Index ($Q_i$)
$$Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1 - v) \cdot \frac{R_i - R^*}{R^- - R^*}$$
Where:
- $S^* = \min_i S_i, \quad S^- = \max_i S_i$
- $R^* = \min_i R_i, \quad R^- = \max_i R_i$
- $v \in [0, 1]$ is the weight of the strategy of maximum group utility:
  - $v > 0.5$: favors majority group benefit.
  - $v < 0.5$: favors individual regret veto.
  - $v = 0.5$: balanced consensus (default).

Alternatives are ranked in ascending order of $Q_i$ ($Q_i \to 0.0$ receives **Rank 1**, optimal compromise).

---

### 3. Compromise Solution Acceptance Conditions (C1 & C2)

Let alternatives sorted by $Q_i$ ascending be $A^{(1)}, A^{(2)}, \dots, A^{(m)}$:

1. **Condition 1 (Acceptable Advantage):**
   $$Q(A^{(2)}) - Q(A^{(1)}) \ge DQ, \quad \text{where } DQ = \frac{1}{m - 1} \quad (m > 1)$$
2. **Condition 2 (Acceptable Stability in Decision Making):**
   Alternative $A^{(1)}$ must also be best-ranked by $S$ (minimum $S$) and/or $R$ (minimum $R$).

**Compromise Set Determination:**
- If **both C1 and C2 are satisfied**: $A^{(1)}$ is declared the **unique stable compromise solution**.
- If **C1 is violated**: A compromise set $\{A^{(1)}, A^{(2)}, \dots, A^{(k)}\}$ is proposed where $Q(A^{(k)}) - Q(A^{(1)}) < DQ$ for maximum $k$.
- If **C2 is violated**: The set $\{A^{(1)}, A^{(2)}\}$ is proposed.

---

### 4. Algorithmic Trade-Offs: Why TOPSIS and VIKOR Disagree

**Ranking disagreement is useful evidence about trade-offs, not a mathematical failure:**

- **Fuzzy TOPSIS** operates on a **geometric compensatory model**. Superior performance on high-weight benefit criteria (e.g. Critical Task Impact) can compensate for poor performance (e.g. High Repair Effort).
- **VIKOR** operates on a **compromise & regret minimization model**. An alternative with extreme weakness on even a single criterion incurs a heavy **Individual Regret ($R_i$)** penalty, preventing it from ranking first unless its benefits overwhelmingly justify the sacrifice.

---

### 5. Established Academic References

1. **Chen, C.-T. (2000).** Extensions of the TOPSIS for group decision-making under fuzzy environment. *Fuzzy Sets and Systems*, 114(1), 1–9. [https://doi.org/10.1016/S0165-0114(97)00377-1](https://doi.org/10.1016/S0165-0114(97)00377-1)
2. **Opricovic, S., & Tzeng, G.-H. (2004).** Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445–455. [https://doi.org/10.1016/S0377-2217(03)00020-1](https://doi.org/10.1016/S0377-2217(03)00020-1)
3. **Opricovic, S. (2011).** Fuzzy VIKOR with an application to water resources planning. *Water Resources Management*, 25(5), 1297–1315. [https://doi.org/10.1007/s11269-010-9745-3](https://doi.org/10.1007/s11269-010-9745-3)
4. **Jahanshahloo, G. R., Hosseinzadeh Lotfi, F., & Izadikhah, M. (2006).** Extension of the TOPSIS method for decision-making problems with fuzzy data. *Applied Mathematics and Computation*, 181(2), 1544–1551. [https://doi.org/10.1016/j.amc.2006.02.057](https://doi.org/10.1016/j.amc.2006.02.057)
5. **Wang, Y.-M., & Elhag, T. M. (2006).** Fuzzy TOPSIS on the basis of alpha level sets. *Fuzzy Sets and Systems*, 157(15), 2095–2108. [https://doi.org/10.1016/j.fss.2006.03.012](https://doi.org/10.1016/j.fss.2006.03.012)


---

## 🚀 Quickstart & Setup Instructions

### Prerequisites
- Python 3.11 or newer (Python 3.13 tested and supported).

---

### 🪟 Windows Setup (PowerShell / Command Prompt)

```powershell
# 1. Clone or navigate into the workspace directory
cd R:\WebEase

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the test suite to verify all mathematical modules
python -m pytest tests/

# 5. Launch the Streamlit dashboard
streamlit run app.py
```

---

### 🍎 macOS Setup (Terminal / zsh)

```bash
# 1. Navigate to the project directory
cd /path/to/WebEase

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
pytest tests/

# 5. Launch the application
streamlit run app.py
```

---

### 🐧 Linux Setup (Ubuntu / Debian / Fedora)

```bash
# 1. Navigate to the project directory
cd /path/to/WebEase

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify test suite execution
pytest tests/ -v

# 5. Launch the application
streamlit run app.py
```

---

## 🧪 Testing

Execute the test suite to validate all mathematical equations and database operations:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_database.py ....                                              [  5%]
tests/test_fuzzy_topsis.py ..................                            [ 30%]
tests/test_importer.py ...                                               [ 34%]
tests/test_mcdm_bridge.py ..                                             [ 36%]
tests/test_models.py ....                                                [ 42%]
tests/test_qa_suite.py .........                                         [ 54%]
tests/test_safe_scanner.py ....                                          [ 60%]
tests/test_sensitivity.py ......                                         [ 68%]
tests/test_tool_adapters.py ....                                         [ 73%]
tests/test_validator.py .......                                          [ 83%]
tests/test_vikor.py ............                                          [100%]
============================= 73 passed in 9.30s ==============================
```

---

## 🧭 Sample End-to-End Workflow

1. **Portfolio Inspection (`1_Overview.py` & `2_Websites_and_Pages.py`):**
   - Review high-level scorecards across registered websites, audited pages, open barriers, and critical journey routes.
   - Drill into a specific institution (e.g. *Metro Transit Authority*) to inspect registered critical journey pages (e.g. *Reduced-Fare Pass Application*).

2. **Audit Evidence & Manual Triage (`3_Accessibility_Issues.py`):**
   - Search, filter, or record manual audit findings.
   - Import automated audit findings from **Google Lighthouse JSON**, **Axe-core CSV**, or **WAVE CSV**.
   - Review candidate findings in the *Audit Triage & Review Queue* and promote confirmed violations to *Human-Verified Finding*.

3. **MCDM Prioritization (`4_Prioritization.py`):**
   - Filter scope to a specific website or essential citizen journey pages.
   - Select a strategic criteria weighting preset (e.g. *Task-Impact Focused* or *Balanced Focus*), or adjust sliders directly.
   - Inspect side-by-side Fuzzy TOPSIS ($CC_i$) and VIKOR ($Q_i$) rankings, alongside deterministic criteria-driven rationales explaining why each barrier ranks high.

4. **Research Evaluation & Comparative Concordance (`5_TOPSIS_vs_VIKOR.py`):**
   - Inspect Spearman's rank correlation ($\rho$) and Kendall's tau ($\tau$).
   - Examine rank concordance scatter plots, paired $CC_i$ vs. $(1 - Q_i)$ bar charts, and VIKOR regret space ($S$ vs. $R$).
   - Review VIKOR compromise acceptance checks (**Condition C1: Acceptable Advantage** and **Condition C2: Acceptable Stability**).
   - Benchmark algorithmic outputs against an external human reviewer reference ranking.

5. **Sensitivity Analysis & Decision Stability (`6_Sensitivity_Analysis.py`):**
   - Evaluate multi-scenario stability across 5 strategic organizational profiles, measuring **Top-3** and **Top-5** appearance frequencies.
   - Run dynamic single-criterion sweeps to observe rank trajectory curves as criteria weights vary.

6. **Reproducibility & Data Export (`7_Data_and_Export.py`):**
   - Download prioritization results as CSV.
   - Export machine-readable JSON archives recording exact weight vectors and algorithm variants.
   - Download publication-ready Markdown research reports for administrative or academic archiving.

---

## 🔒 Security Architecture & Deployment Notice

> [!WARNING]
> **Streamlit Production Deployment Disclaimer:**  
> Streamlit's built-in development web server does **not** provide enterprise user authentication, role-based access control (RBAC), or session token isolation out of the box. For multi-tenant production or administrative deployment:
> - Place the application behind an authenticated upstream reverse proxy (such as **Nginx**, **Caddy**, or **OAuth2-Proxy**) configured with HTTPS/TLS.
> - Restrict network access using VPN, IP allowlisting, or Zero-Trust Access gateways.

### Built-in Security Defenses:
- **SSRF & Private Network Protection:** `core/safe_scanner.py` validates target URLs against RFC 1918 private IP subnets, loopbacks (`127.0.0.1`, `localhost`), link-local (`169.254.0.0/16`), and non-standard schemes before initiating HTTP requests.
- **Robots.txt Compliance & Request Throttling:** Network scans enforce polite crawler identification (`User-Agent`), check `robots.txt` permissions, set strict 8-second socket timeouts, and cap response bodies at 2 MB.
- **SQL Injection Defense:** All database interactions in `core/database.py` use parameterized SQL statements (`?` placeholders).
- **Data Upload Validation:** File uploads are validated for schema conformance, size limits, and allowed extensions (`.csv`, `.json`).

---

## ⚠️ Known Limitations & Methodological Constraints

1. **Static DOM Inspection vs. Client-Side JavaScript SPAs:**
   - The built-in safe URL inspector (`core/safe_scanner.py`) uses Python's standard `html.parser` for fast, secure DOM checking without requiring headless browser binaries (e.g. Chromium).
   - Content rendered entirely by client-side JavaScript frameworks (e.g. React/Vue client-side SPAs) will not be visible to the static parser and should be audited using external tools (Lighthouse / Axe) and imported via CSV/JSON.

2. **Subjectivity in Criteria Weighting:**
   - Multi-Criteria Decision Making models reflect stakeholder value judgements. While sensitivity sweeps demonstrate mathematical stability across scenarios, algorithmic rankings should not be treated as absolute ground truth without empirical usability testing with assistive technology users.

3. **VIKOR Distance Metric Norms:**
   - VIKOR relies on the $L_1$ norm (Manhattan distance for group utility $S$) and $L_\infty$ norm (Chebyshev distance for individual regret $R$). Generalized $L_p$ norms ($1 < p < \infty$) are not parameterized.

---

## ✅ Demonstration & Evaluation Checklist

- [x] **Installation:** `pip install -r requirements.txt` succeeds in clean Python 3.11–3.13 environment.
- [x] **Test Suite:** `pytest tests/ -v` executes all 73 tests across 11 suites with 100% pass rate.
- [x] **Application Launch:** `streamlit run app.py` launches portal without errors or warnings.
- [x] **Navigation:** All 7 dashboard modules navigate smoothly via the Streamlit sidebar.
- [x] **Database Persistence:** SQLite stores, queries, updates, and deletes records persistently.
- [x] **Fuzzy TOPSIS:** Correctly applies Chen (2000) linear scale transformation and Euclidean vertex distance.
- [x] **VIKOR:** Correctly computes utility $S$, regret $R$, index $Q$, and evaluates compromise conditions C1/C2.
- [x] **Sensitivity Analysis:** Proportional weight budget redistribution and scenario stability frequencies function as documented.
- [x] **Evidence Provenance:** Synthetic demonstration data is prominently marked, distinguishing manual and imported findings.
- [x] **Export Capabilities:** CSV, JSON archive, and Markdown report export cleanly for research reproducibility.

---

## 📄 License & Academic Attribution
Developed as part of Soft Computing Research Project **P35: SmartWeb Access**.
Baseline dataset provided is strictly synthetic demonstration data for algorithm evaluation.
