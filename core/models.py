"""Data models, enums, and type definitions for SmartWeb Access.

Defines schemas for accessibility issues, criteria weights, fuzzy scales,
prioritization results, registered websites/pages, and audit verification types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IssueType(str, Enum):
    LOW_COLOR_CONTRAST = "Low color contrast"
    MISSING_FORM_LABELS = "Missing accessible names or form labels"
    KEYBOARD_BARRIERS = "Keyboard navigation barriers"
    MISSING_ALT_TEXT = "Missing alternative text"
    HEADING_HIERARCHY = "Heading hierarchy problems"
    UNCLEAR_LINKS = "Unclear links"
    FORM_VALIDATION = "Form validation and error-message problems"
    FOCUS_VISIBILITY = "Focus visibility problems"


class IssueStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    NEEDS_REVIEW = "Needs Review"


class RecordType(str, Enum):
    SYNTHETIC = "Synthetic Demonstration"
    MANUAL = "Manual Observation"
    IMPORTED = "Imported Audit Evidence"


class AuditMethod(str, Enum):
    AUTOMATED = "Automated Audit (WAVE/Lighthouse/Axe)"
    MANUAL_KEYBOARD = "Manual Keyboard Testing"
    SCREEN_READER = "Screen Reader Testing (NVDA/VoiceOver)"
    MANUAL_INSPECTION = "Manual DOM / Color Inspection"
    SAFE_DOM_SCANNER = "Safe DOM Inspector (Automated)"
    SYNTHETIC_BENCHMARK = "Synthetic Demonstration Benchmark"


class VerificationStatus(str, Enum):
    HUMAN_VERIFIED = "Human-Verified Finding"
    UNVERIFIED_AUTOMATED = "Unverified Automated Detection"
    MANUAL_OBSERVATION = "Manual Observation"
    FALSE_POSITIVE = "False Positive Discarded"


# Triangular Fuzzy Number: (lower, modal, upper)
TFN = Tuple[float, float, float]

# Standard 5-point linguistic scale for benefit criteria (Severity, Task Impact, User Impact, Frequency)
LINGUISTIC_BENEFIT_SCALE: Dict[str, TFN] = {
    "Very Low": (0.0, 0.1, 0.3),
    "Low": (0.1, 0.3, 0.5),
    "Medium": (0.3, 0.5, 0.7),
    "High": (0.5, 0.7, 0.9),
    "Very High": (0.7, 0.9, 1.0),
}

# Standard scale for repair effort (cost criterion).
LINGUISTIC_COST_SCALE: Dict[str, TFN] = {
    "Very Low": (0.0, 0.1, 0.3),
    "Low": (0.1, 0.3, 0.5),
    "Medium": (0.3, 0.5, 0.7),
    "High": (0.5, 0.7, 0.9),
    "Very High": (0.7, 0.9, 1.0),
}

# Crisp defuzzification values (Centroid method) for easy display / baseline comparison
CRISP_DEF_SCALE: Dict[str, float] = {
    "Very Low": 1.0,
    "Low": 2.0,
    "Medium": 3.0,
    "High": 4.0,
    "Very High": 5.0,
}

VALID_SEVERITY_RATINGS = ["Very Low", "Low", "Medium", "High", "Very High"]
VALID_TASK_IMPACT_RATINGS = ["Very Low", "Low", "Medium", "High", "Very High"]
VALID_USER_IMPACT_RATINGS = ["Very Low", "Low", "Medium", "High", "Very High"]
VALID_EFFORT_RATINGS = ["Very Low", "Low", "Medium", "High", "Very High"]
VALID_FREQUENCY_RATINGS = [
    "Rare", "Occasional", "Frequent", "Systematic",
    "Very Low", "Low", "Medium", "High", "Very High"
]
VALID_STATUSES = [s.value for s in IssueStatus]
VALID_RECORD_TYPES = [r.value for r in RecordType]
VALID_VERIFICATION_STATUSES = [v.value for v in VerificationStatus]


@dataclass
class RegisteredWebsite:
    """Represents an audited institutional or public-service web property."""

    website_id: str
    name: str
    base_url: str
    institution_type: str = "Municipal / Civic Government"
    description: str = ""
    contact_email: str = ""
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "website_id": self.website_id,
            "name": self.name,
            "base_url": self.base_url,
            "institution_type": self.institution_type,
            "description": self.description,
            "contact_email": self.contact_email,
            "created_at": self.created_at,
        }


@dataclass
class RegisteredPage:
    """Represents a specific high-priority service page audited under a website."""

    page_id: str
    website_name: str
    page_name: str
    page_url: str
    essential_task: str = ""
    is_critical_journey: bool = True
    notes: str = ""
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "website_name": self.website_name,
            "page_name": self.page_name,
            "page_url": self.page_url,
            "essential_task": self.essential_task,
            "is_critical_journey": 1 if self.is_critical_journey else 0,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass
class AccessibilityIssue:
    """Represents a single detected, audited, or benchmark accessibility issue."""

    issue_id: str
    website: str
    page_url: str
    page_name: str
    issue_type: str
    description: str
    severity: str
    essential_task_impact: str
    affected_pages: int
    user_impact: str
    repair_effort: str
    frequency: str
    evidence: str
    recommended_fix: str
    status: str = IssueStatus.OPEN.value
    record_type: str = RecordType.SYNTHETIC.value
    audit_method: str = AuditMethod.SYNTHETIC_BENCHMARK.value
    verification_status: str = VerificationStatus.HUMAN_VERIFIED.value
    tool_name: str = "Manual Inspection"
    wcag_criterion: str = ""
    wcag_verified: int = 1
    essential_task: str = ""
    notes: str = ""
    reviewer_notes: str = ""
    is_synthetic: int = 1

    def __post_init__(self):
        # Synchronize is_synthetic with record_type
        if self.record_type == RecordType.SYNTHETIC.value:
            self.is_synthetic = 1
        else:
            self.is_synthetic = 0

        # Backwards compatibility: Map "Verified" to "Resolved"
        if self.status == "Verified":
            self.status = IssueStatus.RESOLVED.value

        # Enforce Requirement 9: Do not label as verified WCAG violation unless criterion & evidence present
        if self.verification_status == VerificationStatus.UNVERIFIED_AUTOMATED.value:
            self.wcag_verified = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to a dictionary matching database and export schema."""
        return {
            "issue_id": self.issue_id,
            "website": self.website,
            "page_url": self.page_url,
            "page_name": self.page_name,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "essential_task_impact": self.essential_task_impact,
            "affected_pages": self.affected_pages,
            "user_impact": self.user_impact,
            "repair_effort": self.repair_effort,
            "frequency": self.frequency,
            "evidence": self.evidence,
            "recommended_fix": self.recommended_fix,
            "status": self.status,
            "record_type": self.record_type,
            "audit_method": self.audit_method,
            "verification_status": self.verification_status,
            "tool_name": self.tool_name,
            "wcag_criterion": self.wcag_criterion,
            "wcag_verified": self.wcag_verified,
            "essential_task": self.essential_task,
            "notes": self.notes,
            "reviewer_notes": self.reviewer_notes,
            "is_synthetic": self.is_synthetic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AccessibilityIssue:
        """Create model instance safely from dictionary or pandas row."""
        status = str(data.get("status", IssueStatus.OPEN.value))
        if status == "Verified":
            status = IssueStatus.RESOLVED.value

        rec_type = data.get("record_type")
        if not rec_type:
            rec_type = (
                RecordType.SYNTHETIC.value
                if int(data.get("is_synthetic", 1)) == 1
                else RecordType.MANUAL.value
            )

        verif_status = data.get("verification_status")
        if not verif_status:
            verif_status = (
                VerificationStatus.MANUAL_OBSERVATION.value
                if rec_type == RecordType.MANUAL.value
                else VerificationStatus.HUMAN_VERIFIED.value
            )

        wcag_verif = int(data.get("wcag_verified", 1))
        if verif_status == VerificationStatus.UNVERIFIED_AUTOMATED.value:
            wcag_verif = 0

        return cls(
            issue_id=str(data["issue_id"]).strip(),
            website=str(data["website"]).strip(),
            page_url=str(data["page_url"]).strip(),
            page_name=str(data["page_name"]).strip(),
            issue_type=str(data["issue_type"]).strip(),
            description=str(data["description"]).strip(),
            severity=str(data["severity"]).strip(),
            essential_task_impact=str(data["essential_task_impact"]).strip(),
            affected_pages=int(data.get("affected_pages", 1)),
            user_impact=str(data["user_impact"]).strip(),
            repair_effort=str(data["repair_effort"]).strip(),
            frequency=str(data["frequency"]).strip(),
            evidence=str(data["evidence"]).strip(),
            recommended_fix=str(data["recommended_fix"]).strip(),
            status=status,
            record_type=rec_type,
            audit_method=str(data.get("audit_method", AuditMethod.SYNTHETIC_BENCHMARK.value)),
            verification_status=verif_status,
            tool_name=str(data.get("tool_name", "Manual Inspection")),
            wcag_criterion=str(data.get("wcag_criterion", "") or ""),
            wcag_verified=wcag_verif,
            essential_task=str(data.get("essential_task", "") or ""),
            notes=str(data.get("notes", "") or ""),
            reviewer_notes=str(data.get("reviewer_notes", "") or ""),
            is_synthetic=1 if rec_type == RecordType.SYNTHETIC.value else 0,
        )


@dataclass
class CriterionWeights:
    """Weights configuration for the 6 decision criteria (must sum to 1.0)."""

    c1_severity: float = 0.25
    c2_task_impact: float = 0.30
    c3_affected_pages: float = 0.10
    c4_user_impact: float = 0.15
    c5_repair_effort: float = 0.10
    c6_frequency: float = 0.10

    def as_list(self) -> List[float]:
        return [
            self.c1_severity,
            self.c2_task_impact,
            self.c3_affected_pages,
            self.c4_user_impact,
            self.c5_repair_effort,
            self.c6_frequency,
        ]

    def validate(self, tolerance: float = 1e-3) -> bool:
        return abs(sum(self.as_list()) - 1.0) <= tolerance

    def normalize(self) -> CriterionWeights:
        total = sum(self.as_list())
        if total <= 0:
            return CriterionWeights(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6)
        return CriterionWeights(
            c1_severity=self.c1_severity / total,
            c2_task_impact=self.c2_task_impact / total,
            c3_affected_pages=self.c3_affected_pages / total,
            c4_user_impact=self.c4_user_impact / total,
            c5_repair_effort=self.c5_repair_effort / total,
            c6_frequency=self.c6_frequency / total,
        )


@dataclass
class PrioritizationResult:
    """Consolidated MCDM evaluation output for an accessibility issue."""

    issue_id: str
    website: str
    page_name: str
    issue_type: str
    severity: str
    essential_task_impact: str
    repair_effort: str
    topsis_score: float
    topsis_rank: int
    vikor_s: float
    vikor_r: float
    vikor_q: float
    vikor_rank: int
    rank_diff: int
