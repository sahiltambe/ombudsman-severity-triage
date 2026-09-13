"""Column contracts from the assessment data dictionary.

Includes the two undocumented Yes/No columns. Used by validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "BOOLEAN_FALSE",
    "BOOLEAN_TRUE",
    "CONTRACTS",
    "IDENTIFIER_COLUMN",
    "LEAK_COLUMNS",
    "TARGET_COLUMN",
    "ColumnContract",
    "ColumnKind",
    "ColumnRole",
    "boolean_columns",
    "categorical_columns",
    "date_columns",
    "feature_columns",
    "get",
    "numeric_columns",
    "ordinal_columns",
    "ordinal_levels",
    "undocumented_columns",
]


class ColumnKind(StrEnum):
    """Statistical type of a column, which determines how it is encoded."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"  # nominal, no natural order
    ORDINAL = "ordinal"  # categorical with a natural order
    BOOLEAN = "boolean"  # Yes / No
    DATE = "date"
    IDENTIFIER = "identifier"


class ColumnRole(StrEnum):
    """How the pipeline is allowed to use a column."""

    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    # A column that restates the target and must never reach a model.
    LEAK = "leak"


@dataclass(frozen=True)
class ColumnContract:
    """Schema and business meaning of a single column."""

    name: str
    kind: ColumnKind
    description: str
    role: ColumnRole = ColumnRole.FEATURE
    # For categorical / ordinal / boolean columns. Ordinal tuples are listed
    # from lowest to highest so the index doubles as the ordinal code.
    allowed_values: tuple[str, ...] | None = None
    # For numeric columns.
    min_value: float | None = None
    max_value: float | None = None
    integer: bool = False
    # False for the two columns present in the data but absent from the
    # dictionary. They are still used, but the assumption is recorded.
    documented: bool = True

    def in_range(self, value: float) -> bool:
        """Return True if ``value`` satisfies the documented numeric bounds."""
        if self.min_value is not None and value < self.min_value:
            return False
        return not (self.max_value is not None and value > self.max_value)


BOOLEAN_TRUE = "Yes"
BOOLEAN_FALSE = "No"
_YES_NO = (BOOLEAN_FALSE, BOOLEAN_TRUE)

TARGET_COLUMN = "SeverityScore"
IDENTIFIER_COLUMN = "CaseReference"

# ---------------------------------------------------------------------------
# Helper constructors keep the table below readable.
# ---------------------------------------------------------------------------


def _num(
    name: str, description: str, lo: float, hi: float, *, integer: bool = False
) -> ColumnContract:
    return ColumnContract(
        name=name,
        kind=ColumnKind.NUMERIC,
        description=description,
        min_value=lo,
        max_value=hi,
        integer=integer,
    )


def _cat(name: str, description: str, *values: str) -> ColumnContract:
    return ColumnContract(
        name=name, kind=ColumnKind.CATEGORICAL, description=description, allowed_values=values
    )


def _ord(name: str, description: str, *values: str) -> ColumnContract:
    return ColumnContract(
        name=name, kind=ColumnKind.ORDINAL, description=description, allowed_values=values
    )


def _bool(name: str, description: str, *, documented: bool = True) -> ColumnContract:
    return ColumnContract(
        name=name,
        kind=ColumnKind.BOOLEAN,
        description=description,
        allowed_values=_YES_NO,
        documented=documented,
    )


_FAILURE_TYPES = (
    "Delay",
    "Communication",
    "Policy Application",
    "Record Keeping",
    "Clinical Failure",
    "Administrative Failure",
)

# ---------------------------------------------------------------------------
# The contract table. Order follows the Data Dictionary sheet.
# ---------------------------------------------------------------------------

_CONTRACT_LIST: tuple[ColumnContract, ...] = (
    _num(
        "AdditionalCostsGBP",
        "Additional costs incurred because of the service failure.",
        0,
        50_000,
    ),
    _ord(
        "AgeBand",
        "Age group of the affected person.",
        "Under 18",
        "18-34",
        "35-49",
        "50-64",
        "65-79",
        "80+",
    ),
    _bool("AnxietyReported", "Whether anxiety was reported as an impact."),
    _bool("BereavedPerson", "Indicates whether the complaint involved bereavement."),
    _num("CaseComplexityScore", "Assessed complexity of the case.", 1, 10, integer=True),
    ColumnContract(
        name="CaseCreatedDate",
        kind=ColumnKind.DATE,
        description="Date the complaint was received.",
    ),
    ColumnContract(
        name=IDENTIFIER_COLUMN,
        kind=ColumnKind.IDENTIFIER,
        description="Unique case reference.",
        role=ColumnRole.IDENTIFIER,
    ),
    _cat(
        "ComplaintCategory",
        "Primary complaint category.",
        "Delay",
        "Communication",
        "Clinical Decision",
        "Administrative Error",
        "Incorrect Advice",
        "Service Quality",
        "Lost Information",
        "Eligibility Decision",
    ),
    _num(
        "DelayInResolutionDays",
        "Length of delay before resolution or response.",
        0,
        2_000,
        integer=True,
    ),
    _bool("DepressionReported", "Whether depression was reported as an impact."),
    _bool("DeteriorationInHealth", "Whether health deteriorated due to the issue."),
    _num(
        "DirectFinancialLossGBP",
        "Direct measurable financial losses suffered.",
        0,
        100_000,
    ),
    _num(
        "DurationAffectedDays",
        "Number of days the person was affected.",
        1,
        3_000,
        integer=True,
    ),
    _ord(
        "EmotionalImpactLevel",
        "Assessed emotional impact.",
        "Minimal",
        "Low",
        "Moderate",
        "Significant",
        "Severe",
    ),
    _bool(
        "EscalatedInternally",
        "Whether concerns were escalated within the organisation.",
    ),
    _num(
        "EstimatedImpactScore",
        "Internal assessment of overall impact.",
        1,
        100,
        integer=True,
    ),
    _num(
        "EstimatedRiskScore",
        "Internal assessment of case risk.",
        1,
        100,
        integer=True,
    ),
    _ord(
        "EvidenceStrength",
        "Strength of supporting evidence.",
        "Limited",
        "Moderate",
        "Strong",
    ),
    _num(
        "ExpectedFinancialRedressGBP",
        "Estimated financial remedy recommendation.",
        0,
        75_000,
    ),
    _cat("FailureTypePrimary", "Primary service failure type.", *_FAILURE_TYPES),
    _cat(
        "FailureTypeSecondary",
        "Secondary service failure type if applicable.",
        "None",
        *_FAILURE_TYPES,
    ),
    _cat(
        "InvestigationRoute",
        "Investigation pathway assigned.",
        "Standard",
        "Fast Track",
        "Escalated",
    ),
    _cat(
        "Jurisdiction",
        "Jurisdiction responsible for handling the complaint.",
        "Health",
        "Parliamentary",
    ),
    _bool("LegalChallengePotential", "Whether the case may lead to legal challenge."),
    _bool("LongTermImpactFlag", "Indicates long-term impact identified."),
    _num("LostIncomeGBP", "Income lost as a consequence of the issue.", 0, 100_000),
    _bool("MultipleComplaintThemes", "Whether multiple complaint themes are present."),
    _num(
        "NumberOfDependentsAffected",
        "Number of other people directly affected.",
        0,
        10,
        integer=True,
    ),
    _num(
        "NumberOfOrganisationsInvolved",
        "Organisations involved in the complaint.",
        1,
        5,
        integer=True,
    ),
    _num(
        "NumberOfServiceFailures",
        "Identified service failures within the case.",
        1,
        20,
        integer=True,
    ),
    # Perfect restatement of the triage outcome (severity >= 4) and absent from
    # the holdback set. Kept in the contract so validation can recognise and
    # reject it, but it is never a feature.
    ColumnContract(
        name="OmbudsmanInvestigationRequired",
        kind=ColumnKind.BOOLEAN,
        description="Whether a formal investigation was deemed necessary.",
        role=ColumnRole.LEAK,
        allowed_values=_YES_NO,
    ),
    _cat(
        "OrganisationType",
        "Organisation complained about.",
        "NHS Trust",
        "GP Practice",
        "Integrated Care Board",
        "Government Department",
        "Agency",
        "Local Authority",
    ),
    _ord(
        "PhysicalImpactLevel",
        "Assessed physical or physiological impact.",
        "None",
        "Minor",
        "Moderate",
        "Significant",
        "Severe",
    ),
    _ord(
        "PredictedRemedyBand",
        "Internal remedy recommendation band.",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ),
    _cat(
        "PrimaryVulnerability",
        "Primary vulnerability identified.",
        "None",
        "Physical Health",
        "Mental Health",
        "Learning Disability",
        "Homelessness",
        "Financial Hardship",
        "Caring Responsibilities",
    ),
    _num(
        "PriorComplaintsRaised",
        "Number of related complaints previously raised.",
        0,
        15,
        integer=True,
    ),
    _num("RecoveryTimeMonths", "Estimated recovery period.", 0, 120, integer=True),
    _bool(
        "RepeatedFailurePattern",
        "Indicates recurring failures by the organisation.",
    ),
    _num(
        "RepeatFailuresCount",
        "Number of repeated failures identified.",
        0,
        10,
        integer=True,
    ),
    _bool("SafeguardingConcern", "Presence of safeguarding concerns."),
    _cat(
        "ServiceArea",
        "Service area associated with the complaint.",
        "Acute Care",
        "Primary Care",
        "Mental Health",
        "Community Care",
        "Social Care",
        "Benefits",
        "Immigration",
        "Taxation",
        "Housing",
        "Justice",
    ),
    ColumnContract(
        name=TARGET_COLUMN,
        kind=ColumnKind.NUMERIC,
        description="Target variable representing the assessed severity of injustice.",
        role=ColumnRole.TARGET,
        min_value=1,
        max_value=6,
        integer=True,
    ),
    _bool("SleepDisruptionReported", "Whether sleep disruption was reported."),
    _ord(
        "VulnerabilityLevel",
        "Assessed vulnerability of the affected person.",
        "None",
        "Low",
        "Moderate",
        "High",
    ),
    # --- Present in the data, absent from the dictionary -------------------
    _bool(
        "AdditionalTreatmentRequired",
        "Whether additional treatment was required (undocumented; inferred from name).",
        documented=False,
    ),
    _bool(
        "ImpactOnRelationships",
        "Whether relationships were impacted (undocumented; inferred from name).",
        documented=False,
    ),
)

CONTRACTS: dict[str, ColumnContract] = {c.name: c for c in _CONTRACT_LIST}

LEAK_COLUMNS: tuple[str, ...] = tuple(c.name for c in _CONTRACT_LIST if c.role is ColumnRole.LEAK)

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get(name: str) -> ColumnContract:
    """Return the contract for ``name`` or raise ``KeyError``."""
    return CONTRACTS[name]


def _by(kind: ColumnKind | None = None, role: ColumnRole | None = None) -> list[str]:
    return [
        c.name
        for c in _CONTRACT_LIST
        if (kind is None or c.kind is kind) and (role is None or c.role is role)
    ]


def feature_columns() -> list[str]:
    """All columns a model is permitted to consume."""
    return _by(role=ColumnRole.FEATURE)


def numeric_columns() -> list[str]:
    return _by(kind=ColumnKind.NUMERIC, role=ColumnRole.FEATURE)


def categorical_columns() -> list[str]:
    return _by(kind=ColumnKind.CATEGORICAL, role=ColumnRole.FEATURE)


def ordinal_columns() -> list[str]:
    return _by(kind=ColumnKind.ORDINAL, role=ColumnRole.FEATURE)


def boolean_columns() -> list[str]:
    return _by(kind=ColumnKind.BOOLEAN, role=ColumnRole.FEATURE)


def date_columns() -> list[str]:
    return _by(kind=ColumnKind.DATE, role=ColumnRole.FEATURE)


def undocumented_columns() -> list[str]:
    return [c.name for c in _CONTRACT_LIST if not c.documented]


def ordinal_levels(name: str) -> tuple[str, ...]:
    """Ordered levels for an ordinal column, lowest first."""
    contract = get(name)
    if contract.kind is not ColumnKind.ORDINAL or contract.allowed_values is None:
        raise ValueError(f"{name} is not an ordinal column")
    return contract.allowed_values
