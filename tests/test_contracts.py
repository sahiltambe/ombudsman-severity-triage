from __future__ import annotations

import pytest

from severity_triage import contracts
from severity_triage.contracts import ColumnKind, ColumnRole


def test_contract_count_matches_workbook_columns() -> None:
    # 44 documented fields + 2 undocumented = 46 training columns.
    assert len(contracts.CONTRACTS) == 46
    assert contracts.undocumented_columns() == [
        "AdditionalTreatmentRequired",
        "ImpactOnRelationships",
    ]


def test_roles_are_exclusive_and_complete() -> None:
    features = set(contracts.feature_columns())
    assert contracts.TARGET_COLUMN not in features
    assert contracts.IDENTIFIER_COLUMN not in features
    for leak in contracts.LEAK_COLUMNS:
        assert leak not in features
    assert contracts.LEAK_COLUMNS == ("OmbudsmanInvestigationRequired",)
    assert len(features) == 43


def test_kind_helpers_partition_features() -> None:
    kinds = (
        contracts.numeric_columns()
        + contracts.categorical_columns()
        + contracts.ordinal_columns()
        + contracts.boolean_columns()
        + contracts.date_columns()
    )
    assert sorted(kinds) == sorted(contracts.feature_columns())


@pytest.mark.parametrize(
    ("column", "expected_low", "expected_high"),
    [
        ("AgeBand", "Under 18", "80+"),
        ("EmotionalImpactLevel", "Minimal", "Severe"),
        ("PredictedRemedyBand", "A", "F"),
        ("VulnerabilityLevel", "None", "High"),
    ],
)
def test_ordinal_levels_are_ordered_low_to_high(
    column: str, expected_low: str, expected_high: str
) -> None:
    levels = contracts.ordinal_levels(column)
    assert levels[0] == expected_low
    assert levels[-1] == expected_high


def test_ordinal_levels_rejects_non_ordinal() -> None:
    with pytest.raises(ValueError, match="not an ordinal column"):
        contracts.ordinal_levels("ServiceArea")


def test_numeric_ranges_transcribed_from_dictionary() -> None:
    assert contracts.get("EstimatedImpactScore").min_value == 1
    assert contracts.get("EstimatedImpactScore").max_value == 100
    assert contracts.get("LostIncomeGBP").max_value == 100_000
    assert contracts.get("CaseComplexityScore").integer is True
    assert contracts.get("SeverityScore").role is ColumnRole.TARGET
    assert contracts.get("CaseCreatedDate").kind is ColumnKind.DATE


def test_in_range_helper() -> None:
    c = contracts.get("NumberOfOrganisationsInvolved")  # 1-5
    assert c.in_range(1) and c.in_range(5)
    assert not c.in_range(0) and not c.in_range(7)
