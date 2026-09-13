from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from severity_triage import contracts
from severity_triage.config import Settings
from severity_triage.io import excel_serial_to_datetime
from severity_triage.validation import (
    audit_schema,
    clean,
    quarantine_categoricals,
    quarantine_numeric_ranges,
)


def test_audit_detects_leak_and_undocumented(raw) -> None:
    audit = audit_schema(raw.training, expect_target=True)
    assert audit.ok
    assert audit.target_present
    assert audit.leak_present == ["OmbudsmanInvestigationRequired"]
    assert set(audit.undocumented_present) == {
        "AdditionalTreatmentRequired",
        "ImpactOnRelationships",
    }


def test_audit_holdback_has_no_target(raw) -> None:
    audit = audit_schema(raw.holdback, expect_target=False)
    assert audit.ok
    assert not audit.target_present


def test_audit_reports_missing_and_unexpected(raw) -> None:
    frame = raw.training.drop(columns=["AgeBand"]).assign(Rogue=1)
    audit = audit_schema(frame, expect_target=True)
    assert not audit.ok
    assert audit.expected_missing == ["AgeBand"]
    assert audit.unexpected_present == ["Rogue"]


def test_numeric_quarantine_nulls_and_records() -> None:
    frame = pd.DataFrame(
        {
            "CaseReference": ["A", "B", "C", "D"],
            "EstimatedImpactScore": [50, -3, 108, np.nan],
            "LostIncomeGBP": [100.0, 50.0, 130_000.0, 0.0],
        }
    )
    cleaned, violations = quarantine_numeric_ranges(frame, strategy="null")
    assert cleaned["EstimatedImpactScore"].tolist()[0] == 50
    assert np.isnan(cleaned.loc[1, "EstimatedImpactScore"])
    assert np.isnan(cleaned.loc[2, "EstimatedImpactScore"])
    assert np.isnan(cleaned.loc[2, "LostIncomeGBP"])
    assert len(violations) == 3
    assert set(violations["CaseReference"]) == {"B", "C"}


def test_numeric_quarantine_clip_strategy() -> None:
    frame = pd.DataFrame({"CaseReference": ["A"], "EstimatedRiskScore": [115]})
    cleaned, violations = quarantine_numeric_ranges(frame, strategy="clip")
    assert cleaned.loc[0, "EstimatedRiskScore"] == 100
    assert len(violations) == 1


def test_categorical_quarantine_handles_whitespace_and_unknowns() -> None:
    frame = pd.DataFrame(
        {
            "CaseReference": ["A", "B", "C"],
            "Jurisdiction": [" Health", "Parliamentary", "Mars"],
            "AnxietyReported": ["Yes", "Maybe", "No"],
        }
    )
    cleaned, violations = quarantine_categoricals(frame)
    assert cleaned["Jurisdiction"].tolist()[:2] == ["Health", "Parliamentary"]
    assert pd.isna(cleaned.loc[2, "Jurisdiction"])
    assert pd.isna(cleaned.loc[1, "AnxietyReported"])
    assert sorted(violations["column"]) == ["AnxietyReported", "Jurisdiction"]


def test_excel_serial_conversion() -> None:
    serials = pd.Series([45228, 44946])
    dates = excel_serial_to_datetime(serials)
    assert dates.iloc[0] == pd.Timestamp("2023-10-29")
    assert dates.iloc[1] == pd.Timestamp("2023-01-20")
    already = pd.Series(pd.to_datetime(["2024-01-01"]))
    assert excel_serial_to_datetime(already).equals(already)


def test_clean_training_finds_the_injected_violations(settings: Settings, raw) -> None:
    result = clean(raw.training, settings, is_training=True)
    # Exactly two per numeric feature column: one below the documented minimum
    # and one above the maximum. Ten of the fifteen low-side values are negative;
    # the other five are zeros in columns whose minimum is 1.
    violations = result.range_violations
    assert len(violations) == 30
    per_column = violations.groupby("column").size()
    assert (per_column == 2).all()
    for column, group in violations.groupby("column"):
        contract = contracts.get(str(column))
        values = sorted(group["value"])
        assert values[0] < contract.min_value and values[1] > contract.max_value
    assert (violations["value"] < 0).sum() == 10
    assert "OmbudsmanInvestigationRequired" in result.dropped_columns
    assert "OmbudsmanInvestigationRequired" not in result.frame.columns
    assert str(result.frame[settings.data.target].dtype) == "Int64"
    assert result.frame[settings.data.target].between(1, 6).all()


def test_clean_raises_when_required_column_missing(settings: Settings, raw) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        clean(raw.training.drop(columns=["ServiceArea"]), settings, is_training=True)


def test_loader_preserves_literal_none_category(raw) -> None:
    """pandas treats the string "None" as NA by default; here it is a real level."""
    frame = raw.training
    assert (frame["FailureTypeSecondary"] == "None").sum() == 753
    assert (frame["VulnerabilityLevel"] == "None").sum() == 404
    assert (frame["PrimaryVulnerability"] == "None").sum() == 378
    assert (frame["PhysicalImpactLevel"] == "None").sum() == 289
    # Genuine gaps are still detected: counts verified against the workbook XML.
    assert frame["FailureTypeSecondary"].isna().sum() == 172
    assert frame["PrimaryVulnerability"].isna().sum() == 127
    assert frame["VulnerabilityLevel"].isna().sum() == 0


def test_clean_preserves_row_count_and_ids(settings: Settings, raw) -> None:
    result = clean(raw.holdback, settings, is_training=False)
    assert len(result.frame) == len(raw.holdback) == 500
    assert result.frame[contracts.IDENTIFIER_COLUMN].is_unique
