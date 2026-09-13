"""Schema audit, range quarantine, and data-quality reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from severity_triage import contracts
from severity_triage.config import Settings
from severity_triage.contracts import ColumnKind, ColumnRole
from severity_triage.io import excel_serial_to_datetime

__all__ = [
    "CleanResult",
    "SchemaAudit",
    "audit_schema",
    "clean",
    "missingness_table",
    "quarantine_categoricals",
    "quarantine_numeric_ranges",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema audit
# ---------------------------------------------------------------------------


@dataclass
class SchemaAudit:
    """Outcome of comparing a frame's columns with the contracts."""

    expected_missing: list[str] = field(default_factory=list)
    unexpected_present: list[str] = field(default_factory=list)
    undocumented_present: list[str] = field(default_factory=list)
    leak_present: list[str] = field(default_factory=list)
    target_present: bool = False

    @property
    def ok(self) -> bool:
        return not self.expected_missing and not self.unexpected_present

    def summary(self) -> str:
        lines = [
            f"target present: {self.target_present}",
            f"leak columns present: {self.leak_present or 'none'}",
            f"undocumented columns present: {self.undocumented_present or 'none'}",
            f"expected but missing: {self.expected_missing or 'none'}",
            f"unexpected: {self.unexpected_present or 'none'}",
        ]
        return "\n".join(lines)


def audit_schema(frame: pd.DataFrame, *, expect_target: bool) -> SchemaAudit:
    """Compare ``frame`` against the contracts without modifying it."""
    columns = set(frame.columns)
    audit = SchemaAudit(target_present=contracts.TARGET_COLUMN in columns)

    for name, contract in contracts.CONTRACTS.items():
        if contract.role is ColumnRole.TARGET:
            if expect_target and name not in columns:
                audit.expected_missing.append(name)
            continue
        if contract.role is ColumnRole.LEAK:
            if name in columns:
                audit.leak_present.append(name)
            continue
        if name not in columns:
            audit.expected_missing.append(name)
        elif not contract.documented:
            audit.undocumented_present.append(name)

    audit.unexpected_present = sorted(columns - set(contracts.CONTRACTS))
    return audit


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def _violation_row(
    frame: pd.DataFrame, idx: pd.Index, column: str, reason: str, id_column: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_column: frame.loc[idx, id_column].to_numpy()
            if id_column in frame
            else idx.to_numpy(),
            "column": column,
            "value": frame.loc[idx, column].to_numpy(),
            "reason": reason,
        }
    )


def quarantine_numeric_ranges(
    frame: pd.DataFrame,
    *,
    strategy: str = "null",
    id_column: str = contracts.IDENTIFIER_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handle numeric values outside the documented bounds.

    ``strategy="null"`` converts violations to NaN so the imputer treats them
    like any other missing value. ``"clip"`` snaps them to the bounds instead.
    Either way every violation is returned so it can be reported.
    """
    out = frame.copy()
    records: list[pd.DataFrame] = []
    numeric = [c for c in contracts.numeric_columns() if c in out.columns]
    if contracts.TARGET_COLUMN in out.columns:
        numeric.append(contracts.TARGET_COLUMN)

    for column in numeric:
        contract = contracts.get(column)
        values = pd.to_numeric(out[column], errors="coerce")
        out[column] = values

        too_low = values < contract.min_value if contract.min_value is not None else False
        too_high = values > contract.max_value if contract.max_value is not None else False
        mask = pd.Series(too_low | too_high, index=out.index).fillna(False)
        if not mask.any():
            continue

        idx = out.index[mask]
        records.append(
            _violation_row(
                frame,
                idx,
                column,
                f"outside [{contract.min_value}, {contract.max_value}]",
                id_column,
            )
        )
        if strategy == "clip":
            out.loc[idx, column] = values[mask].clip(contract.min_value, contract.max_value)
        else:
            out.loc[idx, column] = np.nan

    violations = (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame(columns=[id_column, "column", "value", "reason"])
    )
    log.info("Numeric range quarantine: %d violations (%s)", len(violations), strategy)
    return out, violations


def quarantine_categoricals(
    frame: pd.DataFrame, *, id_column: str = contracts.IDENTIFIER_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Null out categorical / ordinal / boolean values not in the allowed set."""
    out = frame.copy()
    records: list[pd.DataFrame] = []
    for column, contract in contracts.CONTRACTS.items():
        if column not in out.columns or contract.allowed_values is None:
            continue
        if contract.kind not in (ColumnKind.CATEGORICAL, ColumnKind.ORDINAL, ColumnKind.BOOLEAN):
            continue
        # Normalise whitespace so " Yes" is not treated as a new category.
        series = out[column].astype("string").str.strip()
        allowed = set(contract.allowed_values)
        mask = series.notna() & ~series.isin(allowed)
        if mask.any():
            idx = out.index[mask]
            records.append(_violation_row(frame, idx, column, "not in allowed values", id_column))
            series = series.mask(mask)
        out[column] = series.astype(object).where(series.notna(), np.nan)

    violations = (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame(columns=[id_column, "column", "value", "reason"])
    )
    log.info("Categorical quarantine: %d violations", len(violations))
    return out, violations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and rates, sorted by rate descending."""
    missing = frame.isna().sum()
    table = pd.DataFrame(
        {
            "missing": missing,
            "missing_pct": (missing / len(frame) * 100).round(2),
            "dtype": frame.dtypes.astype(str),
        }
    )
    return table[table["missing"] > 0].sort_values("missing_pct", ascending=False)


# ---------------------------------------------------------------------------
# Orchestrated clean
# ---------------------------------------------------------------------------


@dataclass
class CleanResult:
    """A cleaned frame plus everything that was changed along the way."""

    frame: pd.DataFrame
    audit: SchemaAudit
    range_violations: pd.DataFrame
    categorical_violations: pd.DataFrame
    dropped_columns: list[str]
    missingness: pd.DataFrame

    def report(self) -> str:
        return "\n".join(
            [
                "Schema audit",
                self.audit.summary(),
                "",
                f"Range violations quarantined : {len(self.range_violations)}",
                f"Categorical violations       : {len(self.categorical_violations)}",
                f"Columns dropped              : {self.dropped_columns or 'none'}",
                f"Columns with missing values  : {len(self.missingness)}",
            ]
        )


def clean(frame: pd.DataFrame, settings: Settings, *, is_training: bool) -> CleanResult:
    """Full cleaning pass: audit, type coercion, quarantine, leak removal.

    No imputation happens here. Missing values are left for the modelling
    pipeline so the imputer is fitted inside cross-validation and never sees
    the holdback set.
    """
    audit = audit_schema(frame, expect_target=is_training)
    if audit.expected_missing:
        raise ValueError(f"Missing required columns: {audit.expected_missing}")
    if audit.unexpected_present:
        log.warning("Unexpected columns will be ignored: %s", audit.unexpected_present)

    out = frame.copy()

    # Dates: stored as Excel serials in the source workbook.
    date_col = settings.features.date_column
    if date_col in out.columns:
        out[date_col] = excel_serial_to_datetime(out[date_col])

    out, range_violations = quarantine_numeric_ranges(
        out,
        strategy=settings.data.range_violation_strategy,
        id_column=settings.data.id_column,
    )
    out, categorical_violations = quarantine_categoricals(out, id_column=settings.data.id_column)

    # Leak columns are removed unconditionally; unexpected columns as well.
    dropped = [c for c in settings.data.leak_columns if c in out.columns]
    dropped += [c for c in audit.unexpected_present if c in out.columns]
    out = out.drop(columns=dropped)

    if is_training:
        target = settings.data.target
        out[target] = pd.to_numeric(out[target], errors="coerce").astype("Int64")

    result = CleanResult(
        frame=out,
        audit=audit,
        range_violations=range_violations,
        categorical_violations=categorical_violations,
        dropped_columns=dropped,
        missingness=missingness_table(out),
    )
    log.info("Clean complete:\n%s", result.report())
    return result
