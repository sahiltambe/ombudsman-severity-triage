"""Feature engineering.

`engineer_features` is a pure transform (same on train and holdback).
`build_preprocessor` imputes / one-hots / scales inside each model pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from severity_triage import contracts
from severity_triage.config import Settings

__all__ = [
    "FeatureGroup",
    "FeatureSpec",
    "build_preprocessor",
    "engineer_features",
    "feature_spec",
    "split_features_target",
]

log = logging.getLogger(__name__)


class FeatureGroup:
    """String constants for feature blocks (used in ablation)."""

    RAW_NUMERIC = "raw_numeric"
    ORDINAL = "ordinal_codes"
    BOOLEAN = "boolean_flags"
    NOMINAL = "nominal"
    DATE = "date_parts"
    FINANCIAL = "composite_financial"
    IMPACT = "composite_impact"
    VULNERABILITY = "composite_vulnerability"
    FAILURE = "composite_failure"
    INTERACTION = "interactions"
    PROXY = "internal_proxies"
    MISSING = "missing_indicators"

    ALL: tuple[str, ...] = (
        RAW_NUMERIC,
        ORDINAL,
        BOOLEAN,
        NOMINAL,
        DATE,
        FINANCIAL,
        IMPACT,
        VULNERABILITY,
        FAILURE,
        INTERACTION,
        PROXY,
        MISSING,
    )


@dataclass
class FeatureSpec:
    """Which engineered columns exist, and which block each belongs to."""

    numeric: list[str] = field(default_factory=list)
    nominal: list[str] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)
    regime: str = "A"

    @property
    def all_columns(self) -> list[str]:
        return [*self.numeric, *self.nominal]

    def columns_in_group(self, group: str) -> list[str]:
        return [c for c, g in self.groups.items() if g == group]

    def without_groups(self, *groups: str) -> FeatureSpec:
        """A copy with every column belonging to ``groups`` removed."""
        drop = set(groups)
        keep = {c for c, g in self.groups.items() if g not in drop}
        return FeatureSpec(
            numeric=[c for c in self.numeric if c in keep],
            nominal=[c for c in self.nominal if c in keep],
            groups={c: g for c, g in self.groups.items() if c in keep},
            regime=self.regime,
        )


# ---------------------------------------------------------------------------
# Encoders for the two structured categorical kinds
# ---------------------------------------------------------------------------


def _encode_ordinal(series: pd.Series, levels: tuple[str, ...]) -> pd.Series:
    """Map ordered category labels to 0..k-1. Unknown labels become NaN."""
    mapping = {label: float(i) for i, label in enumerate(levels)}
    return series.map(mapping).astype(float)


def _encode_boolean(series: pd.Series) -> pd.Series:
    mapping = {contracts.BOOLEAN_TRUE: 1.0, contracts.BOOLEAN_FALSE: 0.0}
    return series.map(mapping).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0, np.nan)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_MONETARY = (
    "DirectFinancialLossGBP",
    "LostIncomeGBP",
    "AdditionalCostsGBP",
    "ExpectedFinancialRedressGBP",
)
_DURATIONS = ("DurationAffectedDays", "DelayInResolutionDays", "RecoveryTimeMonths")
_PSYCHOLOGICAL = (
    "AnxietyReported",
    "DepressionReported",
    "SleepDisruptionReported",
    "ImpactOnRelationships",
)
_HEALTH = ("DeteriorationInHealth", "AdditionalTreatmentRequired", "LongTermImpactFlag")
# Columns with enough missingness in the training data to justify an explicit
# indicator. Chosen from the EDA, kept static so the feature set is stable.
_MISSING_INDICATOR_COLUMNS = (
    "FailureTypeSecondary",
    "PrimaryVulnerability",
    "Jurisdiction",
    "EvidenceStrength",
    "ExpectedFinancialRedressGBP",
    "RecoveryTimeMonths",
    "PhysicalImpactLevel",
    "OrganisationType",
    "AdditionalCostsGBP",
    "EmotionalImpactLevel",
    "EstimatedImpactScore",
    "ServiceArea",
)


def engineer_features(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, FeatureSpec]:
    """Turn a cleaned frame into a model-ready feature frame.

    Returns the frame (numeric columns as float, nominal columns as object)
    and a :class:`FeatureSpec` describing it. The target and identifier are
    never included; use :func:`split_features_target` to obtain ``y``.
    """
    spec = FeatureSpec(regime=settings.features.regime)
    out = pd.DataFrame(index=frame.index)

    def add(name: str, values: pd.Series, group: str, *, nominal: bool = False) -> None:
        out[name] = values.to_numpy() if not nominal else values.astype(object).to_numpy()
        (spec.nominal if nominal else spec.numeric).append(name)
        spec.groups[name] = group

    proxies = set(settings.features.internal_proxy_columns)
    use_proxies = settings.features.regime == "A"

    # --- Raw numerics --------------------------------------------------------
    for column in contracts.numeric_columns():
        if column not in frame:
            continue
        group = FeatureGroup.PROXY if column in proxies else FeatureGroup.RAW_NUMERIC
        if column in proxies and not use_proxies:
            continue
        add(column, pd.to_numeric(frame[column], errors="coerce"), group)

    # --- Ordinal codes -------------------------------------------------------
    for column in contracts.ordinal_columns():
        if column not in frame:
            continue
        if column in proxies and not use_proxies:
            continue
        group = FeatureGroup.PROXY if column in proxies else FeatureGroup.ORDINAL
        add(
            f"{column}_code",
            _encode_ordinal(frame[column], contracts.ordinal_levels(column)),
            group,
        )

    # --- Boolean flags -------------------------------------------------------
    for column in contracts.boolean_columns():
        if column in frame:
            add(column, _encode_boolean(frame[column]), FeatureGroup.BOOLEAN)

    # --- Nominal categoricals (one-hot later) -------------------------------
    for column in contracts.categorical_columns():
        if column in frame:
            add(column, frame[column], FeatureGroup.NOMINAL, nominal=True)

    # --- Date parts ----------------------------------------------------------
    date_col = settings.features.date_column
    if date_col in frame:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        reference = pd.Timestamp(settings.features.reference_date)
        add("case_age_days", (reference - dates).dt.days.astype(float), FeatureGroup.DATE)
        add("case_year", dates.dt.year.astype(float), FeatureGroup.DATE)
        add("case_month", dates.dt.month.astype(float), FeatureGroup.DATE)
        add("case_quarter", dates.dt.quarter.astype(float), FeatureGroup.DATE)
        add("case_dayofweek", dates.dt.dayofweek.astype(float), FeatureGroup.DATE)

    # --- Composite: financial ------------------------------------------------
    money = {c: pd.to_numeric(frame[c], errors="coerce") for c in _MONETARY if c in frame}
    losses = [
        money[c]
        for c in ("DirectFinancialLossGBP", "LostIncomeGBP", "AdditionalCostsGBP")
        if c in money
    ]
    if losses:
        total_loss = pd.concat(losses, axis=1).sum(axis=1, min_count=1)
        add("financial_loss_total_gbp", total_loss, FeatureGroup.FINANCIAL)
        add("log_financial_loss_total", np.log1p(total_loss.clip(lower=0)), FeatureGroup.FINANCIAL)
    for column, series in money.items():
        add(f"log_{column}", np.log1p(series.clip(lower=0)), FeatureGroup.FINANCIAL)
    if "ExpectedFinancialRedressGBP" in money and losses:
        add(
            "redress_to_loss_ratio",
            _safe_ratio(money["ExpectedFinancialRedressGBP"], total_loss),
            FeatureGroup.FINANCIAL,
        )

    # --- Composite: impact ---------------------------------------------------
    psych = [_encode_boolean(frame[c]) for c in _PSYCHOLOGICAL if c in frame]
    if psych:
        add(
            "psychological_impact_count",
            pd.concat(psych, axis=1).sum(axis=1, min_count=1),
            FeatureGroup.IMPACT,
        )
    health = [_encode_boolean(frame[c]) for c in _HEALTH if c in frame]
    if health:
        add(
            "health_impact_count",
            pd.concat(health, axis=1).sum(axis=1, min_count=1),
            FeatureGroup.IMPACT,
        )
    for column in _DURATIONS:
        if column in frame:
            add(
                f"log_{column}",
                np.log1p(pd.to_numeric(frame[column], errors="coerce").clip(lower=0)),
                FeatureGroup.IMPACT,
            )
    if "EmotionalImpactLevel_code" in out and "PhysicalImpactLevel_code" in out:
        add(
            "impact_level_sum",
            out["EmotionalImpactLevel_code"] + out["PhysicalImpactLevel_code"],
            FeatureGroup.IMPACT,
        )

    # --- Composite: vulnerability -------------------------------------------
    vuln_parts: list[pd.Series] = []
    if "VulnerabilityLevel_code" in out:
        vuln_parts.append(out["VulnerabilityLevel_code"])
    if "PrimaryVulnerability" in frame:
        vuln_parts.append(
            (frame["PrimaryVulnerability"].astype(object) != "None")
            .astype(float)
            .where(frame["PrimaryVulnerability"].notna())
        )
    for column in ("SafeguardingConcern", "BereavedPerson"):
        if column in frame:
            vuln_parts.append(_encode_boolean(frame[column]))
    if vuln_parts:
        add(
            "vulnerability_composite",
            pd.concat(vuln_parts, axis=1).sum(axis=1, min_count=1),
            FeatureGroup.VULNERABILITY,
        )
    if "AgeBand_code" in out:
        # Very young and very old are both higher-risk; distance from the middle band.
        add("age_extremity", (out["AgeBand_code"] - 2.5).abs(), FeatureGroup.VULNERABILITY)

    # --- Composite: failure --------------------------------------------------
    fail_parts = [
        pd.to_numeric(frame[c], errors="coerce")
        for c in ("NumberOfServiceFailures", "RepeatFailuresCount")
        if c in frame
    ]
    if "RepeatedFailurePattern" in frame:
        fail_parts.append(_encode_boolean(frame["RepeatedFailurePattern"]))
    if fail_parts:
        add(
            "failure_intensity",
            pd.concat(fail_parts, axis=1).sum(axis=1, min_count=1),
            FeatureGroup.FAILURE,
        )
    if "FailureTypeSecondary" in frame:
        add(
            "has_secondary_failure",
            (frame["FailureTypeSecondary"].astype(object) != "None")
            .astype(float)
            .where(frame["FailureTypeSecondary"].notna()),
            FeatureGroup.FAILURE,
        )
    if "NumberOfOrganisationsInvolved" in frame and "NumberOfServiceFailures" in frame:
        add(
            "failures_per_organisation",
            _safe_ratio(
                pd.to_numeric(frame["NumberOfServiceFailures"], errors="coerce"),
                pd.to_numeric(frame["NumberOfOrganisationsInvolved"], errors="coerce"),
            ),
            FeatureGroup.FAILURE,
        )

    # --- Interactions --------------------------------------------------------
    if "EmotionalImpactLevel_code" in out and "log_DurationAffectedDays" in out:
        add(
            "emotional_x_logduration",
            out["EmotionalImpactLevel_code"] * out["log_DurationAffectedDays"],
            FeatureGroup.INTERACTION,
        )
    if "DelayInResolutionDays" in frame and "DurationAffectedDays" in frame:
        add(
            "delay_to_duration_ratio",
            _safe_ratio(
                pd.to_numeric(frame["DelayInResolutionDays"], errors="coerce"),
                pd.to_numeric(frame["DurationAffectedDays"], errors="coerce"),
            ),
            FeatureGroup.INTERACTION,
        )
    if "vulnerability_composite" in out and "impact_level_sum" in out:
        add(
            "vulnerability_x_impact",
            out["vulnerability_composite"] * out["impact_level_sum"],
            FeatureGroup.INTERACTION,
        )
    if use_proxies and "EstimatedImpactScore" in out and "EstimatedRiskScore" in out:
        add(
            "impact_risk_mean",
            (out["EstimatedImpactScore"] + out["EstimatedRiskScore"]) / 2,
            FeatureGroup.PROXY,
        )
        add(
            "impact_risk_gap",
            out["EstimatedImpactScore"] - out["EstimatedRiskScore"],
            FeatureGroup.PROXY,
        )

    # --- Missing indicators --------------------------------------------------
    for column in _MISSING_INDICATOR_COLUMNS:
        if column in frame and (use_proxies or column not in proxies):
            add(f"{column}_missing", frame[column].isna().astype(float), FeatureGroup.MISSING)

    log.info(
        "Engineered %d features for regime %s (%d numeric, %d nominal)",
        len(spec.all_columns),
        spec.regime,
        len(spec.numeric),
        len(spec.nominal),
    )
    return out, spec


def feature_spec(settings: Settings, sample: pd.DataFrame) -> FeatureSpec:
    """Return the spec that :func:`engineer_features` would produce for ``sample``."""
    return engineer_features(sample.head(5), settings)[1]


def split_features_target(
    frame: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a cleaned training frame into ``(X_raw, y)``."""
    target = settings.data.target
    y = frame[target].astype(int)
    X = frame.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# Fitted preprocessing
# ---------------------------------------------------------------------------


def build_preprocessor(spec: FeatureSpec, *, scale: bool) -> ColumnTransformer:
    """Imputation + encoding (+ scaling) as a single fitted transformer.

    ``scale=True`` for linear and distance-based models; tree ensembles do not
    need it and are marginally more interpretable without it.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    nominal_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5),
            ),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = []
    if spec.numeric:
        transformers.append(("num", Pipeline(numeric_steps), spec.numeric))
    if spec.nominal:
        transformers.append(("nom", nominal_pipeline, spec.nominal))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
