"""Tests that leak columns never reach the model."""

from __future__ import annotations

import pandas as pd
import pytest

from severity_triage import contracts
from severity_triage.config import Settings
from severity_triage.features import build_preprocessor, engineer_features, split_features_target
from severity_triage.models import MODEL_REGISTRY, make_pipeline
from severity_triage.validation import clean

FORBIDDEN = {"OmbudsmanInvestigationRequired", "CaseReference", "SeverityScore"}


def test_leak_column_is_a_perfect_restatement_of_the_decision(raw, settings: Settings) -> None:
    """Documents *why* the column is dropped: the crosstab is exactly diagonal."""
    frame = raw.training
    progressed = frame[settings.data.target] >= settings.data.progress_threshold
    flag = frame["OmbudsmanInvestigationRequired"] == "Yes"
    assert (progressed == flag).all()
    assert raw.holdback["OmbudsmanInvestigationRequired"].isna().all()


@pytest.mark.parametrize("regime", ["A", "B"])
def test_forbidden_columns_never_reach_features(settings: Settings, raw, regime: str) -> None:
    s = settings.model_copy(deep=True)
    s.features.regime = regime
    cleaned = clean(raw.training, s, is_training=True).frame
    X_raw, _ = split_features_target(cleaned, s)
    X, spec = engineer_features(X_raw, s)
    assert not (FORBIDDEN & set(X.columns))
    assert not (FORBIDDEN & set(spec.all_columns))
    transformed = build_preprocessor(spec, scale=False).fit_transform(X)
    assert not any(any(f in col for f in FORBIDDEN) for col in transformed.columns)


def test_contracts_mark_leak_role() -> None:
    assert contracts.get("OmbudsmanInvestigationRequired").role is contracts.ColumnRole.LEAK
    assert "OmbudsmanInvestigationRequired" not in contracts.feature_columns()


def test_pipelines_only_see_spec_columns(features) -> None:
    X, y, spec = features
    # Fit the cheapest family and confirm the preprocessor's input names match the spec.
    pipe = make_pipeline("logistic", spec, seed=0).fit(X.head(300), y.head(300))
    seen = set(pipe.named_steps["pre"].feature_names_in_)
    assert seen == set(spec.all_columns)
    assert not (FORBIDDEN & seen)


def test_registry_covers_all_configured_candidates(settings: Settings) -> None:
    assert set(settings.models.candidates) <= set(MODEL_REGISTRY)
    assert len(MODEL_REGISTRY) >= 8


def test_holdback_never_influences_training_transforms(
    settings: Settings, features, holdback_clean: pd.DataFrame
) -> None:
    """Imputer statistics come from training rows only."""
    X, _, spec = features
    pre = build_preprocessor(spec, scale=False).fit(X)
    medians_fitted = pre.named_transformers_["num"].named_steps["impute"].statistics_
    pre_again = build_preprocessor(spec, scale=False).fit(X)
    assert (
        pre_again.named_transformers_["num"].named_steps["impute"].statistics_ == medians_fitted
    ).all()
    X_h, _ = engineer_features(holdback_clean, settings)
    pre.transform(X_h)  # transforming holdback must not change the fitted statistics
    assert (
        pre.named_transformers_["num"].named_steps["impute"].statistics_ == medians_fitted
    ).all()
