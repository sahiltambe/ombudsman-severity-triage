from __future__ import annotations

import numpy as np
import pandas as pd

from severity_triage.config import Settings
from severity_triage.features import (
    FeatureGroup,
    build_preprocessor,
    engineer_features,
    split_features_target,
)


def test_engineering_is_deterministic(settings: Settings, train_clean: pd.DataFrame) -> None:
    X_raw, _ = split_features_target(train_clean, settings)
    a, spec_a = engineer_features(X_raw, settings)
    b, spec_b = engineer_features(X_raw, settings)
    pd.testing.assert_frame_equal(a, b)
    assert spec_a.all_columns == spec_b.all_columns


def test_target_and_identifier_never_become_features(features) -> None:
    X, _, spec = features
    assert "SeverityScore" not in X.columns
    assert "CaseReference" not in X.columns
    assert "OmbudsmanInvestigationRequired" not in X.columns
    assert set(X.columns) == set(spec.all_columns)


def test_every_feature_has_a_group(features) -> None:
    X, _, spec = features
    assert set(spec.groups) == set(X.columns)
    assert set(spec.groups.values()) <= set(FeatureGroup.ALL)


def test_ordinal_codes_follow_contract_order(settings: Settings, train_clean: pd.DataFrame) -> None:
    X_raw, _ = split_features_target(train_clean, settings)
    X, _ = engineer_features(X_raw, settings)
    codes = X["EmotionalImpactLevel_code"]
    labels = X_raw["EmotionalImpactLevel"]
    assert codes[labels == "Minimal"].dropna().eq(0).all()
    assert codes[labels == "Severe"].dropna().eq(4).all()


def test_regime_b_drops_internal_proxies(settings: Settings, train_clean: pd.DataFrame) -> None:
    b = settings.model_copy(deep=True)
    b.features.regime = "B"
    X_raw, _ = split_features_target(train_clean, settings)
    X_a, _ = engineer_features(X_raw, settings)
    X_b, spec_b = engineer_features(X_raw, b)
    proxies = {
        "EstimatedImpactScore",
        "EstimatedRiskScore",
        "PredictedRemedyBand_code",
        "impact_risk_mean",
        "impact_risk_gap",
    }
    assert proxies <= set(X_a.columns)
    assert not (proxies & set(X_b.columns))
    assert spec_b.columns_in_group(FeatureGroup.PROXY) == []
    assert len(X_b.columns) < len(X_a.columns)


def test_without_groups_removes_whole_block(features) -> None:
    _, _, spec = features
    reduced = spec.without_groups(FeatureGroup.DATE, FeatureGroup.MISSING)
    assert not reduced.columns_in_group(FeatureGroup.DATE)
    assert not reduced.columns_in_group(FeatureGroup.MISSING)
    assert set(reduced.all_columns) < set(spec.all_columns)


def test_preprocessor_outputs_finite_matrix(features) -> None:
    X, _, spec = features
    for scale in (False, True):
        pre = build_preprocessor(spec, scale=scale)
        Xt = pre.fit_transform(X)
        assert isinstance(Xt, pd.DataFrame)
        assert np.isfinite(Xt.to_numpy()).all()
        assert len(Xt) == len(X)


def test_holdback_features_align_with_training(
    settings: Settings, features, holdback_clean: pd.DataFrame
) -> None:
    X, _, spec = features
    X_h, spec_h = engineer_features(holdback_clean, settings)
    assert spec_h.all_columns == spec.all_columns
    pre = build_preprocessor(spec, scale=False).fit(X)
    assert list(pre.transform(X_h).columns) == list(pre.transform(X).columns)


def test_composites_have_expected_ranges(features) -> None:
    X, _, _ = features
    assert X["psychological_impact_count"].dropna().between(0, 4).all()
    assert X["health_impact_count"].dropna().between(0, 3).all()
    assert (X["case_age_days"].dropna() > 0).all()
    assert (X["log_financial_loss_total"].dropna() >= 0).all()
