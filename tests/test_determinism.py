"""Fixed seeds must give bit-identical models and predictions."""

from __future__ import annotations

import numpy as np
import pytest

from severity_triage.models import make_pipeline


@pytest.mark.parametrize("name", ["logistic", "ordered_logit", "lightgbm", "xgboost", "hist_gb"])
def test_refit_is_bit_identical(features, name: str) -> None:
    X, y, spec = features
    X_small, y_small = X.head(600), y.head(600)
    a = make_pipeline(name, spec, seed=42).fit(X_small, y_small)
    b = make_pipeline(name, spec, seed=42).fit(X_small, y_small)
    np.testing.assert_array_equal(a.predict_proba(X.tail(200)), b.predict_proba(X.tail(200)))


def test_different_seeds_change_stochastic_models(features) -> None:
    X, y, spec = features
    a = make_pipeline("random_forest", spec, seed=1).fit(X.head(400), y.head(400))
    b = make_pipeline("random_forest", spec, seed=2).fit(X.head(400), y.head(400))
    assert not np.array_equal(a.predict_proba(X.tail(100)), b.predict_proba(X.tail(100)))
