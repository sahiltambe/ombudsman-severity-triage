from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime, check_grad
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from severity_triage.ordinal import (
    FrankHallClassifier,
    ProportionalOddsClassifier,
    RegressionThresholdClassifier,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    latent = X @ np.array([1.5, -0.8, 0.4, 0.0, 0.9]) + rng.normal(scale=0.5, size=300)
    y = np.digitize(latent, np.quantile(latent, [0.25, 0.5, 0.7, 0.85, 0.95])) + 1
    return X, y


def test_proportional_odds_gradient_matches_finite_differences(toy) -> None:
    X, y = toy
    clf = ProportionalOddsClassifier(alpha=0.3, class_weight=None)
    clf.classes_ = np.unique(y)
    codes = np.searchsorted(clf.classes_, y)
    w = np.ones(len(y))
    p0 = np.random.default_rng(1).normal(scale=0.2, size=X.shape[1] + len(clf.classes_) - 1)
    f = lambda p: clf._objective(p, X, codes, w)[0]  # noqa: E731
    g = lambda p: clf._objective(p, X, codes, w)[1]  # noqa: E731
    err = check_grad(f, g, p0)
    assert err / np.linalg.norm(approx_fprime(p0, f, 1e-6)) < 1e-4


def test_proportional_odds_fits_and_orders_cutpoints(toy) -> None:
    X, y = toy
    clf = ProportionalOddsClassifier().fit(X, y)
    assert clf.converged_
    assert np.all(np.diff(clf.cutpoints_) > 0)
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 6)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Recovers the sign of the true slopes.
    assert clf.coef_[0] > 0 and clf.coef_[1] < 0 and clf.coef_[4] > 0
    assert (clf.predict(X) == y).mean() > 0.6


def test_frank_hall_probabilities_are_valid_and_monotone(toy) -> None:
    X, y = toy
    clf = FrankHallClassifier(LogisticRegression(max_iter=2000)).fit(X, y)
    assert len(clf.estimators_) == 5
    gt = clf.predict_greater_than(X)
    assert np.all(np.diff(gt, axis=1) <= 1e-12)
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 6)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    assert set(np.unique(clf.predict(X))) <= set(range(1, 7))


def test_regression_threshold_learns_ordered_cutpoints(toy) -> None:
    X, y = toy
    clf = RegressionThresholdClassifier(
        HistGradientBoostingRegressor(max_iter=100, random_state=0), random_state=0
    ).fit(X, y)
    assert len(clf.thresholds_) == 5
    assert np.all(np.diff(clf.thresholds_) > 0)
    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= set(range(1, 7))
    assert np.abs(pred - y).mean() < 1.0
