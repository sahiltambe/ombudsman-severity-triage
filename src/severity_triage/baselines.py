"""Simple baselines every model should beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold

from severity_triage.config import Settings
from severity_triage.evaluation import CLASSES, compute_metrics
from severity_triage.features import FeatureGroup, FeatureSpec
from severity_triage.models import make_pipeline

__all__ = ["MajorityClassifier", "SingleFeatureRule", "baseline_ladder"]


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Predict the most frequent training class. Floor for every metric."""

    def fit(self, X: object, y: object) -> MajorityClassifier:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = values
        self.majority_ = values[np.argmax(counts)]
        self.prior_ = counts / counts.sum()
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.full(len(X), self.majority_)  # type: ignore[arg-type]

    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile(self.prior_, (len(X), 1))  # type: ignore[arg-type]


class SingleFeatureRule(BaseEstimator, ClassifierMixin):
    """Bin one numeric feature at learned cut-points.

    Cut-points are the midpoints between consecutive class medians of the
    feature on the training data, which is how a triage officer with a lookup
    table would behave. Missing values fall back to the majority class.
    """

    def __init__(self, feature: str) -> None:
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: object) -> SingleFeatureRule:
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        x = pd.to_numeric(X[self.feature], errors="coerce").to_numpy()
        medians = np.array([np.nanmedian(x[y_arr == c]) for c in self.classes_])
        self.cutpoints_ = (medians[:-1] + medians[1:]) / 2
        values, counts = np.unique(y_arr, return_counts=True)
        self.majority_ = values[np.argmax(counts)]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = pd.to_numeric(X[self.feature], errors="coerce").to_numpy()
        pred = self.classes_[np.searchsorted(self.cutpoints_, x)]
        return np.where(np.isnan(x), self.majority_, pred)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.predict(X)
        proba = np.full((len(pred), len(self.classes_)), 0.02)
        proba[np.arange(len(pred)), np.searchsorted(self.classes_, pred)] = 1.0
        return proba / proba.sum(axis=1, keepdims=True)


def _cv_metrics(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
) -> dict[str, float]:
    splitter = RepeatedStratifiedKFold(
        n_splits=settings.split.cv_folds, n_repeats=1, random_state=settings.project.seed
    )
    y_arr = y.to_numpy()
    rows = []
    for train_idx, test_idx in splitter.split(X, y_arr):
        est = estimator_factory().fit(X.iloc[train_idx], y_arr[train_idx])
        proba = est.predict_proba(X.iloc[test_idx])
        pred = est.predict(X.iloc[test_idx])
        rows.append(
            compute_metrics(
                y_arr[test_idx],
                pred,
                proba,
                classes=CLASSES,
                threshold=settings.data.progress_threshold,
            )
        )
    frame = pd.DataFrame(rows)
    return {
        k: float(frame[k].mean())
        for k in ("qwk", "mae", "macro_f1", "severe_recall", "severe_precision")
    }


def baseline_ladder(
    X: pd.DataFrame, y: pd.Series, spec: FeatureSpec, settings: Settings
) -> pd.DataFrame:
    """Cross-validated metrics for each rung of the baseline ladder."""
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "rung": 0,
            "approach": "Majority class (always predict 1)",
            **_cv_metrics(MajorityClassifier, X, y, settings),
        }
    )

    if "EstimatedImpactScore" in X:
        rows.append(
            {
                "rung": 1,
                "approach": "Single-feature rule on EstimatedImpactScore",
                **_cv_metrics(lambda: SingleFeatureRule("EstimatedImpactScore"), X, y, settings),
            }
        )

    raw_spec = spec.without_groups(
        FeatureGroup.DATE,
        FeatureGroup.FINANCIAL,
        FeatureGroup.IMPACT,
        FeatureGroup.VULNERABILITY,
        FeatureGroup.FAILURE,
        FeatureGroup.INTERACTION,
        FeatureGroup.MISSING,
    )
    rows.append(
        {
            "rung": 2,
            "approach": (
                f"Logistic regression, raw columns only ({len(raw_spec.all_columns)} features)"
            ),
            **_cv_metrics(
                lambda: make_pipeline("logistic", raw_spec, settings.project.seed),
                X[raw_spec.all_columns],
                y,
                settings,
            ),
        }
    )
    rows.append(
        {
            "rung": 3,
            "approach": (
                f"Logistic regression, engineered features ({len(spec.all_columns)} features)"
            ),
            **_cv_metrics(
                lambda: make_pipeline("logistic", spec, settings.project.seed), X, y, settings
            ),
        }
    )
    rows.append(
        {
            "rung": 4,
            "approach": "Ordered logit, engineered features (ordinal structure)",
            **_cv_metrics(
                lambda: make_pipeline("ordered_logit", spec, settings.project.seed), X, y, settings
            ),
        }
    )
    rows.append(
        {
            "rung": 5,
            "approach": "LightGBM, engineered features (non-linear)",
            **_cv_metrics(
                lambda: make_pipeline("lightgbm", spec, settings.project.seed), X, y, settings
            ),
        }
    )

    table = pd.DataFrame(rows)
    table["qwk_gain_vs_previous"] = table["qwk"].diff().round(4)
    return table
