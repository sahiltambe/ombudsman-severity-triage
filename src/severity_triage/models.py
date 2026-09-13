"""Model registry for the families we compare.

Each entry builds a model for a given seed and notes why it was included.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from severity_triage.features import FeatureSpec, build_preprocessor
from severity_triage.ordinal import (
    FrankHallClassifier,
    ProportionalOddsClassifier,
    RegressionThresholdClassifier,
    balanced_sample_weight,
)

__all__ = [
    "MODEL_REGISTRY",
    "REJECTED_ALTERNATIVES",
    "LabelOffsetClassifier",
    "ModelFamily",
    "make_pipeline",
    "model_names",
]


# ---------------------------------------------------------------------------
# Small adapter for libraries that insist on 0-based labels
# ---------------------------------------------------------------------------


class LabelOffsetClassifier(BaseEstimator, ClassifierMixin):
    """Shift labels to 0..K-1 for the wrapped estimator and back on predict.

    Also applies balanced sample weights on ``fit`` when ``balanced=True``,
    which is how XGBoost gets class weighting.
    """

    def __init__(self, estimator: BaseEstimator, balanced: bool = True) -> None:
        self.estimator = estimator
        self.balanced = balanced

    def fit(self, X: Any, y: Any, **fit_params: Any) -> LabelOffsetClassifier:
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        codes = np.searchsorted(self.classes_, y_arr)
        if self.balanced and "sample_weight" not in fit_params:
            fit_params["sample_weight"] = balanced_sample_weight(y_arr)
        self.estimator_ = clone(self.estimator).fit(X, codes, **fit_params)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self.estimator_.feature_importances_)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SearchSpace = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class ModelFamily:
    """One candidate model and how to build it."""

    name: str
    label: str
    build: Callable[[int], BaseEstimator]
    needs_scaling: bool
    ordinal_aware: bool
    rationale: str
    tradeoffs: str
    # Lower = simpler. Used by the one-SE selection rule.
    complexity: int = 5
    search_space: SearchSpace | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def _logistic(seed: int) -> BaseEstimator:
    return LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )


def _ordered_logit(seed: int) -> BaseEstimator:
    return ProportionalOddsClassifier(alpha=1.0, class_weight="balanced")


def _frank_hall(seed: int) -> BaseEstimator:
    # Binary sub-problems converge faster than the 6-class one, so fewer
    # iterations than the standalone HistGB are enough.
    base = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    return FrankHallClassifier(estimator=base)


def _regression_threshold(seed: int) -> BaseEstimator:
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    return RegressionThresholdClassifier(regressor=regressor, n_inner_splits=3, random_state=seed)


def _random_forest(seed: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _hist_gb(seed: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def _xgboost(seed: int) -> BaseEstimator:
    return LabelOffsetClassifier(
        XGBClassifier(
            objective="multi:softprob",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
    )


def _lightgbm(seed: int) -> BaseEstimator:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def _stacked(seed: int) -> BaseEstimator:
    return StackingClassifier(
        estimators=[
            ("xgboost", _xgboost(seed)),
            ("lightgbm", _lightgbm(seed)),
            ("logistic", _logistic(seed)),
            ("ordered_logit", _ordered_logit(seed)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
        cv=3,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def _xgboost_space(trial: Any) -> dict[str, Any]:
    return {
        "clf__estimator__n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "clf__estimator__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "clf__estimator__max_depth": trial.suggest_int("max_depth", 2, 7),
        "clf__estimator__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "clf__estimator__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "clf__estimator__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "clf__estimator__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "clf__estimator__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def _lightgbm_space(trial: Any) -> dict[str, Any]:
    return {
        "clf__n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "clf__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "clf__num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "clf__min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "clf__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "clf__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "clf__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


MODEL_REGISTRY: dict[str, ModelFamily] = {
    "logistic": ModelFamily(
        name="logistic",
        label="Multinomial logistic regression",
        build=_logistic,
        needs_scaling=True,
        ordinal_aware=False,
        complexity=1,
        rationale="Simple linear baseline with readable coefficients.",
        tradeoffs="Ignores class order; no interactions unless engineered.",
        tags=("baseline", "linear"),
    ),
    "ordered_logit": ModelFamily(
        name="ordered_logit",
        label="Proportional-odds (ordered) logit",
        build=_ordered_logit,
        needs_scaling=True,
        ordinal_aware=True,
        complexity=2,
        rationale="Fits ordered severity with one latent score and five cut-points.",
        tradeoffs="Linear; assumes proportional odds.",
        tags=("ordinal", "linear", "interpretable"),
    ),
    "frank_hall": ModelFamily(
        name="frank_hall",
        label="Frank and Hall ordinal decomposition (HistGB base)",
        build=_frank_hall,
        needs_scaling=False,
        ordinal_aware=True,
        complexity=6,
        rationale="Five 'severity > k?' models that respect order.",
        tradeoffs="Heavier to train and explain than a single model.",
        tags=("ordinal", "boosting"),
    ),
    "regression_threshold": ModelFamily(
        name="regression_threshold",
        label="Regression with learned cut-points",
        build=_regression_threshold,
        needs_scaling=False,
        ordinal_aware=True,
        complexity=5,
        rationale="Predict a score, then learn cuts that maximise QWK.",
        tradeoffs="Class probabilities are approximate; slower to tune.",
        tags=("ordinal", "boosting"),
    ),
    "random_forest": ModelFamily(
        name="random_forest",
        label="Random forest",
        build=_random_forest,
        needs_scaling=False,
        ordinal_aware=False,
        complexity=3,
        rationale="Non-linear bagged baseline with simple feature importance.",
        tradeoffs="Ignores order; probabilities can be soft.",
        tags=("bagging",),
    ),
    "hist_gb": ModelFamily(
        name="hist_gb",
        label="Histogram gradient boosting (scikit-learn)",
        build=_hist_gb,
        needs_scaling=False,
        ordinal_aware=False,
        complexity=4,
        rationale="Strong tabular booster with no extra dependency.",
        tradeoffs="Treats classes as unordered.",
        tags=("boosting",),
    ),
    "xgboost": ModelFamily(
        name="xgboost",
        label="XGBoost",
        build=_xgboost,
        needs_scaling=False,
        ordinal_aware=False,
        complexity=4,
        rationale="Common tabular booster; supports monotone constraints.",
        tradeoffs="Needs a label adapter; extra dependency.",
        search_space=_xgboost_space,
        tags=("boosting", "tuned"),
    ),
    "lightgbm": ModelFamily(
        name="lightgbm",
        label="LightGBM",
        build=_lightgbm,
        needs_scaling=False,
        ordinal_aware=False,
        complexity=4,
        rationale="Fast leaf-wise boosting; second boosted view for stacking.",
        tradeoffs="Can overfit small data if leaves are too large.",
        search_space=_lightgbm_space,
        tags=("boosting", "tuned"),
    ),
    "stacked": ModelFamily(
        name="stacked",
        label="Stacked ensemble",
        build=_stacked,
        needs_scaling=True,
        ordinal_aware=False,
        complexity=7,
        rationale="Blend boosters + linear + ordered logit via a meta-learner.",
        tradeoffs="Slowest and hardest to explain; gain is often small.",
        tags=("ensemble",),
    ),
}


REJECTED_ALTERNATIVES: dict[str, str] = {
    "Multi-layer perceptron / deep tabular networks": (
        "Too little data for a clear win over boosting, and harder to explain."
    ),
    "Support vector machine (RBF)": (
        "Awkward probabilities and weak handling of mixed tabular inputs."
    ),
    "k-nearest neighbours": (
        "Distance is weak in high-dimensional mixed features; no class order."
    ),
    "Naive Bayes": (
        "Impact/risk/remedy fields are correlated; independence does not hold."
    ),
    "TabPFN / AutoML frameworks": (
        "Hides the modelling choices the assessment asks us to show."
    ),
    "CatBoost": (
        "Would likely match XGB/LGBM; skipped to avoid a third similar booster."
    ),
}


def model_names() -> list[str]:
    return list(MODEL_REGISTRY)


def make_pipeline(name: str, spec: FeatureSpec, seed: int) -> Pipeline:
    """Preprocessor + estimator for the named family, ready to ``fit``."""
    family = MODEL_REGISTRY[name]
    return Pipeline(
        [
            ("pre", build_preprocessor(spec, scale=family.needs_scaling)),
            ("clf", family.build(seed)),
        ]
    )
