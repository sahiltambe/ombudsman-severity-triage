"""Explainability helpers: importance, SHAP, surrogate rules, cut-points."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from severity_triage.config import Settings
from severity_triage.evaluation import CLASSES, p_severe
from severity_triage.features import FeatureSpec, build_preprocessor
from severity_triage.ordinal import ProportionalOddsClassifier

__all__ = [
    "LatentAnalysis",
    "ShapSummary",
    "TreeShapSummary",
    "expected_severity_shap",
    "latent_contributions",
    "latent_cutpoint_analysis",
    "monotonic_constraint_comparison",
    "permutation_importance_table",
    "pick_illustrative_cases",
    "surrogate_rules",
    "tree_shap_logodds",
]

log = logging.getLogger(__name__)

_QWK_SCORER = make_scorer(cohen_kappa_score, weights="quadratic")

# Features whose direction of effect on severity is known from the brief.
# +1 means "larger value should never decrease predicted severity".
MONOTONE_DIRECTIONS: dict[str, int] = {
    "EstimatedImpactScore": 1,
    "EstimatedRiskScore": 1,
    "PredictedRemedyBand_code": 1,
    "EmotionalImpactLevel_code": 1,
    "PhysicalImpactLevel_code": 1,
    "VulnerabilityLevel_code": 1,
    "NumberOfServiceFailures": 1,
    "RepeatFailuresCount": 1,
    "DurationAffectedDays": 1,
    "RecoveryTimeMonths": 1,
    "ExpectedFinancialRedressGBP": 1,
    "DirectFinancialLossGBP": 1,
    "LostIncomeGBP": 1,
    "AdditionalCostsGBP": 1,
    "log_DurationAffectedDays": 1,
    "log_RecoveryTimeMonths": 1,
    "log_ExpectedFinancialRedressGBP": 1,
    "financial_loss_total_gbp": 1,
    "impact_level_sum": 1,
    "vulnerability_composite": 1,
    "failure_intensity": 1,
    "psychological_impact_count": 1,
    "health_impact_count": 1,
}


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    seed: int,
    top: int | None = None,
) -> pd.DataFrame:
    """Drop in QWK when each engineered feature is shuffled, with spread."""
    result = permutation_importance(
        pipeline,
        X,
        y.to_numpy(),
        scoring=_QWK_SCORER,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table.head(top).reset_index(drop=True) if top else table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


@dataclass
class ShapSummary:
    """Attributions to *expected severity* on the transformed feature matrix.

    ``values[i, j]`` is how much feature ``j`` moved case ``i``'s expected
    severity (``sum_k k * P(y=k)``) away from ``base_value``. Because the
    explainer is model-agnostic these are exact in the additive sense for any
    pipeline: ``base_value + values.sum(1)`` equals the model's expected
    severity for every row.
    """

    feature_names: list[str]
    values: np.ndarray  # (n, features)
    base_value: float
    X_transformed: pd.DataFrame
    row_index: pd.Index

    def mean_abs_table(self, top: int = 25) -> pd.DataFrame:
        contrib = np.abs(self.values).mean(axis=0)
        table = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap_severity": contrib})
        return (
            table.sort_values("mean_abs_shap_severity", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )

    def explanation(self) -> shap.Explanation:
        return shap.Explanation(
            values=self.values,
            base_values=np.full(len(self.values), self.base_value),
            data=self.X_transformed.to_numpy(dtype=float),
            feature_names=self.feature_names,
        )


@dataclass
class TreeShapSummary:
    """Exact per-class log-odds SHAP from a tree booster (fast, tree models only)."""

    feature_names: list[str]
    values: np.ndarray  # (n, features, classes)
    base_values: np.ndarray  # (classes,)
    X_transformed: pd.DataFrame
    classes: np.ndarray

    def mean_abs_table(self, top: int = 25) -> pd.DataFrame:
        """Mean |SHAP| per feature, averaged over classes, plus per-class columns."""
        per_class = np.abs(self.values).mean(axis=0)  # (features, classes)
        table = pd.DataFrame(per_class, columns=[f"class_{c}" for c in self.classes])
        table.insert(0, "feature", self.feature_names)
        table["mean_abs_shap_all_classes"] = per_class.mean(axis=1)
        return (
            table.sort_values("mean_abs_shap_all_classes", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )

    def explanation_for_class(self, cls: int) -> shap.Explanation:
        k = int(np.where(self.classes == cls)[0][0])
        return shap.Explanation(
            values=self.values[:, :, k],
            base_values=np.full(len(self.values), self.base_values[k]),
            data=self.X_transformed.to_numpy(dtype=float),
            feature_names=self.feature_names,
        )


def _unwrap_tree_model(estimator: Any) -> Any | None:
    """Return the underlying tree booster if the estimator has one."""
    inner = getattr(estimator, "estimator_", estimator)  # LabelOffsetClassifier
    if isinstance(
        inner,
        LGBMClassifier | XGBClassifier | RandomForestClassifier | HistGradientBoostingClassifier,
    ):
        return inner
    return None


def tree_shap_logodds(
    pipeline: Pipeline, X: pd.DataFrame, *, max_rows: int = 500, seed: int
) -> TreeShapSummary | None:
    """Exact Tree SHAP in the booster's native (log-odds) output space.

    Returns ``None`` when the final estimator is not a supported tree model.
    """
    pre = pipeline.named_steps["pre"]
    tree = _unwrap_tree_model(pipeline.named_steps["clf"])
    if tree is None:
        return None
    sample = X.sample(n=min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    Xt = pre.transform(sample)
    explainer = shap.TreeExplainer(tree)
    raw = explainer.shap_values(Xt)
    values = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if base.size == 1:
        base = np.repeat(base, values.shape[2])
    return TreeShapSummary(list(Xt.columns), values, base, Xt, np.asarray(pipeline.classes_))


def expected_severity_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    rows: pd.Index | None = None,
    max_rows: int = 100,
    background_rows: int = 30,
    seed: int,
) -> ShapSummary:
    """Model-agnostic SHAP for expected severity, exact in the additive sense.

    Uses a permutation explainer on the transformed matrix so it works for
    every family in the registry, including the stacked ensemble and the
    Frank-Hall decomposition. Cost scales with rows x background x features,
    so the defaults keep it to a couple of minutes; pass ``rows`` to explain
    specific cases (for example the three illustrative ones).
    """
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    classes = np.asarray(pipeline.classes_, dtype=float)

    background = pre.transform(X.sample(n=min(background_rows, len(X)), random_state=seed))
    if rows is not None:
        sample = X.loc[rows]
    else:
        sample = (
            X.sample(n=min(max_rows, len(X)), random_state=seed + 1) if len(X) > max_rows else X
        )
    Xt = pre.transform(sample)
    columns = list(Xt.columns)

    def expected_severity(matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(matrix, columns=columns)
        return clf.predict_proba(frame) @ classes

    masker = shap.maskers.Independent(background.to_numpy(dtype=float), max_samples=background_rows)
    explainer = shap.PermutationExplainer(expected_severity, masker, seed=seed)
    explanation = explainer(Xt.to_numpy(dtype=float), max_evals=2 * Xt.shape[1] + 1, silent=True)
    base = float(np.mean(explanation.base_values))
    return ShapSummary(columns, np.asarray(explanation.values), base, Xt, sample.index)


def latent_contributions(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Exact per-feature contributions to the ordered logit's latent score.

    ``beta_j * (x_j - mean_j)`` for each feature, in latent units. Only
    meaningful for a pipeline whose final step is ``ProportionalOddsClassifier``.
    """
    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, ProportionalOddsClassifier):
        raise TypeError("latent_contributions requires a ProportionalOddsClassifier pipeline")
    Xt = pipeline.named_steps["pre"].transform(X)
    arr = Xt.to_numpy(dtype=float)
    contrib = (arr - arr.mean(axis=0)) * clf.coef_
    return pd.DataFrame(contrib, columns=Xt.columns, index=X.index)


def pick_illustrative_cases(
    proba: np.ndarray, classes: np.ndarray, *, severe_from: int, threshold: float
) -> dict[str, int]:
    """Indices of a clearly mild, a clearly severe and a borderline case."""
    ps = p_severe(proba, list(classes), severe_from)
    return {
        "clearly_not_severe": int(np.argmin(ps)),
        "clearly_severe": int(np.argmax(ps)),
        "borderline": int(np.argmin(np.abs(ps - threshold))),
    }


# ---------------------------------------------------------------------------
# Surrogate rules
# ---------------------------------------------------------------------------


def surrogate_rules(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    severe_from: int,
    threshold: float,
    max_depth: int = 3,
    seed: int,
) -> tuple[str, float, DecisionTreeClassifier]:
    """Distil the model's triage decisions into a shallow tree.

    The tree is fitted on the model's *own* decisions, not the true labels,
    so its accuracy is a fidelity score: how much of the model's behaviour
    three levels of if/else can reproduce.
    """
    proba = pipeline.predict_proba(X)
    decisions = (p_severe(proba, list(pipeline.classes_), severe_from) >= threshold).astype(int)
    numeric = X.select_dtypes(include=[np.number]).fillna(
        X.select_dtypes(include=[np.number]).median()
    )
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=25, random_state=seed)
    tree.fit(numeric, decisions)
    fidelity = float((tree.predict(numeric) == decisions).mean())
    rules = export_text(tree, feature_names=list(numeric.columns), decimals=1, show_weights=False)
    rules = rules.replace("class: 0", "-> CLOSE (not progressed)").replace(
        "class: 1", "-> INVESTIGATE (progressed)"
    )
    return rules, fidelity, tree


# ---------------------------------------------------------------------------
# Monotonic constraints
# ---------------------------------------------------------------------------


def monotonic_constraint_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    settings: Settings,
) -> tuple[pd.DataFrame, list[str]]:
    """LightGBM with and without business-direction monotone constraints.

    Constraints make the model refuse to learn, for example, that more
    financial loss lowers severity, even if noise in a fold suggested it.
    """
    seed = settings.project.seed
    pre = build_preprocessor(spec, scale=False).fit(X)
    names = list(pre.get_feature_names_out())
    constraints = [MONOTONE_DIRECTIONS.get(n, 0) for n in names]
    constrained_features = [n for n, c in zip(names, constraints, strict=True) if c != 0]

    Xt = pre.transform(X)
    cv = StratifiedKFold(n_splits=settings.split.cv_folds, shuffle=True, random_state=seed)
    common: dict[str, Any] = dict(
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
    rows = []
    variants: list[tuple[str, dict[str, Any]]] = [
        ("unconstrained", {}),
        (
            "monotone_constrained",
            {"monotone_constraints": constraints, "monotone_constraints_method": "advanced"},
        ),
    ]
    for label, extra in variants:
        model = LGBMClassifier(**common, **extra)
        scores = cross_val_score(model, Xt, y.to_numpy(), cv=cv, scoring=_QWK_SCORER)
        rows.append({"variant": label, "cv_qwk_mean": scores.mean(), "cv_qwk_std": scores.std()})
    return pd.DataFrame(rows), constrained_features


# ---------------------------------------------------------------------------
# Latent score / cut-point recovery
# ---------------------------------------------------------------------------


@dataclass
class LatentAnalysis:
    cutpoints_latent: np.ndarray
    latent_scores: np.ndarray
    per_class_latent: pd.DataFrame
    univariate_eis_cutpoints: np.ndarray | None
    eis_class_ranges: pd.DataFrame | None


def latent_cutpoint_analysis(
    ordered_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    raw_frame: pd.DataFrame,
) -> LatentAnalysis:
    """Recover the latent severity scale and compare it with EstimatedImpactScore.

    Fits a second, univariate ordered logit on EstimatedImpactScore alone so
    its cut-points can be expressed in the score's own units and laid over the
    empirical class ranges. If the target was generated by binning a latent
    impact score, the two sets of boundaries should nearly coincide.
    """
    clf: ProportionalOddsClassifier = ordered_pipeline.named_steps["clf"]
    Xt = ordered_pipeline.named_steps["pre"].transform(X)
    latent = clf.latent_score(Xt)
    y_arr = y.to_numpy()

    per_class = pd.DataFrame(
        {
            "severity": CLASSES,
            "n": [int(np.sum(y_arr == c)) for c in CLASSES],
            "latent_mean": [float(latent[y_arr == c].mean()) for c in CLASSES],
            "latent_p5": [float(np.percentile(latent[y_arr == c], 5)) for c in CLASSES],
            "latent_p95": [float(np.percentile(latent[y_arr == c], 95)) for c in CLASSES],
        }
    )

    eis_cuts = None
    eis_ranges = None
    if "EstimatedImpactScore" in raw_frame:
        eis = pd.to_numeric(raw_frame["EstimatedImpactScore"], errors="coerce")
        mask = eis.notna().to_numpy()
        uni = ProportionalOddsClassifier(alpha=1e-6, class_weight=None).fit(
            eis[mask].to_numpy().reshape(-1, 1), y_arr[mask]
        )
        # theta_k - beta * x = 0  ->  x* = theta_k / beta   (boundary in score units)
        eis_cuts = uni.cutpoints_ / uni.coef_[0]
        eis_ranges = pd.DataFrame(
            {
                "severity": CLASSES,
                "eis_min": [float(eis[mask][y_arr[mask] == c].min()) for c in CLASSES],
                "eis_p5": [float(eis[mask][y_arr[mask] == c].quantile(0.05)) for c in CLASSES],
                "eis_median": [float(eis[mask][y_arr[mask] == c].median()) for c in CLASSES],
                "eis_p95": [float(eis[mask][y_arr[mask] == c].quantile(0.95)) for c in CLASSES],
                "eis_max": [float(eis[mask][y_arr[mask] == c].max()) for c in CLASSES],
            }
        )
        eis_ranges["recovered_upper_boundary"] = [*np.round(eis_cuts, 1), np.nan]

    return LatentAnalysis(
        cutpoints_latent=clf.cutpoints_,
        latent_scores=latent,
        per_class_latent=per_class,
        univariate_eis_cutpoints=eis_cuts,
        eis_class_ranges=eis_ranges,
    )
