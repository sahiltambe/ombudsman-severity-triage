"""Metrics, CV, tuning, and robustness checks.

Primary metric: quadratic weighted kappa. Also track severe-case recall.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from severity_triage.config import Settings
from severity_triage.features import FeatureGroup, FeatureSpec
from severity_triage.models import MODEL_REGISTRY, make_pipeline

__all__ = [
    "CLASSES",
    "CVResult",
    "ablation_by_group",
    "calibration_table",
    "compare_families",
    "compute_metrics",
    "confusion_tables",
    "cross_validate_family",
    "error_analysis",
    "missingness_injection",
    "noise_perturbation",
    "p_severe",
    "subgroup_performance",
    "tune_family",
]

log = logging.getLogger(__name__)

CLASSES: np.ndarray = np.arange(1, 7)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def p_severe(proba: np.ndarray, classes: Sequence[int] | np.ndarray, threshold: int) -> np.ndarray:
    """``P(severity >= threshold)`` from a class-probability matrix."""
    cols = [i for i, c in enumerate(classes) if c >= threshold]
    # Summing probability columns can overshoot 1.0 by an ulp, which strict
    # metric implementations reject.
    return np.clip(proba[:, cols].sum(axis=1), 0.0, 1.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    *,
    classes: Sequence[int] | np.ndarray = tuple(CLASSES),
    threshold: int = 4,
) -> dict[str, float]:
    """Ordinal, nominal and decision-layer metrics in one flat dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    severe_true = y_true >= threshold
    severe_pred = y_pred >= threshold

    metrics: dict[str, float] = {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=list(classes), zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "exact_or_adjacent": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "severe_recall": recall_score(severe_true, severe_pred, zero_division=0),
        "severe_precision": precision_score(severe_true, severe_pred, zero_division=0),
        "severe_f1": f1_score(severe_true, severe_pred, zero_division=0),
        # Severe cases predicted below the line: the error the brief warns about.
        "severe_missed": float(np.sum(severe_true & ~severe_pred)),
        "non_severe_progressed": float(np.sum(~severe_true & severe_pred)),
    }
    for c in classes:
        mask = y_true == c
        metrics[f"recall_{c}"] = float(np.mean(y_pred[mask] == c)) if mask.any() else np.nan

    if proba is not None:
        ps = p_severe(proba, classes, threshold)
        metrics["severe_roc_auc"] = roc_auc_score(severe_true, ps)
        metrics["severe_pr_auc"] = average_precision_score(severe_true, ps)
        metrics["severe_brier"] = brier_score_loss(severe_true, ps)
        metrics["log_loss"] = log_loss(y_true, proba, labels=list(classes))
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Per-fold metrics plus repeat-averaged out-of-fold probabilities."""

    name: str
    folds: pd.DataFrame
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    fit_seconds: float

    def summary(self) -> pd.Series:
        means = self.folds.drop(columns=["fold", "repeat"]).mean()
        stds = self.folds.drop(columns=["fold", "repeat"]).std()
        out = {"model": self.name, "fit_seconds": self.fit_seconds}
        for key in means.index:
            out[key] = means[key]
            out[f"{key}_std"] = stds[key]
        return pd.Series(out)


def cross_validate_family(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    settings: Settings,
    *,
    params: dict[str, Any] | None = None,
) -> CVResult:
    """Repeated stratified K-fold for one family; returns folds and OOF probabilities."""
    seed = settings.project.seed
    splitter = RepeatedStratifiedKFold(
        n_splits=settings.split.cv_folds,
        n_repeats=settings.split.cv_repeats,
        random_state=seed,
    )
    y_arr = y.to_numpy()
    oof = np.zeros((settings.split.cv_repeats, len(y_arr), len(CLASSES)))
    records: list[dict[str, float]] = []
    started = time.perf_counter()

    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y_arr)):
        repeat, fold = divmod(i, settings.split.cv_folds)
        pipe = make_pipeline(name, spec, seed + repeat)
        if params:
            pipe.set_params(**params)
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])
        pred = pipe.classes_[np.argmax(proba, axis=1)]
        oof[repeat, test_idx] = proba
        row = compute_metrics(
            y_arr[test_idx],
            pred,
            proba,
            classes=pipe.classes_,
            threshold=settings.data.progress_threshold,
        )
        row.update({"fold": fold, "repeat": repeat})
        records.append(row)

    elapsed = time.perf_counter() - started
    oof_mean = oof.mean(axis=0)
    result = CVResult(
        name=name,
        folds=pd.DataFrame(records),
        oof_proba=oof_mean,
        oof_pred=CLASSES[np.argmax(oof_mean, axis=1)],
        fit_seconds=elapsed,
    )
    log.info(
        "CV %-22s qwk=%.3f  severe_recall=%.3f  (%.0fs)",
        name,
        result.folds["qwk"].mean(),
        result.folds["severe_recall"].mean(),
        elapsed,
    )
    return result


def compare_families(
    names: Iterable[str],
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    settings: Settings,
    *,
    params: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, CVResult]]:
    """Cross-validate several families and return a sorted comparison table."""
    results: dict[str, CVResult] = {}
    for name in names:
        results[name] = cross_validate_family(
            name, X, y, spec, settings, params=(params or {}).get(name)
        )
    table = pd.DataFrame([r.summary() for r in results.values()])
    table = table.sort_values(settings.models.primary_metric, ascending=False).reset_index(
        drop=True
    )
    table.insert(1, "label", table["model"].map(lambda n: MODEL_REGISTRY[n].label))
    table.insert(2, "ordinal_aware", table["model"].map(lambda n: MODEL_REGISTRY[n].ordinal_aware))
    return table, results


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_family(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    settings: Settings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Optuna search maximising mean CV QWK. Returns best params and the trial log."""
    family = MODEL_REGISTRY[name]
    if family.search_space is None:
        raise ValueError(f"{name} has no search space defined")

    seed = settings.project.seed
    splitter = StratifiedKFold(n_splits=settings.split.cv_folds, shuffle=True, random_state=seed)
    y_arr = y.to_numpy()
    folds = list(splitter.split(X, y_arr))
    space = family.search_space

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        scores: list[float] = []
        severe: list[float] = []
        for train_idx, test_idx in folds:
            pipe = make_pipeline(name, spec, seed).set_params(**params)
            pipe.fit(X.iloc[train_idx], y_arr[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            scores.append(cohen_kappa_score(y_arr[test_idx], pred, weights="quadratic"))
            severe.append(
                recall_score(
                    y_arr[test_idx] >= settings.data.progress_threshold,
                    pred >= settings.data.progress_threshold,
                )
            )
        trial.set_user_attr("severe_recall", float(np.mean(severe)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
        study_name=f"{name}-{settings.regime_tag}",
    )
    study.optimize(
        objective,
        n_trials=settings.tuning.n_trials,
        timeout=settings.tuning.timeout_seconds,
        show_progress_bar=False,
    )

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    trials = trials.rename(
        columns={"value": "cv_qwk", "user_attrs_severe_recall": "cv_severe_recall"}
    )
    trials["best_so_far"] = trials["cv_qwk"].cummax()

    # Translate Optuna's flat names back into pipeline parameter paths.
    best = space(optuna.trial.FixedTrial(study.best_params))
    log.info("Tuned %s: best CV QWK %.4f over %d trials", name, study.best_value, len(trials))
    return best, trials


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def confusion_tables(
    y_true: np.ndarray, y_pred: np.ndarray, *, threshold: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """6x6 severity confusion and 2x2 triage confusion as labelled frames."""
    six = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASSES),
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )
    two = pd.DataFrame(
        confusion_matrix(np.asarray(y_true) >= threshold, np.asarray(y_pred) >= threshold),
        index=["true_not_progressed", "true_progressed"],
        columns=["pred_not_progressed", "pred_progressed"],
    )
    return six, two


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: int = 4) -> pd.DataFrame:
    """Where do the errors fall, and how many cross the triage boundary?"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_pred - y_true
    n = len(y_true)
    rows = {
        "exact": np.sum(diff == 0),
        "under_by_1": np.sum(diff == -1),
        "over_by_1": np.sum(diff == 1),
        "under_by_2_or_more": np.sum(diff <= -2),
        "over_by_2_or_more": np.sum(diff >= 2),
        "crossed_boundary_downwards (severe -> not)": np.sum(
            (y_true >= threshold) & (y_pred < threshold)
        ),
        "crossed_boundary_upwards (not -> severe)": np.sum(
            (y_true < threshold) & (y_pred >= threshold)
        ),
    }
    table = pd.DataFrame({"count": pd.Series(rows)})
    table["share"] = (table["count"] / n).round(4)
    return table


def calibration_table(y_true_bin: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table for the severe-case probability."""
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "p": p, "y": np.asarray(y_true_bin).astype(float)})
    table = frame.groupby("bin").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    table["gap"] = (table["observed_rate"] - table["mean_predicted"]).round(4)
    return table.reset_index()


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def ablation_by_group(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    settings: Settings,
    *,
    groups: Sequence[str] = FeatureGroup.ALL,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Drop one feature block at a time and measure the change in CV metrics."""
    fast = settings.model_copy(deep=True)
    fast.split.cv_repeats = 1  # ablation is comparative; one repeat is enough
    base = cross_validate_family(name, X, y, spec, fast, params=params).folds
    rows = [
        {
            "dropped_group": "(none)",
            "n_features": len(spec.all_columns),
            "qwk": base["qwk"].mean(),
            "severe_recall": base["severe_recall"].mean(),
            "macro_f1": base["macro_f1"].mean(),
        }
    ]
    for group in groups:
        reduced = spec.without_groups(group)
        if len(reduced.all_columns) == len(spec.all_columns):
            continue  # group not present in this regime
        folds = cross_validate_family(
            name, X[reduced.all_columns], y, reduced, fast, params=params
        ).folds
        rows.append(
            {
                "dropped_group": group,
                "n_features": len(reduced.all_columns),
                "qwk": folds["qwk"].mean(),
                "severe_recall": folds["severe_recall"].mean(),
                "macro_f1": folds["macro_f1"].mean(),
            }
        )
    table = pd.DataFrame(rows)
    base_qwk = float(table["qwk"].iloc[0])
    base_recall = float(table["severe_recall"].iloc[0])
    table["delta_qwk"] = (table["qwk"] - base_qwk).round(4)
    table["delta_severe_recall"] = (table["severe_recall"] - base_recall).round(4)
    return table


def missingness_injection(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    *,
    fractions: Sequence[float],
    seed: int,
    threshold: int = 4,
) -> pd.DataFrame:
    """Randomly blank out a fraction of feature cells and re-score a fitted model."""
    rng = np.random.default_rng(seed)
    rows = []
    y_arr = y.to_numpy()
    for fraction in (0.0, *fractions):
        corrupted = X.copy()
        if fraction > 0:
            mask = rng.random(corrupted.shape) < fraction
            for j, column in enumerate(corrupted.columns):
                col_mask = mask[:, j]
                if column in spec.nominal:
                    corrupted[column] = corrupted[column].astype(object)
                    corrupted.loc[col_mask, column] = None
                else:
                    corrupted.loc[col_mask, column] = np.nan
        proba = pipeline.predict_proba(corrupted)
        pred = pipeline.classes_[np.argmax(proba, axis=1)]
        m = compute_metrics(y_arr, pred, proba, classes=pipeline.classes_, threshold=threshold)
        rows.append(
            {
                "missing_fraction": fraction,
                **{k: m[k] for k in ("qwk", "macro_f1", "severe_recall", "severe_pr_auc")},
            }
        )
    return pd.DataFrame(rows)


def noise_perturbation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    *,
    scales: Sequence[float],
    seed: int,
    threshold: int = 4,
) -> pd.DataFrame:
    """Add Gaussian noise (as a multiple of each column's std) to numeric features."""
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    stds = X[spec.numeric].std(numeric_only=True)
    rows = []
    for scale in (0.0, *scales):
        noisy = X.copy()
        if scale > 0:
            for column in spec.numeric:
                sd = stds.get(column, 0.0)
                if sd and np.isfinite(sd):
                    noisy[column] = noisy[column] + rng.normal(0, scale * sd, size=len(noisy))
        proba = pipeline.predict_proba(noisy)
        pred = pipeline.classes_[np.argmax(proba, axis=1)]
        m = compute_metrics(y_arr, pred, proba, classes=pipeline.classes_, threshold=threshold)
        rows.append(
            {
                "noise_scale": scale,
                **{k: m[k] for k in ("qwk", "macro_f1", "severe_recall", "severe_pr_auc")},
            }
        )
    return pd.DataFrame(rows)


def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: int = 4,
    min_size: int = 20,
) -> pd.DataFrame:
    """Severe recall and QWK per subgroup, to surface uneven error rates."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for column in columns:
        if column not in groups:
            continue
        values = groups[column].astype(object).fillna("(missing)")
        for level in sorted(values.unique(), key=str):
            mask = (values == level).to_numpy()
            n = int(mask.sum())
            if n < min_size:
                continue
            severe_true = y_true[mask] >= threshold
            severe_pred = y_pred[mask] >= threshold
            rows.append(
                {
                    "column": column,
                    "level": level,
                    "n": n,
                    "severe_rate": float(severe_true.mean()),
                    "severe_recall": recall_score(severe_true, severe_pred, zero_division=0)
                    if severe_true.any()
                    else np.nan,
                    "severe_precision": precision_score(severe_true, severe_pred, zero_division=0)
                    if severe_pred.any()
                    else np.nan,
                    "qwk": cohen_kappa_score(y_true[mask], y_pred[mask], weights="quadratic")
                    if len(np.unique(y_true[mask])) > 1
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def refit(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Clone and fit; keeps the original untouched."""
    return clone(pipeline).fit(X, y.to_numpy())
