"""Pipeline stages: validate, train, evaluate, explain, predict.

Artefacts go to data/processed/, reports/, and models/.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from severity_triage import __version__, plots
from severity_triage.baselines import baseline_ladder
from severity_triage.config import Settings, load_settings
from severity_triage.decision import (
    capacity_analysis,
    cost_aware_predict,
    select_operating_point,
    threshold_sweep,
)
from severity_triage.evaluation import (
    CLASSES,
    ablation_by_group,
    calibration_table,
    compare_families,
    compute_metrics,
    confusion_tables,
    cross_validate_family,
    error_analysis,
    missingness_injection,
    noise_perturbation,
    p_severe,
    subgroup_performance,
    tune_family,
)
from severity_triage.explain import (
    expected_severity_shap,
    latent_cutpoint_analysis,
    monotonic_constraint_comparison,
    permutation_importance_table,
    pick_illustrative_cases,
    surrogate_rules,
    tree_shap_logodds,
)
from severity_triage.features import FeatureSpec, engineer_features, split_features_target
from severity_triage.io import load_datasets, sha256_of
from severity_triage.models import MODEL_REGISTRY, make_pipeline
from severity_triage.validation import clean

__all__ = [
    "DataBundle",
    "ModelArtefact",
    "load_artefact",
    "load_bundle",
    "run_all",
    "stage_evaluate",
    "stage_explain",
    "stage_predict",
    "stage_train",
    "stage_validate",
]

log = logging.getLogger(__name__)

ARTEFACT_NAME = "severity_pipeline.joblib"
MANIFEST_NAME = "run_manifest.json"


# ---------------------------------------------------------------------------
# Shared containers
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    """Everything downstream stages need, produced by :func:`stage_validate`."""

    train_clean: pd.DataFrame
    holdback_clean: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_holdback: pd.DataFrame
    spec: FeatureSpec
    workbook_sha256: str

    def split(self, settings: Settings) -> tuple[pd.Index, pd.Index]:
        """Deterministic stratified train / internal-test row indices."""
        train_idx, test_idx = train_test_split(
            self.X.index,
            test_size=settings.split.test_size,
            stratify=self.y,
            random_state=settings.project.seed,
        )
        return pd.Index(train_idx), pd.Index(test_idx)


@dataclass
class ModelArtefact:
    """What gets serialised for scoring: the fitted pipeline and its decision rule."""

    pipeline: Any
    spec: FeatureSpec
    model_name: str
    params: dict[str, Any]
    threshold: float
    cost_ratio: float
    severe_from: int
    regime: str
    classes: list[int]
    package_version: str
    trained_at: str


def _write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, float_format="%.6g")
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Stage 1: validate
# ---------------------------------------------------------------------------


def stage_validate(settings: Settings) -> DataBundle:
    """Load, audit, clean and engineer features for both datasets."""
    started = time.perf_counter()
    raw = load_datasets(settings)
    train = clean(raw.training, settings, is_training=True)
    holdback = clean(raw.holdback, settings, is_training=False)

    quality_dir = settings.paths.reports_dir / "data_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "schema_audit_train.txt").write_text(train.report(), encoding="utf-8")
    (quality_dir / "schema_audit_holdback.txt").write_text(holdback.report(), encoding="utf-8")
    _write_csv(train.range_violations, quality_dir / "range_violations_train.csv")
    _write_csv(holdback.range_violations, quality_dir / "range_violations_holdback.csv")
    _write_csv(train.missingness, quality_dir / "missingness_train.csv", index=True)
    _write_csv(holdback.missingness, quality_dir / "missingness_holdback.csv", index=True)

    X_raw, y = split_features_target(train.frame, settings)
    X, spec = engineer_features(X_raw, settings)
    X_holdback, spec_holdback = engineer_features(holdback.frame, settings)
    if spec_holdback.all_columns != spec.all_columns:
        raise RuntimeError("Feature columns differ between training and holdback")

    processed = settings.paths.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    train.frame.to_parquet(processed / "train_clean.parquet")
    holdback.frame.to_parquet(processed / "holdback_clean.parquet")
    X.to_parquet(processed / f"features_train_{settings.regime_tag}.parquet")
    X_holdback.to_parquet(processed / f"features_holdback_{settings.regime_tag}.parquet")
    _write_json(asdict(spec), processed / f"feature_spec_{settings.regime_tag}.json")
    _write_json({"workbook_sha256": raw.workbook_sha256}, processed / "source.json")

    log.info(
        "validate: %d train rows, %d holdback rows, %d features (%.1fs)",
        len(X),
        len(X_holdback),
        len(spec.all_columns),
        time.perf_counter() - started,
    )
    return DataBundle(train.frame, holdback.frame, X, y, X_holdback, spec, raw.workbook_sha256)


def load_bundle(settings: Settings) -> DataBundle:
    """Rehydrate the validate stage's outputs from ``data/processed``."""
    processed = settings.paths.processed_dir
    tag = settings.regime_tag
    try:
        train_clean = pd.read_parquet(processed / "train_clean.parquet")
        holdback_clean = pd.read_parquet(processed / "holdback_clean.parquet")
        X = pd.read_parquet(processed / f"features_train_{tag}.parquet")
        X_holdback = pd.read_parquet(processed / f"features_holdback_{tag}.parquet")
        spec = FeatureSpec(**json.loads((processed / f"feature_spec_{tag}.json").read_text()))
        source = json.loads((processed / "source.json").read_text())
    except FileNotFoundError:
        log.info("Processed data not found; running validate stage")
        return stage_validate(settings)
    y = train_clean[settings.data.target].astype(int)
    return DataBundle(
        train_clean, holdback_clean, X, y, X_holdback, spec, source["workbook_sha256"]
    )


# ---------------------------------------------------------------------------
# Stage 2: train (baselines + model comparison)
# ---------------------------------------------------------------------------


def stage_train(settings: Settings, bundle: DataBundle | None = None) -> pd.DataFrame:
    """Baseline ladder and repeated-CV comparison of every candidate family."""
    bundle = bundle or load_bundle(settings)
    train_idx, _ = bundle.split(settings)
    X_train, y_train = bundle.X.loc[train_idx], bundle.y.loc[train_idx]
    reports = settings.paths.reports_dir
    figures = settings.paths.figures_dir
    tag = settings.regime_tag

    log.info("train: baseline ladder")
    ladder = baseline_ladder(X_train, y_train, bundle.spec, settings)
    _write_csv(ladder, reports / f"baseline_ladder_{tag}.csv")
    plots.plot_baseline_ladder(ladder, figures / f"baseline_ladder_{tag}.png")

    log.info(
        "train: comparing %d families with %dx%d CV",
        len(settings.models.candidates),
        settings.split.cv_repeats,
        settings.split.cv_folds,
    )
    table, results = compare_families(
        settings.models.candidates, X_train, y_train, bundle.spec, settings
    )
    _write_csv(table, reports / f"model_comparison_{tag}.csv")
    folds = pd.concat(
        [r.folds.assign(model=name) for name, r in results.items()], ignore_index=True
    )
    _write_csv(folds, reports / f"cv_folds_{tag}.csv")
    oof_arrays: dict[str, np.ndarray] = {"index": train_idx.to_numpy()}
    oof_arrays.update({name: r.oof_proba for name, r in results.items()})
    np.savez_compressed(settings.paths.processed_dir / f"oof_{tag}.npz", **oof_arrays)  # type: ignore[arg-type]
    plots.plot_model_comparison(
        table, figures / f"model_comparison_{tag}.png", metric=settings.models.primary_metric
    )
    log.info(
        "train: leaderboard\n%s",
        table[["model", "qwk", "severe_recall", "macro_f1", "fit_seconds"]]
        .round(4)
        .to_string(index=False),
    )
    return table


# ---------------------------------------------------------------------------
# Stage 3: evaluate (tuning, selection, holdout, cost, robustness)
# ---------------------------------------------------------------------------


def _selection_path(settings: Settings) -> Path:
    return settings.paths.models_dir / f"selection_{settings.regime_tag}.json"


def _load_comparison(settings: Settings) -> pd.DataFrame:
    path = settings.paths.reports_dir / f"model_comparison_{settings.regime_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the train stage first")
    return pd.read_csv(path)


def _select_model(comparison: pd.DataFrame, settings: Settings) -> tuple[str, str]:
    """Pick the model to ship and explain why, in one sentence."""
    metric = settings.models.primary_metric
    if settings.models.selected:
        return settings.models.selected, "fixed by configuration (models.selected)"

    best = comparison.iloc[0]
    if settings.models.selection_rule == "best":
        return str(best["model"]), f"highest CV {metric}"

    n_folds = settings.split.cv_folds * settings.split.cv_repeats
    se = float(best[f"{metric}_std"]) / np.sqrt(n_folds)
    tolerance = max(
        settings.models.selection_tolerance_se * se, settings.models.selection_min_tolerance
    )
    floor = float(best[metric]) - tolerance
    within = comparison[comparison[metric] >= floor].copy()
    chosen = within.sort_values(["complexity", metric], ascending=[True, False]).iloc[0]
    reasoning = (
        f"one-standard-error rule with practical-equivalence floor: tolerance "
        f"{tolerance:.4f} = max({settings.models.selection_tolerance_se:g} x SE {se:.4f}, "
        f"{settings.models.selection_min_tolerance:g}); {len(within)} of {len(comparison)} models "
        f"score within it of the best ({best['model']}, {metric}={best[metric]:.4f}); "
        f"{chosen['model']} is the simplest of them ({metric}={chosen[metric]:.4f}, "
        f"complexity {int(chosen['complexity'])} vs {int(best['complexity'])})"
    )
    return str(chosen["model"]), reasoning


def stage_evaluate(settings: Settings, bundle: DataBundle | None = None) -> dict[str, Any]:
    bundle = bundle or load_bundle(settings)
    train_idx, test_idx = bundle.split(settings)
    X_train, y_train = bundle.X.loc[train_idx], bundle.y.loc[train_idx]
    X_test, y_test = bundle.X.loc[test_idx], bundle.y.loc[test_idx]
    reports, figures, tag = (
        settings.paths.reports_dir,
        settings.paths.figures_dir,
        settings.regime_tag,
    )
    threshold_cls = settings.data.progress_threshold
    seed = settings.project.seed

    comparison = _load_comparison(settings)

    # --- Tuning -------------------------------------------------------------
    tuned_params: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, pd.DataFrame] = {}
    if settings.tuning.enabled:
        for name in settings.tuning.models:
            if name not in MODEL_REGISTRY or MODEL_REGISTRY[name].search_space is None:
                continue
            log.info("evaluate: tuning %s (%d trials)", name, settings.tuning.n_trials)
            best, trials = tune_family(name, X_train, y_train, bundle.spec, settings)
            tuned_params[name] = best
            trajectories[name] = trials
            _write_csv(trials, reports / f"tuning_{name}_{tag}.csv")
            tuned_cv = cross_validate_family(
                name, X_train, y_train, bundle.spec, settings, params=best
            )
            row = tuned_cv.summary()
            row["model"] = f"{name}_tuned"
            comparison = pd.concat([comparison, row.to_frame().T], ignore_index=True)
        if trajectories:
            plots.plot_tuning_trajectory(trajectories, figures / f"tuning_trajectory_{tag}.png")
        _write_json(tuned_params, settings.paths.models_dir / f"tuned_params_{tag}.json")

    comparison["label"] = comparison["model"].map(
        lambda n: (
            MODEL_REGISTRY[n.removesuffix("_tuned")].label
            + (" (tuned)" if n.endswith("_tuned") else "")
        )
    )
    comparison["ordinal_aware"] = comparison["model"].map(
        lambda n: MODEL_REGISTRY[n.removesuffix("_tuned")].ordinal_aware
    )
    comparison["complexity"] = comparison["model"].map(
        lambda n: MODEL_REGISTRY[n.removesuffix("_tuned")].complexity
    )
    for col in comparison.columns:
        if col not in ("model", "label", "ordinal_aware"):
            comparison[col] = pd.to_numeric(comparison[col], errors="coerce")
    comparison = comparison.sort_values(
        settings.models.primary_metric, ascending=False
    ).reset_index(drop=True)
    _write_csv(comparison, reports / f"model_comparison_{tag}.csv")
    plots.plot_model_comparison(
        comparison, figures / f"model_comparison_{tag}.png", metric=settings.models.primary_metric
    )

    # --- Selection -----------------------------------------------------------
    selected_key, reasoning = _select_model(comparison, settings)
    family_name = selected_key.removesuffix("_tuned")
    params = tuned_params.get(family_name, {}) if selected_key.endswith("_tuned") else {}
    cv_row = comparison.loc[comparison["model"] == selected_key].iloc[0]
    log.info(
        "evaluate: selected %s (CV %s=%.4f, severe_recall=%.4f) - %s",
        selected_key,
        settings.models.primary_metric,
        cv_row[settings.models.primary_metric],
        cv_row["severe_recall"],
        reasoning,
    )

    # --- Holdout -------------------------------------------------------------
    pipeline = make_pipeline(family_name, bundle.spec, seed)
    if params:
        pipeline.set_params(**params)
    pipeline.fit(X_train, y_train.to_numpy())
    proba_test = pipeline.predict_proba(X_test)
    pred_test = pipeline.classes_[np.argmax(proba_test, axis=1)]
    holdout = compute_metrics(
        y_test.to_numpy(), pred_test, proba_test, classes=pipeline.classes_, threshold=threshold_cls
    )
    _write_csv(
        pd.DataFrame([holdout]).T.rename(columns={0: "value"}),
        reports / f"holdout_metrics_{tag}.csv",
        index=True,
    )

    six, two = confusion_tables(y_test.to_numpy(), pred_test, threshold=threshold_cls)
    _write_csv(six, reports / f"confusion_severity_{tag}.csv", index=True)
    _write_csv(two, reports / f"confusion_triage_{tag}.csv", index=True)
    plots.plot_confusion(
        six,
        two,
        figures / f"confusion_{tag}.png",
        title=f"{MODEL_REGISTRY[family_name].label} on the internal test set (argmax)",
    )
    _write_csv(
        error_analysis(y_test.to_numpy(), pred_test, threshold=threshold_cls),
        reports / f"error_analysis_{tag}.csv",
        index=True,
    )

    # --- Calibration -----------------------------------------------------------
    ps_test = p_severe(proba_test, list(pipeline.classes_), threshold_cls)
    calib_before = calibration_table(y_test.to_numpy() >= threshold_cls, ps_test)
    calibrated = CalibratedClassifierCV(
        make_pipeline(family_name, bundle.spec, seed).set_params(**params)
        if params
        else make_pipeline(family_name, bundle.spec, seed),
        method="isotonic",
        cv=5,
    )
    calibrated.fit(X_train, y_train.to_numpy())
    ps_cal = p_severe(calibrated.predict_proba(X_test), list(calibrated.classes_), threshold_cls)
    calib_after = calibration_table(y_test.to_numpy() >= threshold_cls, ps_cal)
    calib_summary = pd.DataFrame(
        [
            {
                "variant": "uncalibrated",
                "severe_brier": brier_score_loss(y_test >= threshold_cls, ps_test),
                "log_loss": log_loss(y_test, proba_test, labels=list(pipeline.classes_)),
            },
            {
                "variant": "isotonic",
                "severe_brier": brier_score_loss(y_test >= threshold_cls, ps_cal),
                "log_loss": log_loss(
                    y_test, calibrated.predict_proba(X_test), labels=list(calibrated.classes_)
                ),
            },
        ]
    )
    _write_csv(calib_before, reports / f"calibration_uncalibrated_{tag}.csv")
    _write_csv(calib_after, reports / f"calibration_isotonic_{tag}.csv")
    _write_csv(calib_summary, reports / f"calibration_summary_{tag}.csv")
    plots.plot_calibration(calib_before, calib_after, figures / f"calibration_{tag}.png")

    # --- Decision layer: threshold from OOF, reported on holdout -------------
    oof_cv = cross_validate_family(
        family_name, X_train, y_train, bundle.spec, settings, params=params or None
    )
    ps_oof = p_severe(oof_cv.oof_proba, list(CLASSES), threshold_cls)
    sweep = threshold_sweep(
        y_train.to_numpy() >= threshold_cls,
        ps_oof,
        cost_ratios=settings.decision.sensitivity_ratios,
    )
    operating = pd.DataFrame(
        [
            select_operating_point(sweep, cost_ratio=r).as_dict()
            for r in settings.decision.sensitivity_ratios
        ]
    )
    headline = select_operating_point(sweep, cost_ratio=settings.decision.fn_fp_cost_ratio)

    # What the OOF-tuned threshold does on the untouched holdout.
    _, adjusted_h, _ = cost_aware_predict(
        proba_test, pipeline.classes_, severe_from=threshold_cls, threshold=headline.threshold
    )
    holdout_decision = compute_metrics(
        y_test.to_numpy(),
        adjusted_h,
        proba_test,
        classes=pipeline.classes_,
        threshold=threshold_cls,
    )
    decision_compare = pd.DataFrame(
        [
            {
                "rule": "argmax (threshold 0.5-equivalent)",
                **{
                    k: holdout[k]
                    for k in (
                        "qwk",
                        "macro_f1",
                        "severe_recall",
                        "severe_precision",
                        "severe_missed",
                        "non_severe_progressed",
                    )
                },
            },
            {
                "rule": (
                    f"cost-optimal threshold {headline.threshold:.2f} "
                    f"(FN:FP {settings.decision.fn_fp_cost_ratio:g}:1)"
                ),
                **{
                    k: holdout_decision[k]
                    for k in (
                        "qwk",
                        "macro_f1",
                        "severe_recall",
                        "severe_precision",
                        "severe_missed",
                        "non_severe_progressed",
                    )
                },
            },
        ]
    )
    six_d, two_d = confusion_tables(y_test.to_numpy(), adjusted_h, threshold=threshold_cls)
    plots.plot_confusion(
        six_d,
        two_d,
        figures / f"confusion_cost_aware_{tag}.png",
        title=(
            f"{MODEL_REGISTRY[family_name].label} with cost-optimal threshold "
            f"{headline.threshold:.2f}"
        ),
    )
    _write_csv(sweep, reports / f"threshold_sweep_{tag}.csv")
    _write_csv(operating, reports / f"operating_points_{tag}.csv")
    _write_csv(decision_compare, reports / f"decision_rule_comparison_{tag}.csv")
    plots.plot_threshold_sweep(
        sweep,
        operating,
        figures / f"threshold_sweep_{tag}.png",
        headline_ratio=settings.decision.fn_fp_cost_ratio,
    )
    capacity = capacity_analysis(
        y_test.to_numpy() >= threshold_cls, ps_test, fractions=settings.decision.capacity_fractions
    )
    _write_csv(capacity, reports / f"capacity_analysis_{tag}.csv")
    plots.plot_capacity(capacity, figures / f"capacity_{tag}.png")

    # --- Robustness ------------------------------------------------------------
    log.info("evaluate: ablation")
    ablation = ablation_by_group(
        family_name, X_train, y_train, bundle.spec, settings, params=params or None
    )
    _write_csv(ablation, reports / f"ablation_{tag}.csv")
    plots.plot_ablation(ablation, figures / f"ablation_{tag}.png")

    missing = missingness_injection(
        pipeline,
        X_test,
        y_test,
        bundle.spec,
        fractions=settings.robustness.missingness_fractions,
        seed=seed,
        threshold=threshold_cls,
    )
    noise = noise_perturbation(
        pipeline,
        X_test,
        y_test,
        bundle.spec,
        scales=[
            settings.robustness.noise_scale,
            settings.robustness.noise_scale * 2,
            settings.robustness.noise_scale * 5,
        ],
        seed=seed,
        threshold=threshold_cls,
    )
    _write_csv(missing, reports / f"robustness_missingness_{tag}.csv")
    _write_csv(noise, reports / f"robustness_noise_{tag}.csv")
    plots.plot_robustness(missing, noise, figures / f"robustness_{tag}.png")

    subgroups = subgroup_performance(
        y_test.to_numpy(),
        adjusted_h,
        bundle.train_clean.loc[test_idx],
        settings.robustness.subgroup_columns,
        threshold=threshold_cls,
    )
    _write_csv(subgroups, reports / f"subgroup_performance_{tag}.csv")

    # --- Regime comparison (does the model lean on internal assessments?) ----
    other: Literal["A", "B"] = "B" if settings.features.regime == "A" else "A"
    other_settings = settings.model_copy(deep=True)
    other_settings.features.regime = other
    X_raw_train, _ = split_features_target(bundle.train_clean.loc[train_idx], settings)
    X_other, spec_other = engineer_features(X_raw_train, other_settings)
    other_cv = cross_validate_family(
        family_name, X_other, y_train, spec_other, other_settings, params=None
    )
    regime_table = pd.DataFrame(
        [
            {
                "regime": settings.features.regime,
                "n_features": len(bundle.spec.all_columns),
                **{
                    k: oof_cv.folds[k].mean()
                    for k in ("qwk", "mae", "macro_f1", "severe_recall", "severe_pr_auc")
                },
            },
            {
                "regime": other,
                "n_features": len(spec_other.all_columns),
                **{
                    k: other_cv.folds[k].mean()
                    for k in ("qwk", "mae", "macro_f1", "severe_recall", "severe_pr_auc")
                },
            },
        ]
    )
    _write_csv(regime_table, reports / f"regime_comparison_{family_name}.csv")

    selection = {
        "selected": selected_key,
        "family": family_name,
        "selection_reasoning": reasoning,
        "params": params,
        "cv": {
            k: float(cv_row[k])
            for k in ("qwk", "mae", "macro_f1", "severe_recall", "severe_precision")
            if k in cv_row
        },
        "holdout_argmax": holdout,
        "holdout_cost_aware": holdout_decision,
        "operating_point": headline.as_dict(),
        "calibration": calib_summary.to_dict(orient="records"),
    }
    _write_json(selection, _selection_path(settings))
    log.info(
        "evaluate: holdout QWK %.4f | severe recall argmax %.3f -> cost-aware %.3f "
        "(threshold %.2f)",
        holdout["qwk"],
        holdout["severe_recall"],
        holdout_decision["severe_recall"],
        headline.threshold,
    )
    return selection


# ---------------------------------------------------------------------------
# Stage 4: explain
# ---------------------------------------------------------------------------


def _load_selection(settings: Settings) -> dict[str, Any]:
    path = _selection_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the evaluate stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_explain(settings: Settings, bundle: DataBundle | None = None) -> None:
    bundle = bundle or load_bundle(settings)
    selection = _load_selection(settings)
    train_idx, test_idx = bundle.split(settings)
    X_train, y_train = bundle.X.loc[train_idx], bundle.y.loc[train_idx]
    X_test, y_test = bundle.X.loc[test_idx], bundle.y.loc[test_idx]
    reports, figures, tag = (
        settings.paths.reports_dir,
        settings.paths.figures_dir,
        settings.regime_tag,
    )
    seed = settings.project.seed
    threshold_cls = settings.data.progress_threshold
    threshold_p = float(selection["operating_point"]["threshold"])
    family = selection["family"]

    pipeline = make_pipeline(family, bundle.spec, seed)
    if selection["params"]:
        pipeline.set_params(**selection["params"])
    pipeline.fit(X_train, y_train.to_numpy())

    log.info("explain: permutation importance")
    perm = permutation_importance_table(pipeline, X_test, y_test, n_repeats=10, seed=seed)
    _write_csv(perm, reports / f"permutation_importance_{tag}.csv")
    plots.plot_permutation_importance(perm, figures / f"permutation_importance_{tag}.png")

    tree_shap = tree_shap_logodds(pipeline, X_test, max_rows=400, seed=seed)
    if tree_shap is not None:
        log.info("explain: tree SHAP (log-odds)")
        _write_csv(tree_shap.mean_abs_table(30), reports / f"shap_tree_logodds_{tag}.csv")
        for cls in (1, 6):
            shap.plots.beeswarm(tree_shap.explanation_for_class(cls), max_display=15, show=False)
            plt.title(f"Tree SHAP, log-odds of severity {cls}")
            plt.savefig(
                figures / f"shap_beeswarm_class{cls}_{tag}.png", dpi=130, bbox_inches="tight"
            )
            plt.close()

    log.info("explain: expected-severity SHAP (permutation explainer)")
    sev_shap = expected_severity_shap(pipeline, X_test, max_rows=100, background_rows=30, seed=seed)
    _write_csv(sev_shap.mean_abs_table(30), reports / f"shap_expected_severity_{tag}.csv")
    shap.plots.beeswarm(sev_shap.explanation(), max_display=15, show=False)
    plt.title("Contribution to expected severity (1-6)")
    plt.savefig(
        figures / f"shap_expected_severity_beeswarm_{tag}.png", dpi=130, bbox_inches="tight"
    )
    plt.close()

    proba_test = pipeline.predict_proba(X_test)
    cases = pick_illustrative_cases(
        proba_test, pipeline.classes_, severe_from=threshold_cls, threshold=threshold_p
    )
    case_rows = X_test.index[list(cases.values())]
    local = expected_severity_shap(pipeline, X_test, rows=case_rows, background_rows=50, seed=seed)
    case_summary = []
    for (label, pos), row in zip(cases.items(), range(len(case_rows)), strict=True):
        idx = case_rows[row]
        expl = local.explanation()[row]
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.title(
            f"{label.replace('_', ' ')}: case "
            f"{bundle.train_clean.loc[idx, settings.data.id_column]}"
        )
        plt.savefig(figures / f"shap_waterfall_{label}_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close()
        case_summary.append(
            {
                "case_type": label,
                settings.data.id_column: bundle.train_clean.loc[idx, settings.data.id_column],
                "true_severity": int(y_test.loc[idx]),
                "predicted_argmax": int(pipeline.classes_[np.argmax(proba_test[pos])]),
                "p_severe": float(
                    p_severe(proba_test[[pos]], list(pipeline.classes_), threshold_cls)[0]
                ),
                "expected_severity": float(
                    local.base_value + local.values[row].sum()  # noqa: PD011 - ndarray, not a Series
                ),
            }
        )
    _write_csv(pd.DataFrame(case_summary), reports / f"illustrative_cases_{tag}.csv")

    log.info("explain: surrogate rules")
    rules, fidelity, _ = surrogate_rules(
        pipeline, X_train, severe_from=threshold_cls, threshold=threshold_p, seed=seed
    )
    (reports / f"surrogate_rules_{tag}.txt").write_text(
        f"Fidelity to model decisions: {fidelity:.3f}\n\n{rules}", encoding="utf-8"
    )

    log.info("explain: monotonic constraint comparison")
    mono, constrained = monotonic_constraint_comparison(X_train, y_train, bundle.spec, settings)
    mono["n_constrained_features"] = len(constrained)
    _write_csv(mono, reports / f"monotonic_constraints_{tag}.csv")

    log.info("explain: latent cut-point recovery")
    ordered = make_pipeline("ordered_logit", bundle.spec, seed).fit(X_train, y_train.to_numpy())
    raw_train, _ = split_features_target(bundle.train_clean.loc[train_idx], settings)
    latent = latent_cutpoint_analysis(ordered, X_train, y_train, raw_train)
    _write_csv(latent.per_class_latent, reports / f"latent_per_class_{tag}.csv")
    if latent.eis_class_ranges is not None:
        _write_csv(latent.eis_class_ranges, reports / f"latent_eis_boundaries_{tag}.csv")
    plots.plot_latent_scores(
        latent.latent_scores,
        y_train.to_numpy(),
        latent.cutpoints_latent,
        figures / f"latent_scores_{tag}.png",
    )
    log.info("explain: done")


# ---------------------------------------------------------------------------
# Stage 5: predict
# ---------------------------------------------------------------------------


def stage_predict(settings: Settings, bundle: DataBundle | None = None) -> Path:
    """Refit the selected model on all training data and score the holdback set."""
    bundle = bundle or load_bundle(settings)
    selection = _load_selection(settings)
    family, params = selection["family"], selection["params"]
    seed = settings.project.seed
    threshold_cls = settings.data.progress_threshold
    ratio = settings.decision.fn_fp_cost_ratio

    # Operating threshold from OOF probabilities over the *full* training set.
    log.info("predict: OOF probabilities on the full training set for threshold selection")
    oof = cross_validate_family(
        family, bundle.X, bundle.y, bundle.spec, settings, params=params or None
    )
    sweep = threshold_sweep(
        bundle.y.to_numpy() >= threshold_cls,
        p_severe(oof.oof_proba, list(CLASSES), threshold_cls),
        cost_ratios=settings.decision.sensitivity_ratios,
    )
    operating = select_operating_point(sweep, cost_ratio=ratio)

    log.info("predict: fitting %s on %d rows", family, len(bundle.X))
    pipeline = make_pipeline(family, bundle.spec, seed)
    if params:
        pipeline.set_params(**params)
    pipeline.fit(bundle.X, bundle.y.to_numpy())

    proba = pipeline.predict_proba(bundle.X_holdback)
    argmax, adjusted, progress = cost_aware_predict(
        proba, pipeline.classes_, severe_from=threshold_cls, threshold=operating.threshold
    )
    ps = p_severe(proba, list(pipeline.classes_), threshold_cls)

    predictions = pd.DataFrame(
        {
            settings.data.id_column: bundle.holdback_clean[settings.data.id_column].to_numpy(),
            "PredictedSeverityScore": adjusted.astype(int),
            "ProgressDecision": np.where(
                progress, "Progressed for further investigation", "Not progressed"
            ),
            "P_Severe": np.round(ps, 4),
            "ArgmaxSeverityScore": argmax.astype(int),
        }
    )
    for k, c in enumerate(pipeline.classes_):
        predictions[f"P_Severity_{c}"] = np.round(proba[:, k], 4)

    out = settings.paths.predictions_file
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False, lineterminator="\n")
    predictions_sha = sha256_of(out)

    artefact = ModelArtefact(
        pipeline=pipeline,
        spec=bundle.spec,
        model_name=family,
        params=params,
        threshold=operating.threshold,
        cost_ratio=ratio,
        severe_from=threshold_cls,
        regime=settings.features.regime,
        classes=[int(c) for c in pipeline.classes_],
        package_version=__version__,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    settings.paths.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, settings.paths.models_dir / ARTEFACT_NAME, compress=3)

    manifest: dict[str, Any] = {
        "package_version": __version__,
        "trained_at": artefact.trained_at,
        "git_sha": _git_sha(settings.root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "regime": settings.features.regime,
        "workbook_sha256": bundle.workbook_sha256,
        "n_training_rows": len(bundle.X),
        "n_holdback_rows": len(bundle.X_holdback),
        "n_features_engineered": len(bundle.spec.all_columns),
        "model": {"family": family, "label": MODEL_REGISTRY[family].label, "params": params},
        "decision": {**operating.as_dict(), "severe_from": threshold_cls},
        "cv_full_training": {
            k: float(oof.folds[k].mean())
            for k in (
                "qwk",
                "mae",
                "macro_f1",
                "severe_recall",
                "severe_precision",
                "severe_pr_auc",
            )
        },
        "holdout": {
            k: selection["holdout_cost_aware"][k]
            for k in ("qwk", "mae", "macro_f1", "severe_recall", "severe_precision")
        },
        "predictions": {
            "file": _display_path(out, settings.root),
            "sha256": predictions_sha,
            "rows": len(predictions),
            "predicted_distribution": _distribution(predictions["PredictedSeverityScore"]),
            "progressed_share": float(progress.mean()),
        },
        "packages": {
            name: _pkg_version(name)
            for name in (
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "statsmodels",
                "shap",
                "optuna",
            )
        },
    }
    _write_json(manifest, settings.paths.models_dir / MANIFEST_NAME)
    log.info(
        "predict: wrote %d predictions to %s (sha256 %s)",
        len(predictions),
        out,
        predictions_sha[:12],
    )
    log.info(
        "predict: predicted severity distribution %s | progressed %.1f%%",
        manifest["predictions"]["predicted_distribution"],
        100 * progress.mean(),
    )
    return out


def _display_path(path: Path, root: Path) -> str:
    """Repository-relative POSIX path when possible, absolute otherwise."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _distribution(series: pd.Series) -> dict[int, int]:
    counts = series.value_counts().sort_index()
    return {
        int(level): int(count)
        for level, count in zip(counts.index.to_numpy(), counts.to_numpy(), strict=True)
    }


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_artefact(path: Path) -> ModelArtefact:
    artefact = joblib.load(path)
    if not isinstance(artefact, ModelArtefact):
        raise TypeError(f"{path} does not contain a ModelArtefact")
    return artefact


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------


def run_all(settings: Settings) -> Path:
    started = time.perf_counter()
    bundle = stage_validate(settings)
    stage_train(settings, bundle)
    stage_evaluate(settings, bundle)
    stage_explain(settings, bundle)
    out = stage_predict(settings, bundle)
    log.info("all stages complete in %.1f minutes", (time.perf_counter() - started) / 60)
    return out


def default_settings(override: str | None = None) -> Settings:
    return load_settings(override)
