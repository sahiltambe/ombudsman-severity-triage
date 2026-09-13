"""Save figures to disk (Agg backend for headless runs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from severity_triage.evaluation import CLASSES

__all__ = [
    "plot_ablation",
    "plot_baseline_ladder",
    "plot_calibration",
    "plot_capacity",
    "plot_confusion",
    "plot_latent_scores",
    "plot_model_comparison",
    "plot_permutation_importance",
    "plot_robustness",
    "plot_threshold_sweep",
    "plot_tuning_trajectory",
]

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = "viridis"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(six: pd.DataFrame, two: pd.DataFrame, path: Path, *, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    sns.heatmap(six, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Severity 1-6")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    sns.heatmap(two, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1])
    axes[1].set_title("Triage decision")
    axes[1].set_xlabel("Predicted")
    fig.suptitle(title)
    return _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path: Path, *, metric: str = "qwk") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = table.sort_values(metric, ascending=True)
    for ax, (col, label) in zip(
        axes,
        (
            (metric, "Quadratic weighted kappa (CV mean +/- sd)"),
            ("severe_recall", "Severe-case recall (CV mean +/- sd)"),
        ),
        strict=True,
    ):
        ax.barh(
            order["label"],
            order[col],
            xerr=order.get(f"{col}_std"),
            color=sns.color_palette(_PALETTE, len(order)),
        )
        ax.set_xlabel(label)
        lo = max(0.0, order[col].min() - 0.05)
        ax.set_xlim(lo, min(1.0, order[col].max() + 0.02))
    fig.suptitle("Model comparison, repeated stratified cross-validation")
    return _save(fig, path)


def plot_baseline_ladder(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(table["rung"], table["qwk"], marker="o", label="QWK")
    ax.plot(table["rung"], table["severe_recall"], marker="s", label="Severe recall")
    ax.set_xticks(table["rung"])
    ax.set_xticklabels(
        [a.split(",")[0].split("(")[0].strip() for a in table["approach"]], rotation=20, ha="right"
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Baseline ladder: what each idea buys")
    ax.legend()
    return _save(fig, path)


def plot_threshold_sweep(
    sweep: pd.DataFrame, operating_points: pd.DataFrame, path: Path, *, headline_ratio: float
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(sweep["threshold"], sweep["severe_recall"], label="Severe recall")
    ax.plot(sweep["threshold"], sweep["severe_precision"], label="Severe precision")
    ax.plot(sweep["threshold"], sweep["progressed_share"], label="Share progressed", linestyle="--")
    for _, op in operating_points.iterrows():
        ax.axvline(op["threshold"], color="grey", alpha=0.4, linestyle=":")
        ax.text(
            op["threshold"], 0.02, f"{op['cost_ratio']:g}:1", rotation=90, va="bottom", fontsize=8
        )
    ax.set_xlabel("Threshold on P(severity >= 4)")
    ax.set_ylabel("Rate")
    ax.set_title("Operating characteristics")
    ax.legend(loc="center right")

    ax = axes[1]
    cost_cols = [c for c in sweep.columns if c.startswith("cost_ratio_")]
    for col in cost_cols:
        ratio = col.replace("cost_ratio_", "")
        lw = 2.5 if float(ratio) == headline_ratio else 1.2
        ax.plot(sweep["threshold"], sweep[col], label=f"FN:FP = {ratio}:1", linewidth=lw)
    ax.set_xlabel("Threshold on P(severity >= 4)")
    ax.set_ylabel("Expected cost per case")
    ax.set_title("Expected cost by cost ratio")
    ax.legend()
    return _save(fig, path)


def plot_calibration(before: pd.DataFrame, after: pd.DataFrame | None, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
    ax.plot(before["mean_predicted"], before["observed_rate"], marker="o", label="Uncalibrated")
    if after is not None:
        ax.plot(
            after["mean_predicted"], after["observed_rate"], marker="s", label="Isotonic calibrated"
        )
    ax.set_xlabel("Mean predicted P(severe)")
    ax.set_ylabel("Observed severe rate")
    ax.set_title("Reliability of the severe-case probability")
    ax.legend()
    return _save(fig, path)


def plot_capacity(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["capacity_fraction"], table["severe_recall"], marker="o", label="Severe recall")
    ax.plot(
        table["capacity_fraction"], table["severe_precision"], marker="s", label="Severe precision"
    )
    for _, r in table.iterrows():
        ax.annotate(
            f"{int(r['severe_missed'])} missed",
            (r["capacity_fraction"], r["severe_recall"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Fraction of cases the organisation can progress")
    ax.set_ylabel("Rate")
    ax.set_title("Capacity-constrained triage")
    ax.legend()
    return _save(fig, path)


def plot_permutation_importance(table: pd.DataFrame, path: Path, *, top: int = 25) -> Path:
    head = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(head) + 1.5))
    ax.barh(
        head["feature"],
        head["importance_mean"],
        xerr=head["importance_std"],
        color=sns.color_palette(_PALETTE, len(head)),
    )
    ax.set_xlabel("Drop in QWK when shuffled")
    ax.set_title("Permutation importance")
    return _save(fig, path)


def plot_ablation(table: pd.DataFrame, path: Path) -> Path:
    body = table[table["dropped_group"] != "(none)"].sort_values("delta_qwk")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(body) + 1.5))
    colors = ["#c0392b" if d < 0 else "#27ae60" for d in body["delta_qwk"]]
    ax.barh(body["dropped_group"], body["delta_qwk"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in CV QWK when the block is removed")
    ax.set_title("Feature-block ablation")
    return _save(fig, path)


def plot_robustness(missing: pd.DataFrame, noise: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (table, xcol, title) in zip(
        axes,
        (
            (missing, "missing_fraction", "Injected missingness"),
            (noise, "noise_scale", "Gaussian noise (x std)"),
        ),
        strict=True,
    ):
        ax.plot(table[xcol], table["qwk"], marker="o", label="QWK")
        ax.plot(table[xcol], table["severe_recall"], marker="s", label="Severe recall")
        ax.set_xlabel(xcol.replace("_", " "))
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend()
    return _save(fig, path)


def plot_tuning_trajectory(trials: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for name, table in trials.items():
        ax.plot(table["number"], table["best_so_far"], marker=".", label=f"{name} (best so far)")
        ax.scatter(table["number"], table["cv_qwk"], s=10, alpha=0.4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV QWK")
    ax.set_title("Hyperparameter search trajectory")
    ax.legend()
    return _save(fig, path)


def plot_latent_scores(
    latent: np.ndarray, y: np.ndarray, cutpoints: np.ndarray, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    frame = pd.DataFrame({"latent": latent, "severity": y})
    sns.violinplot(
        data=frame,
        x="severity",
        y="latent",
        hue="severity",
        palette=_PALETTE,
        inner="quartile",
        legend=False,
        ax=ax,
    )
    for i, c in enumerate(cutpoints):
        ax.axhline(c, color="grey", linestyle=":", linewidth=1)
        ax.text(5.55, c, f"theta_{i + 1}", va="center", fontsize=8, color="grey")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("True severity")
    ax.set_ylabel("Latent score  x . beta")
    ax.set_title("Ordered logit: latent severity by true class, with fitted cut-points")
    return _save(fig, path)
