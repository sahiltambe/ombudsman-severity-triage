"""Triage decision: turn class probabilities into close vs investigate.

Missing a severe case is costlier than reviewing a mild one. Default FN:FP = 5:1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

__all__ = [
    "OperatingPoint",
    "capacity_analysis",
    "cost_aware_predict",
    "expected_cost",
    "select_operating_point",
    "threshold_sweep",
]


@dataclass(frozen=True)
class OperatingPoint:
    """A chosen probability threshold and what it implies on the data it was tuned on."""

    cost_ratio: float
    threshold: float
    severe_recall: float
    severe_precision: float
    progressed_share: float
    expected_cost_per_case: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cost_ratio": self.cost_ratio,
            "threshold": self.threshold,
            "severe_recall": self.severe_recall,
            "severe_precision": self.severe_precision,
            "progressed_share": self.progressed_share,
            "expected_cost_per_case": self.expected_cost_per_case,
        }


def expected_cost(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, *, fn_fp_ratio: float) -> float:
    """Mean cost per case with a false negative costing ``fn_fp_ratio`` false positives."""
    y_true_bin = np.asarray(y_true_bin, dtype=bool)
    y_pred_bin = np.asarray(y_pred_bin, dtype=bool)
    fn = np.sum(y_true_bin & ~y_pred_bin)
    fp = np.sum(~y_true_bin & y_pred_bin)
    return float((fn_fp_ratio * fn + fp) / len(y_true_bin))


def threshold_sweep(
    y_true_bin: np.ndarray,
    p: np.ndarray,
    *,
    cost_ratios: list[float],
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Recall, precision, progressed share and expected cost for every threshold."""
    y_true_bin = np.asarray(y_true_bin, dtype=bool)
    p = np.asarray(p)
    thresholds = grid if grid is not None else np.round(np.arange(0.02, 0.99, 0.01), 2)
    rows = []
    for t in thresholds:
        pred = p >= t
        row = {
            "threshold": float(t),
            "severe_recall": recall_score(y_true_bin, pred, zero_division=0),
            "severe_precision": precision_score(y_true_bin, pred, zero_division=0),
            "false_negatives": int(np.sum(y_true_bin & ~pred)),
            "false_positives": int(np.sum(~y_true_bin & pred)),
            "progressed_share": float(pred.mean()),
        }
        for ratio in cost_ratios:
            row[f"cost_ratio_{ratio:g}"] = expected_cost(y_true_bin, pred, fn_fp_ratio=ratio)
        rows.append(row)
    return pd.DataFrame(rows)


def select_operating_point(sweep: pd.DataFrame, *, cost_ratio: float) -> OperatingPoint:
    """Threshold with the lowest expected cost for a given ratio.

    Ties are broken towards the higher threshold (fewer progressed cases),
    since at equal cost the organisation prefers less review effort.
    """
    column = f"cost_ratio_{cost_ratio:g}"
    if column not in sweep:
        raise KeyError(f"{column} not in sweep; rerun threshold_sweep with this ratio")
    best = sweep.loc[sweep[column] == sweep[column].min()].iloc[-1]
    return OperatingPoint(
        cost_ratio=cost_ratio,
        threshold=float(best["threshold"]),
        severe_recall=float(best["severe_recall"]),
        severe_precision=float(best["severe_precision"]),
        progressed_share=float(best["progressed_share"]),
        expected_cost_per_case=float(best[column]),
    )


def capacity_analysis(
    y_true_bin: np.ndarray, p: np.ndarray, *, fractions: list[float]
) -> pd.DataFrame:
    """If only a fraction of cases can be progressed, rank by ``p`` and see what is caught."""
    y_true_bin = np.asarray(y_true_bin, dtype=bool)
    order = np.argsort(-np.asarray(p))
    n = len(p)
    rows = []
    for fraction in fractions:
        k = round(fraction * n)
        chosen = np.zeros(n, dtype=bool)
        chosen[order[:k]] = True
        rows.append(
            {
                "capacity_fraction": fraction,
                "cases_progressed": k,
                "implied_threshold": float(np.asarray(p)[order[k - 1]]) if k > 0 else 1.0,
                "severe_recall": recall_score(y_true_bin, chosen, zero_division=0),
                "severe_precision": precision_score(y_true_bin, chosen, zero_division=0),
                "severe_missed": int(np.sum(y_true_bin & ~chosen)),
            }
        )
    return pd.DataFrame(rows)


def cost_aware_predict(
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    severe_from: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Severity prediction made consistent with the cost-optimal triage decision.

    Returns ``(argmax_severity, adjusted_severity, progress_flag)``. The
    adjusted score equals the argmax unless the decision layer disagrees with
    it, in which case it becomes the most likely class on the decided side of
    the boundary. That way anyone reading only the severity column reaches the
    same close / investigate outcome as the decision column.
    """
    classes = np.asarray(classes)
    argmax = classes[np.argmax(proba, axis=1)]
    severe_cols = classes >= severe_from
    ps = proba[:, severe_cols].sum(axis=1)
    progress = ps >= threshold

    adjusted = argmax.copy()
    push_up = progress & (argmax < severe_from)
    push_down = ~progress & (argmax >= severe_from)
    if push_up.any():
        adjusted[push_up] = classes[severe_cols][np.argmax(proba[push_up][:, severe_cols], axis=1)]
    if push_down.any():
        adjusted[push_down] = classes[~severe_cols][
            np.argmax(proba[push_down][:, ~severe_cols], axis=1)
        ]
    return argmax, adjusted, progress
