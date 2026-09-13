from __future__ import annotations

import numpy as np
import pytest

from severity_triage.decision import (
    capacity_analysis,
    cost_aware_predict,
    expected_cost,
    select_operating_point,
    threshold_sweep,
)
from severity_triage.evaluation import compute_metrics, error_analysis, p_severe


def test_expected_cost_weights_false_negatives() -> None:
    y = np.array([1, 1, 0, 0])
    pred_fn = np.array([1, 0, 0, 0])  # one false negative
    pred_fp = np.array([1, 1, 1, 0])  # one false positive
    assert expected_cost(y, pred_fn, fn_fp_ratio=5.0) == pytest.approx(5 / 4)
    assert expected_cost(y, pred_fp, fn_fp_ratio=5.0) == pytest.approx(1 / 4)


def test_operating_point_moves_down_as_fn_cost_rises() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 2000), 0, 1)
    sweep = threshold_sweep(y, p, cost_ratios=[1.0, 5.0, 20.0])
    t1 = select_operating_point(sweep, cost_ratio=1.0).threshold
    t5 = select_operating_point(sweep, cost_ratio=5.0).threshold
    t20 = select_operating_point(sweep, cost_ratio=20.0).threshold
    assert t1 >= t5 >= t20
    assert (
        select_operating_point(sweep, cost_ratio=20.0).severe_recall
        >= select_operating_point(sweep, cost_ratio=1.0).severe_recall
    )


def test_select_operating_point_requires_ratio_column() -> None:
    sweep = threshold_sweep(np.array([1, 0]), np.array([0.9, 0.1]), cost_ratios=[3.0])
    with pytest.raises(KeyError):
        select_operating_point(sweep, cost_ratio=7.0)


def test_capacity_analysis_monotone_in_capacity() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.3
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 500), 0, 1)
    table = capacity_analysis(y, p, fractions=[0.1, 0.3, 0.5])
    assert table["severe_recall"].is_monotonic_increasing
    assert table["severe_missed"].is_monotonic_decreasing
    assert table.loc[0, "cases_progressed"] == 50


def test_cost_aware_predict_is_consistent_with_decision() -> None:
    classes = np.arange(1, 7)
    proba = np.array(
        [
            [0.10, 0.30, 0.35, 0.15, 0.07, 0.03],  # argmax 3 but P(>=4)=0.25
            [0.02, 0.03, 0.05, 0.40, 0.30, 0.20],  # clearly severe
            [0.60, 0.30, 0.05, 0.03, 0.01, 0.01],  # clearly mild
            [0.05, 0.10, 0.40, 0.44, 0.005, 0.005],  # argmax 4 but P(>=4)=0.45
        ]
    )
    argmax, adjusted, progress = cost_aware_predict(proba, classes, severe_from=4, threshold=0.2)
    assert argmax.tolist() == [3, 4, 1, 4]
    assert progress.tolist() == [True, True, False, True]
    assert adjusted.tolist() == [4, 4, 1, 4]
    assert ((adjusted >= 4) == progress).all()

    _, adjusted_hi, progress_hi = cost_aware_predict(proba, classes, severe_from=4, threshold=0.5)
    assert progress_hi.tolist() == [False, True, False, False]
    assert adjusted_hi.tolist() == [3, 4, 1, 3]  # row 4 pushed down to most likely mild class


def test_compute_metrics_and_error_analysis_consistency() -> None:
    y = np.array([1, 2, 3, 4, 5, 6, 4, 3])
    pred = np.array([1, 2, 4, 3, 5, 6, 4, 3])
    proba = np.full((8, 6), 1 / 6)
    m = compute_metrics(y, pred, proba)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["severe_missed"] == 1
    assert m["non_severe_progressed"] == 1
    assert m["exact_or_adjacent"] == 1.0
    table = error_analysis(y, pred)
    assert table.loc["crossed_boundary_downwards (severe -> not)", "count"] == 1
    assert table.loc["crossed_boundary_upwards (not -> severe)", "count"] == 1
    assert p_severe(proba, list(range(1, 7)), 4).tolist() == pytest.approx([0.5] * 8)
