"""End-to-end smoke test on a temp directory (repeatable predictions)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from severity_triage.config import Settings
from severity_triage.io import sha256_of
from severity_triage.pipeline import (
    ARTEFACT_NAME,
    MANIFEST_NAME,
    load_artefact,
    run_all,
    stage_predict,
)


@pytest.mark.slow
def test_full_pipeline_runs_and_is_repeatable(tmp_settings: Settings) -> None:
    out = run_all(tmp_settings)

    # --- deliverable shape -------------------------------------------------
    predictions = pd.read_csv(out)
    assert len(predictions) == 500
    assert predictions["CaseReference"].is_unique
    assert predictions["CaseReference"].str.startswith("HLD-").all()
    assert predictions["PredictedSeverityScore"].between(1, 6).all()
    assert predictions["PredictedSeverityScore"].dtype.kind == "i"
    assert (
        predictions["ProgressDecision"]
        .isin(["Progressed for further investigation", "Not progressed"])
        .all()
    )
    assert predictions["P_Severe"].between(0, 1).all()
    # Decision column and severity column always agree on the triage outcome.
    assert (
        (predictions["PredictedSeverityScore"] >= 4)
        == (predictions["ProgressDecision"] != "Not progressed")
    ).all()

    # --- artefacts -----------------------------------------------------------
    models_dir = tmp_settings.paths.models_dir
    artefact = load_artefact(models_dir / ARTEFACT_NAME)
    assert artefact.classes == [1, 2, 3, 4, 5, 6]
    assert 0 < artefact.threshold < 1
    manifest = json.loads((models_dir / MANIFEST_NAME).read_text())
    assert manifest["predictions"]["rows"] == 500
    assert manifest["predictions"]["sha256"] == sha256_of(out)
    assert manifest["workbook_sha256"]

    reports = tmp_settings.paths.reports_dir
    for name in (
        "baseline_ladder_regime_a.csv",
        "model_comparison_regime_a.csv",
        "holdout_metrics_regime_a.csv",
        "threshold_sweep_regime_a.csv",
        "operating_points_regime_a.csv",
        "ablation_regime_a.csv",
        "subgroup_performance_regime_a.csv",
        "permutation_importance_regime_a.csv",
        "surrogate_rules_regime_a.txt",
        "latent_eis_boundaries_regime_a.csv",
    ):
        assert (reports / name).exists(), name
    assert any(tmp_settings.paths.figures_dir.glob("*.png"))

    # --- repeatability: a second predict run is byte-identical -----------------
    first = sha256_of(out)
    stage_predict(tmp_settings)
    assert sha256_of(out) == first
