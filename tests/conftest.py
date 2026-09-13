"""Shared fixtures. Uses the real workbook; writes outputs to a temp dir."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from severity_triage.config import Settings, find_project_root, load_settings
from severity_triage.features import engineer_features, split_features_target
from severity_triage.io import load_datasets
from severity_triage.validation import clean

ROOT = find_project_root(Path(__file__).parent)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(root=ROOT)


@pytest.fixture
def tmp_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Settings with all outputs redirected to ``tmp_path`` and a fast profile."""
    s = settings.model_copy(deep=True)
    s.paths.processed_dir = tmp_path / "processed"
    s.paths.predictions_file = tmp_path / "predictions" / "holdback_predictions.csv"
    s.paths.models_dir = tmp_path / "models"
    s.paths.reports_dir = tmp_path / "reports"
    s.paths.figures_dir = tmp_path / "reports" / "figures"
    s.split.cv_folds = 3
    s.split.cv_repeats = 1
    s.tuning.enabled = False
    s.models.candidates = ["logistic", "ordered_logit"]
    s.robustness.missingness_fractions = [0.2]
    s.decision.capacity_fractions = [0.3, 0.4]
    return s


@pytest.fixture(scope="session")
def raw(settings: Settings):
    return load_datasets(settings)


@pytest.fixture(scope="session")
def train_clean(settings: Settings, raw) -> pd.DataFrame:
    return clean(raw.training, settings, is_training=True).frame


@pytest.fixture(scope="session")
def holdback_clean(settings: Settings, raw) -> pd.DataFrame:
    return clean(raw.holdback, settings, is_training=False).frame


@pytest.fixture(scope="session")
def features(settings: Settings, train_clean: pd.DataFrame):
    X_raw, y = split_features_target(train_clean, settings)
    X, spec = engineer_features(X_raw, settings)
    return X, y, spec
