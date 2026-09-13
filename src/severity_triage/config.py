"""Settings loaded from configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

__all__ = ["Settings", "find_project_root", "load_settings"]


class ProjectConfig(BaseModel):
    name: str = "severity-triage"
    seed: int = 42


class PathsConfig(BaseModel):
    raw_workbook: Path
    checksums: Path
    processed_dir: Path
    predictions_file: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    def resolve(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path anchored at ``root``."""
        return PathsConfig(
            **{
                name: (root / value if not Path(value).is_absolute() else Path(value))
                for name, value in self.model_dump().items()
            }
        )


class SheetsConfig(BaseModel):
    training: str
    holdback: str
    dictionary: str


class DataConfig(BaseModel):
    sheets: SheetsConfig
    target: str = "SeverityScore"
    id_column: str = "CaseReference"
    progress_threshold: int = 4
    leak_columns: list[str] = Field(default_factory=lambda: ["OmbudsmanInvestigationRequired"])
    range_violation_strategy: Literal["null", "clip"] = "null"


class FeaturesConfig(BaseModel):
    regime: Literal["A", "B"] = "A"
    internal_proxy_columns: list[str] = Field(
        default_factory=lambda: [
            "EstimatedImpactScore",
            "EstimatedRiskScore",
            "PredictedRemedyBand",
        ]
    )
    date_column: str = "CaseCreatedDate"
    reference_date: str = "2026-07-01"


class SplitConfig(BaseModel):
    test_size: float = 0.2
    cv_folds: int = 5
    cv_repeats: int = 3

    @field_validator("test_size")
    @classmethod
    def _check_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1")
        return value


class ModelsConfig(BaseModel):
    candidates: list[str]
    selected: str | None = None
    primary_metric: str = "qwk"
    # "best": highest CV primary metric.
    # "one_se": simplest model whose CV score is within the tolerance of the best,
    #           where tolerance = max(selection_tolerance_se * SE, selection_min_tolerance).
    #           The SE term is Breiman's one-standard-error rule; the floor stops a
    #           large fold count from declaring practically identical models different.
    selection_rule: Literal["best", "one_se"] = "one_se"
    selection_tolerance_se: float = 1.0
    selection_min_tolerance: float = 0.005


class TuningConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 40
    timeout_seconds: int = 600
    models: list[str] = Field(default_factory=lambda: ["xgboost", "lightgbm"])


class DecisionConfig(BaseModel):
    fn_fp_cost_ratio: float = 5.0
    sensitivity_ratios: list[float] = Field(default_factory=lambda: [3.0, 5.0, 10.0])
    capacity_fractions: list[float] = Field(default_factory=lambda: [0.25, 0.30, 0.35, 0.40])


class RobustnessConfig(BaseModel):
    missingness_fractions: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])
    noise_scale: float = 0.1
    subgroup_columns: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    """Root settings object. Paths are absolute once loaded."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    models: ModelsConfig
    tuning: TuningConfig
    decision: DecisionConfig
    robustness: RobustnessConfig
    root: Path

    @property
    def regime_tag(self) -> str:
        return f"regime_{self.features.regime.lower()}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until a directory containing ``configs/base.yaml`` is found.

    Tries ``start`` (default: the working directory) first, then falls back to
    the package's own location so an editable install works from anywhere.
    """
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).resolve().parent]
    for origin in starts:
        current = origin.resolve()
        for candidate in (current, *current.parents):
            if (candidate / "configs" / "base.yaml").exists():
                return candidate
    raise FileNotFoundError(
        "Could not locate configs/base.yaml above " + ", ".join(map(str, starts))
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def load_settings(
    override: str | Path | None = None,
    *,
    root: Path | None = None,
    base: str | Path = "configs/base.yaml",
) -> Settings:
    """Load ``base.yaml``, apply an optional override file, resolve paths."""
    project_root = root or find_project_root()
    raw = _read_yaml(project_root / base)
    if override is not None:
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = project_root / override_path
        raw = _deep_merge(raw, _read_yaml(override_path))

    settings = Settings(root=project_root, **raw)
    settings.paths = settings.paths.resolve(project_root)
    return settings
