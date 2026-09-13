"""Console entrypoint: ``severity-triage <stage> [--config FILE] [--fast]``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from severity_triage import __version__
from severity_triage.config import Settings, load_settings
from severity_triage.pipeline import (
    run_all,
    stage_evaluate,
    stage_explain,
    stage_predict,
    stage_train,
    stage_validate,
)

__all__ = ["main"]

STAGES = ("validate", "train", "evaluate", "explain", "predict", "all")
FAST_OVERRIDE = "configs/fast.yaml"


def _configure_logging(verbose: bool, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("matplotlib", "shap", "optuna", "lightgbm", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="severity-triage",
        description="Severity classification and cost-aware triage pipeline.",
    )
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML override merged on top of configs/base.yaml (e.g. configs/regime_b.yaml)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            f"also merge {FAST_OVERRIDE}: fewer CV repeats and tuning trials, "
            "for CI and smoke tests"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _load(config: str | None, fast: bool) -> Settings:
    settings = load_settings(config)
    if fast:
        fast_settings = load_settings(FAST_OVERRIDE)
        settings.split = fast_settings.split
        settings.tuning = fast_settings.tuning
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = _load(args.config, args.fast)
    _configure_logging(args.verbose, settings.paths.models_dir / "last_run.log")
    log = logging.getLogger("severity_triage")
    log.info(
        "severity-triage %s | stage=%s | regime=%s | seed=%d | fast=%s",
        __version__,
        args.stage,
        settings.features.regime,
        settings.project.seed,
        args.fast,
    )

    if args.stage == "validate":
        stage_validate(settings)
    elif args.stage == "train":
        stage_train(settings)
    elif args.stage == "evaluate":
        stage_evaluate(settings)
    elif args.stage == "explain":
        stage_explain(settings)
    elif args.stage == "predict":
        stage_predict(settings)
    elif args.stage == "all":
        run_all(settings)
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown stage {args.stage}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
