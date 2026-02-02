"""
Experiment runner utilities for ML experimentation pipelines.

This module provides generic helpers for:
- Creating experiment directories
- Running experiment variants
- Aggregating multiple seeds
- Logging metrics and generating reports
"""

import copy
import datetime
import importlib
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reporting import (
    ReportConfig,
    VariantSummary,
    get_default_report_config,
    load_report_config,
    resolve_project_name,
    set_default_report_config,
    write_config_report,
    write_experiment_report,
)
from .utils import get_git_commit

logger = logging.getLogger(__name__)

BASE_DIR = Path("./experiments")
CODE_SNAPSHOT_DIRNAME = "code_snapshot"

# Override from project-specific experiments if needed.
DEFAULT_CODE_SNAPSHOT_DIRS = [Path(__file__).resolve().parent]


def archive_experiment_state(
    output_dir: Path, *, code_dirs: Iterable[Path] | None = None
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = output_dir / CODE_SNAPSHOT_DIRNAME
    snapshot_root.mkdir(parents=True, exist_ok=True)

    dirs_to_save = [
        Path(p).resolve() for p in (code_dirs or DEFAULT_CODE_SNAPSHOT_DIRS)
    ]

    for src_dir in dirs_to_save:
        if not src_dir.exists():
            logger.debug("Snapshot source missing: %s", src_dir)
            continue
        dest_dir = snapshot_root / src_dir.name
        try:
            shutil.copytree(
                src_dir,
                dest_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        except (OSError, shutil.Error) as exc:
            logger.warning("Failed to snapshot %s: %s", src_dir, exc)

    primary_src_dir = (
        dirs_to_save[0] if dirs_to_save else Path(__file__).resolve().parent
    )
    git_commit = get_git_commit(primary_src_dir)
    return git_commit


def prepare_experiment_artifacts(
    config: Any, *, code_dirs: Iterable[Path] | None = None
) -> str:
    return archive_experiment_state(_resolve_output_dir(config), code_dirs=code_dirs)


def _safe_getattr(obj: Any, name: str) -> Any:
    return getattr(obj, name) if hasattr(obj, name) else None


def _resolve_output_dir(config: Any) -> Path:
    if isinstance(config, dict):
        output_dir = config.get("output_dir")
        experiment = (
            config.get("experiment", {})
            if isinstance(config.get("experiment"), dict)
            else None
        )
        if output_dir is None and experiment is not None:
            output_dir = experiment.get("output_dir")
    else:
        experiment = _safe_getattr(config, "experiment")
        output_dir = _safe_getattr(config, "output_dir")
        if output_dir is None and experiment is not None:
            output_dir = _safe_getattr(experiment, "output_dir")

    if output_dir is None:
        raise ValueError("Experiment config must define an output directory")
    return Path(output_dir)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "variant"


def create_experiment_dir(
    experiment_name: str,
    base_dir: Path = Path("./experiments"),
) -> Path:
    """
    Create a new experiment directory with auto-incremented name.
    Example:
        ./experiments/experiment_name_001/2026-01-21_14-32-10
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(experiment_name)}_(\d+)$")
    existing_indices = []

    for d in base_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    experiment_name_inc = f"{experiment_name}_{next_index:03d}"

    output_dir = base_dir / experiment_name_inc

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_experiment_with_variants(
    run_fn: Callable[[Any], Any],
    *,
    base_config: Any,
    variant_builders: Iterable[Callable[[Any], Any]] | None = None,
    code_snapshot_dirs: Iterable[Path] | None = None,
    experiment_label_prefix: str | None = None,
    report_config: ReportConfig | None = None,
) -> list[Any]:
    """
    Generic helper to:
      - expand a base config with variants
      - run each config
      - log markdown results
      - write a per-experiment summary

    Returns the list of raw results (one per config).
    """
    results: list[Any] = []
    summary_rows: list[VariantSummary] = []
    active_report_config = report_config or get_default_report_config()
    summary_title = _safe_getattr(
        _safe_getattr(base_config, "experiment"), "experiment_name"
    )
    summary_title = summary_title or "experiment"
    summary_project = resolve_project_name(base_config, active_report_config)

    experiment_root = _resolve_output_dir(base_config)

    git_sha = prepare_experiment_artifacts(base_config, code_dirs=code_snapshot_dirs)

    for idx, config in enumerate(expand_with_variants(base_config, variant_builders)):
        output_dir = _resolve_output_dir(config)

        experiment = _safe_getattr(config, "experiment")
        experiment_name = _safe_getattr(experiment, "experiment_name") or "experiment"
        logger.info("Running experiment: %s", experiment_name)
        result: Any
        error: Exception | None = None
        try:
            result = run_fn(config)
        except Exception as exc:
            error = exc
            result = {"status": "failed", "error": str(exc)}
            logger.exception("Experiment failed: %s", exc)
        results.append(result)

        config_path = write_config_report(
            output_dir,
            config,
            filename=active_report_config.config_filename,
        )
        experiment_name = _safe_getattr(experiment, "experiment_name") or "variant"
        base_label = "Regular" if idx == 0 else experiment_name
        label = (
            f"{experiment_label_prefix}: {base_label}"
            if experiment_label_prefix
            else base_label
        )
        summary_rows.append(
            VariantSummary(
                label=label,
                config_path=config_path,
                results_path=output_dir / "results.md",
                output_dir=output_dir,
                variant_slug=_slugify(base_label),
                result=result,
                error=error,
            )
        )

    # Use the top-level experiment directory (the timestamped directory for base_config)
    write_experiment_report(
        experiment_root,
        title=summary_title,
        project_name=summary_project,
        report_config=active_report_config,
        variants=summary_rows,
        base_config=base_config,
        git_sha=git_sha,
    )
    return results


def aggregate_seeds_run_experiment_with_variants(
    run_fn: Callable[[Any], Any],
    *,
    base_config: Any,
    num_seeds: int = 10,
    seed_generator_seed: int | None = None,
    variant_builders: Iterable[Callable[[Any], Any]] | None = None,
    code_snapshot_dirs: Iterable[Path] | None = None,
    experiment_label_prefix: str | None = None,
    report_config: ReportConfig | None = None,
):
    """
    Run an ensemble of experiments by varying the random seed and logging only aggregated results.

    This takes a base `dataclass`, samples `num_seeds` random seeds and runs an
    experiment for each of them (and optionally their variants). Results are
    aggregated across seeds per variant; no per-seed output directories are created,
    and logging happens once per variant to the base `output_dir`.

    Parameters
    ----------
    base_config:
        The base configuration from which per-seed runs will be derived.
    num_seeds:
        Number of different seeds / ensemble members.
    seed_generator_seed:
        Optional seed for the seed generator itself (for reproducible ensembles).
    variant_builders:
        Optional iterable of variant builders (see `make_variant_builder`) to
        create additional configurations per seed.
    """

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _add_metrics(accum: Any, value: Any) -> Any:
        if isinstance(value, dict):
            accum = {} if accum is None else accum
            merged: dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(k, str) and k.startswith("__row_label"):
                    merged[k] = accum.get(k, v)
                else:
                    merged[k] = _add_metrics(accum.get(k), v)
            return merged
        if isinstance(value, list):
            accum = [None] * len(value) if accum is None else accum
            if len(accum) != len(value):
                raise ValueError("Metric list lengths differ across seeds.")
            return [_add_metrics(a, b) for a, b in zip(accum, value, strict=False)]
        if value is None:
            return accum
        if _is_number(value):
            return (0.0 if accum is None else float(accum)) + float(value)
        return accum if accum is not None else value

    def _average_metrics(total: Any, divisor: int) -> Any:
        if isinstance(total, dict):
            averaged: dict[str, Any] = {}
            for k, v in total.items():
                if isinstance(k, str) and k.startswith("__row_label"):
                    averaged[k] = v
                else:
                    averaged[k] = _average_metrics(v, divisor)
            return averaged
        if isinstance(total, list):
            return [_average_metrics(v, divisor) for v in total]
        if total is None:
            return None
        if _is_number(total):
            return float(total) / divisor
        return total

    aggregated_results: dict[str, Any] = {}
    summary_rows: list[VariantSummary] = []
    active_report_config = report_config or get_default_report_config()
    summary_title = _safe_getattr(
        _safe_getattr(base_config, "experiment"), "experiment_name"
    )
    summary_title = summary_title or "experiment"
    summary_project = resolve_project_name(base_config, active_report_config)

    experiment_root = _resolve_output_dir(base_config)

    rng = np.random.default_rng(seed_generator_seed)
    seeds = rng.integers(low=1, high=10000, size=num_seeds)

    expanded_configs = expand_with_variants(base_config, variant_builders)

    git_sha = prepare_experiment_artifacts(base_config, code_dirs=code_snapshot_dirs)

    for idx, config in enumerate(expanded_configs):
        totals: Any = None

        for seed in seeds:
            seeded_config = copy.deepcopy(config)
            if hasattr(seeded_config, "experiment") and hasattr(
                seeded_config.experiment, "seed"
            ):
                seeded_config.experiment.seed = int(seed)
            result = run_fn(seeded_config)
            metrics_dict = (
                {"metrics": result} if not isinstance(result, dict) else result
            )
            totals = _add_metrics(totals, metrics_dict)

        averaged = _average_metrics(totals, len(seeds))
        experiment = _safe_getattr(config, "experiment")
        experiment_name = _safe_getattr(experiment, "experiment_name") or "variant"
        aggregated_results[experiment_name] = averaged

        output_dir = _resolve_output_dir(config)
        config_path = write_config_report(
            output_dir,
            config,
            filename=active_report_config.config_filename,
        )
        base_label = "Regular" if idx == 0 else experiment_name
        label = (
            f"{experiment_label_prefix}: {base_label}"
            if experiment_label_prefix
            else base_label
        )
        summary_rows.append(
            VariantSummary(
                label=label,
                config_path=config_path,
                results_path=output_dir / "results.md",
                output_dir=output_dir,
                variant_slug=_slugify(base_label),
                result=averaged,
                error=None,
            )
        )

    write_experiment_report(
        experiment_root,
        title=summary_title,
        project_name=summary_project,
        report_config=active_report_config,
        variants=summary_rows,
        base_config=base_config,
        git_sha=git_sha,
    )
    return aggregated_results


def _load_experiments_module(experiments_module: Any | None = None):
    """
    Resolve the experiments module, falling back to the default import path.
    """
    if experiments_module is not None:
        return experiments_module

    module_path = os.environ.get("EXPERIMENTS_MODULE")
    if module_path:
        return importlib.import_module(module_path)

    msg = (
        "No experiments module provided. Set EXPERIMENTS_MODULE or pass "
        "--experiments-module to experiment_runner.main()."
    )
    raise RuntimeError(msg)


def _discover_experiments(experiments_module: Any) -> dict[str, Callable[..., Any]]:
    """
    Discover experiment functions from the project-specific experiments module.

    Any callable whose name starts with 'experiment_' is registered under the
    name with that prefix stripped, e.g.:

        experiment_baseline_cifar100 -> 'baseline_cifar100'
    """
    import inspect

    registry: dict[str, Callable[..., Any]] = {}

    for name, obj in inspect.getmembers(experiments_module, inspect.isfunction):
        if name.startswith("experiment_"):
            cli_name = name.removeprefix("experiment_")
            registry[cli_name] = obj

    # Also register generic / non-project specific ones defined in this module.
    # We expose 'ensemble' as a top-level experiment name.
    registry["ensemble"] = aggregate_seeds_run_experiment_with_variants

    return registry


def apply_dotted_overrides(config: Any, overrides: dict[str, Any] | None) -> Any:
    """Deep copy config and apply dotted-path overrides (e.g., 'training.lr')."""
    updated = copy.deepcopy(config)
    if not overrides:
        return updated

    for dotted_key, value in overrides.items():
        target = updated
        parts = dotted_key.split(".")
        for attr in parts[:-1]:
            if isinstance(target, dict):
                if attr not in target:
                    raise ValueError(
                        f"Cannot apply override '{dotted_key}': missing '{attr}'"
                    )
                target = target[attr]
            else:
                if not hasattr(target, attr):
                    raise ValueError(
                        f"Cannot apply override '{dotted_key}': missing '{attr}'"
                    )
                target = getattr(target, attr)
        leaf = parts[-1]
        if isinstance(target, dict):
            if leaf not in target:
                raise ValueError(
                    f"Cannot apply override '{dotted_key}': missing '{leaf}'"
                )
            target[leaf] = value
        else:
            if not hasattr(target, leaf):
                raise ValueError(
                    f"Cannot apply override '{dotted_key}': missing '{leaf}'"
                )
            setattr(target, leaf, value)

    return updated


def make_variant_builder(
    *,
    name_suffix: str | None = None,
    output_subdir: str | None = None,
    experiment_suffix: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """
    Factory that returns a variant builder which:
      - applies dotted overrides to any nested dataclass fields
      - optionally appends a suffix to the model name
      - optionally appends a subdirectory to the output_dir
      - optionally appends a suffix to the experiment name
    """

    def _builder(base: Any) -> Any:
        variant = apply_dotted_overrides(base, overrides)

        if (
            name_suffix
            and hasattr(variant, "model")
            and hasattr(variant.model, "model_name")
        ):
            variant.model.model_name = f"{variant.model.model_name}{name_suffix}"
        if (
            output_subdir
            and hasattr(variant, "experiment")
            and hasattr(variant.experiment, "output_dir")
        ):
            variant.experiment.output_dir = (
                Path(variant.experiment.output_dir) / output_subdir
            )
        if (
            experiment_suffix
            and hasattr(variant, "experiment")
            and hasattr(variant.experiment, "experiment_name")
        ):
            variant.experiment.experiment_name = (
                f"{variant.experiment.experiment_name}{experiment_suffix}"
            )

        return variant

    return _builder


def expand_with_variants(
    config: Any,
    variant_builders: Iterable[Callable[[Any], Any]] | None = None,
) -> list[Any]:
    """Return the base config plus any additional variants created by user-supplied builders."""
    base = copy.deepcopy(config)
    configs: list[Any] = [base]

    for build_variant in variant_builders or []:
        built = build_variant(copy.deepcopy(base))
        if is_dataclass(built):
            configs.append(built)
        elif isinstance(built, Iterable) and not isinstance(built, (str, bytes)):
            configs.extend(built)
        else:
            raise TypeError(
                "Variant builder must return a dataclass or iterable of dataclass,"
                f" got {type(built)}"
            )

    return configs


def list_experiments(experiments: dict[str, Callable[..., Any]]):
    """Print all available experiments."""
    print("\n" + "=" * 80)
    print("Available Experiments".center(80))
    print("=" * 80)

    for name, func in experiments.items():
        doc = func.__doc__ or "No description"
        print(f"\n{name}")
        print(f"   {doc.strip()}")

    print("\n" + "=" * 80)


def main(experiments_module: Any | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run predefined experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        default="list",
        help="Experiment to run (use 'list' to see all options, 'all' to run everything)",
    )
    parser.add_argument(
        "--experiments-module",
        type=str,
        default=None,
        help="Python module path containing experiment_* functions (e.g. 'my_pkg.experiments').",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project name used in experiment reports.",
    )
    parser.add_argument(
        "--report-config",
        type=str,
        default=None,
        help="Path to a JSON report configuration file.",
    )

    args = parser.parse_args()

    if args.report_config:
        set_default_report_config(load_report_config(Path(args.report_config)))
    if args.project_name:
        updated = get_default_report_config()
        updated.project_name = args.project_name
        set_default_report_config(updated)

    if args.experiments_module:
        experiments_module = importlib.import_module(args.experiments_module)

    experiments_module = _load_experiments_module(experiments_module)
    experiments = _discover_experiments(experiments_module)

    if args.experiment not in list(experiments.keys()) + ["all", "list"]:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    if args.experiment == "list":
        list_experiments(experiments)
        return

    if args.experiment == "all":
        print("Running ALL experiments".center(80))
        print("-" * 40)

        results = {}
        for name, func in experiments.items():
            print(f"\n{'=' * 80}")
            print(f"Experiment: {name}".center(80))
            print(f"{'=' * 80}")

            try:
                result = func()
                results[name] = result
                print(f"\n{name} completed successfully")
            except Exception as e:
                print(f"\n {name} failed with error: {e}")
                results[name] = None

        # Print summary
        print("\n" + "=" * 80)
        print("Experiment Summary".center(80))
        print("=" * 80)
        for name, result in results.items():
            status = "Success" if result is not None else "Failed"
            print(f"{name:30} {status}")
        print("=" * 80)

    else:
        # Run specific experiment
        if args.experiment in experiments:
            experiments[args.experiment]()
        else:
            print(f"Unknown experiment: {args.experiment}")
            list_experiments(experiments)


if __name__ == "__main__":
    main()
