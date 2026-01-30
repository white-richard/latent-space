"""
Experiment runner with predefined dataclass configurations.

See `./experiements.py` for how to use this module to define and run experiments.
"""

import copy
import datetime
import importlib
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_space.utils.markdown_results import (
    MarkdownTableLogger,
    merge_row,
)

from .utils import get_git_commit

BASE_DIR = Path("./experiments")
CODE_SNAPSHOT_DIRNAME = "code_snapshot"
GIT_COMMIT_FILENAME = "git_commit.txt"

# Replace and pass from experiments.py
DEFAULT_CODE_SNAPSHOT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "../data_utils",
    Path(__file__).resolve().parent / "../models",
    Path(__file__).resolve().parent / "../utils",
    Path(__file__).resolve().parent / "../schedulers",
]


def archive_experiment_state(output_dir: Path, *, code_dirs: Iterable[Path] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = output_dir / CODE_SNAPSHOT_DIRNAME
    snapshot_root.mkdir(parents=True, exist_ok=True)

    dirs_to_save = [Path(p).resolve() for p in (code_dirs or DEFAULT_CODE_SNAPSHOT_DIRS)]

    for src_dir in dirs_to_save:
        if not src_dir.exists():
            continue
        dest_dir = snapshot_root / src_dir.name
        shutil.copytree(
            src_dir,
            dest_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    primary_src_dir = dirs_to_save[0] if dirs_to_save else Path(__file__).resolve().parent
    git_commit = get_git_commit(primary_src_dir)
    (output_dir / GIT_COMMIT_FILENAME).write_text(f"{git_commit}\n", encoding="utf-8")


def prepare_experiment_artifacts(
    config: dataclass, *, code_dirs: Iterable[Path] | None = None
) -> None:
    archive_experiment_state(Path(config.experiment.output_dir), code_dirs=code_dirs)


def render_summary_section(title: str, results_md_path: Path) -> list[str]:
    if results_md_path.exists():
        table = results_md_path.read_text(encoding="utf-8").strip()
    else:
        table = "_No results logged yet._"
    return [f"### {title}", table]


def write_experiment_summary(experiment_root: Path, records: list[tuple[str, Path]]) -> None:
    if not records:
        return

    link_target = (
        experiment_root.parent if experiment_root.parent != experiment_root else experiment_root
    )
    link_path = f"./{link_target.as_posix().lstrip('./')}"
    timestamp = experiment_root.name
    lines: list[str] = [
        "# --- Experiment Path ---",
        f"[Experiemnt path]({link_path})",
        "",
        "## Experiemnt summary",
        f"**Datetime** {timestamp}",
        "",
    ]

    base_label, base_results_path = records[0]
    base_section_title = "Regular" if base_label.lower() == "regular" else f"Varient {base_label}"
    lines.extend(render_summary_section(base_section_title, base_results_path))

    for label, results_path in records[1:]:
        lines.append("")
        lines.extend(render_summary_section(f"Varient {label}", results_path))

    lines.append("")

    summary_path = Path(experiment_root) / "experiment_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_dir / experiment_name_inc / timestamp

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _extract_row_label_payload(result: Any) -> tuple[Sequence[Any], str, dict[str, Any]] | None:
    if not isinstance(result, dict):
        return None
    row_labels = result.get("__row_labels")
    if row_labels is None:
        return None
    row_label_header = result.get("__row_label_header", "name")
    columns = {
        k: v for k, v in result.items() if not (isinstance(k, str) and k.startswith("__row_label"))
    }
    return row_labels, row_label_header, columns


def run_experiment_with_variants(
    run_fn: Callable[[Any], Any],
    *,
    base_config: Any,
    variant_builders: Iterable[Callable[[Any], Any]] | None = None,
    code_snapshot_dirs: Iterable[Path] | None = None,
    experiment_label_prefix: str | None = None,
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
    summary_rows: list[tuple[str, Path]] = []

    for idx, config in enumerate(expand_with_variants(base_config, variant_builders)):
        prepare_experiment_artifacts(config, code_dirs=code_snapshot_dirs)
        results_path = Path(config.experiment.output_dir) / "results.md"
        logger = MarkdownTableLogger(results_path)

        print(f"\nExperiment ({config.experiment.experiment_name})")
        result = run_fn(config)
        results.append(result)

        row_label_payload = _extract_row_label_payload(result)
        if row_label_payload:
            row_labels, row_label_header, columns = row_label_payload
            logger.append_columns(
                columns,
                row_labels=row_labels,
                row_label_header=row_label_header,
            )
        else:
            row = merge_row(
                {"metrics": result} if not isinstance(result, dict) else result,
            )
            logger.append(row)

        base_label = "Regular" if idx == 0 else config.experiment.experiment_name
        label = (
            f"{experiment_label_prefix}: {base_label}" if experiment_label_prefix else base_label
        )
        summary_rows.append((label, results_path))

    # Use the top-level experiment directory (the timestamped directory for base_config)
    experiment_root = Path(base_config.experiment.output_dir)
    write_experiment_summary(experiment_root, summary_rows)
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
            return 0.0 if accum is None else accum
        return (0.0 if accum is None else float(accum)) + float(value)

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
        return float(total) / divisor

    aggregated_results: dict[str, Any] = {}
    summary_rows: list[tuple[str, Path]] = []

    rng = np.random.default_rng(seed_generator_seed)
    seeds = rng.integers(low=1, high=10000, size=num_seeds)

    expanded_configs = expand_with_variants(base_config, variant_builders)

    for idx, config in enumerate(expanded_configs):
        prepare_experiment_artifacts(config, code_dirs=code_snapshot_dirs)
        totals: Any = None

        for seed in seeds:
            seeded_config = copy.deepcopy(config)
            if hasattr(seeded_config, "experiment") and hasattr(seeded_config.experiment, "seed"):
                seeded_config.experiment.seed = int(seed)
            result = run_fn(seeded_config)
            metrics_dict = {"metrics": result} if not isinstance(result, dict) else result
            totals = _add_metrics(totals, metrics_dict)

        averaged = _average_metrics(totals, len(seeds))
        aggregated_results[config.experiment.experiment_name] = averaged

        results_path = Path(config.experiment.output_dir) / "results.md"
        logger = MarkdownTableLogger(results_path)
        row_label_payload = _extract_row_label_payload(averaged)
        if row_label_payload:
            row_labels, row_label_header, columns = row_label_payload
            logger.append_columns(
                columns,
                row_labels=row_labels,
                row_label_header=row_label_header,
            )
        else:
            row = merge_row(
                {"metrics": averaged} if not isinstance(averaged, dict) else averaged,
            )
            logger.append(row)

        base_label = "Regular" if idx == 0 else config.experiment.experiment_name
        label = (
            f"{experiment_label_prefix}: {base_label}" if experiment_label_prefix else base_label
        )
        summary_rows.append((label, results_path))

    experiment_root = Path(base_config.experiment.output_dir)
    write_experiment_summary(experiment_root, summary_rows)
    return aggregated_results


def _load_experiments_module(experiments_module: Any | None = None):
    """
    Resolve the experiments module, falling back to the default import path.
    """
    if experiments_module is not None:
        return experiments_module

    try:
        return importlib.import_module("src.cifar_lightning.experiments")
    except ModuleNotFoundError as e:
        msg = (
            "No experiments module provided to experiment_runner.main() and "
            "default import 'src.cifar_lightning.experiments' failed."
        )
        raise RuntimeError(msg) from e


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

    # Optional: keep a short alias for a common baseline, if present.
    if "baseline_cifar10" in registry:
        registry["baseline"] = registry["baseline_cifar10"]

    return registry


def apply_dotted_overrides(config: dataclass, overrides: dict[str, Any] | None) -> dataclass:
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
                    raise ValueError(f"Cannot apply override '{dotted_key}': missing '{attr}'")
                target = target[attr]
            else:
                if not hasattr(target, attr):
                    raise ValueError(f"Cannot apply override '{dotted_key}': missing '{attr}'")
                target = getattr(target, attr)
        leaf = parts[-1]
        if isinstance(target, dict):
            if leaf not in target:
                raise ValueError(f"Cannot apply override '{dotted_key}': missing '{leaf}'")
            target[leaf] = value
        else:
            if not hasattr(target, leaf):
                raise ValueError(f"Cannot apply override '{dotted_key}': missing '{leaf}'")
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

        if name_suffix and hasattr(variant, "model") and hasattr(variant.model, "model_name"):
            variant.model.model_name = f"{variant.model.model_name}{name_suffix}"
        if (
            output_subdir
            and hasattr(variant, "experiment")
            and hasattr(variant.experiment, "output_dir")
        ):
            variant.experiment.output_dir = Path(variant.experiment.output_dir) / output_subdir
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

    experiments_module = _load_experiments_module(experiments_module)
    experiments = _discover_experiments(experiments_module)

    parser = argparse.ArgumentParser(
        description="Run predefined experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        choices=list(experiments.keys()) + ["all", "list"],
        default="list",
        help="Experiment to run (use 'list' to see all options, 'all' to run everything)",
    )

    args = parser.parse_args()

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
