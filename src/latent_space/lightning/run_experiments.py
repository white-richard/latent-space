"""
Experiment runner with predefined configurations.
"""

import copy
import datetime
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from latent_space.utils.markdown_results import (
    MarkdownTableLogger,
    merge_row,
)

from .config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .train import train
from .utils import get_git_commit

BASE_DIR = Path("./experiments")
LIGHTNING_SRC_DIR = Path(__file__).resolve().parent
CODE_SNAPSHOT_DIRNAME = "code_snapshot"
GIT_COMMIT_FILENAME = "git_commit.txt"

OTHER_SAVE_DIRS = [
    Path("../data_utils"),
    Path("../models"),
    Path("../utils"),
    Path("../schedulers"),
]


def archive_experiment_state(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = output_dir / CODE_SNAPSHOT_DIRNAME
    snapshot_root.mkdir(parents=True, exist_ok=True)

    dirs_to_save = [LIGHTNING_SRC_DIR] + [
        (LIGHTNING_SRC_DIR / rel_path).resolve() for rel_path in OTHER_SAVE_DIRS
    ]

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

    git_commit = get_git_commit(LIGHTNING_SRC_DIR)
    (output_dir / GIT_COMMIT_FILENAME).write_text(f"{git_commit}\n", encoding="utf-8")


def prepare_experiment_artifacts(config: Config) -> None:
    archive_experiment_state(Path(config.experiment.output_dir))


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


def experiment_baseline_cifar100(
    variant_builders: Iterable[Callable[[Config], Config]] | None = None,
):
    experiment_name = experiment_baseline_cifar100.__name__.removeprefix("experiment_")

    output_dir = create_experiment_dir(experiment_name, BASE_DIR)
    base_config = Config(
        data=DataConfig(
            batch_size=256,
            num_workers=16,
            use_cifar100=True,
        ),
        model=ModelConfig(
            model_name="vit_tiny",
        ),
        training=TrainingConfig(
            epochs=400,
            lr=0.003,
            weight_decay=0.05,
            clip_norm=5.0,
            use_bfloat16=True,
            scheduler_name="warmup_hold_decay",
            # scheduler_name="cosine",
            start_cooldown_immediately=False,  # Use on a ckpt when you want to start cooldown
            auto_trigger_cooldown=True,
        ),
        experiment=ExperimentConfig(
            experiment_name=experiment_name,
            seed=42,
            debug_mode=False,
            output_dir=output_dir,
        ),
    )

    variant_builders = [
        make_variant_builder(
            # name_suffix="_mHC", # optional suffix to model name
            output_subdir="mHC",
            experiment_suffix="_mHC",
            overrides={
                "experiment.run_mhc_variant": True,
            },
        )
    ]

    results = []
    summary_rows: list[tuple[str, Path]] = []
    for idx, config in enumerate(expand_with_variants(base_config, variant_builders)):
        prepare_experiment_artifacts(config)
        results_path = Path(config.experiment.output_dir) / "results.md"
        logger = MarkdownTableLogger(results_path)
        print(f"\nRunning Baseline Experiment ({config.model.model_name})")
        result = train(config)
        results.append(result)

        # For markdown logging
        row = merge_row(
            {"metrics": result} if not isinstance(result, dict) else result,  # adapt
        )
        logger.append(row)
        label = "Regular" if idx == 0 else config.experiment.experiment_name
        summary_rows.append((label, results_path))

    write_experiment_summary(output_dir, summary_rows)
    return results


def experiment_baseline_cifar10(
    variant_builders: Iterable[Callable[[Config], Config]] | None = None,
):
    experiment_name = experiment_baseline_cifar10.__name__.removeprefix("experiment_")

    output_dir = create_experiment_dir(experiment_name, BASE_DIR)
    base_config = Config(
        data=DataConfig(
            batch_size=256,
            num_workers=16,
            use_cifar100=False,
        ),
        model=ModelConfig(
            model_name="vit_tiny",
        ),
        training=TrainingConfig(
            epochs=200,
            lr=0.003,
            weight_decay=0.01,
            clip_norm=0.0,
            use_bfloat16=True,
            scheduler_name="warmup_hold_decay",
            # scheduler_name="cosine",
            start_cooldown_immediately=False,  # Use on a ckpt when you want to start cooldown
            auto_trigger_cooldown=True,
        ),
        experiment=ExperimentConfig(
            experiment_name=experiment_name,
            seed=42,
            debug_mode=False,
            output_dir=output_dir,
        ),
    )

    results = []
    summary_rows: list[tuple[str, Path]] = []
    for idx, config in enumerate(expand_with_variants(base_config, variant_builders)):
        prepare_experiment_artifacts(config)
        results_path = Path(config.experiment.output_dir) / "results.md"
        logger = MarkdownTableLogger(results_path)
        print(f"\nRunning Baseline Experiment ({config.model.model_name})")
        result = train(config)
        results.append(result)

        # For markdown logging
        row = merge_row(
            {"metrics": result} if not isinstance(result, dict) else result,  # adapt
        )
        logger.append(row)
        label = "Regular" if idx == 0 else config.experiment.experiment_name
        summary_rows.append((label, results_path))

    write_experiment_summary(output_dir, summary_rows)
    return results


def experiment_ensemble_seeds(
    base_config: Config,
    num_seeds: int = 5,
    seed_generator_seed: int | None = None,
    variant_builders: Iterable[Callable[[Config], Config]] | None = None,
):
    results = []
    summary_rows: list[tuple[str, Path]] = []
    rng = np.random.default_rng(seed_generator_seed)
    seeds = rng.integers(low=1, high=10000, size=num_seeds)

    for seed in seeds:
        per_seed_base = copy.deepcopy(base_config)
        per_seed_base.experiment.seed = int(seed)

        per_seed_base.experiment.output_dir = base_config.experiment.output_dir / f"seed_{seed}"

        for config in expand_with_variants(per_seed_base, variant_builders):
            prepare_experiment_artifacts(config)
            results_path = Path(config.experiment.output_dir) / "results.md"
            logger = MarkdownTableLogger(results_path)
            print(f"\nRunning Ensemble Experiment (Seed {seed}) ({config.model.model_name})")
            result = train(config)
            result_row = merge_row(
                {"seed": int(seed), "model_name": config.model.model_name},
                {"metrics": result} if not isinstance(result, dict) else result,
            )
            logger.append(result_row)
            results.append(
                {"seed": int(seed), "model_name": config.model.model_name, "result": result}
            )
            summary_rows.append((f"seed_{seed}_{config.experiment.experiment_name}", results_path))

    write_experiment_summary(Path(base_config.experiment.output_dir), summary_rows)
    return results


# Experiment registry
EXPERIMENTS = {
    "baseline": experiment_baseline_cifar10,
    "baseline_cifar100": experiment_baseline_cifar100,
    "ensemble": experiment_ensemble_seeds,
}


def apply_dotted_overrides(config: Config, overrides: dict[str, Any] | None) -> Config:
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
) -> Callable[[Config], Config]:
    """
    Factory that returns a variant builder which:
      - applies dotted overrides to any nested dataclass fields
      - optionally appends a suffix to the model name
      - optionally appends a subdirectory to the output_dir
      - optionally appends a suffix to the experiment name
    """

    def _builder(base: Config) -> Config:
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
    config: Config,
    variant_builders: Iterable[Callable[[Config], Config | Iterable[Config]]] | None = None,
) -> list[Config]:
    """Return the base config plus any additional variants created by user-supplied builders."""
    base = copy.deepcopy(config)
    configs: list[Config] = [base]

    for build_variant in variant_builders or []:
        built = build_variant(copy.deepcopy(base))
        if isinstance(built, Config):
            configs.append(built)
        elif isinstance(built, Iterable) and not isinstance(built, (str, bytes)):
            configs.extend(built)
        else:
            raise TypeError(
                f"Variant builder must return a Config or iterable of Config, got {type(built)}"
            )

    return configs


def list_experiments():
    """Print all available experiments."""
    print("\n" + "=" * 80)
    print("Available Experiments".center(80))
    print("=" * 80)

    for name, func in EXPERIMENTS.items():
        doc = func.__doc__ or "No description"
        print(f"\n{name}")
        print(f"   {doc.strip()}")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run predefined experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        choices=list(EXPERIMENTS.keys()) + ["all", "list"],
        default="list",
        help="Experiment to run (use 'list' to see all options, 'all' to run everything)",
    )

    args = parser.parse_args()

    if args.experiment == "list":
        list_experiments()
        return

    if args.experiment == "all":
        print("Running ALL experiments".center(80))
        print("-" * 40)

        results = {}
        for name, func in EXPERIMENTS.items():
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
        if args.experiment in EXPERIMENTS:
            EXPERIMENTS[args.experiment]()
        else:
            print(f"Unknown experiment: {args.experiment}")
            list_experiments()


if __name__ == "__main__":
    main()
