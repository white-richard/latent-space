"""
Experiment runner with predefined configurations.
"""

import copy
import datetime
import re
import shutil
from pathlib import Path

import numpy as np

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


def experiment_baseline():
    """Baseline configuration"""
    use_cifar100 = False
    experiment_name = experiment_baseline.__name__.removeprefix("experiment_")
    dataset_name = "cifar100" if use_cifar100 else "cifar10"
    experiment_name = experiment_name + f"_{dataset_name}"

    output_dir = create_experiment_dir(experiment_name, BASE_DIR)
    base_config = Config(
        data=DataConfig(
            batch_size=256,
            num_workers=16,
            use_cifar100=use_cifar100,
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
            start_cooldown_immediately=False,  # Use on a ckpt when you want to start cooldown
            auto_trigger_cooldown=True,
        ),
        experiment=ExperimentConfig(
            experiment_name=experiment_name,
            seed=42,
            debug_mode=False,
            output_dir=output_dir,
            run_mhc_variant=True,
        ),
    )
    with_mhc = base_config.experiment.run_mhc_variant

    print("\nRunning Baseline Experimentwith MHC" if with_mhc else "")
    # return train(config)
    results = []
    for config in expand_w_mhc(base_config):
        prepare_experiment_artifacts(config)
        print(f"\nRunning Baseline Experiment ({config.model.model_name})")
        results.append(train(config))

    return results


def experiment_ensemble_seeds(
    base_config: Config, num_seeds: int = 5, seed_generator_seed: int | None = None
):
    results = []
    rng = np.random.default_rng(seed_generator_seed)
    seeds = rng.integers(low=1, high=10000, size=num_seeds)

    for seed in seeds:
        per_seed_base = copy.deepcopy(base_config)
        per_seed_base.experiment.seed = int(seed)

        per_seed_base.experiment.output_dir = base_config.experiment.output_dir / f"seed_{seed}"

        for config in expand_w_mhc(per_seed_base):
            prepare_experiment_artifacts(config)
            print(f"\nRunning Ensemble Experiment (Seed {seed}) ({config.model.model_name})")
            result = train(config)
            results.append(
                {"seed": int(seed), "model_name": config.model.model_name, "result": result}
            )

    return results


# Experiment registry
EXPERIMENTS = {
    "baseline": experiment_baseline,
    "ensemble": experiment_ensemble_seeds,
}


def expand_w_mhc(config: Config) -> list[Config]:
    """Return configs including optional mhc variant based on ExperimentConfig."""
    configs: list[Config] = [copy.deepcopy(config)]

    if config.experiment.run_mhc_variant:
        mhc = copy.deepcopy(config)

        # avoid double-append
        if not mhc.model.model_name.endswith("_mhc"):
            mhc.model.model_name = f"{mhc.model.model_name}_mhc"

        mhc.experiment.output_dir = mhc.experiment.output_dir / "mhc"
        mhc.experiment.is_mhc = True
        configs.append(mhc)

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
                print(f"\n❌ {name} failed with error: {e}")
                results[name] = None

        # Print summary
        print("\n" + "=" * 80)
        print("Experiment Summary".center(80))
        print("=" * 80)
        for name, result in results.items():
            status = "Success" if result is not None else "❌ Failed"
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
