"""
Experiment runner with predefined configurations.
"""

import copy

import numpy as np

from .config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .train import train


def experiment_baseline():
    """Baseline configuration - standard settings."""
    suffix = experiment_baseline.__name__.removeprefix("experiment_")
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
            epochs=100,
            lr=0.001,
            weight_decay=0.01,
            clip_norm=0.0,
            use_bfloat16=True,
            scheduler_name="warmup_hold_decay",
            start_cooldown_immediately=False,  # Use on a ckpt when you want to start cooldown
            auto_trigger_cooldown=True
        ),
        experiment=ExperimentConfig(
            seed=42,
            debug_mode=True,
            checkpoint_dir="./experiments/baseline/checkpoints",
            output_dir=f"./experiments/{suffix}",
            run_mhc_variant=True,
        ),
    )
    with_mhc = base_config.experiment.run_mhc_variant

    print("\nRunning Baseline Experimentwith MHC" if with_mhc else "")
    # return train(config)
    results = []
    for config in expand_w_mhc(base_config):
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

        per_seed_base.experiment.output_dir = f"{base_config.experiment.output_dir}/seed_{seed}"
        per_seed_base.experiment.checkpoint_dir = (
            f"{base_config.experiment.checkpoint_dir}/seed_{seed}"
        )

        for config in expand_w_mhc(per_seed_base):
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
    """Return configs including optional _mhc variant based on ExperimentConfig."""
    configs: list[Config] = [copy.deepcopy(config)]

    if config.experiment.run_mhc_variant:
        mhc = copy.deepcopy(config)

        # avoid double-append if someone already set it
        if not mhc.model.model_name.endswith("_mhc"):
            mhc.model.model_name = f"{mhc.model.model_name}_mhc"

        mhc.experiment.output_dir += "_mhc"
        mhc.experiment.checkpoint_dir += "_mhc"
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
