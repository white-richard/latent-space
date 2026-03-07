"""
Argparse adapter example.

Demonstrates a custom ConfigAdapter that converts a dataclass config
to an argparse.Namespace before passing it to train_fn.

Run from the repo root:
    python -m latent_space.ml_experiment_runner.examples.argparse_repo.example
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from pathlib import Path

from latent_space.ml_experiment_runner import (
    Experiment,
    ExperimentSuite,
    RunnerConfig,
)


@dataclasses.dataclass
class RawConfig:
    seed: int = 0
    lr: float = 1e-3
    epochs: int = 10


@dataclasses.dataclass
class Metrics:
    final_loss: float
    best_accuracy: float


class ArgparseAdapter:
    """Converts RawConfig to argparse.Namespace for train_fn."""

    def inject_seed(self, config: RawConfig, seed: int) -> RawConfig:
        import copy

        seeded = copy.deepcopy(config)
        seeded.seed = seed
        return seeded

    def to_native(self, config: RawConfig) -> argparse.Namespace:
        return argparse.Namespace(
            seed=config.seed,
            lr=config.lr,
            epochs=config.epochs,
        )


def fake_train(args: argparse.Namespace) -> Metrics:
    rng = random.Random(args.seed)
    loss = 1.0 / (args.lr * 1000) * (1 + rng.gauss(0, 0.05))
    acc = min(0.99, 0.7 + args.lr * 100 + rng.gauss(0, 0.03))
    return Metrics(final_loss=loss, best_accuracy=acc)


def main():
    adapter = ArgparseAdapter()

    experiments = [
        Experiment(
            name="lr_1e-3",
            config=RawConfig(lr=1e-3),
            train_fn=fake_train,
            seeds=[10, 20, 30],
            adapter=adapter,
        ),
        Experiment(
            name="lr_1e-2",
            config=RawConfig(lr=1e-2),
            train_fn=fake_train,
            seeds=[10, 20, 30],
            adapter=adapter,
        ),
    ]

    output_dir = Path("tmp")

    cfg = RunnerConfig(
        suite_name="argparse_lr_sweep",
        export_formats=["markdown", "latex", "excel"],
        output_dir=output_dir,
    )
    suite = ExperimentSuite(experiments, runner_config=cfg)
    result = suite.run()

    for name, agg in result.results.items():
        acc = agg["best_accuracy"]
        print(f"{name}: best_accuracy = {acc.mean:.4f} ± {acc.std:.4f}")


if __name__ == "__main__":
    main()
