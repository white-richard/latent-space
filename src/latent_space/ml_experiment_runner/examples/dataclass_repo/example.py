"""
Dataclass adapter example.

Demonstrates using DataclassAdapter with a nested dataclass config.
Run from the repo root:
    python -m latent_space.ml_experiment_runner.examples.dataclass_repo.example
"""

from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path

from latent_space.ml_experiment_runner import (
    DataclassAdapter,
    Experiment,
    ExperimentSuite,
    RunnerConfig,
)


@dataclasses.dataclass
class ExperimentSection:
    seed: int = 0
    name: str = "baseline"


@dataclasses.dataclass
class ModelSection:
    hidden_dim: int = 64
    lr: float = 1e-3


@dataclasses.dataclass
class TrainConfig:
    experiment: ExperimentSection = dataclasses.field(default_factory=ExperimentSection)
    model: ModelSection = dataclasses.field(default_factory=ModelSection)


@dataclasses.dataclass
class TrainMetrics:
    val_accuracy: float
    train_loss: float


def fake_train(config: TrainConfig) -> TrainMetrics:
    """Simulated training: accuracy improves with seed parity, loss is random-ish."""
    rng = random.Random(config.experiment.seed)
    noise = rng.gauss(0, 0.02)
    accuracy = 0.85 + noise
    loss = math.exp(-accuracy * 2) + abs(noise) * 0.1
    return TrainMetrics(val_accuracy=accuracy, train_loss=loss)


def main():
    base_config = TrainConfig()
    adapter = DataclassAdapter(seed_field="experiment.seed")

    experiments = [
        Experiment(
            name="baseline",
            config=base_config,
            train_fn=fake_train,
            seeds=[1, 2, 3, 4, 5],
            adapter=adapter,
        ),
        Experiment(
            name="wider_model",
            config=TrainConfig(model=ModelSection(hidden_dim=128, lr=5e-4)),
            train_fn=fake_train,
            seeds=[1, 2, 3, 4, 5],
            adapter=adapter,
        ),
    ]

    output_dir = Path("tmp")

    cfg = RunnerConfig(
        suite_name="dataclass_example",
        export_formats=["markdown", "latex", "excel"],
        metric_directions={
            "val_accuracy": "higher_is_better",
            "train_loss": "lower_is_better",
        },
        output_dir=output_dir,
    )

    suite = ExperimentSuite(experiments, runner_config=cfg)
    result = suite.run()

    print(f"Completed suite with experiments: {result.experiments}")
    for exp_name, agg in result.results.items():
        acc = agg["val_accuracy"]
        print(f"  {exp_name}: val_accuracy = {acc.mean:.4f} ± {acc.std:.4f}")


if __name__ == "__main__":
    main()
