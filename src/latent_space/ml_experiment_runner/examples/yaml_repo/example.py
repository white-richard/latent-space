"""
YAML / dict adapter example.

Demonstrates a custom ConfigAdapter that works with plain dict configs
(as you'd load from a YAML file).

Run from the repo root:
    python -m latent_space.ml_experiment_runner.examples.yaml_repo.example
"""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path

from latent_space.ml_experiment_runner import (
    Experiment,
    ExperimentSuite,
    RunnerConfig,
)


@dataclasses.dataclass
class Metrics:
    accuracy: float
    loss: float


class YamlDictAdapter:
    """Adapter for plain dict configs (e.g. loaded from YAML)."""

    def __init__(self, seed_key: str = "seed") -> None:
        self.seed_key = seed_key

    def inject_seed(self, config: dict, seed: int) -> dict:
        seeded = copy.deepcopy(config)
        # Support dotted path like "training.seed"
        parts = self.seed_key.split(".")
        node = seeded
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = seed
        return seeded

    def to_native(self, config: dict) -> dict:
        return config


def fake_train(config: dict) -> Metrics:
    rng = random.Random(config["seed"])
    lr = config.get("lr", 1e-3)
    acc = min(0.99, 0.75 + lr * 50 + rng.gauss(0, 0.02))
    loss = 1 - acc + abs(rng.gauss(0, 0.01))
    return Metrics(accuracy=acc, loss=loss)


def main():
    # Simulate configs loaded from YAML files
    config_a = {"seed": 0, "lr": 1e-3, "batch_size": 32}
    config_b = {"seed": 0, "lr": 5e-3, "batch_size": 64}

    adapter = YamlDictAdapter(seed_key="seed")

    experiments = [
        Experiment(
            name="small_lr",
            config=config_a,
            train_fn=fake_train,
            seeds=[1, 2, 3],
            adapter=adapter,
        ),
        Experiment(
            name="larger_lr",
            config=config_b,
            train_fn=fake_train,
            seeds=[1, 2, 3],
            adapter=adapter,
        ),
    ]

    output_dir = Path("tmp")

    cfg = RunnerConfig(
        suite_name="yaml_lr_comparison",
        export_formats=["markdown", "latex", "excel"],
        output_dir=output_dir,
    )
    suite = ExperimentSuite(experiments, runner_config=cfg)
    result = suite.run()

    for name, agg in result.results.items():
        acc = agg["accuracy"]
        print(f"{name}: accuracy = {acc.mean:.4f} ± {acc.std:.4f}")


if __name__ == "__main__":
    main()
