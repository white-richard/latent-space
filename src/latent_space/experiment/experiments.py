"""
Generic experiment definitions template.

This file shows how to define experiment entrypoints that can be auto-discovered
by `experiment_runner` using the `experiment_{name}` naming convention.

How to use:
- Copy/paste one of the sample functions below.
- Replace the placeholder values with your dataset/model/training specifics.
- Keep the function name prefixed with `experiment_` so it is auto-registered.
- Each function should:
  1) Derive an `experiment_name` from its own function name.
  2) Create an `output_dir` via `create_experiment_dir`.
  3) Build a `Config` (data, model, training, experiment sections).
  4) Optionally define `variant_builders` via `make_variant_builder`.
  5) Call `run_experiment_with_variants(...)` and return the results.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from latent_space.experiment.experiment_runner import (
    BASE_DIR,
    DEFAULT_CODE_SNAPSHOT_DIRS,
    create_experiment_dir,
    make_variant_builder,
    run_experiment_with_variants,
)
from latent_space.experiment.reporting import ReportConfig


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    # Add other data-related configurations as needed


@dataclass
class ModelConfig:
    model_name: str = "default_model"
    # Add other model-related configurations as needed


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    epochs: int = 10
    # Add optimizer/scheduler settings as needed


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    seed: int = 42
    project_name: str = "Example Project"
    hardware: str = "Unknown"


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig


def _example_runner(config: Config) -> dict[str, float]:
    """Replace with your real training/eval function."""
    _ = config
    return {
        "status": "training-complete",
        "metrics": {
            "final/acc": 0.82,
            "final/loss": 1.23,
        },
        "plots": ["plot_loss_curve.png", "plot_accuracy.png"],
    }


def experiment_example_task():
    """
    Example experiment entrypoint.

    Steps to adapt:
    - Replace the Config(...) sections with your dataset/model/training settings.
    - Adjust defaults (seed, debug_mode, epochs, lr, etc.) to your needs.
    - Add or remove variant builders to explore hyperparameters or naming tweaks.
    """
    # Derive a clean experiment name from this function's name
    experiment_name = experiment_example_task.__name__.removeprefix("experiment_")
    output_dir = create_experiment_dir(experiment_name, base_dir=BASE_DIR)

    # TODO: Replace the placeholders below with your real configuration
    base_config = Config(
        data=DataConfig(
            batch_size=64,
            num_workers=8,
        ),
        model=ModelConfig(
            model_name="your_model",
        ),
        training=TrainingConfig(
            lr=3e-4,
            epochs=20,
        ),
        experiment=ExperimentConfig(
            experiment_name=experiment_name,
            output_dir=str(output_dir),
            project_name="Example Project",
            hardware="RTX 4090",
        ),
    )

    variant_builders: Iterable[Callable[[Config], Config]] | None = None

    # Optional: define variant builders to create multiple configs from the base
    if variant_builders is None:
        variant_builders = [
            make_variant_builder(
                name_suffix="_v1",  # uncomment to suffix model_name
                output_subdir="variant1",  # uncomment to write to subdir
                experiment_suffix="_v1",  # uncomment to suffix experiment name
                overrides={
                    "training.lr": 0.0005,  # example override path -> value
                },
            ),
        ]

    report_config = ReportConfig(
        project_name=base_config.experiment.project_name,
        hypothesis="Adding gradient checkpointing reduces VRAM without hurting accuracy.",
        parameters={"Hardware": base_config.experiment.hardware},
    )

    return run_experiment_with_variants(
        _example_runner,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Example Task",
        code_snapshot_dirs=DEFAULT_CODE_SNAPSHOT_DIRS,
        report_config=report_config,
    )


# ---------------------------------------------------------------------------
# Add more experiments below by following the same pattern:
#
# def experiment_my_new_dataset(variant_builders=None):
#     experiment_name = experiment_my_new_dataset.__name__.removeprefix("experiment_")
#     output_dir = create_experiment_dir(experiment_name, base_dir=BASE_DIR)
#     base_config = Config(...)
#     if variant_builders is None:
#         variant_builders = [make_variant_builder(...), ...]
#     return run_experiment_with_variants(
#         base_config=base_config,
#         variant_builders=variant_builders,
#         experiment_label_prefix="My New Dataset",
#     )
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import latent_space.experiment.experiment_runner as experiment_runner

    experiment_runner.main(experiments_module=sys.modules[__name__])
