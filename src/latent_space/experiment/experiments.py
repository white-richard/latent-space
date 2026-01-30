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


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    # Add other data-related configurations as needed


@dataclass
class ModelConfig:
    model_name: str = "default_model"
    # Add other model-related configurations as needed


# ... other dataclasses


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig
    model: ModelConfig


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
    _ = output_dir  # Use output_dir as needed in your runner

    # TODO: Replace the placeholders below with your real configuration
    base_config = Config(
        data=DataConfig(
            batch_size=64,
            num_workers=8,
        ),
        model=ModelConfig(
            model_name="your_model",
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

    # Temporaty run function to replace
    runner = None

    return run_experiment_with_variants(
        runner,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Example Task",
        code_snapshot_dirs=DEFAULT_CODE_SNAPSHOT_DIRS,
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
