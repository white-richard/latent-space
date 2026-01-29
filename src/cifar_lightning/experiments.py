from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from latent_space.experiment.experiment_runner import (
    BASE_DIR,
    create_experiment_dir,
    make_variant_builder,
    run_experiment_with_variants,
)

from .config import Config, DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .train import train


def experiment_baseline_cifar100():
    experiment_name = experiment_baseline_cifar100.__name__.removeprefix("experiment_")

    output_dir = create_experiment_dir(experiment_name, base_dir=BASE_DIR)
    base_config = Config(
        data=DataConfig(
            batch_size=256,
            num_workers=16,
            use_cifar100=True,
        ),
        model=ModelConfig(
            model_name="vit_small",
        ),
        training=TrainingConfig(
            epochs=400,
            lr=0.001,
            weight_decay=0.05,
            clip_norm=None,
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
    variant_builders: Iterable[Callable[[Config], Config]] | None = None

    if variant_builders is None:
        variant_builders = [
            make_variant_builder(
                # name_suffix="_mHC",
                output_subdir="mHC",
                experiment_suffix="_mHC",
                overrides={
                    "experiment.run_mhc_variant": True,
                },
            )
        ]

    return run_experiment_with_variants(
        train,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Baseline CIFAR100",
    )


def experiment_baseline_cifar10():
    experiment_name = experiment_baseline_cifar10.__name__.removeprefix("experiment_")

    output_dir = create_experiment_dir(experiment_name, base_dir=BASE_DIR)
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
    variant_builders: Iterable[Callable[[Config], Config]] | None = None

    return run_experiment_with_variants(
        train,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Baseline CIFAR10",
    )


if __name__ == "__main__":
    import latent_space.experiment.experiment_runner as experiment_runner

    experiment_runner.main(experiments_module=sys.modules[__name__])
