from __future__ import annotations

import pathlib
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
from latent_space.experiment.reporting import ReportConfig

from .config import (
    Config,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    LossItemConfig,
    ModelConfig,
    TrainingConfig,
)
from .train import train

DEFAULT_CODE_SNAPSHOT_DIRS = [
    pathlib.Path(__file__).parent,
    pathlib.Path(__file__).parent.parent / "latent_space",
]


def experiment_metric_cifar100():
    experiment_name = experiment_metric_cifar100.__name__.removeprefix("experiment_")
    hypothesis = "Class clustering will become more apparent due to circle loss"

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
            epochs=320,
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
        loss=LossConfig(
            losses=[
                LossItemConfig(
                    name="circle",
                    weight=0.025,
                    start_epoch=50,
                ),
                LossItemConfig(
                    name="cross_entropy",
                    weight=1.0,
                ),
            ],
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
                    "model.use_mhc": True,
                },
            )
        ]

    return run_experiment_with_variants(
        train,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Baseline CIFAR100",
        code_snapshot_dirs=DEFAULT_CODE_SNAPSHOT_DIRS,
        report_config=ReportConfig(
            project_name="CIFAR Lightning",
            hypothesis=hypothesis,
            parameters={"Hardware": "RTX 3090"},
        ),
    )


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
            epochs=320,
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
                    "model.use_mhc": True,
                },
            )
        ]

    return run_experiment_with_variants(
        train,
        base_config=base_config,
        variant_builders=variant_builders,
        experiment_label_prefix="Baseline CIFAR100",
        code_snapshot_dirs=DEFAULT_CODE_SNAPSHOT_DIRS,
        report_config=ReportConfig(
            project_name="CIFAR Lightning",
            hypothesis="mHC improves representation quality without slowing convergence.",
            parameters={"Hardware": "RTX 3090"},
        ),
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
        code_snapshot_dirs=DEFAULT_CODE_SNAPSHOT_DIRS,
        report_config=ReportConfig(
            project_name="CIFAR Lightning",
            hypothesis="Baseline ViT-Tiny achieves stable convergence on CIFAR-10.",
            parameters={"Hardware": "RTX 3090"},
        ),
    )


if __name__ == "__main__":
    import latent_space.experiment.experiment_runner as experiment_runner

    experiment_runner.main(experiments_module=sys.modules[__name__])
