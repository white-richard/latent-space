from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal


@dataclass
class ExperimentConfig:
    """Experiment-level parameters shared across entry points."""

    experiment_name: str = ""
    output_dir: str = "./experiments"
    seed: int = 42  # Use -1 to disable deterministic seeding.
    debug_mode: bool = True

    save_embeddings: bool = True
    umap_n_neighbors: list[int] = field(default_factory=lambda: [5, 15, 50])
    umap_min_dist: float = 0.1

    run_variants: bool = False
    variant_label: str | None = None
    is_variant_run: bool = False

    # Legacy compatibility flags
    run_mhc_variant: bool = False

    def __post_init__(self) -> None:
        if self.variant_label is None and self.run_variants:
            self.variant_label = "variant"

        if self.run_mhc_variant:
            self.run_variants = True

        # keep attribute expected by older experiment scripts
        object.__setattr__(self, "is_mhc", self.is_variant_run)


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    dataset_name: str = "cifar"
    use_cifar100: bool | None = None  # Retained for CIFAR-specific helpers.
    num_classes: int | None = None

    data_dir: str = "./data"
    image_size: int | tuple[int, int] = 32
    batch_size: int = 256
    num_workers: int = 16
    pin_memory: bool = True

    reprob: float = 0.25
    jitter: float = 0.1
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    _KNOWN_DATASETS: ClassVar[dict[str, int]] = {
        "cifar10": 10,
        "cifar100": 100,
    }

    def __post_init__(self) -> None:
        dataset_key = (self.dataset_name or "").lower()

        if dataset_key in self._KNOWN_DATASETS:
            if self.num_classes is None:
                self.num_classes = self._KNOWN_DATASETS[dataset_key]
            if self.use_cifar100 is None:
                self.use_cifar100 = dataset_key == "cifar100"
        elif self.use_cifar100 is not None:
            dataset_key = "cifar100" if self.use_cifar100 else "cifar10"
            if self.num_classes is None:
                self.num_classes = self._KNOWN_DATASETS[dataset_key]
        elif self.num_classes is None:
            raise ValueError("Please provide `data.num_classes` for non-predefined datasets.")

        self.dataset_name = dataset_key or self.dataset_name or "dataset"
        if self.use_cifar100 is None:
            self.use_cifar100 = False


@dataclass
class ModelConfig:
    """Model construction parameters."""

    model_name: str = "vit_tiny"
    variant_name: str | None = None
    patch_size: int = 4
    num_classes: int | None = None
    num_batches: int | None = None  # Populated at runtime from the datamodule.


@dataclass
class TrainingConfig:
    """Optimization hyperparameters."""

    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-2
    clip_norm: float = 0.0
    use_bfloat16: bool = True

    scheduler_name: Literal["cosine", "warmup_hold_decay"] = "warmup_hold_decay"
    lr_min_factor: float = 0.01
    
    # Warmup-hold-decay specific parameters
    frac_warmup: float = 0.1
    decay_type: str = "1-sqrt"
    start_cooldown_immediately: bool = False
    auto_trigger_cooldown: bool = False
    
    # Runtime populated. 
    num_batches: int | None = None  # Optional override for schedulers.


@dataclass
class Config:
    """Unified configuration container used across the Lightning template."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        data_num_classes = self.data.num_classes

        if self.model.num_classes is None:
            if data_num_classes is None:
                raise ValueError(
                    "Unable to infer `model.num_classes`; please set `data.num_classes` "
                    "or provide `model.num_classes` explicitly."
                )
            self.model.num_classes = data_num_classes
        elif data_num_classes is not None and self.model.num_classes != data_num_classes:
            raise ValueError(
                "`model.num_classes` and `data.num_classes` must match "
                f"({self.model.num_classes} != {data_num_classes})."
            )

        inferred_batches = self.model.num_batches or self.training.num_batches
        if inferred_batches is not None:
            self.model.num_batches = inferred_batches
            self.training.num_batches = inferred_batches
