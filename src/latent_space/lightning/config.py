from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    use_cifar100: bool = True
    batch_size: int = 256
    num_workers: int = 16
    data_dir: str = "./temp/data"
    pin_memory: bool = True

    # Data augmentation parameters
    scale: float = 0.75
    reprob: float = 0.25
    jitter: float = 0.1


@dataclass
class ModelConfig:
    model_name: Literal["vit_tiny", "vit_tiny_mhc"] = "vit_tiny"
    patch_size: int = 4
    num_classes: int | None = None


@dataclass
class TrainingConfig:
    epochs: int = 100
    num_batches: int | None = None
    lr: float = 0.001
    weight_decay: float = 0.01
    clip_norm: float = 0.0
    use_bfloat16: bool = True

    # Scheduler parameters
    scheduler_name: Literal["cosine", "warmup_hold_decay"] = "warmup_hold_decay"
    # Cosine
    cosine_eta_min_factor: float = 0.01  # lr * this factor = min lr
    # Warmup Hold Decay
    frac_warmup: float = 0.1
    final_lr_factor: float = 0.1
    decay_type: str = "1-sqrt"
    start_cooldown_immediately: bool = False
    auto_trigger_cooldown: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 42
    debug_mode: bool = True
    checkpoint_dir: str = "./temp/checkpoints"
    output_dir: str = "./temp"
    run_mhc_variant:bool = False

    # Visualization parameters
    save_embeddings: bool = True
    umap_n_neighbors: list[int] = field(default_factory=lambda: [5, 15, 50])
    umap_min_dist: float = 0.1


@dataclass
class Config:
    """Main configuration class combining all configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        # If model.num_classes wasn't provided, set it from data.use_cifar100
        if self.model.num_classes is None:
            self.model.num_classes = 100 if self.data.use_cifar100 else 10
        else:
            expected = 100 if self.data.use_cifar100 else 10
            if self.model.num_classes != expected:
                raise ValueError(
                    f"`model.num_classes` ({self.model.num_classes}) is inconsistent with "
                    f"`data.use_cifar100` ({self.data.use_cifar100}); expected {expected}"
                )
