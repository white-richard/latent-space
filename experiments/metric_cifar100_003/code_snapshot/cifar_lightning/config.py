from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExperimentConfig:
    experiment_name: str = ""

    output_dir: str = "./experiments"
    seed: int = 42  # -1 to disable
    debug_mode: bool = False

    # Visualization parameters
    save_embeddings: bool = True
    umap_n_neighbors: list[int] = field(default_factory=lambda: [5, 15, 50])
    umap_min_dist: float = 0.1


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
    model_name: Literal["vit_tiny", "vit_small"] = "vit_tiny"
    patch_size: int = 4
    num_classes: int | None = None
    use_mhc: bool = False


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

    lr_min_factor: float = 0.1  # TODO abalate? # lr * this factor = min lr
    # Warmup Hold Decay
    frac_warmup: float = 0.1
    decay_type: str = "1-sqrt"
    start_cooldown_immediately: bool = False
    auto_trigger_cooldown: bool = False


@dataclass
class LossItemConfig:
    name: Literal["cross_entropy", "circle"] = "cross_entropy"
    weight: float = 1.0
    # Start applying this loss at the given epoch (0-based)
    start_epoch: int = 0
    # Linearly ramp loss weight from 0 to full weight over this many epochs
    warmup_epochs: int = 0
    # Circle loss parameters
    circle_m: float = 0.25
    circle_gamma: float = 256.0


@dataclass
class LossConfig:
    losses: list[LossItemConfig] = field(default_factory=lambda: [LossItemConfig()])


@dataclass
class Config:
    """Main configuration class combining all configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
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
