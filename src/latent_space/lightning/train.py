import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from latent_space.PaCMAP import visualize_embedding
from latent_space.umap_features import plot_umap

from .config import Config, ExperimentConfig
from .datamodule import CIFARDataModule
from .lightning_module import VisionTransformerModule

torch.backends.cudnn.benchmark = True


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def create_callbacks(config: ExperimentConfig):
    """Create PyTorch Lightning callbacks."""
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir / "checkpoints",
        filename="{epoch}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=35,
        save_last=True,
        every_n_epochs=15,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Rich progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    return callbacks


def generate_visualizations(
    module: VisionTransformerModule,
    datamodule: CIFARDataModule,
    config: ExperimentConfig,
):
    """Generate embedding visualizations after training."""
    print("\nGenerating embedding visualizations...")

    # Get embeddings
    test_dataloader = datamodule.test_dataloader()
    embeddings, labels = module.get_embeddings(test_dataloader)

    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate PaCMAP visualization
    pacmap_path = output_dir / "pacmap_emb.png"
    visualize_embedding(embeddings, labels, save_path=str(pacmap_path))
    print(f"Saved PaCMAP visualization to: {pacmap_path}")

    # Generate UMAP visualizations with different n_neighbors
    for n_neighbors in config.umap_n_neighbors:
        umap_path = output_dir / f"umap_nn{n_neighbors}_emb.png"
        plot_umap(
            embeddings,
            labels,
            save_path=str(umap_path),
            n_neighbors=n_neighbors,
            min_dist=config.umap_min_dist,
            is_hyperbolic=False,
            manifold=None,
            device=module.device,
        )
        print(f"Saved UMAP (n_neighbors={n_neighbors}) visualization to: {umap_path}")


def train(config: Config):
    """Main training function."""

    # Set random seeds
    if config.experiment.seed != -1:
        set_all_seeds(config.experiment.seed)

    # Initialize data module
    datamodule = CIFARDataModule(config.data)
    datamodule.setup()
    config.model.num_batches = len(datamodule.train_dataset)

    # Initialize model
    module = VisionTransformerModule(config=config)

    logger = TensorBoardLogger(
        save_dir=config.experiment.output_dir,
        name=config.experiment.experiment_name,
        version=None,
    )

    # Create callbacks
    callbacks = create_callbacks(config.experiment)

    # Configure precision
    precision = "bf16-mixed" if config.training.use_bfloat16 else "32-true"

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        gradient_clip_val=config.training.clip_norm if config.training.clip_norm > 0 else None,
        fast_dev_run=config.experiment.debug_mode,
    )

    # Train the model
    trainer.fit(module, datamodule=datamodule)

    # Test the model
    trainer.test(module, datamodule=datamodule)

    # Generate visualizations if requested
    if config.experiment.save_embeddings and not config.experiment.debug_mode:
        generate_visualizations(module, datamodule, config.experiment)

    print("\nTraining complete!")
    return module, datamodule, trainer
