import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from latent_space.PaCMAP import visualize_embedding
from latent_space.set_seeds import set_seeds
from latent_space.umap_features import plot_umap

from .datamodule import CIFARDataModule
from .lightning_module import VisionTransformerModule


def generate_visualizations(
    module: VisionTransformerModule,
    datamodule: CIFARDataModule,
    experiment,
) -> None:
    """Generate embedding visualizations after training."""
    print("\nGenerating embedding visualizations...")

    test_dataloader = datamodule.test_dataloader()
    embeddings, labels = module.get_embeddings(test_dataloader)

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pacmap_path = output_dir / "pacmap_emb.png"
    visualize_embedding(embeddings, labels, save_path=str(pacmap_path))
    print(f"Saved PaCMAP visualization to: {pacmap_path}")

    # umap_n_neighbors expected as a list on experiment
    for n_neighbors in getattr(experiment, "umap_n_neighbors", [15]):
        umap_path = output_dir / f"umap_nn{n_neighbors}_emb.png"
        plot_umap(
            embeddings,
            labels,
            save_path=str(umap_path),
            n_neighbors=n_neighbors,
            min_dist=getattr(experiment, "umap_min_dist", 0.1),
            is_hyperbolic=False,
            manifold=None,
            device=module.device,
        )
        print(f"Saved UMAP (n_neighbors={n_neighbors}) visualization to: {umap_path}")


def train(config):
    """Run training given a flat dataclass `config` (produced by namespace_to_dataclass).

    If dinov3 weights/repo are missing and we're in debug mode, use a small
    in-memory LightningModule fallback so the debug run can complete without
    the heavyweight dinov3 model.
    """
    # Seed (flat config)
    if getattr(config, "seed", -1) != -1:
        set_seeds(config.seed)

    # Data module: pass the flat config directly (CIFARDataModule reads attributes like data_dir, batch_size)
    datamodule = CIFARDataModule(config)
    datamodule.setup()

    # set number of batches for schedulers on the same flat config object
    config.num_batches = len(datamodule.train_dataloader())

    # Simplified logic: when debug_mode is enabled, force use of a small dummy
    # LightningModule so debug/test runs can be executed quickly without loading
    # the heavyweight dinov3 model and its repo dependencies. Otherwise,
    # instantiate the real VisionTransformerModule (which will try to load dinov3).
    if getattr(config, "debug_mode", False):

        class DummyVisionTransformerModule(pl.LightningModule):
            def __init__(self, cfg) -> None:
                super().__init__()
                self.save_hyperparameters()
                self.config = cfg
                # CIFAR images: 3x32x32, compute in_features accordingly
                in_ch = 3
                img_h = 32
                img_w = 32
                in_features = in_ch * img_h * img_w
                emb_dim = 64
                self.backbone = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features, emb_dim),
                    nn.ReLU(),
                )
                self.head = nn.Linear(emb_dim, getattr(cfg, "num_classes", 10))
                self.loss_fn = nn.CrossEntropyLoss()

            def forward_cls(self, x):
                return self.backbone(x)

            def forward_head(self, emb):
                return self.head(emb)

            def forward(self, x):
                emb = self.forward_cls(x)
                return self.forward_head(emb)

            def training_step(self, batch, batch_idx):
                X, y = batch
                logits = self.forward(X)
                loss = self.loss_fn(logits, y)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                X, y = batch
                logits = self.forward(X)
                loss = self.loss_fn(logits, y)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                self.log("val/loss", loss, on_epoch=True, prog_bar=True)
                self.log("val/acc", acc, on_epoch=True, prog_bar=True)
                return {"val_loss": loss, "val_acc": acc}

            def test_step(self, batch, batch_idx):
                X, y = batch
                logits = self.forward(X)
                loss = self.loss_fn(logits, y)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                self.log("test/loss", loss, on_epoch=True)
                self.log("test/acc", acc, on_epoch=True)
                return {"test_loss": loss, "test_acc": acc}

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=getattr(self.config, "lr", 1e-3))

            def get_embeddings(self, dataloader):
                self.eval()
                embs = []
                labels = []
                with torch.no_grad():
                    for X, y in dataloader:
                        X = X.to(self.device)
                        emb = self.forward_cls(X)
                        embs.append(emb.cpu())
                        labels.append(y)
                embs = torch.cat(embs, dim=0)
                labels = torch.cat(labels, dim=0)
                return embs.numpy(), labels.numpy()

        module = DummyVisionTransformerModule(config)
    else:
        # Model: pass the same flat config (VisionTransformerModule supports flat or nested)
        module = VisionTransformerModule(config=config)

    logger = TensorBoardLogger(
        save_dir=getattr(config, "output_dir", "./runs"),
        name=getattr(config, "experiment_name", "default"),
        version=None,
    )

    precision = "bf16-mixed" if getattr(config, "use_bfloat16", False) else "32-true"

    trainer = pl.Trainer(
        max_epochs=getattr(config, "epochs", 10),
        accelerator="auto",
        devices=1,
        precision=precision,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        gradient_clip_val=getattr(config, "clip_norm", 0.0)
        if getattr(config, "clip_norm", 0.0) and getattr(config, "clip_norm", 0.0) > 0
        else None,
        fast_dev_run=getattr(config, "debug_mode", False),
        overfit_batches=getattr(config, "overfit_batches", 0.0),
    )

    trainer.fit(module, datamodule=datamodule)

    test_metrics_list = trainer.test(module, datamodule=datamodule)
    test_metrics = test_metrics_list[0] if test_metrics_list else {}

    # Visualizations: pass flat config (generate_visualizations reads fields like output_dir, umap_n_neighbors)
    if getattr(config, "save_embeddings", False) and not getattr(config, "debug_mode", False):
        generate_visualizations(module, datamodule, config)

    print("\nTraining complete!")
    return test_metrics


def namespace_to_dataclass(args: argparse.Namespace) -> Any:
    args_dict = vars(args)

    field_definitions = [
        (key, type(value), dataclasses.field(default=value)) for key, value in args_dict.items()
    ]

    Config = dataclasses.make_dataclass("Config", field_definitions)
    return Config()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightning training")

    # Data
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--use-cifar100", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--scale", type=float, default=0.08)

    # Model
    parser.add_argument(
        "--model-name",
        choices=["vit_tiny", "vit_small", "vit_base"],
        default="vit_tiny",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--use-mhc", action="store_true")
    parser.add_argument(
        "--dinov3-weights-path",
        default="./model_weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        help="Path to dinov3 weights file",
    )
    parser.add_argument(
        "--dinov3-repo-path",
        default="./repos/dinov3",
        help="Path to dinov3 repo directory",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-bfloat16", action="store_true")
    parser.add_argument("--clip-norm", type=float, default=0.0)
    parser.add_argument("--frac-warmup", type=float, default=0.0)
    parser.add_argument("--lr-min-factor", type=float, default=0.0)
    parser.add_argument(
        "--scheduler-name",
        choices=["cosine", "warmup_hold_decay"],
        default="cosine",
    )
    parser.add_argument("--decay-type", default="cosine")
    parser.add_argument("--start-cooldown-immediately", action="store_true")
    parser.add_argument("--auto-trigger-cooldown", action="store_true")

    # Experiment/logging
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--overfit-batches", type=float, default=0.0)
    parser.add_argument("--save-embeddings", action="store_true")

    # UMAP visualization
    parser.add_argument(
        "--umap-n-neighbors",
        default="15",
        help="Comma-separated list, e.g. 5,15,50",
    )
    parser.add_argument("--umap-min-dist", type=float, default=0.1)

    # Loss: allow multiple --loss entries
    parser.add_argument(
        "--loss",
        action="append",
        default=["cross_entropy"],
        help="Loss spec name[:weight[:start_epoch[:warmup_epochs[:circle_m[:circle_gamma]]]]]",
    )

    parsed = parser.parse_args(list(argv) if argv else None)

    # Parse umap list into ints and use an immutable default (tuple) so dataclasses can accept it
    umap_raw = getattr(parsed, "umap_n_neighbors", None)
    try:
        umap_n_neighbors = tuple(int(x) for x in umap_raw.split(","))
    except Exception:
        umap_n_neighbors = (15,)

    # Parse losses into structured namespaces and produce an immutable tuple default
    losses_list = []
    for loss_spec in getattr(parsed, "loss", []) or []:
        parts = loss_spec.split(":")
        name = parts[0] if len(parts) > 0 and parts[0] != "" else "cross_entropy"
        weight = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 1.0
        start_epoch = int(parts[2]) if len(parts) > 2 and parts[2] != "" else 0
        warmup_epochs = int(parts[3]) if len(parts) > 3 and parts[3] != "" else 0
        circle_m = float(parts[4]) if len(parts) > 4 and parts[4] != "" else 0.25
        circle_gamma = float(parts[5]) if len(parts) > 5 and parts[5] != "" else 256.0
        losses_list.append(
            argparse.Namespace(
                name=name,
                weight=weight,
                start_epoch=start_epoch,
                warmup_epochs=warmup_epochs,
                circle_m=circle_m,
                circle_gamma=circle_gamma,
            ),
        )

    losses_tuple = tuple(losses_list)

    # Build a flat namespace with all config values (include dinov3 fields)
    return argparse.Namespace(
        # Data
        data_dir=parsed.data_dir,
        use_cifar100=parsed.use_cifar100,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        pin_memory=parsed.pin_memory,
        scale=parsed.scale,
        # Model
        model_name=parsed.model_name,
        patch_size=parsed.patch_size,
        num_classes=parsed.num_classes,
        use_mhc=parsed.use_mhc,
        dinov3_weights_path=parsed.dinov3_weights_path,
        dinov3_repo_path=parsed.dinov3_repo_path,
        num_batches=None,
        # Training
        epochs=parsed.epochs,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        use_bfloat16=parsed.use_bfloat16,
        clip_norm=parsed.clip_norm,
        frac_warmup=parsed.frac_warmup,
        lr_min_factor=parsed.lr_min_factor,
        scheduler_name=parsed.scheduler_name,
        decay_type=parsed.decay_type,
        start_cooldown_immediately=parsed.start_cooldown_immediately,
        auto_trigger_cooldown=parsed.auto_trigger_cooldown,
        # Experiment
        output_dir=parsed.output_dir,
        experiment_name=parsed.experiment_name,
        seed=parsed.seed,
        debug_mode=parsed.debug_mode,
        overfit_batches=parsed.overfit_batches,
        save_embeddings=parsed.save_embeddings,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=parsed.umap_min_dist,
        # Losses (immutable tuple)
        losses=losses_tuple,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = namespace_to_dataclass(args)
    metrics = train(cfg)
    print("Test metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
