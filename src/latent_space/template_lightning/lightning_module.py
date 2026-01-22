from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from latent_space.models.vision_transformer.vision_transformer import vit_tiny
from latent_space.schedulers.warm_hold_decay_scheduler import WHDScheduler

from .config import Config

ModelBuilder = Callable[..., nn.Module]


def _default_model_registry() -> dict[str, ModelBuilder]:
    """Return the built-in registry of model builders."""
    return {
        "vit_tiny": vit_tiny,
    }


class VisionTransformerModule(pl.LightningModule):
    """Generic LightningModule wrapper around the configured backbone."""

    def __init__(
        self,
        config: Config,
        criterion: nn.Module,
        model_registry: Mapping[str, ModelBuilder] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_registry", "criterion"])

        self.config = config
        self.model_registry = dict(_default_model_registry())
        if model_registry is not None:
            self.model_registry.update(model_registry)

        self.model = self._build_model()
        self.criterion = criterion

    # --------------------------------------------------------------------- #
    # Lightning hooks
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="train")

        current_lr = None
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        batch_size = batch[0].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if current_lr is not None:
            self.log("train/lr", current_lr, on_step=True, prog_bar=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="val")
        batch_size = batch[0].size(0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="test")
        batch_size = batch[0].size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test/acc", acc, on_step=False, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        scheduler_config = self._build_scheduler(optimizer)
        if scheduler_config is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        if self.config.training.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.training.clip_norm,
            )

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _build_model(self) -> nn.Module:
        model_name = self.config.model.model_name
        variant_name = self.config.model.variant_name
        base_name = model_name

        if variant_name is None and model_name.endswith("_mhc"):
            base_name = model_name.removesuffix("_mhc")
            variant_name = "mhc"

        builder = self.model_registry.get(base_name)
        if builder is None:
            available = ", ".join(sorted(self.model_registry))
            raise ValueError(
                f"Unknown `model.model_name`: {model_name}. Available models: {available}"
            )

        extra_kwargs: dict[str, Any] = {}
        if variant_name == "mhc":
            extra_kwargs["use_mhc"] = True

        model = builder(
            patch_size=self.config.model.patch_size,
            num_classes=self.config.model.num_classes,
            **extra_kwargs,
        )

        if hasattr(model, "init_weights"):
            model.init_weights()

        return model

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean()

        return loss, acc

    def _build_scheduler(self, optimizer) -> dict[str, Any] | None:
        cfg = self.config.training
        total_batches = cfg.num_batches or self.config.model.num_batches
        if total_batches is None:
            return None

        total_iterations = total_batches * cfg.epochs

        if cfg.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_iterations,
                eta_min=cfg.lr * cfg.lr_min_factor,
            )
            return {"scheduler": scheduler, "interval": "step", "frequency": 1}

        if cfg.scheduler_name == "warmup_hold_decay":
            scheduler = WHDScheduler(
                optimizer,
                n_iterations=total_iterations,
                frac_warmup=cfg.frac_warmup,
                final_lr_factor=cfg.lr_min_factor,
                decay_type=cfg.decay_type,
                start_cooldown_immediately=cfg.start_cooldown_immediately,
                auto_trigger_cooldown=cfg.auto_trigger_cooldown,
            )
            return {"scheduler": scheduler, "interval": "step", "frequency": 1}

        raise ValueError(f"Unsupported scheduler: {cfg.scheduler_name}")

    # --------------------------------------------------------------------- #
    # Embedding utilities
    # --------------------------------------------------------------------- #
    def get_embeddings(self, dataloader):
        """Extract embeddings by temporarily removing the classification head."""
        self.eval()

        if not hasattr(self.model, "head"):
            raise AttributeError(
                "Model does not expose a `head` attribute required for embeddings."
            )

        original_head = self.model.head
        self.model.head = nn.Identity()

        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                emb = self(inputs)
                embeddings.append(emb.detach().cpu())
                labels.append(targets.detach().cpu())

        self.model.head = original_head  # Restore original head

        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.cat(labels, dim=0)

        return embeddings_tensor.numpy(), labels_tensor.numpy()
