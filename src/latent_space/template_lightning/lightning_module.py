from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
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
    def forward_head(self, x):
        """
        Forward pass through only the head of the model.
        """
        return self.model.head(x)

    def forward_cls(self, x):
        """
        Forward pass through the model up to the feature extraction.
        """
        return self.model.forward_cls(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="train")

        current_lr = None
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        batch_size = batch[0].size(0)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        if current_lr is not None:
            self.log("train/lr", current_lr, on_step=True, prog_bar=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="val")
        batch_size = batch[0].size(0)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, acc = self._shared_step(batch, stage="test")
        batch_size = batch[0].size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test/acc", acc, on_step=False, on_epoch=True, batch_size=batch_size)
    
    def on_test_end(self):
        embeddings, labels = self.get_embeddings(self.trainer.datamodule.test_dataloader())

        knn_acc, y_knn = self.knn_accuracy_in_embedding_space(embeddings, labels, k=1)
        # cm, label_list, per_class = self.per_class_confusion_matrix(labels, y_knn, normalize=None)
        overall_sil, sil_by_class, sil_per_sample = self.silhouette_score_by_class(
            embeddings, labels
        )
        self.log("test/knn_acc", knn_acc, prog_bar=False)
        self.log("test/silhouette", overall_sil, prog_bar=False)

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
        total_batches = self.config.model.num_batches
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

        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                emb = self.forward_cls(inputs)
                embeddings.append(emb.detach().cpu())
                labels.append(targets.detach().cpu())

        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.cat(labels, dim=0)

        return embeddings_tensor.numpy(), labels_tensor.numpy()

    def knn_accuracy_in_embedding_space(self, embeddings, y_true, k=1, metric="euclidean"):
        X = np.asarray(embeddings)
        y = np.asarray(y_true)

        if X.ndim != 2:
            raise ValueError("embeddings must be a 2D array (n_samples, n_dims).")
        if len(y) != X.shape[0]:
            raise ValueError("y_true length must match number of rows in embeddings.")
        if k < 1:
            raise ValueError("k must be >= 1.")

        # +1 because the nearest neighbor of a point is itself (distance 0)
        nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric=metric)
        nn.fit(X)
        neigh_idx = nn.kneighbors(X, return_distance=False)

        # drop self-neighbor (first column)
        neigh_idx = neigh_idx[:, 1:]
        if neigh_idx.shape[1] < k:
            # small dataset edge case: reduce k
            k_eff = neigh_idx.shape[1]
            if k_eff == 0:
                raise ValueError("Not enough samples to compute neighbors (need at least 2).")
        else:
            k_eff = k

        # majority vote
        y_pred = np.empty_like(y, dtype=object if y.dtype.kind in {"U", "S", "O"} else y.dtype)
        for i in range(X.shape[0]):
            neighbor_labels = y[neigh_idx[i, :k_eff]]
            # Count occurrences
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            # Tie-break: pick the first max (deterministic by np.unique sort)
            y_pred[i] = vals[np.argmax(counts)]

        acc = float(np.mean(y_pred == y))
        return acc, y_pred

    def per_class_confusion_matrix(self, y_true, y_pred, labels=None, normalize=None):
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        if labels is None:
            label_list = np.unique(np.concatenate([y_t, y_p]))
        else:
            label_list = np.asarray(labels)

        cm = confusion_matrix(y_t, y_p, labels=label_list, normalize=normalize)

        # For per-class metrics, use the *unnormalized* confusion matrix
        cm_raw = confusion_matrix(y_t, y_p, labels=label_list, normalize=None)
        per_class = {}

        for i, lab in enumerate(label_list):
            tp = cm_raw[i, i]
            fn = cm_raw[i, :].sum() - tp
            fp = cm_raw[:, i].sum() - tp
            tn = cm_raw.sum() - (tp + fn + fp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            per_class[lab] = {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "support": int(cm_raw[i, :].sum()),
            }

        return cm, label_list, per_class

    def silhouette_score_by_class(self, embeddings, y_true, metric="euclidean"):
        """
        Silhouette analysis using class labels as cluster IDs.
        """
        X = np.asarray(embeddings)
        y = np.asarray(y_true)

        unique, counts = np.unique(y, return_counts=True)
        if unique.shape[0] < 2:
            raise ValueError("Silhouette requires at least 2 distinct classes.")
        if np.any(counts < 2):
            bad = unique[counts < 2]
            raise ValueError(
                f"Silhouette requires at least 2 samples per class. Classes with <2 samples: {bad}"
            )

        per_sample = silhouette_samples(X, y, metric=metric)
        overall = float(silhouette_score(X, y, metric=metric))

        by_class = {}
        for lab in unique:
            by_class[lab] = float(np.mean(per_sample[y == lab]))

        return overall, by_class, per_sample
