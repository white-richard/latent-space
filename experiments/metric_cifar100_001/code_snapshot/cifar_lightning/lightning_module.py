import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

from latent_space.loss.circle_loss import CircleLoss, convert_label_to_similarity
from latent_space.models.vision_transformer.vision_transformer import vit_small, vit_tiny, vit_base

from .config import Config


class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        if self.config.model.model_name == "vit_tiny":
            self.model = vit_tiny(
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
                use_mhc=self.config.model.use_mhc,
            )
        elif self.config.model.model_name == "vit_small":
            self.model = vit_small(
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
                use_mhc=self.config.model.use_mhc,
            )
        elif self.config.model.model_name == "vit_base":
            self.model = vit_base(
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
                use_mhc=self.config.model.use_mhc,
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.config.model.model_name}")

        self.model.init_weights()

        # Loss functions
        self.loss_items = self._build_loss_items()

        # Metrics tracking
        self.train_acc = []
        self.val_acc = []

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

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _build_loss_items(self):
        items = []
        for item in self.config.loss.losses:
            if item.name == "cross_entropy":
                items.append(
                    {
                        "name": "cross_entropy",
                        "weight": float(item.weight),
                        "fn": nn.CrossEntropyLoss(),
                    }
                )
            elif item.name == "circle":
                items.append(
                    {
                        "name": "circle",
                        "weight": float(item.weight),
                        "fn": CircleLoss(m=item.circle_m, gamma=item.circle_gamma),
                    }
                )
            else:
                raise ValueError(f"Unknown loss name: {item.name}")
        return items

    def compute_losses(self, embeddings, logits, labels):
        losses = {}
        for item in self.loss_items:
            name = item["name"]
            if name == "cross_entropy":
                losses[name] = item["fn"](logits, labels)
            elif name == "circle":
                normed = F.normalize(embeddings, dim=1)
                sp, sn = convert_label_to_similarity(normed, labels)
                losses[name] = item["fn"](sp, sn)
        return losses

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch
        embeddings = self.forward_cls(X)
        output = self.forward_head(embeddings)
        loss_dict = self.compute_losses(embeddings, output, y)
        loss = sum(self._loss_weight(name) * val for name, val in loss_dict.items())

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        for name, val in loss_dict.items():
            self.log(f"train/loss_{name}", val, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = batch
        embeddings = self.forward_cls(X)
        output = self.forward_head(embeddings)
        loss_dict = self.compute_losses(embeddings, output, y)
        loss = sum(self._loss_weight(name) * val for name, val in loss_dict.items())

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"val/loss_{name}", val, on_step=False, on_epoch=True, prog_bar=False)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """Test step."""
        X, y = batch
        embeddings = self.forward_cls(X)
        logits = self.forward_head(embeddings)
        loss_dict = self.compute_losses(embeddings, logits, y)
        loss = sum(self._loss_weight(name) * val for name, val in loss_dict.items())

        # Calculate accuracy
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        for name, val in loss_dict.items():
            self.log(f"test/loss_{name}", val, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        embeddings, labels = self.get_embeddings(self.trainer.datamodule.test_dataloader())

        # Compute metrics
        knn_acc, _ = self.knn_accuracy_in_embedding_space(embeddings, labels, k=1)
        overall_sil, _, _ = self.silhouette_score_by_class(embeddings, labels)

        # Log so they show up in the test metrics dict
        self.log("test/knn_acc", knn_acc, prog_bar=False)
        self.log("test/silhouette", overall_sil, prog_bar=False)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        num_batches = self.config.model.num_batches
        if num_batches is None:
            raise ValueError("TrainingConfig.num_batches must be set before training.")

        n_iters = num_batches * self.config.training.epochs
        warmup_steps = int(self.config.training.frac_warmup * n_iters)

        base_lr = self.config.training.lr
        min_lr = base_lr * self.config.training.lr_min_factor

        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = (current_step - warmup_steps) / float(max(1, n_iters - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine_decay)

        if self.config.training.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
        elif self.config.training.scheduler_name == "warmup_hold_decay":
            from latent_space.schedulers.warm_hold_decay_scheduler import WHDScheduler

            scheduler = WHDScheduler(
                optimizer,
                n_iterations=n_iters,
                frac_warmup=self.config.training.frac_warmup,
                final_lr_factor=self.config.training.lr_min_factor,
                decay_type=self.config.training.decay_type,
                start_cooldown_immediately=self.config.training.start_cooldown_immediately,
                auto_trigger_cooldown=self.config.training.auto_trigger_cooldown,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unknown scheduler name: {self.config.training.scheduler_name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update per step for smooth lr changes
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        pass

    def _loss_weight(self, name: str) -> float:
        for item in self.loss_items:
            if item["name"] == name:
                return float(item["weight"])
        return 1.0

    def get_embeddings(self, dataloader):
        """Extract embeddings from the model (without classification head)."""
        self.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                emb = self.forward_cls(X)
                embeddings.append(emb.cpu())
                labels.append(y)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        return embeddings.numpy(), labels.numpy()

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

        label_list = np.unique(np.concatenate([y_t, y_p])) if labels is None else np.asarray(labels)

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
