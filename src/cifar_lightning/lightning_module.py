import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch import nn

from latent_space.loss.circle_loss import CircleLoss, convert_label_to_similarity
from latent_space.loss.koleo_loss import KoLeoLoss
from latent_space.models.dinov3 import get_dinov3


class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        # Need to define model — weights/repo path are configurable via config
        dinov3_weights = getattr(
            config,
            "dinov3_weights_path",
            "./model_weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        )
        dinov3_repo = getattr(config, "dinov3_repo_path", "./repos/dinov3")
        dinov3_resize = getattr(config, "dinov3_resize", 384)
        self.model = get_dinov3(
            dinov3_weights_path=dinov3_weights,
            dinov3_repo_path=dinov3_repo,
            resize=dinov3_resize,
        )
        self.model.init_weights()

        # Loss functions
        self.loss_items = self._build_loss_items()

        # Metrics tracking
        self.train_acc = []
        self.val_acc = []

    def forward_head(self, x):
        """Forward pass through only the head of the model."""
        return self.model.head(x)

    def forward_cls(self, x):
        """Forward pass through the model up to the feature extraction."""
        return self.model.forward_cls(x)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _build_loss_items(self):
        items = []
        # Support either nested config.loss.losses or flat config.losses (list of specs)
        if hasattr(self.config, "loss") and hasattr(self.config.loss, "losses"):
            raw_losses = self.config.loss.losses
        else:
            raw_losses = getattr(self.config, "losses", None)

        if not raw_losses:
            # default to a single cross_entropy if nothing provided
            raw_losses = ["cross_entropy"]

        for item in raw_losses:
            # Normalize an item into common fields (support string, Namespace, or dict)
            if isinstance(item, str):
                name = item
                weight = 1.0
                start_epoch = 0
                warmup_epochs = 0
                circle_m = 0.25
                circle_gamma = 256.0
            # item might be an argparse.Namespace or a dict-like
            elif isinstance(item, dict):
                name = item.get("name", "cross_entropy")
                weight = float(item.get("weight", 1.0))
                start_epoch = int(item.get("start_epoch", 0))
                warmup_epochs = int(item.get("warmup_epochs", 0))
                circle_m = float(item.get("circle_m", 0.25))
                circle_gamma = float(item.get("circle_gamma", 256.0))
            else:
                name = getattr(item, "name", "cross_entropy")
                weight = float(getattr(item, "weight", 1.0))
                start_epoch = int(getattr(item, "start_epoch", 0))
                warmup_epochs = int(getattr(item, "warmup_epochs", 0))
                circle_m = float(getattr(item, "circle_m", 0.25))
                circle_gamma = float(getattr(item, "circle_gamma", 256.0))

            if name == "cross_entropy":
                items.append(
                    {
                        "name": "cross_entropy",
                        "weight": float(weight),
                        "start_epoch": int(start_epoch),
                        "warmup_epochs": int(warmup_epochs),
                        "fn": nn.CrossEntropyLoss(),
                    },
                )
            elif name == "circle":
                items.append(
                    {
                        "name": "circle",
                        "weight": float(weight),
                        "start_epoch": int(start_epoch),
                        "warmup_epochs": int(warmup_epochs),
                        "fn": CircleLoss(m=circle_m, gamma=circle_gamma),
                    },
                )
            elif name == "koleo":
                items.append(
                    {
                        "name": "koleo",
                        "weight": float(weight),
                        "start_epoch": int(start_epoch),
                        "warmup_epochs": int(warmup_epochs),
                        "fn": KoLeoLoss(),
                    },
                )
            else:
                msg = f"Unknown loss name: {name}"
                raise ValueError(msg)
        return items

    def compute_losses(self, norm_emb, logits, labels):
        losses = {}
        for item in self.loss_items:
            name = item["name"]
            if not self._is_loss_active(name):
                continue
            if name == "cross_entropy":
                losses[name] = item["fn"](logits, labels)
            elif name == "circle":
                normed = F.normalize(norm_emb, dim=1)
                sp, sn = convert_label_to_similarity(normed, labels)
                losses[name] = item["fn"](sp, sn)
            elif name == "koleo":
                losses[name] = item["fn"](norm_emb)
        return losses

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch
        embeddings = self.forward_cls(X)
        normed_emb = F.normalize(embeddings, dim=1)
        output = self.forward_head(embeddings)
        loss_dict = self.compute_losses(normed_emb, output, y)
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

        self._log_grad_norms_per_loss(loss_dict)

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

    def on_test_epoch_end(self) -> None:
        embeddings, labels = self.get_embeddings(self.trainer.datamodule.test_dataloader())

        # Compute metrics
        knn_acc, _ = self.knn_accuracy_in_embedding_space(embeddings, labels, k=1)
        overall_sil, _, _ = self.silhouette_score_by_class(embeddings, labels)

        # Log so they show up in the test metrics dict
        self.log("test/knn_acc", knn_acc, prog_bar=False)
        self.log("test/silhouette", overall_sil, prog_bar=False)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        This function supports both a nested config (config.training / config.model)
        and a flat config (top-level attributes like lr, epochs, num_batches, etc.).
        Preference: use nested attributes if present, otherwise fall back to flat.
        """
        # choose training/model view (nested or flat)
        training_cfg = getattr(self.config, "training", None) or self.config
        model_cfg = getattr(self.config, "model", None) or self.config

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=getattr(training_cfg, "lr", getattr(self.config, "lr", 1e-3)),
            weight_decay=getattr(
                training_cfg,
                "weight_decay",
                getattr(self.config, "weight_decay", 0.0),
            ),
        )

        num_batches = getattr(model_cfg, "num_batches", getattr(self.config, "num_batches", None))
        if num_batches is None:
            msg = "TrainingConfig.num_batches must be set before training."
            raise ValueError(msg)

        n_iters = num_batches * getattr(training_cfg, "epochs", getattr(self.config, "epochs", 1))
        warmup_steps = int(
            getattr(training_cfg, "frac_warmup", getattr(self.config, "frac_warmup", 0.0))
            * n_iters,
        )

        base_lr = getattr(training_cfg, "lr", getattr(self.config, "lr", 1e-3))
        min_lr = base_lr * getattr(
            training_cfg,
            "lr_min_factor",
            getattr(self.config, "lr_min_factor", 0.0),
        )

        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = (current_step - warmup_steps) / float(max(1, n_iters - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine_decay)

        scheduler_name = getattr(
            training_cfg,
            "scheduler_name",
            getattr(self.config, "scheduler_name", "cosine"),
        )
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
        elif scheduler_name == "warmup_hold_decay":
            from latent_space.schedulers.warm_hold_decay_scheduler import WHDScheduler

            scheduler = WHDScheduler(
                optimizer,
                n_iterations=n_iters,
                frac_warmup=getattr(
                    training_cfg,
                    "frac_warmup",
                    getattr(self.config, "frac_warmup", 0.0),
                ),
                final_lr_factor=getattr(
                    training_cfg,
                    "lr_min_factor",
                    getattr(self.config, "lr_min_factor", 0.0),
                ),
                decay_type=getattr(
                    training_cfg,
                    "decay_type",
                    getattr(self.config, "decay_type", "cosine"),
                ),
                start_cooldown_immediately=getattr(
                    training_cfg,
                    "start_cooldown_immediately",
                    False,
                ),
                auto_trigger_cooldown=getattr(training_cfg, "auto_trigger_cooldown", False),
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
            msg = f"Unknown scheduler name: {scheduler_name}"
            raise ValueError(msg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update per step for smooth lr changes
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        pass

    def _log_grad_norms_per_loss(self, loss_dict) -> None:
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return

        for name, loss in loss_dict.items():
            if loss is None:
                continue

            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            grad_norms = [g.norm(2) for g in grads if g is not None]
            if not grad_norms:
                continue

            total_norm = torch.norm(torch.stack(grad_norms), p=2)
            self.log(
                f"train/grad_norm_{name}",
                total_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def _is_loss_active(self, name: str) -> bool:
        for item in self.loss_items:
            if item["name"] == name:
                start_epoch = int(item.get("start_epoch", 0))
                return self.current_epoch >= start_epoch
        return True

    def _loss_weight(self, name: str) -> float:
        for item in self.loss_items:
            if item["name"] == name:
                base_weight = float(item["weight"])
                start_epoch = int(item.get("start_epoch", 0))
                warmup_epochs = int(item.get("warmup_epochs", 0))
                if warmup_epochs <= 0:
                    return base_weight
                progress = (self.current_epoch - start_epoch) / float(warmup_epochs)
                factor = max(0.0, min(1.0, progress))
                return base_weight * factor
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
            msg = "embeddings must be a 2D array (n_samples, n_dims)."
            raise ValueError(msg)
        if len(y) != X.shape[0]:
            msg = "y_true length must match number of rows in embeddings."
            raise ValueError(msg)
        if k < 1:
            msg = "k must be >= 1."
            raise ValueError(msg)

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
                msg = "Not enough samples to compute neighbors (need at least 2)."
                raise ValueError(msg)
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
        """Silhouette analysis using class labels as cluster IDs."""
        X = np.asarray(embeddings)
        y = np.asarray(y_true)

        unique, counts = np.unique(y, return_counts=True)
        if unique.shape[0] < 2:
            msg = "Silhouette requires at least 2 distinct classes."
            raise ValueError(msg)
        if np.any(counts < 2):
            bad = unique[counts < 2]
            msg = (
                f"Silhouette requires at least 2 samples per class. Classes with <2 samples: {bad}"
            )
            raise ValueError(
                msg,
            )

        per_sample = silhouette_samples(X, y, metric=metric)
        overall = float(silhouette_score(X, y, metric=metric))

        by_class = {}
        for lab in unique:
            by_class[lab] = float(np.mean(per_sample[y == lab]))

        return overall, by_class, per_sample
