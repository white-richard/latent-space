import pytorch_lightning as pl
import torch
import torch.nn as nn

from latent_space.models.vision_transformer.vision_transformer import vit_tiny

from .config import Config


class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        # Initialize model
        if self.config.model.model_name == "vit_tiny":
            self.model = vit_tiny(
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
            )

        elif self.config.model.model_name == "vit_tiny_mhc":
            self.model = vit_tiny(
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
                use_mhc=True,
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.config.model.model_name}")

        self.model.init_weights()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_acc = []
        self.val_acc = []

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch
        output = self(X)
        loss = self.criterion(output, y)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = batch
        output = self(X)
        loss = self.criterion(output, y)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """Test step."""
        X, y = batch
        output = self(X)
        loss = self.criterion(output, y)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        num_batches = self.config.model.num_batches
        n_iters = num_batches * self.config.training.epochs

        if self.config.training.scheduler_name == "cosine":
            raise NotImplementedError("Cosine scheduler not implemented")

        elif self.config.training.scheduler_name == "warmup_hold_decay":
            from latent_space.schedulers.warm_hold_decay_scheduler import WHDScheduler

            scheduler = WHDScheduler(
                optimizer,
                n_iterations=n_iters,
                frac_warmup=self.config.training.frac_warmup,
                final_lr_factor=self.config.training.lr_min_factor,
                decay_type=self.config.training.decay_type,
                start_cooldown_immediately=self.config.training.start_cooldown_immediately,
                auto_trigger_cooldown=self.config.training.auto_trigger_cooldown
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
        """Clip gradients if configured."""
        if self.config.training.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.training.clip_norm,
            )

    def get_embeddings(self, dataloader):
        """Extract embeddings from the model (without classification head)."""
        self.eval()

        # Temporarily replace head with identity
        original_head = self.model.head
        self.model.head = nn.Identity()

        embeddings = []
        labels = []

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                emb = self(X)
                embeddings.append(emb.cpu())
                labels.append(y)

        # Restore original head
        self.model.head = original_head

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        return embeddings.numpy(), labels.numpy()
