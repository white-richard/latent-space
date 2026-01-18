#!/usr/bin/env python3
"""
PyTorch Lightning adaptation of DINOv3 fine-tuning
Exact replica of dinov3/dinov3/train functionality but using Lightning framework
"""

import os
import sys
import math
import logging
from pathlib import Path
from functools import partial
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from omegaconf import OmegaConf
from geoopt import ManifoldParameter

try:
    from hypercore.optimizers import RiemannianAdam
except ImportError:
    raise ImportError(
        "Please install hypercore to use RiemannianAdam optimizer: pip install hypercore"
    )

# Import DINOv3 modules
sys.path.append("dinov3")
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.configs import setup_config, get_default_config
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.data import DataAugmentationDINO
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3_lightning")


class DINOv3LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping DINOv3 SSL training
    Maintains exact same functionality as original DINOv3 training
    """

    def __init__(
        self,
        cfg_path: str,
        checkpoint_path: Optional[str] = None,
        accumulate_grad_batches: int = 1,
    ):
        super().__init__()

        # Load configuration
        if isinstance(cfg_path, str):
            self.cfg = OmegaConf.load(cfg_path)
            # Merge with default config
            default_cfg = get_default_config()
            self.cfg = OmegaConf.merge(default_cfg, self.cfg)
        else:
            self.cfg = cfg_path

        # Store paths
        self.checkpoint_path = checkpoint_path

        # Manual gradient accumulation (Lightning doesn't support auto accumulation with manual optimization)
        self.accumulate_grad_batches = accumulate_grad_batches

        # Initialize model components
        self._build_model()

        # Learning rate schedules
        self.lr_schedules = None
        self.current_iteration = 0

        # Save hyperparameters
        self.save_hyperparameters(ignore=["cfg"])

        # Manual optimization for exact DINOv3 behavior
        self.automatic_optimization = False

    def _build_model(self):
        """Build the DINOv3 SSL model architecture"""
        logger.info("Building DINOv3 SSL model...")

        # Create model with meta device first (like original), unless using a custom
        # backbone that can't be constructed on `meta` (e.g. hypercore LViT).
        if getattr(self.cfg.student, "arch", None) == "lvit_small":
            self.ssl_model = SSLMetaArch(self.cfg)
        else:
            with torch.device("meta"):
                self.ssl_model = SSLMetaArch(self.cfg)

        # Distributed training setup will be handled in setup() method
        # when Lightning has initialized the distributed environment

        # Initialize with NaN values (like original)
        self.ssl_model._apply(
            lambda t: torch.full_like(
                t,
                fill_value=(
                    math.nan
                    if t.dtype.is_floating_point
                    else (2 ** (t.dtype.itemsize * 8 - 1))
                ),
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            recurse=True,
        )

        logger.info(f"Model built: {self.ssl_model}")

    def setup(self, stage: str):
        """Setup model weights and schedules"""
        if stage == "fit":
            # Lightning handles distributed training automatically
            logger.info(f"Training setup (world_size: {self.trainer.world_size})")

            # Initialize weights
            self.ssl_model.init_weights()
            self._scan_for_invalid_tensors(tag="post_init_weights")

            # Load checkpoint if provided
            if self.checkpoint_path:
                logger.info(f"Loading checkpoint from {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

                if "teacher" in checkpoint:
                    # This is a full SSL checkpoint with teacher/student structure
                    self.ssl_model.load_state_dict(checkpoint["teacher"], strict=True)
                else:
                    # This is a pretrained backbone checkpoint - map keys to student/teacher/ema
                    backbone_state = checkpoint
                    ssl_state = {}

                    # Map backbone weights to student, teacher, and model_ema
                    for key, value in backbone_state.items():
                        # Skip storage_tokens as it's not in all model variants
                        if key == "storage_tokens":
                            continue

                        ssl_state[f"student.backbone.{key}"] = value.clone()
                        ssl_state[f"teacher.backbone.{key}"] = value.clone()
                        ssl_state[f"model_ema.backbone.{key}"] = value.clone()

                    self.ssl_model.load_state_dict(ssl_state, strict=False)
            self._scan_for_invalid_tensors(tag="post_checkpoint_load")

            # Build learning rate schedules
            self._build_schedules()

            # Apply torch.compile if enabled in config
            self._apply_compile()

    def _build_schedules(self):
        """Build learning rate and other schedules (exact copy from original)"""
        if "schedules" in self.cfg:
            logger.info("Using schedules v2")
            self.lr_schedules = self._build_schedules_v2()
        else:
            OFFICIAL_EPOCH_LENGTH = self.cfg.train.OFFICIAL_EPOCH_LENGTH
            lr = dict(
                base_value=self.cfg.optim["lr"],
                final_value=self.cfg.optim["min_lr"],
                total_iters=self.cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
                warmup_iters=self.cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
                start_warmup_value=0,
                trunc_extra=self.cfg.optim.get("schedule_trunc_extra", False),
            )
            wd = dict(
                base_value=self.cfg.optim["weight_decay"],
                final_value=self.cfg.optim["weight_decay_end"],
                total_iters=self.cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
                trunc_extra=self.cfg.optim.get("schedule_trunc_extra", False),
            )
            momentum = dict(
                base_value=self.cfg.teacher["momentum_teacher"],
                final_value=self.cfg.teacher["final_momentum_teacher"],
                total_iters=self.cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
                trunc_extra=self.cfg.optim.get("schedule_trunc_extra", False),
            )
            teacher_temp = dict(
                base_value=self.cfg.teacher["teacher_temp"],
                final_value=self.cfg.teacher["teacher_temp"],
                total_iters=self.cfg.teacher["warmup_teacher_temp_epochs"]
                * OFFICIAL_EPOCH_LENGTH,
                warmup_iters=self.cfg.teacher["warmup_teacher_temp_epochs"]
                * OFFICIAL_EPOCH_LENGTH,
                start_warmup_value=self.cfg.teacher["warmup_teacher_temp"],
            )

            lr_schedule = CosineScheduler(**lr)
            wd_schedule = CosineScheduler(**wd)
            momentum_schedule = CosineScheduler(**momentum)
            teacher_temp_schedule = CosineScheduler(**teacher_temp)
            last_layer_lr_schedule = CosineScheduler(**lr)

            last_layer_lr_schedule.schedule[
                : self.cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
            ] = 0

            self.lr_schedules = (
                lr_schedule,
                wd_schedule,
                momentum_schedule,
                teacher_temp_schedule,
                last_layer_lr_schedule,
            )

        logger.info("Schedules ready.")

    def _build_schedules_v2(self):
        """Build schedules v2 (exact copy from original)"""
        iter_per_epoch = self.cfg.train.OFFICIAL_EPOCH_LENGTH
        total_iterations = self.cfg.train.OFFICIAL_EPOCH_LENGTH * self.cfg.optim.epochs
        logger.info(f"Total training iterations {total_iterations}")

        # LR scaling rules
        lr_peak = self.cfg.schedules.lr.peak
        lr_end = self.cfg.schedules.lr.end

        if self.cfg.optim.scaling_rule == "linear_wrt_256":
            world_size = (
                self.trainer.world_size if hasattr(self.trainer, "world_size") else 1
            )
            lr_peak *= self.cfg.train.batch_size_per_gpu * world_size / 256.0
            lr_end *= self.cfg.train.batch_size_per_gpu * world_size / 256.0
            logger.info(
                f"Scaling rule {self.cfg.optim.scaling_rule}, LR peak {self.cfg.schedules.lr.peak} -> {lr_peak}, LR end {self.cfg.schedules.lr.end} -> {lr_end}"
            )
        elif self.cfg.optim.scaling_rule == "sqrt_wrt_1024":
            world_size = (
                self.trainer.world_size if hasattr(self.trainer, "world_size") else 1
            )
            lr_peak *= 4 * math.sqrt(
                self.cfg.train.batch_size_per_gpu * world_size / 1024.0
            )
            lr_end *= 4 * math.sqrt(
                self.cfg.train.batch_size_per_gpu * world_size / 1024.0
            )
            logger.info(
                f"Scaling rule {self.cfg.optim.scaling_rule}, LR peak {self.cfg.schedules.lr.peak} -> {lr_peak}, LR end {self.cfg.schedules.lr.end} -> {lr_end}"
            )

        lr = linear_warmup_cosine_decay(
            start=self.cfg.schedules.lr.start,
            peak=lr_peak,
            end=lr_end,
            warmup_iterations=iter_per_epoch * self.cfg.schedules.lr.warmup_epochs,
            total_iterations=total_iterations,
            cosine_iterations=(
                iter_per_epoch * self.cfg.schedules.lr.cosine_epochs
                if "cosine_epochs" in self.cfg.schedules.lr
                else None
            ),
        )
        last_layer_lr = lr.copy()
        last_layer_lr[
            : iter_per_epoch * self.cfg.schedules.lr.freeze_last_layer_epochs
        ] = 0

        weight_decay = linear_warmup_cosine_decay(
            start=self.cfg.schedules.weight_decay.start,
            peak=self.cfg.schedules.weight_decay.peak,
            end=self.cfg.schedules.weight_decay.end,
            warmup_iterations=iter_per_epoch
            * self.cfg.schedules.weight_decay.warmup_epochs,
            total_iterations=total_iterations,
            cosine_iterations=(
                iter_per_epoch * self.cfg.schedules.weight_decay.cosine_epochs
                if "cosine_epochs" in self.cfg.schedules.weight_decay
                else None
            ),
        )

        momentum = linear_warmup_cosine_decay(
            start=self.cfg.schedules.momentum.start,
            peak=self.cfg.schedules.momentum.peak,
            end=self.cfg.schedules.momentum.end,
            warmup_iterations=iter_per_epoch
            * self.cfg.schedules.momentum.warmup_epochs,
            total_iterations=total_iterations,
            cosine_iterations=(
                iter_per_epoch * self.cfg.schedules.momentum.cosine_epochs
                if "cosine_epochs" in self.cfg.schedules.momentum
                else None
            ),
        )

        teacher_temp = linear_warmup_cosine_decay(
            start=self.cfg.schedules.teacher_temp.start,
            peak=self.cfg.schedules.teacher_temp.peak,
            end=self.cfg.schedules.teacher_temp.end,
            warmup_iterations=iter_per_epoch
            * self.cfg.schedules.teacher_temp.warmup_epochs,
            total_iterations=total_iterations,
            cosine_iterations=(
                iter_per_epoch * self.cfg.schedules.teacher_temp.cosine_epochs
                if "cosine_epochs" in self.cfg.schedules.teacher_temp
                else None
            ),
        )

        return lr, weight_decay, momentum, teacher_temp, last_layer_lr

    def _scan_for_invalid_tensors(self, tag: str) -> None:
        """Log and fail fast if any params/buffers contain NaN/Inf after init/load."""
        invalid = []
        for name, param in self.ssl_model.named_parameters():
            if param.is_floating_point() and not torch.isfinite(param).all():
                invalid.append(f"param:{name}")
        for name, buf in self.ssl_model.named_buffers():
            if buf.is_floating_point() and not torch.isfinite(buf).all():
                invalid.append(f"buffer:{name}")
        if invalid:
            logger.error("Invalid tensors detected (%s): %s", tag, ", ".join(invalid))
            raise ValueError(
                f"NaN/Inf detected in ssl_model ({tag}). See logs for tensor names."
            )

    def _scan_for_invalid_grads(self, tag: str) -> None:
        """Log and fail fast if any parameter gradients contain NaN/Inf."""
        invalid = []
        for name, param in self.ssl_model.named_parameters():
            if param.grad is None or not param.grad.is_floating_point():
                continue
            if not torch.isfinite(param.grad).all():
                invalid.append(f"grad:{name}")
        if invalid:
            logger.error("Invalid gradients detected (%s): %s", tag, ", ".join(invalid))
            raise ValueError(
                f"NaN/Inf detected in gradients ({tag}). See logs for tensor names."
            )

    def _apply_compile(self):
        """Apply torch.compile to model components if enabled in config"""
        if hasattr(self.cfg.train, "compile") and self.cfg.train.compile:
            logger.info("Applying torch.compile to model components...")

            # Apply compile to specific model blocks (like original DINOv3)
            def compile_block(
                block: nn.Module, is_backbone_block: bool = True
            ) -> nn.Module:
                """Compile individual blocks based on configuration"""
                try:
                    if torch.cuda.is_available():
                        # Use optimized compilation for CUDA
                        block = torch.compile(
                            block,
                            fullgraph=True,
                            dynamic=False,
                            options={"triton.cudagraphs": True},
                        )
                    else:
                        # Standard compilation for CPU
                        block = torch.compile(block)
                    logger.info(
                        f"Successfully compiled {'backbone' if is_backbone_block else 'head'} block"
                    )
                    return block
                except Exception as e:
                    logger.warning(
                        f"Failed to compile block: {e}, continuing without compilation"
                    )
                    return block

            # Apply compile exactly like original DINOv3
            def wrap_compile_module(
                m: nn.Module, is_backbone: bool = True
            ) -> nn.Module:
                """Compile module exactly like original DINOv3"""
                try:
                    if (
                        is_backbone
                        and hasattr(self.cfg.train, "cudagraphs")
                        and self.cfg.train.cudagraphs
                    ):
                        m = torch.compile(
                            m,
                            fullgraph=True,
                            dynamic=False,
                            options={"triton.cudagraphs": True},
                        )
                    else:
                        m = torch.compile(m)
                    return m
                except Exception as e:
                    logger.warning(f"Failed to compile module: {e}")
                    return m

            # Compile student backbone (exact replica of original logic)
            if hasattr(self.ssl_model.student, "backbone"):
                self.ssl_model.student.backbone = wrap_compile_module(
                    self.ssl_model.student.backbone, is_backbone=True
                )
                logger.info("Successfully compiled student backbone")

            # Compile teacher backbone
            if hasattr(self.ssl_model.teacher, "backbone"):
                self.ssl_model.teacher.backbone = wrap_compile_module(
                    self.ssl_model.teacher.backbone, is_backbone=True
                )
                logger.info("Successfully compiled teacher backbone")

            # Compile EMA backbone
            if hasattr(self.ssl_model.model_ema, "backbone"):
                try:
                    self.ssl_model.model_ema.backbone = wrap_compile_module(
                        self.ssl_model.model_ema.backbone, is_backbone=True
                    )
                    logger.info("Successfully compiled EMA backbone")
                except Exception as e:
                    logger.warning(f"Failed to compile EMA backbone: {e}")

            # Compile other components like loss functions if they exist
            if hasattr(self.ssl_model, "ibot_loss") and hasattr(
                self.ssl_model.ibot_loss, "sinkhorn_knopp_teacher"
            ):
                try:
                    self.ssl_model.ibot_loss.sinkhorn_knopp_teacher = torch.compile(
                        self.ssl_model.ibot_loss.sinkhorn_knopp_teacher
                    )
                    logger.info("Successfully compiled iBOT sinkhorn_knopp_teacher")
                except Exception as e:
                    logger.warning(f"Failed to compile sinkhorn_knopp_teacher: {e}")

            logger.info("torch.compile application completed")
        else:
            logger.info("torch.compile disabled in config (train.compile: false)")

    def configure_optimizers(self):
        """Configure optimizer exactly like original DINOv3 with support for hyperbolic parameters"""
        # Separate Euclidean and Manifold (hyperbolic) parameters
        euc_params = []
        hyp_params = []

        for n, p in self.ssl_model.named_parameters():
            if p.requires_grad:
                if isinstance(p, ManifoldParameter):
                    hyp_params.append(p)
                else:
                    euc_params.append(p)

        # Get hyperbolic optimizer settings from config or use defaults
        hyp_lr = getattr(self.cfg.optim, "hyperbolic_lr", self.cfg.optim.lr)
        hyp_weight_decay = getattr(
            self.cfg.optim, "hyperbolic_weight_decay", self.cfg.optim.weight_decay
        )

        # Check if we have hyperbolic parameters
        if hyp_params:
            logger.info(
                f"Found {len(hyp_params)} hyperbolic parameters (ManifoldParameter)"
            )
            logger.info(f"Found {len(euc_params)} Euclidean parameters")

            # Create optimizer for Euclidean parameters using param groups
            # Note: We need to filter param_groups to only include Euclidean params
            params_groups = self.ssl_model.get_params_groups()

            # Create Euclidean optimizer (AdamW)
            optimizer_euc = AdamW(
                params_groups,
                betas=(self.cfg.optim.adamw_beta1, self.cfg.optim.adamw_beta2),
            )

            # Create hyperbolic optimizer (RiemannianAdam)
            optimizer_hyp = RiemannianAdam(
                hyp_params, lr=hyp_lr, stabilize=1, weight_decay=hyp_weight_decay
            )

            # Return both optimizers - Lightning will handle them
            return [optimizer_euc, optimizer_hyp]
        else:
            logger.info(
                "No hyperbolic parameters found, using standard AdamW optimizer"
            )
            # No hyperbolic parameters, use standard configuration
            params_groups = self.ssl_model.get_params_groups()
            optimizer = AdamW(
                params_groups,
                betas=(self.cfg.optim.adamw_beta1, self.cfg.optim.adamw_beta2),
            )
            return optimizer

    def training_step(self, batch, batch_idx):
        """Training step - exact replica of DINOv3 training loop with manual gradient accumulation"""
        optimizers = self.optimizers()
        # Handle both single optimizer and multiple optimizers (when hyperbolic params exist)
        if isinstance(optimizers, list):
            optimizer_euc, optimizer_hyp = optimizers
        else:
            optimizer_euc = optimizers
            optimizer_hyp = None

        # Determine if this is an accumulation step or optimizer step.
        num_batches = getattr(self.trainer, "num_training_batches", None)
        is_last_batch = (
            isinstance(num_batches, int)
            and num_batches > 0
            and (batch_idx + 1) >= num_batches
        )
        is_accumulating = (
            (batch_idx + 1) % self.accumulate_grad_batches != 0
        ) and not is_last_batch

        # Get current iteration (increments only on optimizer steps, not accumulation steps)
        it = self.current_iteration

        # Apply learning rate schedules
        if self.lr_schedules is not None:
            if isinstance(self.lr_schedules[0], torch.Tensor):
                # Schedules v2
                lr, wd, mom, teacher_temp, last_layer_lr = self.lr_schedules
                lr_val = lr[it].item() if it < len(lr) else lr[-1].item()
                wd_val = wd[it].item() if it < len(wd) else wd[-1].item()
                mom_val = mom[it].item() if it < len(mom) else mom[-1].item()
                teacher_temp_val = (
                    teacher_temp[it].item()
                    if it < len(teacher_temp)
                    else teacher_temp[-1].item()
                )
                last_layer_lr_val = (
                    last_layer_lr[it].item()
                    if it < len(last_layer_lr)
                    else last_layer_lr[-1].item()
                )
            else:
                # Schedules v1
                (
                    lr_schedule,
                    wd_schedule,
                    momentum_schedule,
                    teacher_temp_schedule,
                    last_layer_lr_schedule,
                ) = self.lr_schedules
                lr_val = lr_schedule[it]
                wd_val = wd_schedule[it]
                mom_val = momentum_schedule[it]
                teacher_temp_val = teacher_temp_schedule[it]
                last_layer_lr_val = last_layer_lr_schedule[it]

            self._apply_optim_scheduler(
                optimizer_euc, lr_val, wd_val, last_layer_lr_val
            )
            # Apply schedule to hyperbolic optimizer if it exists
            if optimizer_hyp is not None:
                # Use hyperbolic-specific LR or scale the main LR
                hyp_lr_val = getattr(self.cfg.optim, "hyperbolic_lr", lr_val)
                hyp_wd_val = getattr(self.cfg.optim, "hyperbolic_weight_decay", wd_val)
                self._apply_optim_scheduler_hyperbolic(
                    optimizer_hyp, hyp_lr_val, hyp_wd_val
                )
        else:
            # Default values
            teacher_temp_val = self.cfg.teacher.teacher_temp
            mom_val = self.cfg.teacher.momentum_teacher

        # Add global batch size to batch data (effective batch size considering accumulation)
        batch["global_batch_size"] = (
            self.cfg.train.batch_size_per_gpu
            * self.trainer.world_size
            * self.accumulate_grad_batches
        )

        # Zero gradients only at the start of accumulation
        if batch_idx % self.accumulate_grad_batches == 0:
            optimizer_euc.zero_grad(set_to_none=True)
            if optimizer_hyp is not None:
                optimizer_hyp.zero_grad(set_to_none=True)

        # Forward and backward pass (gradients accumulate automatically)
        total_loss, metrics_dict = self.ssl_model.forward_backward(
            batch, teacher_temp=teacher_temp_val, iteration=it
        )
        self._scan_for_invalid_grads(tag=f"post_backward_it{it}_batch{batch_idx}")

        # Only perform optimizer step and EMA update after accumulating enough gradients
        if not is_accumulating:
            # Gradient clipping (if enabled)
            if self.cfg.optim.clip_grad:
                for k, v in self.ssl_model.student.items():
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        v.parameters(),
                        max_norm=self.cfg.optim.clip_grad,
                    )
                    metrics_dict[f"{k}_grad_norm"] = grad_norm.item()

            # Optimizer step for Euclidean parameters
            optimizer_euc.step()

            # Optimizer step for hyperbolic parameters if they exist
            if optimizer_hyp is not None:
                optimizer_hyp.step()

            # Update EMA
            self.ssl_model.update_ema(mom_val)

            # Handle GRAM updates if needed (only on optimizer steps)
            if (
                hasattr(self.ssl_model, "gram_use_loss")
                and self.ssl_model.gram_use_loss
                and hasattr(self.ssl_model, "gram_rep_update")
                and self.ssl_model.gram_rep_update
                and (it + 1) >= self.ssl_model.gram_it_first_update
                and (it + 1) % self.ssl_model.gram_update_frequency == 0
                and (
                    not hasattr(self.ssl_model.cfg.gram, "max_updates")
                    or self.ssl_model.cfg.gram.max_updates is None
                    or self.ssl_model.num_gram_updates
                    < self.ssl_model.cfg.gram.max_updates
                )
            ):
                logger.info(
                    f"Updating Gram teacher from EMA teacher after iteration {it}"
                )
                self.ssl_model.update_gram()
                if not hasattr(self.ssl_model, "num_gram_updates"):
                    self.ssl_model.num_gram_updates = 0
                self.ssl_model.num_gram_updates += 1

            # Increment iteration counter only on optimizer steps
            self.current_iteration += 1

        # Log metrics with detailed individual losses
        self.log("total_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("iteration", float(it), prog_bar=True, sync_dist=False)

        # Log individual losses prominently
        loss_components = {}
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                val = value.item() if value.numel() == 1 else value.mean()
                self.log(f"train/{key}", val, sync_dist=True)

                # Track individual loss components for progress bar
                if "loss" in key.lower():
                    loss_components[key] = val
            else:
                self.log(f"train/{key}", value, sync_dist=True)
                if "loss" in key.lower():
                    loss_components[key] = value

        # Store loss components for progress bar display
        if hasattr(self, "current_loss_components"):
            self.current_loss_components.update(loss_components)
        else:
            self.current_loss_components = loss_components

        # Log learning rates
        if self.lr_schedules is not None:
            self.log("train/lr", lr_val, sync_dist=True)
            self.log("train/wd", wd_val, sync_dist=True)
            self.log("train/momentum", mom_val, sync_dist=True)
            self.log("train/teacher_temp", teacher_temp_val, sync_dist=True)

        return total_loss

    def _apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        """Apply learning rate and weight decay schedules (exact copy from original)"""
        for param_group in optimizer.param_groups:
            is_last_layer = param_group.get("is_last_layer", False)
            lr_multiplier = param_group.get("lr_multiplier", 1.0)
            wd_multiplier = param_group.get("wd_multiplier", 1.0)
            param_group["weight_decay"] = wd * wd_multiplier
            if is_last_layer:
                param_group["lr"] = last_layer_lr * lr_multiplier
            else:
                param_group["lr"] = lr * lr_multiplier

    def _apply_optim_scheduler_hyperbolic(self, optimizer, lr, wd):
        """Apply learning rate and weight decay schedules for hyperbolic optimizer"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            if "weight_decay" in param_group:
                param_group["weight_decay"] = wd

    def on_train_epoch_end(self):
        """Handle end of training epoch"""
        # Force garbage collection (like in original)
        import gc

        gc.collect()

    def forward(self, x):
        """Forward pass - not used during training but required for Lightning"""
        # This is primarily used during inference
        return self.ssl_model.teacher.backbone(x, is_training=False)

    def get_model_for_checkpoint(self):
        """Get the model state dict for checkpointing"""
        return self.ssl_model.state_dict()

    def load_from_checkpoint_dict(self, checkpoint_dict):
        """Load model from checkpoint dictionary"""
        self.ssl_model.load_state_dict(checkpoint_dict, strict=False)
