#!/usr/bin/env python3
"""
Enhanced ModelCheckpoint callback for DINOv3 training
Supports step-based checkpointing with detailed naming
"""

import os
import math
import time
import sys
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer


class DINOv3ModelCheckpoint(ModelCheckpoint):
    """
    Simple ModelCheckpoint for DINOv3 using Lightning's native saving with custom filename
    """
    
    
    def __init__(
        self,
        dirpath: Optional[Union[str, os.PathLike]] = None,
        filename: Optional[str] = "model_epoch_{epoch:02d}_step_{step:06d}_loss_{total_loss:.6f}",
        monitor: Optional[str] = "total_loss",
        verbose: bool = False,
        save_last: Optional[bool] = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[Union[int, float]] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        **kwargs
    ):
            
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs
        )
        
        if verbose and hasattr(torch.distributed, 'is_available') and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"DINOv3ModelCheckpoint initialized:", flush=True)
                print(f"  dirpath: {dirpath}", flush=True)
                print(f"  filename: {filename}", flush=True)
                print(f"  every_n_train_steps: {every_n_train_steps}", flush=True)
                print(f"  save_top_k: {save_top_k}", flush=True)
                print(f"  monitor: {monitor}", flush=True)


def load_dinov3_checkpoint(
    checkpoint_path: str,
    cfg_path: Optional[str] = None,
    model: Optional[LightningModule] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    save_path: Optional[str] = None,
) -> Optional[tuple[LightningModule, list, list]]:
    # Ensure dinov3 package is importable from the local `tmp/dinov3` tree
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "dinov3"))
    )
    from dinov3.train.ssl_meta_arch import SSLMetaArch
    from omegaconf import OmegaConf

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build model if not provided
    if model is None:
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "ssl_lvit_small.yaml")
        cfg = OmegaConf.load(cfg_path)
        model = SSLMetaArch(cfg)

    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
    else:
        raise ValueError(
            "Unsupported checkpoint format: expected dict with 'state_dict' or raw state_dict"
        )

    if any(k.startswith("ssl_model.") for k in state_dict.keys()):
        state_dict = {k.replace("ssl_model.", ""): v for k, v in state_dict.items()}

    # Load into model. Use assign=True so that meta parameters are assigned instead of silently ignored.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    model = model.student.backbone

    if missing:
        print("Missing keys when loading checkpoint:")
        for key in missing:
            print(f"  {key}")
        raise ValueError("Some model parameters are missing after loading checkpoint.")
    if unexpected:
        print("Unexpected keys when loading checkpoint:")
        for key in unexpected:
            print(f"  {key}")
        raise ValueError("Some unexpected parameters were found after loading checkpoint.")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to: {save_path}")

    return model