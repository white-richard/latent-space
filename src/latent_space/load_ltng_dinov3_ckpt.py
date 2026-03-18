import os
import sys

import torch
from dinov3.train.ssl_meta_arch import SSLMetaArch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule


def load_dinov3_checkpoint(
    checkpoint_path: str,
    cfg_path: str | None = None,
    model: LightningModule | None = None,
    map_location: str | torch.device | None = "cpu",
) -> tuple[LightningModule, list, list]:

    # Ensure dinov3 package is importable from the local `tmp/dinov3` tree
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "dinov3")))

    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

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
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
        state_dict = ckpt
    else:
        msg = "Unsupported checkpoint format: expected dict with 'state_dict' or raw state_dict"
        raise ValueError(
            msg,
        )

    if any(k.startswith("ssl_model.") for k in state_dict):
        state_dict = {k.replace("ssl_model.", ""): v for k, v in state_dict.items()}

    # Load into model. Use assign=True so that meta parameters
    # are assigned instead of silently ignored.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    model = model.student.backbone

    if missing:
        print("Missing keys when loading checkpoint:")
        for key in missing:
            print(f"  {key}")
        msg = "Some model parameters are missing after loading checkpoint."
        raise ValueError(msg)
    if unexpected:
        print("Unexpected keys when loading checkpoint:")
        for key in unexpected:
            print(f"  {key}")
        msg = "Some unexpected parameters were found after loading checkpoint."
        raise ValueError(msg)

    return model
