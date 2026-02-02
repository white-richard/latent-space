from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torchvision

import torch
from torch import nn
from torchvision.transforms import v2

__all__ = ["get_dinov3", "_make_dino_transform"]


def _make_dino_transform(
    *, resize_size: int = 384, transforms_list: list[torchvision.transforms.v2.Transform] = None
) -> torchvision.transforms.v2.Transform:
    """
    Build the DINOv3 preprocessing pipeline.

    Args:
        resize_size: Target square resize dimension.

    Returns:
        A torchvision v2 transform that converts input images to normalized float tensors.
    """
    if transforms_list is None:
        transforms_list = []
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, *transforms_list, to_float, normalize])


def _infer_model_name(weights_path: pathlib.Path) -> str:
    """
    Infer the DINOv3 model name from the official weight filename.
    Expects filenames like: `vitb16_pretrain.pth`, `vitl14_pretrain.pth`, etc.
    """
    model_name = weights_path.name.split("_pretrain")[0]
    if not model_name:
        raise ValueError(
            "dinov3_weights_path not in official format; model_name cannot be inferred."
        )
    return model_name


def get_dinov3(
    *,
    dinov3_weights_path: str | pathlib.Path,
    dinov3_repo_path: str,
    resize: int = 384,
) -> tuple[nn.Module, Any]:
    """
    Load a DINOv3 model and its matching preprocessing transform.

    Args:
        dinov3_weights_path: Path to the pretrained weights file (local).
        dinov3_repo_path: Path or repo identifier passed to torch.hub.load (e.g., the
            directory containing the DINOv3 hubconf.py).
        resize: Target resize dimension for preprocessing.

    Returns:
        (model, transform): The loaded model (in eval mode) and the preprocessing transform.
    """
    weights_path = pathlib.Path(dinov3_weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"dinov3_weights_path does not exist: {weights_path}")

    model_name = _infer_model_name(weights_path)

    model = torch.hub.load(
        dinov3_repo_path,
        str(model_name),
        weights=str(weights_path),
        source="local",
    )
    model.eval()

    transform = _make_dino_transform(resize_size=resize)
    return model, transform


if __name__ == "__main__":
    # Example usage
    model, transform = get_dinov3(
        dinov3_weights_path="/home/richiewhite/.code/model_weights/dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        dinov3_repo_path="/home/richiewhite/.code/latent-space/repos/dinov3",
        resize=384,
    )
    print(model)
    print(transform)
