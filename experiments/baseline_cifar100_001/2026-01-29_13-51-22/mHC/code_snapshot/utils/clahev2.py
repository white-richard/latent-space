from typing import Any

import torch
import numpy as np
import cv2
from pathlib import Path


class CLAHEv2(torch.nn.Module):
    """ Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to single-channel images. 
    Supports uint8 and uint16 input images, with output dtype choice.
    
    Can be used as a transform in torch transform v2 pipelines."""
    def __init__(
        self,
        clip_limit: float = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        output_dtype: torch.dtype = torch.uint8,
    ):
        super().__init__()
        allowed = {torch.uint8, torch.uint16}
        if output_dtype not in allowed:
            raise ValueError(f"output_dtype must be one of {allowed}")
        self.output_dtype = output_dtype
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def _apply_single(self, img: torch.Tensor) -> torch.Tensor:
        # Accept [H,W] or [1,H,W]
        add_channel_back = False
        if img.ndim == 3:
            if img.shape[0] != 1:
                raise ValueError(f"Expected shape [1,H,W] but got {tuple(img.shape)}")
            img = img[0]
            add_channel_back = True
        elif img.ndim != 2:
            raise ValueError(f"Expected 2D [H,W] or 3D [1,H,W] but got {img.ndim}D")

        # Torch -> NumPy (CPU)
        img_np = img.detach().cpu().numpy()

        # Ensure uint8 for CLAHE
        if img_np.dtype == np.uint16:
            img8 = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            img8 = img_np
        else:
            raise ValueError(f"Input tensor dtype must be uint8 or uint16 but got {img_np.dtype}")

        # Apply CLAHE (expects uint8 single-channel)
        img8_clahe = self._clahe.apply(img8)

        # Output dtype choice
        if self.output_dtype == torch.uint16:
            img_out = cv2.normalize(img8_clahe, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        else:
            img_out = img8_clahe  # uint8

        out = torch.from_numpy(img_out)

        if add_channel_back:
            out = out.unsqueeze(0)

        return out

    def forward(self, *inputs: Any) -> Any:
        # Typical v2 pattern: (img) or (img, target)
        if len(inputs) == 1:
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                return self._apply_single(x)
            if isinstance(x, (list, tuple)):
                return type(x)(self._apply_single(t) if isinstance(t, torch.Tensor) else t for t in x)
            return x

        # If (img, target, ...) style: only transform tensor-like items
        return tuple(self._apply_single(x) if isinstance(x, torch.Tensor) else x for x in inputs)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def imread_any(path: Path) -> np.ndarray | None:
    """
    Read image without changing bit depth.
    Returns:
      - grayscale uint8/uint16 array [H,W]
      - or None if unreadable
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # If color, convert to grayscale (CLAHE expects single-channel)
    if img.ndim == 3:
        # Handles BGR/BGRA
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Now img should be [H,W]
    return img

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Directory containing images")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--clip-limit", type=float, default=4.0)
    ap.add_argument("--tile", type=int, nargs=2, default=(8, 8), metavar=("W", "H"))
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"Not a directory: {in_dir}")

    out_dir = in_dir.parent / f"{in_dir.name}_clahe_uint8"
    out_dir.mkdir(parents=True, exist_ok=True)

    clahe = CLAHEv2(
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile[0], args.tile[1]),
        output_dtype=torch.uint8,
    )

    # Pick files
    it = in_dir.rglob("*") if args.recursive else in_dir.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    if not files:
        print(f"No images found in {in_dir}")
        return

    n_ok, n_skip = 0, 0
    for src in files:
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".png")  # save as PNG to preserve uint8 reliably
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = imread_any(src)
        if img is None:
            print(f"SKIP (unreadable): {src}")
            n_skip += 1
            continue

        if img.dtype not in (np.uint8, np.uint16):
            print(f"SKIP (dtype {img.dtype}): {src}")
            n_skip += 1
            continue

        t = torch.from_numpy(img)  # [H,W]
        t_out = clahe(t)
        out_np = t_out.numpy()

        ok = cv2.imwrite(str(dst), out_np)
        if not ok:
            print(f"SKIP (write failed): {dst}")
            n_skip += 1
            continue

        n_ok += 1

    print(f"Done. Wrote {n_ok} images to: {out_dir}")
    if n_skip:
        print(f"Skipped: {n_skip}")


if __name__ == "__main__":
    main()