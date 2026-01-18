#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "dinov3"))

from dinov3.models.lvit import lvit_base  # noqa: E402


def main() -> None:
    model = lvit_base(
        embed_dim=768+12,
        pos_embed_type="rope",
    )
    model.init_weights()
    model.eval()

    global_crop = torch.randn(2, 3, 518, 518)
    local_crop = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model.forward_features([global_crop, local_crop], [None, None])

    for name, out in zip(["global", "local"], outputs):
        print(f"{name} x_norm_clstoken: {tuple(out['x_norm_clstoken'].shape)}")
        print(f"{name} x_norm_patchtokens: {tuple(out['x_norm_patchtokens'].shape)}")


if __name__ == "__main__":
    main()
