from __future__ import annotations

from pathlib import Path

import torch
from core.vision_encoder import pe, transforms
from PIL import Image, ImageDraw


class PEEncoder:
    """Wraps a PE VisionTransformer for single-image and tiled feature extraction."""

    DEFAULT_CONFIG = "PE-Lang-L14-448-Tiling"

    def __init__(
        self,
        config: str = DEFAULT_CONFIG,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pe.VisionTransformer.from_config(config, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)

    @staticmethod
    def available_configs() -> list[str]:
        """Return all available PE VisionTransformer config names."""
        return pe.VisionTransformer.available_configs()

    def encode(self, image: str | Path | Image.Image) -> torch.Tensor:
        """Encode a single image (no tiling) and return patch tokens.

        Args:
            image: File path or a pre-loaded PIL image.

        Returns:
            Tensor of shape [1, num_patches, embed_dim].

        """
        pil = self._load(image)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            return self.model.forward_features(tensor, strip_cls_token=True)

    def encode_tiled(
        self,
        image: str | Path | Image.Image,
        tile_size: int = 448,
        max_tiles: int = 6,
    ) -> torch.Tensor:
        """Encode a high-res image by splitting it into tiles plus a thumbnail."""
        pil = self._load(image)
        all_tiles = self._build_tiles(pil, tile_size=tile_size, max_tiles=max_tiles)
        with torch.no_grad(), torch.autocast(self.device):
            return self.model.forward_features(all_tiles, strip_cls_token=True)

    def save_tiling_collage(
        self,
        image: str | Path | Image.Image,
        output_path: str | Path = "tiling_collage.png",
        tile_size: int = 448,
        max_tiles: int = 6,
    ) -> None:
        """Save a diagnostic collage showing the tiling grid for a given image."""
        pil = self._load(image)
        output_path = Path(output_path)
        cols, rows, boxes, thumbnail, tiles = self._compute_tiling(
            pil,
            tile_size=tile_size,
            max_tiles=max_tiles,
        )

        # Annotate original with grid overlay
        annotated = pil.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, pil.width // 200)
        for idx, box in enumerate(boxes):
            draw.rectangle(box, outline=(255, 80, 80), width=line_width)
            draw.text((box[0] + 6, box[1] + 4), f"T{idx + 1}", fill=(255, 80, 80))

        # Layout constants
        pad, label_h = 8, 20
        cell_w = tile_size + pad * 2
        cell_h = tile_size + label_h + pad * 2

        scale = (tile_size * rows) / pil.height
        orig_w, orig_h = int(pil.width * scale), int(pil.height * scale)
        original_scaled = annotated.resize((orig_w, orig_h))

        total_w = max(orig_w + cell_w + pad, cols * cell_w)
        total_h = orig_h + rows * cell_h + pad * 3 + label_h * 2
        collage = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        cdraw = ImageDraw.Draw(collage)

        def paste_cell(img: Image.Image, x: int, y: int, label: str) -> None:
            collage.paste(img, (x + pad, y + label_h))
            cdraw.text((x + pad, y + 2), label, fill=(200, 200, 200))

        # Original + header
        collage.paste(original_scaled, (pad, label_h))
        cdraw.text(
            (pad, 2),
            f"Original ({pil.width}x{pil.height})  →  {rows}x{cols} grid + thumbnail",
            fill=(200, 200, 200),
        )

        # Thumbnail next to original
        paste_cell(thumbnail, orig_w + pad, 0, "Tile 0 (thumbnail)")

        # Grid tiles below
        y_start = orig_h + label_h + pad * 2
        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            paste_cell(
                tile,
                c * cell_w,
                y_start + r * cell_h,
                f"Tile {idx + 1}  (row {r}, col {c})",
            )

        collage.save(output_path)
        print(
            f"Saved tiling collage → {output_path} "
            f"({cols}x{rows} grid, {len(tiles) + 1} total tiles)",
        )

    @staticmethod
    def _load(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(image)

    def _compute_tiling(
        self,
        pil: Image.Image,
        tile_size: int,
        max_tiles: int,
    ) -> tuple[int, int, list[tuple[int, int, int, int]], Image.Image, list[Image.Image]]:
        """Return (cols, rows, boxes, thumbnail, tiles) for the given image."""
        W, H = pil.size
        cols = max(1, round((W / H) ** 0.5 * max_tiles**0.5))
        rows = max(1, round((H / W) ** 0.5 * max_tiles**0.5))
        tile_w, tile_h = W // cols, H // rows

        thumbnail = pil.resize((tile_size, tile_size))
        boxes, tiles = [], []
        for r in range(rows):
            for c in range(cols):
                box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
                boxes.append(box)
                tiles.append(pil.crop(box).resize((tile_size, tile_size)))

        return cols, rows, boxes, thumbnail, tiles

    def _build_tiles(
        self,
        pil: Image.Image,
        tile_size: int,
        max_tiles: int,
    ) -> torch.Tensor:
        """Return a batched tensor of [thumbnail, *grid_tiles] ready for the model."""
        _, _, _, thumbnail, tiles = self._compute_tiling(pil, tile_size, max_tiles)
        tensors = [self.preprocess(t).unsqueeze(0).to(self.device) for t in [thumbnail, *tiles]]
        return torch.cat(tensors, dim=0)
