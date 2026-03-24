import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from latent_space.process_anyres_image import (
    PatchMergeStrategy,
    divide_to_patches,
    merge_patch_features,
    process_anyres_image,
    resize_and_pad_image,
    select_best_resolution,
)

mpl.use("Agg")

# ── config ────────────────────────────────────────────────────────────────────
PRESAMPLE_SIZE = (4000, 2000)
batch_size = 64
learning_rate = 3e-4
epochs = 10
collage_samples = 4  # how many images to show in the collage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ── model ─────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
model.to(device)

encoder_input_size = model.default_cfg["input_size"][1]  # 224

grid_pinpoints = [
    (encoder_input_size * 1, encoder_input_size * 1),
    (encoder_input_size * 2, encoder_input_size * 1),
    (encoder_input_size * 1, encoder_input_size * 2),
    (encoder_input_size * 2, encoder_input_size * 2),
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

patch_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ],
)

# ── denormalization ────────────────────────────────────────────────────────────
_mean = torch.tensor(MEAN)[:, None, None]
_std = torch.tensor(STD)[:, None, None]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) normalised tensor → (H, W, C) uint8 numpy array."""
    img = (tensor.cpu().float() * _std + _mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── visualization helpers ──────────────────────────────────────────────────────
def get_patching_intermediates(image: Image.Image):
    """Returns the intermediate states needed for the collage:
    - padded_with_grid  : PIL image at best resolution with grid lines drawn
    - patch_tensors     : list of (C, H, W) tensors, first is global context.
    """
    best_res = select_best_resolution(image.size, grid_pinpoints)
    padded = resize_and_pad_image(image, best_res)
    raw_patches = divide_to_patches(padded, encoder_input_size)

    # Draw grid lines on a copy of the padded image
    padded_grid = padded.copy()
    draw = ImageDraw.Draw(padded_grid)
    w, h = padded.size
    for x in range(0, w, encoder_input_size):
        draw.line([(x, 0), (x, h)], fill="red", width=4)
    for y in range(0, h, encoder_input_size):
        draw.line([(0, y), (w, y)], fill="red", width=4)

    # Global context thumbnail + patches, all as tensors
    global_thumb = image.resize((encoder_input_size, encoder_input_size))
    all_patches = [global_thumb, *raw_patches]
    patch_tensors = [patch_transform(p) for p in all_patches]

    return padded_grid, patch_tensors


def save_collage(dataset_raw, indices: list[int], path: str = "collage.png") -> None:
    """Saves a three-column collage:
    col 1 — original image at PRESAMPLE_SIZE
    col 2 — padded image with patch grid overlaid
    col 3+ — each patch as the model sees it (denormalized).
    """
    max_patches = 5  # global + up to 4 grid patches
    n_cols = 2 + max_patches
    n_rows = len(indices)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.3},
    )
    # Always 2-D array of axes
    if n_rows == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, label = dataset_raw[idx]  # PIL image (already 1200×900)
        class_name = CIFAR10_CLASSES[label]

        padded_grid, patch_tensors = get_patching_intermediates(image)

        # Col 0: original
        ax = axes[row, 0]
        ax.imshow(image)
        ax.set_title(f"original\n{class_name}", fontsize=8)
        ax.axis("off")

        # Col 1: padded + grid
        ax = axes[row, 1]
        ax.imshow(padded_grid)
        ax.set_title(f"patched\n{padded_grid.size}", fontsize=8)
        ax.axis("off")

        # Cols 2+: individual patches as model sees them
        for col_offset, t in enumerate(patch_tensors[:max_patches]):
            ax = axes[row, 2 + col_offset]
            ax.imshow(denormalize(t))
            label_str = "global" if col_offset == 0 else f"patch {col_offset}"
            ax.set_title(label_str, fontsize=8)
            ax.axis("off")

        # Hide unused patch columns
        for col_offset in range(len(patch_tensors), max_patches):
            axes[row, 2 + col_offset].axis("off")

    # Column headers on the first row
    for col, title in enumerate(
        ["original", "grid overlay"] + [f"model view {i}" for i in range(max_patches)],
    ):
        axes[0, col].set_xlabel(title, fontsize=7, labelpad=2)

    fig.suptitle("AnyRes patching visualisation", fontsize=11, y=1.01)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Collage saved → {path}")


# ── dataset ───────────────────────────────────────────────────────────────────
class AnyResCIFAR10(Dataset):
    def __init__(self, train: bool) -> None:
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.resize(PRESAMPLE_SIZE, Image.LANCZOS)  # ← 1200×900
        image_patches = process_anyres_image(
            image,
            patch_transform,
            grid_pinpoints,
            encoder_input_size,
        )
        return image_patches, label


train_dataset = AnyResCIFAR10(train=True)
test_dataset = AnyResCIFAR10(train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ── training loop ──────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
MERGE_STRATEGY = PatchMergeStrategy.POOL

for epoch in trange(epochs):
    model.train()
    running_loss = 0.0

    for image_patches, labels in tqdm(train_loader):
        B, N, C, H, W = image_patches.shape
        labels = labels.to(device)
        flat_patches = image_patches.view(B * N, C, H, W).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            patch_features = model.forward_features(flat_patches)  # (B*N, D)
            patch_features = patch_features[:, 0]  # (B*N, 192)

            patch_features = patch_features.view(B, N, -1)  # (B,   N, D)
            merged = merge_patch_features(patch_features, MERGE_STRATEGY)
            logits = model.head(merged)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if DEBUG:
            break

    if DEBUG:
        break

    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

# ── collage ───────────────────────────────────────────────────────────────────
# Use the raw dataset (no patch transform) so we have PIL images for display
raw_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=None)


# Wrap it so images are pre-resized to 1200×900, matching training
class ResizedRawCIFAR10(Dataset):
    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img.resize(PRESAMPLE_SIZE, Image.LANCZOS), label


save_collage(
    dataset_raw=ResizedRawCIFAR10(raw_dataset),
    indices=list(range(collage_samples)),
    path="tmp/anyres_collage.png",
)
