from dataclasses import dataclass

import torch
import torchvision

@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader

def get_imagenet_loaders(
    data_dir: str,
    *
    train_transforms: list,
    eval_transforms: list,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoaders:
    train_imagenet_data = torchvision.datasets.ImageNet(
        data_dir,
        split="train",
        transform=train_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_imagenet_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_imagenet_data = torchvision.datasets.ImageNet(
        data_dir,
        split="val",
        transform=eval_transforms
    )

    val_loader = torch.utils.data.DataLoader(
        val_imagenet_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = DataLoaders(train=train_loader, val=val_loader)

    return loaders

if __name__ == "__main__":
    data_dir = "~/.code/datasets/imagenet"
    train_transform, val_transform = torch.nn.Identity(), torch.nn.Identity()
    loaders = get_imagenet_loaders(data_dir, train_transforms=train_transform, eval_transforms=val_transform, batch_size=16, num_workers=0)
    val_loader = loaders["val"]

    for images, labels in val_loader:
        print(images.shape, labels.shape)
        break
