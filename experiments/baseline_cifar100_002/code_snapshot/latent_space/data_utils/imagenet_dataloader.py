from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Subset


def get_imagenet_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transforms: list = None,
    limit_data: bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    train_imagenet_data = torchvision.datasets.ImageNet(
        data_dir,
        split="train",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                train_transforms if train_transforms is not None else None,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    if limit_data:
        train_imagenet_data = limit_dataset_with_stratified_sampling(train_imagenet_data)

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
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    if limit_data:
        val_imagenet_data = limit_dataset_with_stratified_sampling(val_imagenet_data)

    val_loader = torch.utils.data.DataLoader(
        val_imagenet_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = {"train": train_loader, "val": val_loader}
    return loaders


def limit_dataset_with_stratified_sampling(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.Dataset:
    targets = torch.tensor(dataset.targets)
    fraction = 0.1
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label.item()].append(idx)
    subset_indices = []
    for label, idxs in class_to_indices.items():
        idxs = torch.tensor(idxs)
        n = max(1, int(len(idxs) * fraction))
        perm = torch.randperm(len(idxs))[:n]
        subset_indices.extend(idxs[perm].tolist())
    return Subset(dataset, subset_indices)


if __name__ == "__main__":
    data_dir = "/home/richw/.code/datasets/imagenet"
    loaders = get_imagenet_loaders(data_dir, batch_size=16, num_workers=4, limit_val=True)
    val_loader = loaders["val"]

    for images, labels in val_loader:
        print(images.shape, labels.shape)
        break
