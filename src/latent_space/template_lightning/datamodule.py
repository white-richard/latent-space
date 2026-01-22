from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .config import DataConfig


def _resolve_dataset(config: DataConfig) -> type[Dataset[Any]]:
    """Return the dataset class associated with the provided config."""
    dataset_key = (config.dataset_name or "").lower()

    if dataset_key in {"cifar10", "cifar"} and not config.use_cifar100:
        return datasets.CIFAR10
    if dataset_key in {"cifar100"} or config.use_cifar100:
        return datasets.CIFAR100

    raise NotImplementedError(
        "This template currently supports CIFAR-style datasets out of the box. "
        "Provide your own datamodule or extend `_resolve_dataset` for other datasets."
    )


def _build_transforms(config: DataConfig, is_train: bool) -> T.Compose:
    """Construct a torchvision transform pipeline from the config."""
    transform_ops: list[Any] = []
    norm = T.Normalize(config.normalization_mean, config.normalization_std)
    if isinstance(config.image_size, int):
        config.image_size = (config.image_size, config.image_size)

    if is_train:
        transform_ops.extend(
            [
                T.RandomResizedCrop(
                    size=config.image_size,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandAugment(),
                T.ColorJitter(
                    brightness=config.jitter,
                    contrast=config.jitter,
                    saturation=config.jitter,
                    hue=min(config.jitter, 0.5),
                ),
                T.ToTensor(),
                norm,
                T.RandomErasing(p=config.reprob),
            ]
        )
    else:
        transform_ops.extend(
            [
                T.ToTensor(),
                norm,
            ]
        )

    return T.Compose(transform_ops)


class GenericLightningDataModule(pl.LightningDataModule):
    """Lightning DataModule that defaults to CIFAR but can be extended."""

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    def prepare_data(self) -> None:
        """Download/prepare data if needed."""
        dataset_cls = _resolve_dataset(self.config)
        dataset_cls(
            root=self.config.data_dir,
            train=True,
            download=True,
        )
        dataset_cls(
            root=self.config.data_dir,
            train=False,
            download=True,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create dataset instances for the requested stage."""
        dataset_cls = _resolve_dataset(self.config)

        if stage in {None, "fit"}:
            self.train_dataset = dataset_cls(
                root=self.config.data_dir,
                train=True,
                download=False,
                transform=_build_transforms(self.config, is_train=True),
            )
            self.val_dataset = dataset_cls(
                root=self.config.data_dir,
                train=False,
                download=False,
                transform=_build_transforms(self.config, is_train=False),
            )

        if stage in {None, "test"}:
            self.test_dataset = dataset_cls(
                root=self.config.data_dir,
                train=False,
                download=False,
                transform=_build_transforms(self.config, is_train=False),
            )

    def _dataloader(
        self,
        dataset: Dataset[Any] | None,
        shuffle: bool,
    ) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError("Attempted to create a dataloader before calling `setup`.")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.test_dataset, shuffle=False)
