import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .config import DataConfig


class CIFARDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR dataset."""

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        # cifar normalization constants
        self.cifar_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_std = (0.2471, 0.2435, 0.2616)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """Setup datasets for each stage."""

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, scale=(self.config.scale, 1.0), ratio=(1.0, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std),
                transforms.RandomErasing(),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std),
            ]
        )
        use_cifar100 = self.config.use_cifar100

        DatasetClass = (
            torchvision.datasets.CIFAR100 if use_cifar100 else torchvision.datasets.CIFAR10
        )

        if stage == "fit" or stage is None:
            self.train_dataset = DatasetClass(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=train_transform,
            )

            self.val_dataset = DatasetClass(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=test_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DatasetClass(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=test_transform,
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
