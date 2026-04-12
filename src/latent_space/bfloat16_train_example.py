import argparse

import mlflow
import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from latent_space import mlflow_helper


def get_args():
    parser = argparse.ArgumentParser(description="Train ViT on CIFAR10 with bfloat16 autocast")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--data-dir", type=str, default="./datasets/cifar")
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device, use_bfloat16):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Torch amp autocast for bfloat16
        # All you do is wrap your forward pass model() and loss criterion
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device, use_bfloat16):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def main() -> None:
    args = get_args()
    mlflow.log_params(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_bfloat16 = True
    best_eval_acc = float("-inf")
    sample_input = next(iter(train_loader))[0][:1].to(device)

    for epoch in trange(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, use_bfloat16)
        eval_loss, eval_acc = evaluate(model, test_loader, criterion, device, use_bfloat16)

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            mlflow_helper.log_model(model, sample_input, name=f"checkpoint_{epoch}")

        mlflow.log_metrics(
            {"train_loss": train_loss, "test_loss": eval_loss, "test_acc": eval_acc},
            step=epoch,
        )


if __name__ == "__main__":
    experiment_name = "Test Expr 2"
    mlflow_helper.setup(experiment_name=experiment_name)
    mlflow_helper.test_connection()

    with mlflow.start_run() as run:
        main()
