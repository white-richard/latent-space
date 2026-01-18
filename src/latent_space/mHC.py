"""
Implementation of mHC (https://arxiv.org/pdf/2512.24880)

x_{l+1} = H_l^{res} x_l + H_l^{post,T} F(H_l^{pre} x_l, W_l)

with the key constraints:

H_res: doubly stochastic (Birkhoff polytope; entries ≥ 0, rows sum to 1, cols sum to 1), via Sinkhorn-Knopp.
H_pre, H_post: non-negative mixing maps.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm.models


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="ConvMixer")

parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--scale", default=0.75, type=float)
parser.add_argument("--reprob", default=0.25, type=float)
parser.add_argument("--jitter", default=0.1, type=float)

parser.add_argument("--wd", default=0.01, type=float)
parser.add_argument("--epochs", default=75, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--workers", default=8, type=int)

args = parser.parse_args()


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=args.reprob),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
)


def lr_schedule(t):
    return np.interp(
        [t],
        [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
        [0, args.lr_max, args.lr_max / 20.0, 0],
    )[0]


opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1) / len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    print(
        f"[{args.name}] Epoch: {epoch} | Train Acc: {train_acc / n:.4f},"
        "Test Acc: {test_acc / m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}"
    )
