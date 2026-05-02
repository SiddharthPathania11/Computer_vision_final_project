"""
Training script for the weather classifier.
Configs 1-4 control augmentation and synthetic data inclusion.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, save_checkpoint
from utils import plot_curves

CONFIG_NAMES = {
    1: 'Baseline',
    2: 'Augmentation',
    3: 'Synthesis',
    4: 'Synthesis + Augmentation',
}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return loss_sum / n, correct / n


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.config, args.batch_size, args.num_workers
    )
    print(
        f'Config {args.config} ({CONFIG_NAMES[args.config]}): '
        f'{len(train_loader.dataset)} train | {len(val_loader.dataset)} val'
    )

    model = build_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.results_dir) / f'config{args.config}'
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        print(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'train loss {tr_loss:.4f} acc {tr_acc:.3f}  '
            f'val loss {va_loss:.4f} acc {va_acc:.3f}'
        )

        if va_acc > best_val:
            best_val = va_acc
            save_checkpoint(model, epoch, va_acc, args.config,
                            out_dir / 'best_model.pt')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_curves(history, out_dir / 'curves.png',
                title=f'Config {args.config}: {CONFIG_NAMES[args.config]}')
    print(f'Best val acc: {best_val:.4f}  |  results -> {out_dir}')
    return history, best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train weather classifier')
    parser.add_argument('--config', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--backbone', default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    train(parser.parse_args())
