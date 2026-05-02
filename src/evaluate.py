"""
Loads saved checkpoints and evaluates them on the held-out test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASSES, get_dataloaders
from model import build_model, load_checkpoint
from utils import compare_confusion_matrices, plot_confusion_matrix


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_config(config_num, data_root, results_dir, backbone, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(data_root, config_num, batch_size)

    ckpt_path = Path(results_dir) / f'config{config_num}' / 'best_model.pt'
    model = build_model(backbone)
    model, epoch, best_val_acc = load_checkpoint(model, ckpt_path, device)
    model = model.to(device)

    preds, labels = get_predictions(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True)

    out_dir = Path(results_dir) / f'config{config_num}'
    plot_confusion_matrix(
        cm, CLASSES,
        out_dir / 'confusion_matrix.png',
        title=f'Config {config_num} - Test Set',
    )
    with open(out_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nConfig {config_num} | epoch of best ckpt: {epoch} | '
          f'best val acc: {best_val_acc:.4f} | test acc: {report["accuracy"]:.4f}')
    print(classification_report(labels, preds, target_names=CLASSES))
    return cm, report


def main(args):
    all_cms, done_configs = [], []

    for cfg in args.configs:
        ckpt = Path(args.results_dir) / f'config{cfg}' / 'best_model.pt'
        if not ckpt.exists():
            print(f'Config {cfg}: no checkpoint at {ckpt}, skipping')
            continue
        cm, report = evaluate_config(
            cfg, args.data_root, args.results_dir, args.backbone, args.batch_size
        )
        all_cms.append(cm)
        done_configs.append(cfg)

    if len(all_cms) > 1:
        compare_confusion_matrices(
            all_cms,
            [f'Config {c}' for c in done_configs],
            CLASSES,
            Path(args.results_dir) / 'confusion_matrix_comparison.png',
        )
        print(f'\nComparison figure -> {args.results_dir}/confusion_matrix_comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained weather classifiers')
    parser.add_argument('--configs', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--backbone', default='efficientnet_b0')
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())
