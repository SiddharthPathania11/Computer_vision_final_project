import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_curves(history: dict, save_path, title: str = 'Training') -> None:
    """Save a two-panel loss/accuracy curve figure."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path,
    title: str = 'Confusion Matrix',
) -> None:
    """Save side-by-side count and normalised confusion matrices."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    ax1.set_title(f'{title} (counts)')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
    ax2.set_title(f'{title} (row-normalised)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_confusion_matrices(
    cms: list[np.ndarray],
    titles: list[str],
    class_names: list[str],
    save_path,
) -> None:
    """Save a row of normalised confusion matrices for comparison."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, cm, title in zip(axes, cms, titles):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_val_curves(results_dir, configs: list[int], save_path) -> None:
    """Overlay validation loss and accuracy for every config."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cfg in configs:
        hist_path = Path(results_dir) / f'config{cfg}' / 'history.json'
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = range(1, len(h['val_acc']) + 1)
        ax1.plot(epochs, h['val_loss'], label=f'Config {cfg}')
        ax2.plot(epochs, h['val_acc'], label=f'Config {cfg}')

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Validation Loss - All Configs'); ax1.legend(); ax1.grid(True)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy - All Configs'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_robustness_results(results: dict, save_path) -> None:
    """Bar chart of accuracy per perturbation, grouped by config."""
    perturbations = list(results.keys())
    configs = sorted({k for v in results.values() for k in v})

    x = np.arange(len(perturbations))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cfg in enumerate(configs):
        accs = [results[p].get(cfg, 0.0) for p in perturbations]
        ax.bar(x + i * width, accs, width, label=f'Config {cfg}')

    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels([p.replace('_', '\n') for p in perturbations], fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Robustness Suite Results')
    ax.legend(); ax.set_ylim(0, 1); ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
