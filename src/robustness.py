"""
Robustness evaluation suite -- applies four perturbations to the test set
and reports per-config accuracy for each one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD, WeatherDataset
from model import build_model, load_checkpoint
from utils import plot_robustness_results


def gaussian_blur(img: Image.Image) -> Image.Image:
    return TF.gaussian_blur(img, kernel_size=9, sigma=3.0)


def brightness_reduction(img: Image.Image) -> Image.Image:
    return TF.adjust_brightness(img, brightness_factor=0.5)


def rectangular_occlusion(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    rh, rw = h // 4, w // 4
    y = np.random.randint(0, h - rh)
    x = np.random.randint(0, w - rw)
    arr[y:y + rh, x:x + rw] = 128
    return Image.fromarray(arr)


def gaussian_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 25, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


PERTURBATIONS: dict[str, callable] = {
    'gaussian_blur': gaussian_blur,
    'brightness_reduction': brightness_reduction,
    'rectangular_occlusion': rectangular_occlusion,
    'gaussian_noise': gaussian_noise,
}


class PerturbedDataset(Dataset):
    """Wraps a WeatherDataset, applying a PIL-level perturbation before ToTensor."""

    _resize = T.Resize((224, 224))
    _to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def __init__(self, base: WeatherDataset, perturbation_fn: callable):
        self.samples = base.samples
        self.perturbation = perturbation_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self._resize(img)
        img = self.perturbation(img)
        return self._to_tensor(img), label


@torch.no_grad()
def eval_perturbed(model, dataset, device, batch_size=32) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return float(accuracy_score(labels, preds))


def run_robustness_suite(args) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {p: {} for p in PERTURBATIONS}

    for cfg in args.configs:
        ckpt_path = Path(args.results_dir) / f'config{cfg}' / 'best_model.pt'
        if not ckpt_path.exists():
            print(f'Config {cfg}: no checkpoint, skipping')
            continue

        model = build_model(args.backbone)
        model, _, _ = load_checkpoint(model, ckpt_path, device)
        model = model.to(device)

        base_test = WeatherDataset(args.data_root, 'test')

        for pert_name, pert_fn in PERTURBATIONS.items():
            acc = eval_perturbed(model, PerturbedDataset(base_test, pert_fn), device, args.batch_size)
            results[pert_name][cfg] = acc
            print(f'Config {cfg} | {pert_name:<25}: {acc:.4f}')

    out_dir = Path(args.results_dir)
    with open(out_dir / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_robustness_results(results, out_dir / 'robustness_plot.png')
    print(f'\nRobustness results -> {out_dir}/robustness_results.json')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robustness perturbation suite')
    parser.add_argument('--configs', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--backbone', default='efficientnet_b0')
    parser.add_argument('--batch-size', type=int, default=32)
    run_robustness_suite(parser.parse_args())
