"""
Builds train/val/test split JSON files from the organised image directories.
Captured images go to val/test only; synthetic images are train-only.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CLASSES = ['clear', 'cloudy', 'foggy', 'rainy', 'snowy']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def collect_real(data_root: Path) -> tuple[list, list]:
    """Return (seed_pool_entries, captured_entries) - synthetic excluded."""
    seed_pool, captured = [], []
    for source, bucket in [('raw', seed_pool), ('captured', captured)]:
        src_dir = data_root / source
        if not src_dir.exists():
            continue
        for cls in CLASSES:
            for img in sorted((src_dir / cls).glob('*')):
                if img.suffix.lower() in EXTENSIONS:
                    bucket.append({
                        'path': str(img.relative_to(data_root)),
                        'label': cls,
                        'source': source,
                        'synthetic': False,
                    })
    return seed_pool, captured


def collect_synthetic(data_root: Path) -> list:
    syn_dir = data_root / 'synthetic'
    entries = []
    if not syn_dir.exists():
        return entries
    for cls in CLASSES:
        for img in sorted((syn_dir / cls).glob('*')):
            if img.suffix.lower() in EXTENSIONS:
                entries.append({
                    'path': str(img.relative_to(data_root)),
                    'label': cls,
                    'source': 'synthetic',
                    'synthetic': True,
                })
    return entries


def stratified_split(entries: list, train_r: float, val_r: float, rng: random.Random):
    by_class = defaultdict(list)
    for e in entries:
        by_class[e['label']].append(e)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * train_r)
        n_va = int(n * val_r)
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr + n_va])
        test.extend(items[n_tr + n_va:])
    return train, val, test


def make_splits(data_root: str, train_r: float = 0.70, val_r: float = 0.15, seed: int = 42):
    data_root = Path(data_root)
    rng = random.Random(seed)

    seed_pool, captured = collect_real(data_root)
    synthetic = collect_synthetic(data_root)

    # Stratified split on the seed pool
    tr, va, te = stratified_split(seed_pool, train_r, val_r, rng)

    # Self-captured images go to val / test only (deliberate distribution shift)
    by_cls_cap = defaultdict(list)
    for e in captured:
        by_cls_cap[e['label']].append(e)
    for cls, items in by_cls_cap.items():
        rng.shuffle(items)
        half = len(items) // 2
        va.extend(items[:half])
        te.extend(items[half:])

    # Synthetic entries go into train (WeatherDataset filters via include_synthetic flag)
    tr.extend(synthetic)

    splits_dir = data_root / 'splits'
    splits_dir.mkdir(exist_ok=True)

    summary = {}
    for name, entries in [('train', tr), ('val', va), ('test', te)]:
        with open(splits_dir / f'{name}.json', 'w') as f:
            json.dump(entries, f, indent=2)
        # per-class breakdown
        by_cls = defaultdict(int)
        for e in entries:
            by_cls[e['label']] += 1
        summary[name] = dict(by_cls)
        print(f'{name:5s}: {len(entries):4d} images  {dict(by_cls)}')

    print(f'\nSplit files written to {splits_dir}/')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare train/val/test splits')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    make_splits(args.data_root, args.train_ratio, args.val_ratio, args.seed)
