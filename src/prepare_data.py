"""
Scan data/raw/ and data/captured/ for real images and data/synthetic/ for
synthetic images, then write stratified train/val/test split JSON files to
data/splits/.

Split JSON format (one file per split):
    [{"path": "raw/clear/001.jpg", "label": "clear", "synthetic": false}, ...]

Paths are relative to --data-root so the dataset loader can resolve them.
"""

import argparse
import json
import random
from pathlib import Path

CLASSES = ['clear', 'cloudy', 'foggy', 'rainy', 'snowy']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# 70 / 15 / 15 split on real images; all synthetic goes to train only
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def collect_images(source_dir: Path, label: str, synthetic: bool) -> list[dict]:
    if not source_dir.exists():
        return []
    entries = []
    for p in sorted(source_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            entries.append({
                'path': str(p.relative_to(source_dir.parent.parent)),
                'label': label,
                'synthetic': synthetic,
            })
    return entries


def stratified_split(entries: list[dict], seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for e in entries:
        by_class[e['label']].append(e)

    train, val, test = [], [], []
    for cls_entries in by_class.values():
        rng.shuffle(cls_entries)
        n = len(cls_entries)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(cls_entries[:n_train])
        val.extend(cls_entries[n_train:n_train + n_val])
        test.extend(cls_entries[n_train + n_val:])

    return train, val, test


def main(data_root: str, seed: int) -> None:
    root = Path(data_root)
    splits_dir = root / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    real_entries = []
    synthetic_entries = []

    for label in CLASSES:
        real_entries.extend(collect_images(root / 'raw' / label, label, synthetic=False))
        real_entries.extend(collect_images(root / 'captured' / label, label, synthetic=False))
        synthetic_entries.extend(collect_images(root / 'synthetic' / label, label, synthetic=True))

    train, val, test = stratified_split(real_entries, seed)
    train.extend(synthetic_entries)

    random.Random(seed).shuffle(train)

    splits = {'train': train, 'val': val, 'test': test}
    for name, entries in splits.items():
        out = splits_dir / f'{name}.json'
        with open(out, 'w') as f:
            json.dump(entries, f, indent=2)
        print(f'{name:5s}: {len(entries):5d} samples  ->  {out}')

    print(f'\nSplit files written to {splits_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare train/val/test split JSON files')
    parser.add_argument('--data-root', default='data', help='Root of the data directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args.data_root, args.seed)
