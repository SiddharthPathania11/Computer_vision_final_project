import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

CLASSES = ['clear', 'cloudy', 'foggy', 'rainy', 'snowy']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, augment: bool = False) -> T.Compose:
    if split == 'train' and augment:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class WeatherDataset(Dataset):
    """
    Reads images listed in data/splits/{split}.json.
    Each JSON entry: {"path": "raw/clear/001.jpg", "label": "clear", "synthetic": false}
    """

    def __init__(
        self,
        root: str,
        split: str,
        augment: bool = False,
        include_synthetic: bool = False,
    ):
        self.root = Path(root)
        self.transform = get_transforms(split, augment=(split == 'train' and augment))

        split_file = self.root / 'splits' / f'{split}.json'
        if not split_file.exists():
            raise FileNotFoundError(
                f'Split file not found: {split_file}\n'
                'Run: python scripts/prepare_data.py --data-root data'
            )

        with open(split_file) as f:
            entries = json.load(f)

        self.samples: list[tuple[str, int]] = []
        for e in entries:
            if e.get('synthetic', False) and not include_synthetic:
                continue
            img_path = self.root / e['path']
            if img_path.exists():
                self.samples.append((str(img_path), CLASS_TO_IDX[e['label']]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def get_dataloaders(
    data_root: str,
    config_num: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for a given config number."""
    augment = config_num in (2, 4)
    include_synthetic = config_num in (3, 4)

    train_ds = WeatherDataset(data_root, 'train', augment=augment, include_synthetic=include_synthetic)
    val_ds = WeatherDataset(data_root, 'val', augment=False, include_synthetic=False)
    test_ds = WeatherDataset(data_root, 'test', augment=False, include_synthetic=False)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)
