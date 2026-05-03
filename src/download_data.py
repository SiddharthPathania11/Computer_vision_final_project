"""
Downloads the Kaggle weather dataset and organises images into the project
class folders. Requires kaggle CLI and ~/.kaggle/kaggle.json.
Note: the dataset has no snowy class, so those must be added manually.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CLASSES = ['clear', 'cloudy', 'foggy', 'rainy', 'snowy']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Kaggle dataset slug -> {source_folder: project_class}
# jehanbhathena/weather-dataset has 11 classes; we map them to our 5
DATASETS = [
    {
        'slug': 'jehanbhathena/weather-dataset',
        'class_map': {
            'dew': 'clear',
            'rainbow': 'clear',
            'lightning': 'clear',
            'fog smog': 'foggy',
            'rain': 'rainy',
            'hail': 'rainy',
            'snow': 'snowy',
            'frost': 'snowy',
            'rime': 'snowy',
            'glaze': 'snowy',
            'sandstorm': 'foggy',
        },
    },
]


def check_kaggle() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        print('ERROR: kaggle not installed.  Run: pip install kaggle')
        return False


def download_and_unzip(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', slug, '-p', str(dest), '--unzip']
    print(f'Downloading {slug} ...')
    subprocess.run(cmd, check=True)


def organise(src_root: Path, dst_root: Path, class_map: dict) -> int:
    # find all matching folders anywhere in the extracted tree
    moved = 0
    for src_name, dst_cls in class_map.items():
        matches = [p for p in src_root.rglob(src_name) if p.is_dir()]
        if not matches:
            continue
        dst_dir = dst_root / dst_cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_dir in matches:
            for img in src_dir.iterdir():
                if img.suffix.lower() in EXTENSIONS:
                    dst = dst_dir / img.name
                    if not dst.exists():
                        shutil.copy2(img, dst)
                        moved += 1
    return moved


def main(args):
    if not check_kaggle():
        sys.exit(1)

    out_dir = Path(args.output_dir)
    tmp_dir = out_dir / '_download_tmp'

    for ds in DATASETS:
        try:
            download_and_unzip(ds['slug'], tmp_dir)
            n = organise(tmp_dir, out_dir, ds['class_map'])
            print(f'Copied {n} images -> {out_dir}/')
        except subprocess.CalledProcessError as exc:
            print(f'Download failed: {exc}')

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print('\nNext step: python src/prepare_data.py --data-root data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Kaggle weather dataset')
    parser.add_argument('--output-dir', default='data/raw')
    parser.add_argument('--keep-tmp', action='store_true')
    main(parser.parse_args())
