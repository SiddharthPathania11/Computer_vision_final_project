"""
Runs inference on a single image and prints class probabilities.
"""

import argparse

import torch
from PIL import Image

from dataset import CLASSES, get_transforms
from model import build_model, load_checkpoint


def predict(image_path: str, checkpoint: str, backbone: str = 'efficientnet_b0'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(backbone)
    model, _, _ = load_checkpoint(model, checkpoint, device)
    model = model.to(device).eval()

    transform = get_transforms('val')
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    results = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
    print(f'\nPrediction for: {image_path}')
    for cls, p in results:
        bar = '#' * int(p * 40)
        print(f'  {cls:<8} {p:.3f}  {bar}')
    print(f'\nTop prediction: {results[0][0]}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict weather class for an image')
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default='results/config2/best_model.pt')
    parser.add_argument('--backbone', default='efficientnet_b0')
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.backbone)
