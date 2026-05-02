from pathlib import Path

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 5


def build_model(
    backbone: str = 'efficientnet_b0',
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create a timm model with a freshly initialised classification head."""
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(model: nn.Module, epoch: int, val_acc: float, config: int, path):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_acc': val_acc,
            'config': config,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module, path, device='cpu'
) -> tuple[nn.Module, int, float]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model, state.get('epoch', 0), state.get('best_val_acc', 0.0)
