"""Weight quantization for CPU inference."""

import torch

from .model import SeraModel


def quantize_dynamic(model: SeraModel) -> SeraModel:
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)



