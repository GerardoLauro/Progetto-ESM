from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TextureLoss(nn.Module):
    """Texture loss from Wavelet-SRNet.

    Penalizes high-frequency predicted wavelet coefficients whose energy is too
    small compared with the target coefficients.
    """

    def __init__(self, *, image_channels: int = 3, alpha: float = 1.2, margin: float = 0.0) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.alpha = float(alpha)
        self.margin = float(margin)

    def forward(self, pred_high: Tensor, target_high: Tensor) -> Tensor:
        if pred_high.shape != target_high.shape:
            raise ValueError(f"Shape mismatch: {pred_high.shape} vs {target_high.shape}")
        if pred_high.shape[1] % self.image_channels != 0:
            raise ValueError("Channel dimension must be divisible by image_channels")

        pred = pred_high.contiguous().view(
            pred_high.shape[0], -1, self.image_channels, pred_high.shape[-2], pred_high.shape[-1]
        )
        target = target_high.contiguous().view_as(pred)
        pred_energy = torch.sum(pred * pred, dim=2)
        target_energy = torch.sum(target * target, dim=2)
        return F.relu(target_energy.mul(self.alpha) - pred_energy + self.margin).mean()


def texture_loss(pred_high: Tensor, target_high: Tensor, *, image_channels: int = 3, alpha: float = 1.2, margin: float = 0.0) -> Tensor:
    return TextureLoss(image_channels=image_channels, alpha=alpha, margin=margin)(pred_high, target_high)
