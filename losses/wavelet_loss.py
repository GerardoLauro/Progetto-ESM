from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .texture_loss import TextureLoss
from .perceptual_loss import VGGPerceptualLoss


@dataclass
class WaveletSRLossOutput:
    total: Tensor
    low: Tensor
    high: Tensor
    texture: Tensor
    image: Tensor
    perceptual: Tensor


class WaveletSRLoss(nn.Module):
    """Unified Wavelet-SRNet loss.

    This combines:
    - low-frequency wavelet MSE with a small weight;
    - high-frequency wavelet MSE;
    - texture loss on high-frequency bands;
    - full-image reconstruction MSE.
    - perceptual VGG-19 loss on the reconstructed image.
    """

    def __init__(
        self,
        *,
        image_channels: int = 3,
        lambda_low: float = 0.01,
        lambda_high: float = 1.0,
        texture_weight: float = 1.0,
        image_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        texture_alpha: float = 1.2,
        texture_margin: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        self.texture_weight = float(texture_weight)
        self.image_weight = float(image_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.reduction = reduction
        self.texture = TextureLoss(
            image_channels=image_channels,
            alpha=texture_alpha,
            margin=texture_margin,
        )
        self.perceptual = VGGPerceptualLoss() if self.perceptual_weight > 0.0 else None

    def _mse(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.reduction == "mean":
            return F.mse_loss(pred, target, reduction="mean")
        if self.reduction == "sum_batch":
            return torch.sum((pred - target) ** 2) / (pred.shape[0] * 2.0)
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(self, pred_wavelets: Tensor, target_wavelets: Tensor, pred_image: Tensor, target_image: Tensor) -> WaveletSRLossOutput:
        if pred_wavelets.shape != target_wavelets.shape:
            raise ValueError(f"Wavelet shape mismatch: {pred_wavelets.shape} vs {target_wavelets.shape}")
        low_pred = pred_wavelets[:, : self.image_channels]
        low_target = target_wavelets[:, : self.image_channels]
        high_pred = pred_wavelets[:, self.image_channels :]
        high_target = target_wavelets[:, self.image_channels :]

        low = self._mse(low_pred, low_target)
        high = self._mse(high_pred, high_target)
        texture = self.texture(high_pred, high_target) if high_pred.numel() > 0 else high.new_zeros(())
        image = self._mse(pred_image, target_image)
        
        perceptual = high.new_zeros(())
        if self.perceptual is not None and self.perceptual_weight > 0.0:
            perceptual = self.perceptual(pred_image, target_image)
            
        total = (
            self.lambda_low * low
            + self.lambda_high * high
            + self.texture_weight * texture
            + self.image_weight * image
            + self.perceptual_weight * perceptual
        )
        return WaveletSRLossOutput(total=total, low=low, high=high, texture=texture, image=image, perceptual=perceptual)


def wavelet_prediction_loss(pred_wavelets: Tensor, target_wavelets: Tensor, *, image_channels: int = 3, lambda_low: float = 0.01, lambda_high: float = 1.0) -> Tensor:
    low = F.mse_loss(pred_wavelets[:, :image_channels], target_wavelets[:, :image_channels])
    high = F.mse_loss(pred_wavelets[:, image_channels:], target_wavelets[:, image_channels:])
    return lambda_low * low + lambda_high * high
