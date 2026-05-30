from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def rgb_to_y(x: Tensor) -> Tensor:
    """Convert RGB image tensor in [0, 1] to luminance Y."""
    if x.shape[1] != 3:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


@torch.no_grad()
def psnr(pred: Tensor, target: Tensor, *, data_range: float = 1.0, luminance: bool = True) -> float:
    pred = pred.detach().clamp(0, data_range)
    target = target.detach().clamp(0, data_range)
    if luminance:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    return window


@torch.no_grad()
def ssim(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    luminance: bool = False,
) -> float:
    pred = pred.detach().clamp(0, data_range)
    target = target.detach().clamp(0, data_range)
    if luminance:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return score.mean().item()
