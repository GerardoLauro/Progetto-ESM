from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class LPIPSMetric:
    """Optional LPIPS wrapper.

    Install ``lpips`` to enable it. Inputs are expected in [0, 1] and are
    internally mapped to [-1, 1].
    """

    def __init__(self, net: str = "vgg", device: Optional[torch.device | str] = None) -> None:
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError("LPIPS is optional. Install it with: pip install lpips") from exc

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        pred = pred.to(self.device).clamp(0, 1) * 2 - 1
        target = target.to(self.device).clamp(0, 1) * 2 - 1
        return self.model(pred, target).mean().item()
