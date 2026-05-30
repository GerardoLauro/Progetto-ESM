from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ArcFaceSimilarity:
    """Cosine similarity wrapper for an ArcFace-compatible backbone.

    The backbone must return one embedding per image. This class deliberately
    does not download weights; pass your own verified ArcFace model.
    """

    def __init__(self, backbone: nn.Module, *, device: Optional[torch.device | str] = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        pred_features = F.normalize(self.backbone(pred.to(self.device)), dim=1)
        target_features = F.normalize(self.backbone(target.to(self.device)), dim=1)
        return torch.sum(pred_features * target_features, dim=1).mean().item()
