from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """Cosine identity loss for face SR.

    Pass an already-loaded ArcFace/face-recognition backbone that maps images to
    embeddings. When no backbone is supplied, the module returns zero, making it
    safe to keep in configs before adding ArcFace.
    """

    def __init__(self, backbone: Optional[nn.Module] = None, *, weight: float = 1.0, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.weight = float(weight)
        if self.backbone is not None and freeze:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, pred_image: Tensor, target_image: Tensor) -> Tensor:
        if self.backbone is None or self.weight == 0.0:
            return pred_image.new_zeros(())

        pred_features = F.normalize(self.backbone(pred_image), dim=1)
        with torch.no_grad():
            target_features = F.normalize(self.backbone(target_image), dim=1)
        loss = 1.0 - torch.sum(pred_features * target_features, dim=1).mean()
        return loss * self.weight
