from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss based on VGG-19 feature maps.
    
    Extracts features from the early/mid layers of VGG-19 to penalize 
    blurry textures and structural differences, pushing for photorealism.
    """

    def __init__(self, feature_layers: tuple[int, ...] = (2, 7, 12, 21, 30)) -> None:
        super().__init__()
        self.feature_layers = feature_layers
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        self.slices = nn.ModuleList()
        start_idx = 0
        for end_idx in self.feature_layers:
            self.slices.append(vgg[start_idx : end_idx])
            start_idx = end_idx
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, pred_image: Tensor, target_image: Tensor) -> Tensor:
        loss = 0.0
        x = (pred_image - self.mean) / self.std
        y = (target_image - self.mean) / self.std
        
        for slice_module in self.slices:
            x = slice_module(x)
            with torch.no_grad():
                y = slice_module(y)
            loss += F.mse_loss(x, y)
            
        return loss