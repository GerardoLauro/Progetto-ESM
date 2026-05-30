from .texture_loss import TextureLoss, texture_loss
from .wavelet_loss import WaveletSRLoss, WaveletSRLossOutput, wavelet_prediction_loss
from .identity_loss import IdentityLoss
from .perceptual_loss import VGGPerceptualLoss

__all__ = [
    "TextureLoss",
    "texture_loss",
    "WaveletSRLoss",
    "WaveletSRLossOutput",
    "wavelet_prediction_loss",
    "IdentityLoss",
    "VGGPerceptualLoss",
]
