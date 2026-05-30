from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image
import yaml
import os
import urllib.request

from dataset import create_dataloader, scan_images
from metrics import psnr, ssim, LPIPSMetric, ArcFaceSimilarity
from models.waveletsrnet import HaarWaveletTransform, WaveletSRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate modern Wavelet-SRNet")
    parser.add_argument("--config", type=str, default="configs/config_8x.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg: dict[str, Any], levels: int) -> WaveletSRNet:
    model_cfg = cfg.get("model", {})
    return WaveletSRNet(
        levels=levels,
        image_channels=model_cfg.get("image_channels", 3),
        embedding_channels=tuple(model_cfg.get("embedding_channels", [64, 128, 256, 512, 1024])),
        num_embedding_blocks=model_cfg.get("num_embedding_blocks", 2),
        wavelet_channels=model_cfg.get("wavelet_channels", 32),
        num_branch_blocks=model_cfg.get("num_branch_blocks", 1),
    )


def load_model(path: str | Path, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)


def safe_name(path: str, index: int) -> str:
    stem = Path(path).stem if path else f"sample_{index:06d}"
    return f"{index:06d}_{stem}.png"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    # Fissa lo stesso seed del training per avere lo stesso identico ordine di shuffle
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    
    data_cfg = cfg["data"]
    test_cfg = cfg.get("test", {})
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = int(data_cfg.get("scale", 8))
    levels = int(round(math.log2(scale)))
    if 2**levels != scale:
        raise ValueError("scale must be a power of two")

    # --- LOGICA DI AUTO-SPLIT ---
    dataset_root = data_cfg.get("dataset_root")
    if dataset_root is not None:
        all_images = scan_images(dataset_root)
        random.shuffle(all_images)
        val_split = float(data_cfg.get("val_split", 0.1))
        test_split = float(data_cfg.get("test_split", 0.1))
        val_size = int(len(all_images) * val_split)
        test_size = int(len(all_images) * test_split)
        test_list = all_images[val_size:val_size + test_size]
        test_root = dataset_root
        test_list_path = None
        print(f"Auto-split: usa {len(test_list)} immagini per il test.")
    else:
        test_list = None
        test_root = data_cfg.get("test_root", data_cfg.get("val_root"))
        test_list_path = data_cfg.get("test_list", data_cfg.get("val_list"))

    loader = create_dataloader(
        root=test_root,
        image_list=test_list,
        list_path=test_list_path,
        max_images=data_cfg.get("test_max_images"),
        hr_size=data_cfg.get("hr_size", 128),
        scale=scale,
        resize_before_crop=data_cfg.get("resize_before_crop"),
        crop_size=None,
        random_crop=False,
        hflip=False,
        batch_size=test_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=test_cfg.get("num_workers", 4),
    )

    wavelet_weights_path = cfg.get("wavelet_weights_path") or cfg.get("model", {}).get("wavelet_weights_path")
    model = build_model(cfg, levels).to(device).eval()
    load_model(args.checkpoint, model, device)
    wavelet_rec = HaarWaveletTransform(levels, inverse=True, params_path=wavelet_weights_path).to(device)
    if wavelet_weights_path is not None:
        print(f"Using wavelet weights from: {wavelet_weights_path}")

    # Optional metrics
    metrics_cfg = cfg.get("metrics", {})
    lpips_metric = None
    arcface_metric = None
    if metrics_cfg.get("lpips", {}).get("enabled", False):
        try:
            lpips_net = metrics_cfg.get("lpips", {}).get("net", "alex")
            lpips_metric = LPIPSMetric(net=lpips_net, device=device)
            print("LPIPS metric enabled for test.")
        except Exception as exc:
            print("LPIPS not available for test:", exc)
    if metrics_cfg.get("arcface", {}).get("enabled", False):
        backbone_path = metrics_cfg.get("arcface", {}).get("backbone")
        backbone_url = metrics_cfg.get("arcface", {}).get("backbone_url")
        if backbone_path:
            if not Path(backbone_path).exists() and backbone_url:
                try:
                    print(f"Downloading ArcFace backbone from {backbone_url} -> {backbone_path}")
                    os.makedirs(Path(backbone_path).parent, exist_ok=True)
                    urllib.request.urlretrieve(backbone_url, backbone_path)
                except Exception as exc:
                    print("Failed to download ArcFace backbone for test:", exc)
            try:
                backbone = torch.load(backbone_path, map_location=device)
                if isinstance(backbone, torch.nn.Module):
                    arcface_metric = ArcFaceSimilarity(backbone, device=device)
                    print("ArcFace metric enabled for test.")
                else:
                    print("Loaded ArcFace backbone is not an nn.Module; skipping arcface metric.")
            except Exception as exc:
                print("Failed to load ArcFace backbone for test:", exc)
        else:
            print("ArcFace enabled but no backbone path provided in config; skipping ArcFace metric.")

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    arcface_values: list[float] = []
    image_index = 0

    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        pred_wavelets = model(lr)
        sr = wavelet_rec(pred_wavelets).clamp(0, 1)

        psnr_values.append(psnr(sr.float(), hr.float(), luminance=True))
        ssim_values.append(ssim(sr.float(), hr.float(), luminance=False))
        if lpips_metric is not None:
            try:
                lpips_values.append(lpips_metric(sr, hr))
            except Exception:
                lpips_values.append(0.0)
        if arcface_metric is not None:
            try:
                arcface_values.append(arcface_metric(sr, hr))
            except Exception:
                arcface_values.append(0.0)

        if args.save_images:
            for item_idx in range(sr.shape[0]):
                save_image(sr[item_idx], output_dir / safe_name(str(batch["path"][item_idx]), image_index))
                image_index += 1

    print(f"PSNR={float(np.mean(psnr_values)):.3f} dB")
    print(f"SSIM={float(np.mean(ssim_values)):.4f}")
    if lpips_metric is not None:
        print(f"LPIPS={float(np.mean(lpips_values)):.4f}")
    if arcface_metric is not None:
        print(f"ArcFace={float(np.mean(arcface_values)):.4f}")


if __name__ == "__main__":
    main()
