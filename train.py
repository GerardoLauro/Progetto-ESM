from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any
import os
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torchvision.utils import save_image
import yaml

from dataset import create_dataloader, scan_images
from losses import WaveletSRLoss
from metrics import psnr, ssim, LPIPSMetric, ArcFaceSimilarity
from models.waveletsrnet import HaarWaveletTransform, WaveletSRNet


# Gestisce gli argomenti passati da riga di comando quando avvii lo script
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modern Wavelet-SRNet")
    parser.add_argument("--config", type=str, default="configs/config_8x.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# Carica i parametri di configurazione (es. learning rate, batch size) dal file YAML
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# Fissa i seed casuali per garantire che gli esperimenti siano riproducibili
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Inizializza l'architettura della rete neurale leggendo i parametri dal config
def build_model(cfg: dict[str, Any]) -> WaveletSRNet:
    model_cfg = cfg.get("model", {})
    return WaveletSRNet(
        levels=model_cfg.get("wavelet_levels", 3),
        image_channels=model_cfg.get("image_channels", 3),
        embedding_channels=tuple(model_cfg.get("embedding_channels", [64, 128, 256, 512, 1024])),
        num_embedding_blocks=model_cfg.get("num_embedding_blocks", 2),
        wavelet_channels=model_cfg.get("wavelet_channels", 32),
        num_branch_blocks=model_cfg.get("num_branch_blocks", 1),
    )


# Ripristina un addestramento interrotto ricaricando i pesi e lo stato dell'ottimizzatore
def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


# Salva i progressi del modello su disco (pesi e ottimizzatore)
def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


# Ciclo di validazione: disattiva il calcolo dei gradienti per risparmiare memoria
# Serve per testare oggettivamente il modello (calcolando PSNR e SSIM) su dati non visti
@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    wavelet_rec: HaarWaveletTransform,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_dir: Path,
    epoch: int,
    amp: bool,
    max_batches: int | None = None,
    lpips_metric: LPIPSMetric | None = None,
    arcface_metric: ArcFaceSimilarity | None = None,
) -> dict[str, float]:
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    arcface_values: list[float] = []
    sample_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        # Calcola le previsioni usando la mixed precision (più veloce su GPU moderne)
        with autocast(enabled=amp and device.type == "cuda"):
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

        # Salva un'immagine di esempio (confronto LR -> SR generata -> HR reale) del primo batch
        if batch_idx == 0:
            lr_up = torch.nn.functional.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            grid = torch.cat([lr_up[:4], sr[:4], hr[:4]], dim=0)
            save_image(grid, sample_dir / f"epoch_{epoch:04d}.png", nrow=4)

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    model.train()
    result: dict[str, float] = {
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
    }
    if lpips_metric is not None:
        result["lpips"] = float(np.mean(lpips_values)) if lpips_values else 0.0
    if arcface_metric is not None:
        result["arcface"] = float(np.mean(arcface_values)) if arcface_values else 0.0
    return result


# Funzione principale: orchestra l'intero processo di training
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    device = torch.device(args.device)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    out_dir = Path(cfg.get("output_dir", "results"))
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"

    # Calcola il numero di livelli Wavelet in base al fattore di scala (es. 8x -> 3 livelli)
    scale = int(data_cfg.get("scale", 8))
    levels = int(round(math.log2(scale)))
    if 2**levels != scale:
        raise ValueError("scale must be a power of two for Haar packet reconstruction")

    # --- LOGICA DI AUTO-SPLIT ---
    dataset_root = data_cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("Specificare 'dataset_root' nel config per usare l'auto-split.")
    
    all_images = scan_images(dataset_root)
    # Mescola le immagini in modo riproducibile (grazie a seed_everything all'inizio del file)
    random.shuffle(all_images)
    
    val_split = float(data_cfg.get("val_split", 0.1))
    test_split = float(data_cfg.get("test_split", 0.1))
    val_size = int(len(all_images) * val_split)
    test_size = int(len(all_images) * test_split)
    
    val_list = all_images[:val_size]
    test_list = all_images[val_size:val_size + test_size]
    train_list = all_images[val_size + test_size:]
    print(f"Auto-split completato: {len(train_list)} per train, {len(val_list)} per validazione, {len(test_list)} per test.")

    # Prepara i caricatoli dei dati (dataloader) per train e validation
    train_loader = create_dataloader(
        root=dataset_root,
        image_list=train_list,
        max_images=data_cfg.get("train_max_images"),
        hr_size=data_cfg.get("hr_size", 128),
        scale=scale,
        resize_before_crop=data_cfg.get("resize_before_crop"),
        crop_size=data_cfg.get("crop_size"),
        random_crop=True,
        hflip=True,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
    )
    val_loader = create_dataloader(
        root=dataset_root,
        image_list=val_list,
        max_images=data_cfg.get("val_max_images"),
        hr_size=data_cfg.get("hr_size", 128),
        scale=scale,
        resize_before_crop=data_cfg.get("resize_before_crop"),
        crop_size=None,
        random_crop=False,
        hflip=False,
        batch_size=train_cfg.get("val_batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    cfg.setdefault("model", {})["wavelet_levels"] = levels
    wavelet_weights_path = cfg.get("wavelet_weights_path") or cfg.get("model", {}).get("wavelet_weights_path")
    model = build_model(cfg).to(device)
    # Abilita il training multi-GPU se disponibile
    if torch.cuda.device_count() > 1 and train_cfg.get("data_parallel", False):
        model = nn.DataParallel(model)

    # Inizializza i moduli matematici per calcolare e invertire la trasformata Wavelet
    wavelet_dec = HaarWaveletTransform(levels, inverse=False, params_path=wavelet_weights_path).to(device)
    wavelet_rec = HaarWaveletTransform(levels, inverse=True, params_path=wavelet_weights_path).to(device)
    if wavelet_weights_path is not None:
        print(f"Using wavelet weights from: {wavelet_weights_path}")
    # Inizializza la nostra Loss personalizzata (Wavelet, Image, Texture e Perceptual)
    criterion = WaveletSRLoss(**cfg.get("loss", {})).to(device)

    # Configura l'algoritmo di ottimizzazione (Adam) che aggiornerà i pesi
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
    )
    scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    # --- Metrics (optional LPIPS / ArcFace) ---
    metrics_cfg = cfg.get("metrics", {})
    lpips_metric = None
    arcface_metric = None
    if metrics_cfg.get("lpips", {}).get("enabled", False):
        try:
            lpips_net = metrics_cfg.get("lpips", {}).get("net", "alex")
            lpips_metric = LPIPSMetric(net=lpips_net, device=device)
            print("LPIPS metric enabled.")
        except Exception as exc:  # pylint: disable=broad-except
            print("LPIPS metric not available:", exc)

    if metrics_cfg.get("arcface", {}).get("enabled", False):
        backbone_path = metrics_cfg.get("arcface", {}).get("backbone")
        backbone_url = metrics_cfg.get("arcface", {}).get("backbone_url")
        if backbone_path:
            # If the file doesn't exist but a URL is provided, try to download it automatically
            if not Path(backbone_path).exists() and backbone_url:
                try:
                    print(f"Downloading ArcFace backbone from {backbone_url} -> {backbone_path}")
                    os.makedirs(Path(backbone_path).parent, exist_ok=True)
                    urllib.request.urlretrieve(backbone_url, backbone_path)
                except Exception as exc:  # pylint: disable=broad-except
                    print("Failed to download ArcFace backbone:", exc)

            try:
                backbone = torch.load(backbone_path, map_location=device)
                if isinstance(backbone, nn.Module):
                    arcface_metric = ArcFaceSimilarity(backbone, device=device)
                    print("ArcFace metric enabled.")
                else:
                    print("Loaded ArcFace backbone is not an nn.Module; skipping arcface metric.")
            except Exception as exc:  # pylint: disable=broad-except
                print("Failed to load ArcFace backbone:", exc)
        else:
            print("ArcFace enabled but no backbone path provided in config; skipping ArcFace metric.")

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume, device, optimizer) + 1

    epochs = int(train_cfg.get("epochs", 100))
    log_every = int(train_cfg.get("log_every", 20))
    val_every = int(train_cfg.get("val_every", 1))
    save_every = int(train_cfg.get("save_every", 5))
    max_val_batches = train_cfg.get("max_val_batches")
    global_step = 0

    # --- LOOP DELLE EPOCHE ---
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        start = time.time()
        # --- LOOP DEI BATCH ---
        for batch_idx, batch in enumerate(train_loader):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            # Azzera i gradienti accumulati al giro precedente
            optimizer.zero_grad(set_to_none=True)
            # Forward pass: calcola le predizioni
            with autocast(enabled=scaler.is_enabled()):
                target_wavelets = wavelet_dec(hr)
                pred_wavelets = model(lr)
                pred_image = wavelet_rec(pred_wavelets)
                loss_out = criterion(pred_wavelets, target_wavelets, pred_image, hr)

            # Backward pass: calcola i gradienti (di quanto modificare i pesi) per minimizzare la loss
            scaler.scale(loss_out.total).backward()
            if train_cfg.get("grad_clip_norm") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            # Applica le modifiche ai pesi
            scaler.step(optimizer)
            scaler.update()

            # Stampa a schermo le statistiche di tanto in tanto
            global_step += 1
            if global_step % log_every == 0:
                elapsed = time.time() - start
                print(
                    f"epoch={epoch:03d} step={global_step:07d} "
                    f"loss={loss_out.total.item():.4f} low={loss_out.low.item():.4f} "
                    f"high={loss_out.high.item():.4f} texture={loss_out.texture.item():.4f} "
                    f"image={loss_out.image.item():.4f} perceptual={loss_out.perceptual.item():.4f} "
                    f"time={elapsed:.1f}s"
                )

        # A fine epoca, valuta il modello sui dati di test
        if epoch % val_every == 0:
            metrics = validate(
                model=model,
                wavelet_rec=wavelet_rec,
                loader=val_loader,
                device=device,
                sample_dir=sample_dir,
                epoch=epoch,
                amp=scaler.is_enabled(),
                max_batches=max_val_batches,
                lpips_metric=lpips_metric,
                arcface_metric=arcface_metric,
            )
            msg = f"validation epoch={epoch:03d} PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f}"
            if "lpips" in metrics:
                msg += f" LPIPS={metrics['lpips']:.4f}"
            if "arcface" in metrics:
                msg += f" ArcFace={metrics['arcface']:.4f}"
            print(msg)

        # Salva un checkpoint del modello periodicamente
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                path=ckpt_dir / f"waveletsrnet_epoch_{epoch:04d}.pth",
            )


if __name__ == "__main__":
    main()
