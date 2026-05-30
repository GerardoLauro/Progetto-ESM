# WaveletSRNet modernized

Struttura PyTorch moderna per Wavelet-SRNet:

```text
WaveletSRNet/
├── train.py
├── test.py
├── dataset.py
├── models/
│   └── waveletsrnet.py
├── losses/
│   ├── wavelet_loss.py
│   ├── texture_loss.py
│   └── identity_loss.py
├── metrics/
│   ├── psnr_ssim.py
│   ├── lpips_metric.py
│   └── arcface_metric.py
├── configs/
│   └── config_8x.yaml
└── results/
```

## Setup

Puoi usare `conda` (scelta consigliata) oppure `venv` (Python standard).

**Opzione A: Conda**
```bash
conda create --name waveletsrnet python=3.10 -y
conda activate waveletsrnet
pip install torch torchvision pillow pyyaml numpy lpips
```

**Opzione B: venv**
```bash
python -m venv .venv
source .venv/bin/activate  # Su Windows usa: .venv\Scripts\activate
pip install torch torchvision pillow pyyaml numpy lpips
```

## Training 16x16 -> 128x128

Modifica i path in `configs/config_8x.yaml`, poi:

```bash
python train.py --config configs/config_8x.yaml
```

## Test

```bash
python test.py --config configs/config_8x.yaml \
  --checkpoint results/8x/checkpoints/waveletsrnet_epoch_0100.pth \
  --save_images
```

## Note

- `data.scale: 8` corrisponde a 3 livelli Haar, quindi 64 coefficienti wavelet per canale.
- Per `8x8 -> 64x64` tieni `scale: 8` e imposta `hr_size: 64`.
- Per `8x8 -> 128x128` usa `scale: 16` e `hr_size: 128`.
- `identity_loss.py` e `arcface_metric.py` sono wrapper: richiedono un backbone ArcFace caricato da te.

## Utilizzo su Google Colab

1. Carica l'intera cartella del progetto sul tuo Google Drive in formato `.zip` oppure sincronizzala.
2. Apri un nuovo notebook su Google Colab e imposta l'hardware su **GPU** (Runtime -> Cambia tipo di runtime -> T4 GPU).
3. Monta il tuo Drive eseguendo questa cella:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Spostati nella cartella del progetto:
   ```bash
   %cd /content/drive/MyDrive/percorso/della/cartella/WaveletSRNet_modernized
   ```
5. Installa le dipendenze extra (PyTorch è già incluso nativamente):
   ```bash
   !pip install lpips pyyaml
   ```
6. Avvia l'addestramento (assicurati che i percorsi nel file `configs/config_8x.yaml` siano corretti):
   ```bash
   !python train.py --config configs/config_8x.yaml
   ```
