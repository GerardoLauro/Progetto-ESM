# WaveletSRNet modernized

This version is organized for Python 3 and modern PyTorch.

## Original wavelet weights

The project includes the original `wavelet_weights_c2.pkl` in:

```text
weights/wavelet_weights_c2.pkl
```

The config uses it by default:

```yaml
wavelet_weights_path: weights/wavelet_weights_c2.pkl
```

These are fixed Haar wavelet filters used by the decomposition and reconstruction transforms. They are **not** trained CNN checkpoint weights.

For the first baseline, keep:

```yaml
data:
  hr_size: 128
  scale: 8
```

This corresponds to `16x16 -> 128x128` and uses the `rec8` filters.

## Training

```bash
pip install torch torchvision pillow pyyaml numpy
python train.py --config configs/config_8x.yaml
```

## Testing

```bash
python test.py \
  --config configs/config_8x.yaml \
  --checkpoint results/8x/checkpoints/waveletsrnet_epoch_0005.pth \
  --save_images
```
