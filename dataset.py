from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    BICUBIC = Image.BICUBIC

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image_list(list_path: Optional[str | Path], *, max_images: Optional[int] = None) -> list[str]:
    """Load image paths from a text file.

    Each line may contain ``path`` or ``path label``; labels are ignored.
    """
    if list_path is None:
        return []
    list_path = Path(list_path)
    paths: list[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            paths.append(line.split()[0])
            if max_images is not None and len(paths) >= max_images:
                break
    return paths


def scan_images(root: str | Path) -> list[str]:
    root = Path(root)
    return [str(path.relative_to(root)) for path in sorted(root.rglob("*")) if is_image_file(path)]


def _pair(value: int | Iterable[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    value = tuple(int(v) for v in value)
    if len(value) != 2:
        raise ValueError("Expected int or pair")
    return value[0], value[1]


class FaceSRDataset(Dataset[dict[str, Tensor | str]]):
    """Image dataset for face super-resolution.

    Pipeline:
    1. open RGB image;
    2. optional resize before crop;
    3. random or center crop;
    4. resize to HR size;
    5. bicubic downsample to LR size.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        image_list: Optional[Iterable[str]] = None,
        list_path: Optional[str | Path] = None,
        max_images: Optional[int] = None,
        hr_size: int | Iterable[int] = 128,
        scale: int = 8,
        resize_before_crop: int | Iterable[int] | None = None,
        crop_size: int | Iterable[int] | None = None,
        random_crop: bool = True,
        hflip: bool = True,
        scale_back: bool = False,
    ) -> None:
        self.root = Path(root)
        if image_list is None:
            loaded = load_image_list(list_path, max_images=max_images)
            self.image_paths = loaded if loaded else scan_images(self.root)
        else:
            self.image_paths = list(image_list)
            if max_images is not None:
                self.image_paths = self.image_paths[:max_images]

        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")

        self.hr_size = _pair(hr_size)
        assert self.hr_size is not None
        self.scale = int(scale)
        self.resize_before_crop = _pair(resize_before_crop)
        self.crop_size = _pair(crop_size)
        self.random_crop = bool(random_crop)
        self.hflip = bool(hflip)
        self.scale_back = bool(scale_back)

        if self.hr_size[0] % self.scale != 0 or self.hr_size[1] % self.scale != 0:
            raise ValueError(f"hr_size={self.hr_size} must be divisible by scale={self.scale}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _crop(self, image: Image.Image) -> Image.Image:
        if self.crop_size is None:
            return image
        crop_w, crop_h = self.crop_size[1], self.crop_size[0]
        width, height = image.size
        if width < crop_w or height < crop_h:
            image = ImageOps.fit(image, (max(width, crop_w), max(height, crop_h)), method=BICUBIC)
            width, height = image.size

        if self.random_crop:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
        else:
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        rel_path = self.image_paths[index]
        image_path = Path(rel_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path

        image = Image.open(image_path).convert("RGB")
        if self.hflip and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.resize_before_crop is not None:
            image = image.resize((self.resize_before_crop[1], self.resize_before_crop[0]), BICUBIC)
        image = self._crop(image)

        hr_h, hr_w = self.hr_size
        hr = image.resize((hr_w, hr_h), BICUBIC)
        lr = hr.resize((hr_w // self.scale, hr_h // self.scale), BICUBIC)
        if self.scale_back:
            lr = lr.resize((hr_w, hr_h), BICUBIC)

        return {
            "lr": TF.to_tensor(lr),
            "hr": TF.to_tensor(hr),
            "path": str(rel_path),
        }


def create_dataloader(
    *,
    root: str | Path,
    image_list: Optional[Iterable[str]] = None,
    list_path: Optional[str | Path] = None,
    max_images: Optional[int] = None,
    hr_size: int | Iterable[int] = 128,
    scale: int = 8,
    resize_before_crop: int | Iterable[int] | None = None,
    crop_size: int | Iterable[int] | None = None,
    random_crop: bool = True,
    hflip: bool = True,
    scale_back: bool = False,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = FaceSRDataset(
        root=root,
        image_list=image_list,
        list_path=list_path,
        max_images=max_images,
        hr_size=hr_size,
        scale=scale,
        resize_before_crop=resize_before_crop,
        crop_size=crop_size,
        random_crop=random_crop,
        hflip=hflip,
        scale_back=scale_back,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
