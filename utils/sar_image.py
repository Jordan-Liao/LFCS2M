"""SAR image input/output helpers for LFCS2M inference."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_dir: str | Path) -> List[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def _to_three_channels(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return array


def _linear_normalize(array: np.ndarray) -> np.ndarray:
    return array / 127.5 - 1.0


def _sar_mean_clip_normalize(array: np.ndarray) -> np.ndarray:
    # Paper-style SAR magnitude normalization: non-positive values map to -1,
    # values above the image mean map to 1, and the rest are linearly scaled.
    mean_val = float(np.mean(array[array > 0])) if np.any(array > 0) else 1.0
    mean_val = max(mean_val, 1e-6)
    normalized = 2.0 * array / mean_val - 1.0
    normalized[array <= 0] = -1.0
    normalized[array >= mean_val] = 1.0
    return normalized


def load_sar_image(path: str | Path, image_size: int = 256, normalization: str = "linear") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None and image_size > 0:
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    array = np.asarray(image).astype(np.float32)
    array = _to_three_channels(array)

    if normalization == "linear":
        array = _linear_normalize(array)
    elif normalization == "sar_mean_clip":
        array = _sar_mean_clip_normalize(array)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).float()


def save_sar_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("save_sar_image expects a single image tensor")
        tensor = tensor[0]
    array = tensor.permute(1, 2, 0).numpy()
    array = ((array + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    if array.shape[-1] == 3 and np.allclose(array[..., 0], array[..., 1]) and np.allclose(array[..., 1], array[..., 2]):
        array = array[..., 0]
    Image.fromarray(array).save(path)
