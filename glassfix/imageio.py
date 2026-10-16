"""Image loading/saving: JPG/PNG/HEIC, EXIF rotation, working-resolution resize."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

try:  # HEIC support is optional (pillow-heif bundles LGPL libheif; see LICENSES.md)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as uint8 RGB with EXIF orientation applied."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"} and not HEIF_AVAILABLE:
        raise RuntimeError(f"{path.name}: HEIC support needs `pip install -r requirements-heic.txt`")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB")).copy()


def to_working(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the long side is <= max_side. Returns (image, scale) with scale <= 1."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img, 1.0
    size = (round(w * scale), round(h * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def resize_to(arr: np.ndarray, hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    if arr.shape[:2] == tuple(hw):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (hw[1], hw[0]), interpolation=interp)


def save_jpg(path: str | Path, rgb: np.ndarray, quality: int) -> None:
    Image.fromarray(rgb).save(path, quality=quality, subsampling=0)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    m = mask if mask.dtype == np.uint8 and mask.max() > 1 else (np.clip(mask, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(m).save(path)


def list_car_photos(root: str | Path) -> dict[str, list[Path]]:
    """test_photos/<car_id>/*.jpg -> {car_id: [paths]}. Loose files go under '_ungrouped'."""
    root = Path(root)
    cars: dict[str, list[Path]] = {}
    for p in sorted(root.iterdir()):
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if files:
                cars[p.name] = files
        elif p.suffix.lower() in IMAGE_EXTS:
            cars.setdefault("_ungrouped", []).append(p)
    return cars
