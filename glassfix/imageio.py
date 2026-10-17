"""Image loading (JPG/PNG/WEBP, optional HEIC) with EXIF rotation, and car-folder listing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

try:  # HEIC (iPhone) support is optional: pillow-heif bundles LGPL libheif, see LICENSES.md
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


def load_image(path: str | Path) -> Image.Image:
    """Open an image with its EXIF orientation applied."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"} and not HEIF_AVAILABLE:
        raise ValueError(f"{path.name}: HEIC support needs `pip install -r requirements-heic.txt`")
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).copy()


def list_car_photos(root: str | Path) -> dict[str, list[Path]]:
    """<root>/<car_id>/*.jpg -> {car_id: [paths]}. Loose files go under '_ungrouped'."""
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
