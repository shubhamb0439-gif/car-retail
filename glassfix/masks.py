"""Glass-mask cleanup and feathering (OpenCV morphology)."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

Box = Sequence[float]  # x0, y0, x1, y1


def expand_box(box: Box, frac: float, hw: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    h, w = hw
    return (max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(np.ceil(x1 + dx))), min(h, int(np.ceil(y1 + dy))))


def clip_to_box(mask: np.ndarray, box: Box, expand: float) -> np.ndarray:
    """Zero the mask outside its prompt box (grown by `expand`) so SAM spill-over is cut off."""
    x0, y0, x1, y1 = expand_box(box, expand, mask.shape[:2])
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def _ellipse(px: int) -> np.ndarray:
    k = max(1, int(px)) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(raw: np.ndarray, car_boxes: list[Box], cfg: dict[str, Any]) -> np.ndarray:
    """Close gaps, remove specks, fill holes, and keep only regions plausibly on a car.

    Returns a uint8 mask with values 0/1.
    """
    m = (raw > 0).astype(np.uint8)
    if cfg["close_px"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(cfg["close_px"]))
    if cfg["open_px"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(cfg["open_px"]))
    if cfg["fill_holes"]:
        m = binary_fill_holes(m).astype(np.uint8)

    h, w = m.shape
    car_zone = None
    if car_boxes:
        car_zone = np.zeros_like(m)
        for b in car_boxes:
            x0, y0, x1, y1 = expand_box(b, cfg["car_box_expand"], (h, w))
            car_zone[y0:y1, x0:x1] = 1

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    min_area = cfg["min_region_frac"] * h * w
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if car_zone is not None:
            inside = car_zone[labels == i].mean()
            if inside < cfg["min_inside_car"]:
                continue
        keep[i] = True
    return keep[labels].astype(np.uint8)


def feather(mask: np.ndarray, px: float) -> np.ndarray:
    """Inward-only feather: alpha ramps 0->1 over `px` pixels inside the mask, exactly 0 outside."""
    m = (mask > 0).astype(np.uint8)
    if px <= 0:
        return m.astype(np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return np.clip(dist / float(px), 0.0, 1.0).astype(np.float32)


def components(mask: np.ndarray) -> list[np.ndarray]:
    """Split a binary mask into per-region boolean masks (largest first)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [labels == i for i in order]
