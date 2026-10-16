"""Split the glass L channel (LAB) into low-frequency (reflections/glare) and high-frequency (detail/cracks)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or float(0..1) RGB -> float32 LAB with L in 0..100."""
    f = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
    return cv2.cvtColor(f, cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 LAB -> float32 RGB clipped to 0..1."""
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)


def masked_gaussian(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised-convolution blur: only pixels inside the mask contribute (no bleed from the car body)."""
    m = (mask > 0).astype(np.float32)
    num = cv2.GaussianBlur(img * m, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    out = img.copy()
    ok = den > 1e-4
    out[ok] = num[ok] / den[ok]
    return out


def separate(L: np.ndarray, mask: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) with low + high == L inside the mask and high == 0 outside."""
    L = L.astype(np.float32)
    long_side = max(L.shape)
    sigma = max(1.0, cfg["sigma_frac"] * long_side)
    low = masked_gaussian(L, mask, sigma)

    if cfg["method"] == "guided":
        # Edge-preserving low-pass keeps glare edges in the low layer. Outside-mask pixels are replaced
        # by the masked blur so the car body does not leak in.
        filled = np.where(mask > 0, L, low) / 100.0
        radius = max(1, int(round(cfg["guided_radius_frac"] * long_side)))
        low = cv2.ximgproc.guidedFilter(filled, filled, radius, float(cfg["guided_eps"])) * 100.0
    elif cfg["method"] != "gaussian":
        raise ValueError(f"layers.method must be gaussian or guided, got {cfg['method']!r}")

    inside = mask > 0
    high = np.where(inside, L - low, 0.0).astype(np.float32)
    low = np.where(inside, low, L).astype(np.float32)
    return low, high
