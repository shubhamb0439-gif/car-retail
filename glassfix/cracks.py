"""Crack detection on the high-frequency layer with scikit-image ridge filters."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.filters import frangi, sato
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .layers import masked_gaussian


def _robust_z(resp: np.ndarray, inside: np.ndarray) -> np.ndarray:
    vals = resp[inside]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    return (resp - med) / (mad + 1e-6)


def _ridge_z(high: np.ndarray, inside: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    sigmas = list(cfg["sigmas"])
    filters = {"sato": [sato], "frangi": [frangi], "both": [sato, frangi]}[cfg["method"]]
    polarities = [b for b, on in ((False, cfg["detect_bright"]), (True, cfg["detect_dark"])) if on]
    z = np.full(high.shape, -np.inf, dtype=np.float32)
    for f in filters:
        for black in polarities:
            r = f(high, sigmas=sigmas, black_ridges=black, mode="reflect").astype(np.float32)
            z = np.maximum(z, _robust_z(r, inside))
    return z


def detect_cracks(high: np.ndarray, mask: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (crack_mask uint8 0/1, info). Only thin, long components inside the glass survive."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    info = {"crack_detected": False, "crack_length_px": 0, "crack_components": 0}

    margin = int(cfg["edge_margin_px"])
    inner = (mask > 0).astype(np.uint8)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        inner = cv2.erode(inner, k)
    if inner.sum() < 50:
        return out, info

    # Work on the bounding box of the glass only (ridge filters are the slowest step).
    ys, xs = np.nonzero(inner)
    pad = int(max(cfg["sigmas"]) * 4) + 2
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    hi = high[y0:y1, x0:x1].astype(np.float32)
    inside = inner[y0:y1, x0:x1] > 0

    z = _ridge_z(hi, inside, cfg)
    cand = inside & (z > cfg["threshold_z"]) & (np.abs(hi) >= cfg["min_contrast"])
    if cfg["local_ratio"] > 0:
        # A crack is a thin line on otherwise smooth glass. Interior detail seen through the glass is
        # busy texture, so require the pixel to stand out against its own neighbourhood's energy.
        local_rms = np.sqrt(masked_gaussian(hi * hi, inside, cfg["local_sigma_px"]))
        cand &= np.abs(hi) >= cfg["local_ratio"] * local_rms

    keep = np.zeros_like(cand)
    total_len = 0
    n_kept = 0
    for region in regionprops(label(cand, connectivity=2)):
        if region.area < cfg["min_size_px"]:
            continue
        length = int(skeletonize(region.image).sum())
        if length < cfg["min_length_px"]:
            continue
        if region.area / max(length, 1) > cfg["max_width_px"]:
            continue  # blob, not a line
        r0, c0, r1, c1 = region.bbox
        keep[r0:r1, c0:c1] |= region.image
        total_len += length
        n_kept += 1

    crack = keep.astype(np.uint8)
    if cfg["dilate_px"] > 0 and n_kept:
        d = int(cfg["dilate_px"])
        crack = cv2.dilate(crack, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
        crack &= (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    out[y0:y1, x0:x1] = crack

    info.update(
        crack_detected=bool(total_len >= cfg["min_total_length_px"]),
        crack_length_px=int(total_len),
        crack_components=n_kept,
    )
    return out, info
