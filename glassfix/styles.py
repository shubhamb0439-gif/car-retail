"""New-glass styles: 'tint' and 'studio'. Deterministic: same config -> same pixels."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .layers import lab_to_rgb, masked_gaussian, rgb_to_lab
from .masks import components


# ---------------------------------------------------------------- clear: keep the glass, remove reflections
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def reflection_veil(work_rgb: np.ndarray, mask: np.ndarray, clear_cfg: dict[str, Any]) -> np.ndarray:
    """Estimate the additive reflection/glare veil on the glass (linear light), at working resolution.

    A reflection adds light on top of what is behind the glass, so even the darkest pixel in a small
    neighbourhood still carries it (the dark-channel idea from image dehazing). The veil is that local
    lower envelope, smoothed. Thin bright details (interior edges, cracks) never survive the envelope,
    so they are not treated as reflection.

    veil_color "per_channel" estimates R, G, B separately, so a blue sky reflection is removed as blue
    light (a grey estimate would leave a saturated blue cast). Returns (H, W, 3) or (H, W, 1).
    """
    long_side = max(mask.shape)
    inside = mask > 0
    lin = srgb_to_linear(work_rgb.astype(np.float32) / 255.0)
    chans = [lin[..., c] for c in range(3)] if clear_cfg["veil_color"] == "per_channel" else [lin.min(axis=2)]
    k = max(3, int(round(clear_cfg["min_window_frac"] * long_side)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Opening (erode then dilate) is a lower envelope that follows a sloping glare gradient without
    # the half-window shift a plain erode has.
    op = cv2.MORPH_OPEN if clear_cfg["envelope"] == "open" else cv2.MORPH_ERODE
    sigma = max(1.0, clear_cfg["veil_sigma_frac"] * long_side)
    veils = []
    for ch in chans:
        ch = np.where(inside, ch, 1.0).astype(np.float32)  # outside the glass never lowers the minimum
        ch = cv2.morphologyEx(ch, op, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1.0)
        veils.append(np.where(inside, masked_gaussian(ch, mask, sigma), 0.0))
    return np.stack(veils, axis=-1).astype(np.float32)


def remove_veil(rgb: np.ndarray, veil: np.ndarray, clear_cfg: dict[str, Any]) -> np.ndarray:
    """Subtract `strength` of the veil in linear light and restore some lost contrast. RGB float 0..1.

    `rgb` is uint8 (0..255) or float (0..1).
    """
    f = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
    lin = srgb_to_linear(f)
    v = float(clear_cfg["strength"]) * veil
    out = (lin - v) / np.maximum(1.0 - float(clear_cfg["restore_contrast"]) * v, 1e-3)
    return linear_to_srgb(out)


# ---------------------------------------------------------------- dereflect fallback (C)
def fallback_map(work_rgb: np.ndarray, reflection: np.ndarray, mask: np.ndarray,
                 fb_cfg: dict[str, Any]) -> np.ndarray:
    """Where the glass shows (almost) only reflection, or is blown out, nothing behind it was captured.

    Returns a 0..1 map (working resolution) of how much clean glass should replace the result there.
    """
    inside = mask > 0
    if not fb_cfg["enabled"]:
        return np.zeros(mask.shape, np.float32)
    sigma = max(1.0, fb_cfg["smooth_frac"] * max(mask.shape))
    lum = work_rgb.astype(np.float32).mean(axis=2) / 255.0
    r_lum = reflection.mean(axis=2)
    dominance = masked_gaussian(r_lum, mask, sigma) / np.maximum(masked_gaussian(lum, mask, sigma), 1e-3)
    f = np.maximum(_smoothstep(fb_cfg["dominance_low"], fb_cfg["dominance_high"], dominance),
                   _smoothstep(fb_cfg["glare_low"], fb_cfg["glare_high"], masked_gaussian(lum, mask, sigma)))
    f = masked_gaussian(f.astype(np.float32), mask, sigma)
    return np.where(inside, np.clip(f, 0.0, 1.0), 0.0).astype(np.float32)


# ---------------------------------------------------------------- studiowhite
def studio_glass_base(mask: np.ndarray, orig_L: np.ndarray, sw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per window: light studio glass, or dark glass if the window itself is tinted.

    Returns (RGB float base, per-pixel 'is clear window' 0/1 map), both at the mask's resolution.
    """
    out = np.zeros(mask.shape + (3,), np.float32)
    clear_map = np.zeros(mask.shape, np.float32)
    light = np.array(sw["clear_glass_rgb"], np.float32) / 255.0
    dark = np.array(sw["tinted_glass_rgb"], np.float32) / 255.0
    for region in components(mask):
        tinted = float(np.percentile(orig_L[region], 90)) < sw["tinted_p90_below_L"]
        col = dark if tinted else light
        ys = np.nonzero(region)[0]
        y0, y1 = ys.min(), ys.max()
        t = (np.arange(y0, y1 + 1, dtype=np.float32) - y0) / max(y1 - y0, 1)
        shade = 1.0 - sw["top_darken"] * (1.0 - t)          # slightly darker at the top
        rows = out[y0:y1 + 1]
        rows[region[y0:y1 + 1]] = (col[None, None, :] * shade[:, None, None]).repeat(rows.shape[1], 1)[region[y0:y1 + 1]]
        if not tinted:
            clear_map[region] = 1.0
    return out, clear_map


# ---------------------------------------------------------------- softbox asset
def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@lru_cache(maxsize=8)
def _softbox_cached(key: str) -> np.ndarray:
    cfg = json.loads(key)
    s = int(cfg["texture_size"])
    v, u = np.mgrid[0:s, 0:s].astype(np.float32) / (s - 1)
    a = np.deg2rad(cfg["angle_deg"])
    d = (u - 0.5) * np.cos(a) + (v - 0.5) * np.sin(a) + 0.5  # distance across the tilted strips
    tex = np.zeros((s, s), np.float32)
    for st in cfg["strips"]:
        half = st["width"] / 2
        tex += st["intensity"] * (1.0 - _smoothstep(half, half + st["softness"], np.abs(d - st["center"])))
    tex *= 1.0 - cfg["vertical_falloff"] * v
    tex = np.clip(tex, 0.0, 1.0)
    tex.setflags(write=False)
    return tex


def softbox_texture(studio_cfg: dict[str, Any]) -> np.ndarray:
    """Procedural studio-softbox highlight: soft white diagonal light strips on black, values 0..1."""
    keys = ("texture_size", "angle_deg", "vertical_falloff", "strips")
    return _softbox_cached(json.dumps({k: studio_cfg[k] for k in keys}, sort_keys=True))


# ---------------------------------------------------------------- geometry
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (image space)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def region_quad(region: np.ndarray, eps_frac: float) -> np.ndarray:
    """Fit a quadrilateral to a glass region: approxPolyDP if it gives a convex 4-gon, else minAreaRect."""
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, eps_frac * cv2.arcLength(cnt, True), True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
    ordered = _order_corners(quad)
    if len({tuple(p) for p in ordered.tolist()}) < 4:  # degenerate ordering -> fall back to rect
        ordered = _order_corners(cv2.boxPoints(cv2.minAreaRect(cnt)))
    return ordered


def studio_overlay(mask: np.ndarray, studio_cfg: dict[str, Any]) -> np.ndarray:
    """Warp the softbox texture onto every glass region's quad. Returns float32 (H, W) 0..1."""
    tex = softbox_texture(studio_cfg)
    s = tex.shape[0]
    src = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], np.float32)
    h, w = mask.shape
    out = np.zeros((h, w), np.float32)
    min_area = studio_cfg["min_region_frac"] * h * w
    for region in components(mask):
        if region.sum() < min_area:
            continue
        ys, xs = np.nonzero(region)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        quad = region_quad(region, studio_cfg["approx_eps_frac"]) - np.array([x0, y0], np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(tex, M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        crop = out[y0:y1, x0:x1]
        np.maximum(crop, warped * region[y0:y1, x0:x1], out=crop)
    return out


# ---------------------------------------------------------------- styles
def tint_base(mask: np.ndarray, tint_cfg: dict[str, Any]) -> np.ndarray:
    """Dark smoked tint with a vertical sky-to-ground gradient fitted to each region. RGB float 0..1."""
    top = np.array(tint_cfg["top_rgb"], np.float32) / 255.0
    bottom = np.array(tint_cfg["bottom_rgb"], np.float32) / 255.0
    inv_gamma = 1.0 / float(tint_cfg["gradient_gamma"])
    out = np.zeros(mask.shape + (3,), np.float32)
    out[:] = bottom
    for region in components(mask):
        ys = np.nonzero(region)[0]
        y0, y1 = ys.min(), ys.max()
        t = (np.arange(y0, y1 + 1, dtype=np.float32) - y0) / max(y1 - y0, 1)
        col = top + (bottom - top) * (t ** inv_gamma)[:, None]  # (rows, 3)
        rows = out[y0:y1 + 1]
        reg = region[y0:y1 + 1]
        rows[reg] = np.broadcast_to(col[:, None, :], rows.shape)[reg]
    return out


def _region_mean_shading(low: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """low minus its mean within each glass region (zero outside)."""
    out = np.zeros_like(low, dtype=np.float32)
    for region in components(mask):
        out[region] = low[region] - low[region].mean()
    return out


def render_glass_lab(mask: np.ndarray, low: np.ndarray, style_cfg: dict[str, Any]) -> np.ndarray:
    """Build the new glass (without cracks/detail) as float32 LAB at the mask's resolution."""
    name = style_cfg["name"]
    if name not in ("tint", "studio"):
        raise ValueError(f"render_glass_lab handles tint/studio only, got {name!r}")
    rgb = tint_base(mask, style_cfg["tint"])
    if name == "studio":
        hl = studio_overlay(mask, style_cfg["studio"]) * float(style_cfg["studio"]["strength"])
        rgb = 1.0 - (1.0 - rgb) * (1.0 - hl[..., None])  # screen blend
    lab = rgb_to_lab(rgb)
    lab[..., 0] += float(style_cfg["keep_shading"]) * _region_mean_shading(low, mask)
    return lab


def finish_glass(lab: np.ndarray, high: np.ndarray, crack_mask: np.ndarray | None,
                 style_cfg: dict[str, Any], crack_cfg: dict[str, Any]) -> np.ndarray:
    """Add back fine detail and preserved cracks to L, return RGB float 0..1."""
    lab = lab.copy()
    if style_cfg["detail_keep"] > 0:
        lab[..., 0] += float(style_cfg["detail_keep"]) * high
    if style_cfg["preserve_cracks"] and crack_mask is not None:
        detail = np.abs(high) if crack_cfg["render"] == "bright" else high
        lab[..., 0] += float(crack_cfg["strength"]) * crack_mask.astype(np.float32) * detail
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    return lab_to_rgb(lab)
