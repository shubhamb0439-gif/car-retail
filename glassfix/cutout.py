"""Cut-out output: the car as an RGBA PNG on a transparent background, with studio glass.

Inside each window:
  - interior_mode "none" (default): plain studio glass, nothing seen through it is kept;
    other modes keep interior parts (seats, steering wheel...) from the original, see config.yaml;
  - everything else in the glass (reflections, the street seen through the car) becomes slightly grey,
    semi-transparent glass, so the client's own background shows through;
  - windows that are themselves tinted become darker and less transparent;
  - wipers stay solid; preserved cracks are drawn on the glass.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .cracks import detect_cracks
from .layers import rgb_to_lab, separate
from .masks import components
from .restyle import RestyleResult, upsample_crop
from .styles import reflection_veil, remove_veil


def _refine_alpha(mask: np.ndarray, guide_rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Soft, edge-aligned alpha from a binary mask using a guided filter (OpenCV contrib)."""
    m = mask.astype(np.float32)
    r = int(cfg["edge_refine_radius_px"])
    if r <= 0:
        return m
    guide = guide_rgb.astype(np.float32) / 255.0
    return np.clip(cv2.ximgproc.guidedFilter(guide, m, r, float(cfg["edge_refine_eps"])), 0.0, 1.0)


def make_cutout(full_rgb: np.ndarray, work_rgb: np.ndarray, glass_w: np.ndarray, cfg: dict[str, Any],
                car_w: np.ndarray, interior_w: np.ndarray | None,
                exclude_w: np.ndarray | None) -> tuple[RestyleResult, np.ndarray]:
    """Return (result with an on-background preview as .clean, RGBA uint8 cut-out at full resolution)."""
    c = cfg["cutout"]
    t: dict[str, float] = {}
    H, W = full_rgb.shape[:2]
    sy, sx = work_rgb.shape[0] / H, work_rgb.shape[1] / W
    full_box = (0, H, 0, W)

    # ---- car alpha (working resolution, refined, then upsampled)
    t0 = time.perf_counter()
    car_alpha_w = _refine_alpha(car_w | glass_w, work_rgb, c)
    car_alpha = upsample_crop(car_alpha_w, sy, sx, full_box)
    t["car_alpha"] = time.perf_counter() - t0

    # ---- glass analysis at working resolution
    t0 = time.perf_counter()
    L_w = rgb_to_lab(work_rgb)[..., 0]
    _, high_w = separate(L_w, glass_w, cfg["layers"])
    crack_w, crack_info = detect_cracks(high_w, glass_w, cfg["cracks"])
    veil_w = reflection_veil(work_rgb, glass_w, cfg["style"]["clear"])
    clear_win = np.zeros(glass_w.shape, np.float32)          # 1 = clear window, 0 = tinted window
    for region in components(glass_w):
        if float(np.percentile(L_w[region], 90)) >= c["tinted_p90_below_L"]:
            clear_win[region] = 1.0
    mode = c["interior_mode"]
    detected = (interior_w > 0).astype(np.float32) if interior_w is not None else np.zeros_like(clear_win)
    if mode in ("luminance", "both"):
        # Keep what is lit from behind the glass: pixels whose light is mostly NOT reflection
        # (veil / original) and that are not bright sky/glare in the original.
        from .styles import _smoothstep, srgb_to_linear

        lin = srgb_to_linear(work_rgb.astype(np.float32) / 255.0).mean(axis=2)
        share = np.clip(veil_w.mean(axis=2) * cfg["style"]["clear"]["strength"] / np.maximum(lin, 1e-4), 0, 1)
        lum_keep = ((1.0 - _smoothstep(c["reflection_share_low"], c["reflection_share_high"], share))
                    * (1.0 - _smoothstep(c["highlight_low"], c["highlight_high"], L_w))).astype(np.float32)
    else:
        lum_keep = np.zeros_like(clear_win)
    inter = {"none": np.zeros_like(clear_win), "detect": detected, "luminance": lum_keep,
             "both": np.maximum(detected, lum_keep)}[mode]
    inter = inter * clear_win * (glass_w > 0)
    if c["interior_feather_px"] > 0:
        inter = cv2.GaussianBlur(inter, (0, 0), c["interior_feather_px"])
    t["glass_analysis"] = time.perf_counter() - t0

    # ---- full-resolution glass, only inside the glass bounding box
    t0 = time.perf_counter()
    ys, xs = np.nonzero(glass_w)
    pad = int(np.ceil(2.0 / min(sx, sy))) + 2
    y0, y1 = max(0, int(ys.min() / sy) - pad), min(H, int(np.ceil((ys.max() + 1) / sy)) + pad)
    x0, x1 = max(0, int(xs.min() / sx) - pad), min(W, int(np.ceil((xs.max() + 1) / sx)) + pad)
    box = (y0, y1, x0, x1)
    glass = upsample_crop(glass_w.astype(np.float32), sy, sx, box)
    if exclude_w is not None and exclude_w.any():                 # wipers stay solid original pixels
        glass *= 1.0 - (upsample_crop(exclude_w.astype(np.float32), sy, sx, box) >= 0.5)
    glass = np.clip(glass, 0.0, 1.0)[..., None]
    clear_c = upsample_crop(clear_win, sy, sx, box)[..., None]
    inter_c = np.clip(upsample_crop(inter, sy, sx, box), 0.0, 1.0)[..., None] * c["interior_opacity"]
    crack_c = np.clip(upsample_crop(crack_w.astype(np.float32), sy, sx, box), 0.0, 1.0)[..., None]

    full_c = full_rgb[y0:y1, x0:x1]
    interior_rgb = remove_veil(full_c, upsample_crop(veil_w, sy, sx, box), cfg["style"]["clear"])
    interior_rgb = interior_rgb * np.array(c["interior_multiply"], np.float32)
    glass_rgb = (clear_c * np.array(c["glass_rgb"], np.float32) / 255.0
                 + (1 - clear_c) * np.array(c["tinted_glass_rgb"], np.float32) / 255.0)
    glass_a = clear_c * c["glass_alpha"] + (1 - clear_c) * c["tinted_glass_alpha"]
    # Interior over glass ("over" compositing): interior is opaque where present.
    a_g = inter_c + glass_a * (1 - inter_c)
    rgb_g = (interior_rgb * inter_c + glass_rgb * glass_a * (1 - inter_c)) / np.maximum(a_g, 1e-4)
    if cfg["style"]["preserve_cracks"]:
        crack_rgb = np.array(c["crack_rgb"], np.float32) / 255.0
        rgb_g = rgb_g * (1 - crack_c) + crack_rgb * crack_c
        a_g = np.maximum(a_g, crack_c * c["crack_alpha"])

    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = full_rgb
    rgba[..., 3] = np.clip(np.rint(car_alpha * 255), 0, 255).astype(np.uint8)
    region = rgba[y0:y1, x0:x1].astype(np.float32) / 255.0
    region[..., :3] = region[..., :3] * (1 - glass) + rgb_g * glass
    region[..., 3:] = region[..., 3:] * (1 - glass) + a_g * glass
    rgba[y0:y1, x0:x1] = np.clip(np.rint(region * 255), 0, 255).astype(np.uint8)

    bg = np.array(c["preview_background_rgb"], np.float32)
    a = rgba[..., 3:].astype(np.float32) / 255.0
    preview = np.clip(np.rint(rgba[..., :3] * a + bg * (1 - a)), 0, 255).astype(np.uint8)
    t["glass_render"] = time.perf_counter() - t0

    glass_full = np.zeros((H, W), np.uint8)
    glass_full[y0:y1, x0:x1] = (glass[..., 0] > 0.5).astype(np.uint8) * 255
    crack_full = np.zeros((H, W), np.uint8)
    crack_full[y0:y1, x0:x1] = (crack_c[..., 0] >= 0.5).astype(np.uint8) * 255
    return RestyleResult(clean=preview, glass_mask=glass_full, crack_mask=crack_full,
                         crack_info=crack_info, timings=t), rgba
