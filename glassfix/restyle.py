"""Model-free part of the pipeline: layers -> cracks -> new glass -> composite.

Smooth layers (low-frequency shading, tint, highlight) are computed at working resolution and
upsampled; crack detail is taken from the full-resolution original so it stays sharp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .cracks import detect_cracks
from .layers import lab_to_rgb, rgb_to_lab, separate
from .masks import feather
from .styles import (_smoothstep, fallback_map, srgb_to_linear, studio_glass_base, finish_glass, reflection_veil, remove_veil, render_glass_lab,
                     tint_base)


def upsample_crop(arr: np.ndarray, sy: float, sx: float, box: tuple[int, int, int, int]) -> np.ndarray:
    """Bilinear upsample of a working-resolution array, evaluated only on a full-resolution crop.

    Uses the same pixel-centre convention as cv2.resize, so it equals resize-then-crop.
    """
    y0, y1, x0, x1 = box
    xs = ((np.arange(x0, x1, dtype=np.float32) + 0.5) * sx - 0.5)
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5) * sy - 0.5)
    map_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (y1 - y0, x1 - x0)))
    map_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (y1 - y0, x1 - x0)))
    return cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


STYLES = ("studiowhite", "cleanglass", "dereflect", "clear", "tint", "studio", "external")


@dataclass
class RestyleResult:
    clean: np.ndarray            # uint8 RGB, full resolution
    glass_mask: np.ndarray       # uint8 0/255, full resolution
    crack_mask: np.ndarray       # uint8 0/255, full resolution
    crack_info: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    fallback_pct: float = 0.0    # dereflect: share of the glass replaced by clean glass (C)


def restyle(full_rgb: np.ndarray, work_rgb: np.ndarray, mask_w: np.ndarray, cfg: dict[str, Any],
            reflection_w: np.ndarray | None = None, exclude_w: np.ndarray | None = None,
            external_w: np.ndarray | None = None, interior_w: np.ndarray | None = None) -> RestyleResult:
    """reflection_w: DSIT reflection layer at working resolution (required for style 'dereflect').
    external_w: glass edited by an external (generative) tool, working resolution RGB 0..1 (style 'external').
    exclude_w: objects lying on the glass (wipers). The new glass meets them with a hard edge instead
    of the usual feather, so no halo of old reflective glass shows around them.
    """
    t: dict[str, float] = {}
    H, W = full_rgb.shape[:2]
    scale = work_rgb.shape[1] / W

    t0 = time.perf_counter()
    L_w = rgb_to_lab(work_rgb)[..., 0]
    low_w, high_w = separate(L_w, mask_w, cfg["layers"])
    t["layers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crack_w, crack_info = detect_cracks(high_w, mask_w, cfg["cracks"])
    t["cracks"] = time.perf_counter() - t0

    style = cfg["style"]["name"]
    if style not in STYLES:
        raise ValueError(f"style.name must be one of {STYLES}, got {style!r}")
    t0 = time.perf_counter()
    fallback_w = None
    if style == "external":
        if external_w is None:
            raise ValueError("style 'external' needs the edited glass (external_w)")
    elif style == "dereflect":
        if reflection_w is None:
            raise ValueError("style 'dereflect' needs the DSIT reflection layer (reflection_w)")
        dc = cfg["dereflect"]
        if dc["then_clear"]:
            rest = np.clip(work_rgb.astype(np.float32) / 255.0 - dc["strength"] * reflection_w, 0.0, 1.0)
            veil_w = reflection_veil((rest * 255.0).astype(np.uint8), mask_w, cfg["style"]["clear"])
        fallback_w = fallback_map(work_rgb, reflection_w, mask_w, dc["fallback"])
        glass_w = tint_base(mask_w, cfg["style"]["tint"])
    elif style == "clear":
        veil_w = reflection_veil(work_rgb, mask_w, cfg["style"]["clear"])
    elif style == "studiowhite":
        sw = cfg["style"]["studiowhite"]
        veil_w = reflection_veil(work_rgb, mask_w, cfg["style"]["clear"])
        haze_free_L = rgb_to_lab(remove_veil(work_rgb, veil_w, cfg["style"]["clear"]))[..., 0]
        base_w, clear_win_w = studio_glass_base(mask_w, L_w, sw)
        inter = interior_w if interior_w is not None else np.zeros_like(mask_w)
        inter_w = (inter > 0).astype(np.float32) * clear_win_w      # silhouettes only in clear windows
        sel = inter_w > 0
        p_lo, p_hi = (np.percentile(haze_free_L[sel], [5, 95]) if sel.sum() > 50 else (0.0, 100.0))
    elif style == "cleanglass":
        veil_w = reflection_veil(work_rgb, mask_w, cfg["style"]["clear"])
        tint_cfg = {**cfg["style"], "name": "tint", "keep_shading": 0.0}
        base_lab_w = render_glass_lab(mask_w, low_w, tint_cfg)
    else:
        base_lab_w = render_glass_lab(mask_w, low_w, cfg["style"])
    t["style"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Full-resolution work is done only on the glass bounding box (memory and speed).
    sy, sx = work_rgb.shape[0] / H, work_rgb.shape[1] / W
    ys, xs = np.nonzero(mask_w)
    pad = int(np.ceil(2.0 / min(sx, sy))) + 2
    y0, y1 = max(0, int(ys.min() / sy) - pad), min(H, int(np.ceil((ys.max() + 1) / sy)) + pad)
    x0, x1 = max(0, int(xs.min() / sx) - pad), min(W, int(np.ceil((xs.max() + 1) / sx)) + pad)
    box = (y0, y1, x0, x1)

    mask_c = (upsample_crop(mask_w.astype(np.float32), sy, sx, box) >= 0.5).astype(np.uint8)
    if exclude_w is not None and exclude_w.any():
        excl_c = (upsample_crop(exclude_w.astype(np.float32), sy, sx, box) >= 0.5).astype(np.uint8)
        alpha = feather(mask_c | excl_c, cfg["mask"]["feather_px"] / scale) * mask_c
    else:
        alpha = feather(mask_c, cfg["mask"]["feather_px"] / scale)
    crack_c = np.clip(upsample_crop(crack_w.astype(np.float32), sy, sx, box), 0.0, 1.0) * mask_c
    full_c = full_rgb[y0:y1, x0:x1]
    if style == "external":
        new_rgb = np.clip(upsample_crop(external_w.astype(np.float32), sy, sx, box), 0.0, 1.0)
    elif style == "dereflect":
        dc = cfg["dereflect"]
        new_rgb = np.clip(full_c.astype(np.float32) / 255.0
                          - dc["strength"] * upsample_crop(reflection_w, sy, sx, box), 0.0, 1.0)
        if dc["then_clear"]:
            new_rgb = remove_veil(new_rgb, upsample_crop(veil_w, sy, sx, box), cfg["style"]["clear"])
        f = upsample_crop(fallback_w, sy, sx, box)[..., None]
        new_rgb = new_rgb * (1.0 - f) + upsample_crop(glass_w, sy, sx, box) * f
    elif style == "studiowhite":
        sw = cfg["style"]["studiowhite"]
        base = upsample_crop(base_w, sy, sx, box)
        clear_rgb = remove_veil(full_c, upsample_crop(veil_w, sy, sx, box), cfg["style"]["clear"])
        lab = rgb_to_lab(clear_rgb)
        # Interior tones: darkest -> interior_dark_L, lightest -> the glass's own lightness (soft silhouettes).
        glass_L = rgb_to_lab(base)[..., 0]
        tnorm = np.clip((lab[..., 0] - p_lo) / max(p_hi - p_lo, 1e-3), 0.0, 1.0)
        lab[..., 0] = sw["interior_dark_L"] + (glass_L - sw["interior_dark_L"]) * tnorm
        lab[..., 1:] *= sw["interior_saturation"]
        interior_rgb = lab_to_rgb(lab)
        a_int = upsample_crop(inter_w, sy, sx, box)
        fpx = sw["interior_feather_px"] / scale
        if fpx > 0:
            a_int = cv2.GaussianBlur(a_int, (0, 0), fpx)
        a_int = (np.clip(a_int, 0.0, 1.0) * sw["interior_strength"])[..., None]
        mixed = base * (1.0 - a_int) + interior_rgb * a_int
        low_c = upsample_crop(low_w, sy, sx, box)
        high_c = np.where(mask_c > 0, rgb_to_lab(full_c)[..., 0] - low_c, 0.0).astype(np.float32)
        new_rgb = finish_glass(rgb_to_lab(mixed), high_c, crack_c, {**cfg["style"], "detail_keep": 0.0}, cfg["cracks"])
    elif style == "cleanglass":
        cg = cfg["style"]["cleanglass"]
        clear_rgb = remove_veil(full_c, upsample_crop(veil_w, sy, sx, box), cfg["style"]["clear"])
        tint_lab = upsample_crop(base_lab_w, sy, sx, box)
        # Bright content on glass is reflection (sky, clouds, glare): fade it into the tint. Dark and
        # mid tones are what the interior looks like through glass: let a share of it show.
        veil_c = upsample_crop(veil_w, sy, sx, box)
        orig_lin = srgb_to_linear(full_c.astype(np.float32) / 255.0).mean(axis=2)
        share = np.clip(veil_c.mean(axis=2) * cfg["style"]["clear"]["strength"] / np.maximum(orig_lin, 1e-4), 0, 1)
        L_orig = rgb_to_lab(full_c)[..., 0]
        show = (cg["interior_visibility"]
                * (1.0 - _smoothstep(cg["reflection_share_low"], cg["reflection_share_high"], share))
                * (1.0 - _smoothstep(cg["highlight_low"], cg["highlight_high"], L_orig)))
        grey = clear_rgb.mean(axis=2, keepdims=True)
        interior = grey + cg["interior_saturation"] * (clear_rgb - grey)
        mixed = lab_to_rgb(tint_lab) * (1.0 - show[..., None]) + interior * show[..., None]
        low_c = upsample_crop(low_w, sy, sx, box)
        high_c = np.where(mask_c > 0, rgb_to_lab(full_c)[..., 0] - low_c, 0.0).astype(np.float32)
        no_detail = {**cfg["style"], "detail_keep": 0.0}
        new_rgb = finish_glass(rgb_to_lab(mixed), high_c, crack_c, no_detail, cfg["cracks"])
    elif style == "clear":
        # Original glass kept: only the veil is removed, so detail and cracks survive untouched.
        new_rgb = remove_veil(full_c, upsample_crop(veil_w, sy, sx, box), cfg["style"]["clear"])
    else:
        low_c = upsample_crop(low_w, sy, sx, box)
        high_c = np.where(mask_c > 0, rgb_to_lab(full_c)[..., 0] - low_c, 0.0).astype(np.float32)
        new_rgb = finish_glass(upsample_crop(base_lab_w, sy, sx, box), high_c, crack_c, cfg["style"], cfg["cracks"])

    clean = full_rgb.copy()
    region = clean[y0:y1, x0:x1]
    inside = alpha > 0  # pixels outside the mask are copied untouched
    a = alpha[inside][:, None]
    blended = full_c[inside].astype(np.float32) * (1.0 - a) + new_rgb[inside] * 255.0 * a
    region[inside] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    glass_mask = np.zeros((H, W), np.uint8)
    glass_mask[y0:y1, x0:x1] = mask_c * 255
    crack_mask = np.zeros((H, W), np.uint8)
    crack_mask[y0:y1, x0:x1] = (crack_c >= 0.5).astype(np.uint8) * 255
    t["composite"] = time.perf_counter() - t0

    return RestyleResult(
        clean=clean,
        glass_mask=glass_mask,
        crack_mask=crack_mask,
        crack_info=crack_info,
        timings=t,
        fallback_pct=round(100.0 * float(fallback_w[mask_w > 0].mean()), 2) if fallback_w is not None else 0.0,
    )
