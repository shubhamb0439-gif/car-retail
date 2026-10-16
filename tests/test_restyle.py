"""Synthetic tests: a fake window with a gradient reflection and a thin drawn crack."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from glassfix.config import load_config, style_fingerprint, with_overrides
from glassfix.imageio import to_working
from glassfix.layers import rgb_to_lab
from glassfix.masks import clean_mask, feather
from glassfix.restyle import restyle
from glassfix.styles import softbox_texture

H, W = 512, 768
WIN = (120, 140, 640, 400)  # x0, y0, x1, y1 of the fake window


def make_scene(crack: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 150.0, np.float32)            # car body
    img += rng.normal(0, 3, img.shape)                      # sensor noise everywhere
    x0, y0, x1, y1 = WIN
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 1

    xs = np.arange(x0, x1, dtype=np.float32)
    reflection = 110.0 * (xs - x0) / (x1 - x0)              # smooth left->right glare
    glass = np.full((y1 - y0, x1 - x0, 3), 45.0, np.float32) + reflection[None, :, None]
    glass += rng.normal(0, 2, glass.shape)
    img[y0:y1, x0:x1] = glass

    line = np.zeros((H, W), np.uint8)
    if crack:
        pts = np.array([[230, 180], [330, 240], [420, 270], [520, 350]], np.int32)
        cv2.polylines(line, [pts], False, 1, thickness=2)
        cv2.line(line, (330, 240), (360, 170), 1, thickness=2)  # a branch
        img[line > 0] += 40.0
    return np.clip(img, 0, 255).astype(np.uint8), mask, line > 0


def cfg_for(style="tint", preserve=True, keep=0.15, max_side=2048):
    return with_overrides(load_config(), **{
        "style.name": style, "style.preserve_cracks": preserve,
        "style.keep_shading": keep, "processing.max_side": max_side,
    })


def run(img, mask, cfg):
    work, _ = to_working(img, cfg["processing"]["max_side"])
    mask_w = cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
    return restyle(img, work, mask_w, cfg)


def interior(mask, margin=20):
    k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv2.erode(mask, k) > 0


def horizontal_range(L, region, exclude):
    """Range of column-mean luminance across the region (measures left->right gradient)."""
    sel = region & ~exclude
    cols = [L[:, c][sel[:, c]].mean() for c in range(W) if sel[:, c].sum() > 20]
    return float(np.max(cols) - np.min(cols))


@pytest.mark.parametrize("keep", [0.0, 0.15])
def test_reflection_gradient_removed(keep):
    img, mask, line = make_scene(crack=False)
    res = run(img, mask, cfg_for(keep=keep))
    inner = interior(mask)
    no_line = np.zeros_like(inner)
    before = horizontal_range(rgb_to_lab(img)[..., 0], inner, no_line)
    after = horizontal_range(rgb_to_lab(res.clean)[..., 0], inner, no_line)
    assert before > 30, "test scene should have a strong reflection"
    assert after <= before * (keep + 0.08), f"gradient not removed: {before:.1f} -> {after:.1f}"


def test_crack_pixels_survive():
    img, mask, line = make_scene(crack=True)
    res = run(img, mask, cfg_for(preserve=True))
    assert res.crack_info["crack_detected"]
    inner = interior(mask, 10)
    truth = line & inner
    found = (res.crack_mask > 0)[truth].mean()
    assert found >= 0.8, f"only {found:.0%} of crack pixels detected"

    L = rgb_to_lab(res.clean)[..., 0]
    near = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = near & ~cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & inner
    contrast = L[truth].mean() - L[ring].mean()
    assert contrast >= 8.0, f"crack not visible in result (contrast {contrast:.1f} L)"


def test_crack_removed_when_preservation_off():
    img, mask, line = make_scene(crack=True)
    res = run(img, mask, cfg_for(preserve=False))
    L = rgb_to_lab(res.clean)[..., 0]
    inner = interior(mask, 10)
    near = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = near & ~cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & inner
    assert abs(L[line & inner].mean() - L[ring].mean()) < 3.0


def test_no_false_crack_on_clean_glass():
    img, mask, _ = make_scene(crack=False)
    res = run(img, mask, cfg_for())
    assert not res.crack_info["crack_detected"]
    assert res.crack_mask.sum() == 0


def test_busy_interior_seen_through_glass_is_not_a_crack():
    """Seats/dashboard edges behind the glass are thin lines too, but dense; they must not be flagged."""
    img, mask, _ = make_scene(crack=False)
    rng = np.random.default_rng(1)
    x0, y0, x1, y1 = WIN
    busy = img.copy()
    for _ in range(400):
        p = (int(rng.integers(x0 + 10, x1 - 10)), int(rng.integers(y0 + 10, y1 - 10)))
        q = (p[0] + int(rng.integers(-40, 40)), p[1] + int(rng.integers(-40, 40)))
        shade = int(rng.integers(-60, 60))
        color = tuple(int(np.clip(int(v) + shade, 0, 255)) for v in busy[p[1], p[0]])
        cv2.line(busy, p, q, color, thickness=int(rng.integers(1, 3)))
    busy[mask == 0] = img[mask == 0]
    res = run(busy, mask, cfg_for())
    assert not res.crack_info["crack_detected"], res.crack_info


@pytest.mark.parametrize("style", ["tint", "studio"])
def test_outside_mask_untouched(style):
    img, mask, _ = make_scene()
    res = run(img, mask, cfg_for(style=style))
    assert np.array_equal(res.clean[mask == 0], img[mask == 0])
    assert not np.array_equal(res.clean[mask > 0], img[mask > 0])


@pytest.mark.parametrize("style", ["tint", "studio"])
def test_deterministic(style):
    img, mask, _ = make_scene()
    a = run(img, mask, cfg_for(style=style)).clean
    b = run(img, mask, cfg_for(style=style)).clean
    assert np.array_equal(a, b)


def test_downscaled_processing_returns_full_resolution():
    img, mask, line = make_scene(crack=True)
    res = run(img, mask, cfg_for(max_side=512))
    assert res.clean.shape == img.shape
    assert res.glass_mask.shape == mask.shape
    out_mask = res.glass_mask > 0
    iou = (out_mask & (mask > 0)).sum() / (out_mask | (mask > 0)).sum()
    assert iou > 0.98
    # The saved glass mask is the contract: nothing outside it changes.
    assert np.array_equal(res.clean[~out_mask], img[~out_mask])
    assert res.crack_info["crack_detected"]


def test_studio_highlight_inside_glass_only():
    img, mask, _ = make_scene(crack=False)
    tint = run(img, mask, cfg_for(style="tint")).clean.astype(int)
    studio = run(img, mask, cfg_for(style="studio")).clean.astype(int)
    diff = np.abs(studio - tint).sum(-1)
    assert diff[mask > 0].max() > 30          # a visible highlight was added
    assert diff[mask == 0].max() == 0


def test_softbox_texture_is_fixed():
    cfg = load_config()["style"]["studio"]
    t1, t2 = softbox_texture(cfg), softbox_texture(dict(cfg))
    assert t1 is t2 or np.array_equal(t1, t2)
    assert 0.0 <= t1.min() and t1.max() <= 1.0 and t1.max() > 0.5


def test_style_fingerprint_stable():
    assert style_fingerprint(load_config()) == style_fingerprint(load_config())
    assert style_fingerprint(cfg_for(style="tint")) != style_fingerprint(cfg_for(style="studio"))


def test_feather_is_inward_only():
    m = np.zeros((50, 50), np.uint8)
    m[10:40, 10:40] = 1
    a = feather(m, 3)
    assert a[m == 0].max() == 0
    assert a[25, 25] == 1.0 and 0 < a[10, 25] < 1


def test_clean_mask_drops_specks_and_off_car_regions():
    cfg = load_config()["mask"]
    m = np.zeros((400, 400), np.uint8)
    m[100:200, 100:250] = 1         # window on the car
    m[300:303, 300:303] = 1         # speck
    m[10:60, 330:390] = 1           # region off the car
    out = clean_mask(m, [(80, 80, 280, 260)], cfg)
    assert out[150, 150] == 1 and out[301, 301] == 0 and out[30, 350] == 0


# ------------------------------------------------------------------ clear style (original glass kept)
def test_clear_removes_reflection_but_keeps_glass_detail():
    img, mask, line = make_scene(crack=True)
    res = run(img, mask, cfg_for(style="clear"))
    inner = interior(mask)
    L0, L1 = rgb_to_lab(img)[..., 0], rgb_to_lab(res.clean)[..., 0]
    excl = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    before, after = horizontal_range(L0, inner, excl), horizontal_range(L1, inner, excl)
    assert after <= before * 0.5, f"reflection not reduced enough: {before:.1f} -> {after:.1f}"

    # Fine detail (the crack line) is original glass content: it must stay clearly visible.
    near = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = near & ~cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & inner
    truth = line & inner
    c0, c1 = L0[truth].mean() - L0[ring].mean(), L1[truth].mean() - L1[ring].mean()
    assert c1 >= 0.7 * c0, f"detail lost: contrast {c0:.1f} -> {c1:.1f}"
    assert res.crack_info["crack_detected"]


def test_clear_keeps_interior_texture():
    """Interior seen through the glass must survive (unlike tint/studio)."""
    img, mask, _ = make_scene(crack=False)
    rng = np.random.default_rng(2)
    x0, y0, x1, y1 = WIN
    interior_img = img.copy()
    for _ in range(40):  # dark 'seat/dashboard' shapes behind the glass
        cx, cy = int(rng.integers(x0 + 30, x1 - 30)), int(rng.integers(y0 + 30, y1 - 30))
        cv2.circle(interior_img, (cx, cy), int(rng.integers(6, 16)), (20, 20, 20), -1)
    interior_img[mask == 0] = img[mask == 0]
    res = run(interior_img, mask, cfg_for(style="clear"))
    inner = interior(mask)
    g0 = cv2.Laplacian(rgb_to_lab(interior_img)[..., 0], cv2.CV_32F)[inner].std()
    g1 = cv2.Laplacian(rgb_to_lab(res.clean)[..., 0], cv2.CV_32F)[inner].std()
    assert g1 >= 0.8 * g0, f"interior detail flattened: {g0:.2f} -> {g1:.2f}"


def test_clear_outside_untouched_and_deterministic():
    img, mask, _ = make_scene()
    a = run(img, mask, cfg_for(style="clear")).clean
    b = run(img, mask, cfg_for(style="clear")).clean
    assert np.array_equal(a, b)
    assert np.array_equal(a[mask == 0], img[mask == 0])


# ------------------------------------------------------------------ cleanglass (no reflections at all)
def test_cleanglass_removes_bright_reflection_and_keeps_crack():
    img, mask, line = make_scene(crack=True)
    res = run(img, mask, cfg_for(style="cleanglass"))
    inner = interior(mask)
    excl = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    L0, L1 = rgb_to_lab(img)[..., 0], rgb_to_lab(res.clean)[..., 0]
    before, after = horizontal_range(L0, inner, excl), horizontal_range(L1, inner, excl)
    assert after <= before * 0.25, f"reflection gradient left: {before:.1f} -> {after:.1f}"
    near = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = near & ~cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & inner
    assert L1[line & inner].mean() - L1[ring].mean() >= 8.0
    assert np.array_equal(res.clean[mask == 0], img[mask == 0])


def test_cleanglass_deterministic():
    img, mask, _ = make_scene()
    a = run(img, mask, cfg_for(style="cleanglass")).clean
    b = run(img, mask, cfg_for(style="cleanglass")).clean
    assert np.array_equal(a, b)
