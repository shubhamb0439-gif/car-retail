"""Synthetic tests for the cut-out deliverable (option A): car PNG with studio glass."""

from __future__ import annotations

import cv2
import numpy as np

from glassfix.config import load_config, with_overrides
from glassfix.cutout import make_cutout

H, W = 480, 800
CAR = (80, 120, 720, 420)        # x0, y0, x1, y1 of the car body
CLEAR = (180, 150, 380, 250)     # clear window: bright sky reflection over a mid-grey interior
TINTED = (420, 150, 620, 250)    # tinted window: dark
WIPER = (200, 236, 360, 244)     # wiper lying on the clear window


def scene(crack: bool = False):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 90, np.float32) + rng.normal(0, 4, (H, W, 3))       # street
    car = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = CAR
    car[y0:y1, x0:x1] = 1
    img[car > 0] = (170, 40, 40)                                                   # red paint
    glass = np.zeros((H, W), np.uint8)
    for (gx0, gy0, gx1, gy1), base in ((CLEAR, 110.0), (TINTED, 25.0)):
        glass[gy0:gy1, gx0:gx1] = 1
        xs = np.linspace(0, 1, gx1 - gx0, dtype=np.float32)
        img[gy0:gy1, gx0:gx1] = base + (90.0 if base > 50 else 10.0) * xs[None, :, None]   # reflection
    excl = np.zeros((H, W), np.uint8)
    wx0, wy0, wx1, wy1 = WIPER
    excl[wy0:wy1, wx0:wx1] = 1
    img[excl > 0] = (15, 15, 15)
    line = np.zeros((H, W), np.uint8)
    if crack:
        cv2.line(line, (220, 165), (350, 225), 1, 2)
        img[line > 0] += 45
    return np.clip(img, 0, 255).astype(np.uint8), car, glass & (1 - excl), excl, line > 0


def cfg_for(preserve=True):
    cfg = load_config()
    return with_overrides(cfg, **{"style.name": "cutout", "style.preserve_cracks": preserve,
                                  "cutout.interior_mode": "none", "processing.max_side": 2048})


def run(img, car, glass, excl, cfg):
    return make_cutout(img, img, glass, cfg, car, None, excl)


def inner(box, m=8):
    x0, y0, x1, y1 = box
    return (slice(y0 + m, y1 - m), slice(x0 + m, x1 - m))


def test_background_is_transparent_and_body_untouched():
    img, car, glass, excl, _ = scene()
    _, rgba = run(img, car, glass, excl, cfg_for())
    assert rgba[:40, :40, 3].max() == 0 and rgba[-40:, -40:, 3].max() == 0
    body = inner((CAR[0], 300, CAR[2], CAR[3]))                 # lower body, away from glass and edges
    assert rgba[body][..., 3].min() == 255
    assert np.array_equal(rgba[body][..., :3], img[body])


def test_clear_and_tinted_glass_get_studio_colour_and_opacity():
    img, car, glass, excl, _ = scene()
    c = cfg_for()["cutout"]
    res, rgba = run(img, car, glass, excl, cfg_for())
    clear = rgba[inner((CLEAR[0], CLEAR[1], CLEAR[2], 230))]
    tinted = rgba[inner(TINTED)]
    assert abs(clear[..., 3].astype(int).mean() - round(255 * c["glass_alpha"])) <= 2
    assert abs(tinted[..., 3].astype(int).mean() - round(255 * c["tinted_glass_alpha"])) <= 2
    assert np.abs(clear[..., :3].reshape(-1, 3).mean(0) - np.array(c["glass_rgb"])).max() <= 3
    # The reflection gradient is gone: the glass is uniform.
    assert clear[..., :3].reshape(-1, 3).astype(int).std(axis=0).max() < 2.0
    # On a white background the clear glass reads as light studio glass (the client's reference).
    prev = res.clean[inner((CLEAR[0], CLEAR[1], CLEAR[2], 230))].reshape(-1, 3).mean(0)
    assert (prev > 200).all()


def test_wiper_stays_original_and_opaque():
    img, car, glass, excl, _ = scene()
    _, rgba = run(img, car, glass, excl, cfg_for())
    w = inner(WIPER, 2)
    assert rgba[w][..., 3].min() == 255
    assert np.array_equal(rgba[w][..., :3], img[w])


def test_crack_drawn_on_glass_when_preserved():
    img, car, glass, excl, line = scene(crack=True)
    res, rgba = run(img, car, glass, excl, cfg_for(preserve=True))
    assert res.crack_info["crack_detected"]
    on = line & (glass > 0)
    assert rgba[on][..., 3].mean() > 200                      # crack is (nearly) opaque on clear glass
    _, rgba_off = run(img, car, glass, excl, cfg_for(preserve=False))
    assert rgba_off[on][..., 3].mean() < 100


def test_deterministic():
    img, car, glass, excl, _ = scene()
    a = run(img, car, glass, excl, cfg_for())[1]
    b = run(img, car, glass, excl, cfg_for())[1]
    assert np.array_equal(a, b)
