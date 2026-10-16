"""One-off quality test: OpenAI GPT Image edit for studio car glass (paid API, not part of the pod pipeline).

Two variants on one photo:
  whole    - the whole photo -> transparent-background PNG of the car with semi-transparent tinted
             glass, lighting fixed, using the client's example image as a style reference.
  windows  - the BiRefNet cut-out + a glass mask: only the windows are edited; afterwards only the
             glass pixels are copied back, so the car body stays the original photo's pixels.

Usage (OPENAI_API_KEY must be set; never print it):
  python tools/openai_glass_test.py --photo test_photos/ford_ecosport_user/side.png \
      --cutout results/openai_test/ecosport_birefnet_cutout.png \
      --glass-mask results/cutmodes/detect/ford_ecosport_user/side_glass_mask.png \
      --reference reference/target_look_cutout.png --out results/openai_test --quality medium

Prints the token usage and an estimated cost per call (GPT Image 2.5 token rates).
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "gpt-image-2.5-sunburst"   # OpenAI's model for precise edits (docs, 2026-10)
RATES = {"image_in": 8.0, "text_in": 5.0, "image_out": 30.0}   # USD per 1M tokens (pricing page, 2026-10)

PROMPT_WHOLE = (
    "Turn this car photo into a clean studio product image with a fully transparent background. "
    "Keep the car itself exactly the same: same model, shape, paint colour, wheels, badges, lights, trim and "
    "camera angle. Remove the background, the ground and all shadows. Even out the lighting on the car like a "
    "soft white studio. Replace every window with clean, slightly grey tinted glass that is semi-transparent, "
    "with no reflections of sky, trees or buildings; the car's interior (seats, headrests, steering wheel) "
    "stays faintly visible through it. Match the glass and overall look of the second (reference) image."
)
PROMPT_WINDOWS = (
    "Edit only the car's window glass. Make every window clean, slightly grey tinted, semi-transparent glass "
    "with no reflections at all (no sky, clouds, trees, buildings or glare). Keep the car's real interior "
    "(seats, headrests, steering wheel, dashboard) faintly visible through the glass. Anything seen through "
    "the car to the other side becomes plain white. Do not change anything outside the windows."
)


def png_bytes(img: Image.Image, name: str) -> tuple[str, bytes, str]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return (name, buf.getvalue(), "image/png")


def cost(usage) -> str:
    if usage is None:
        return "usage not reported"
    det = getattr(usage, "input_tokens_details", None)
    img_in = getattr(det, "image_tokens", 0) or 0
    txt_in = getattr(det, "text_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    usd = (img_in * RATES["image_in"] + txt_in * RATES["text_in"] + out * RATES["image_out"]) / 1e6
    return f"input image {img_in} + text {txt_in} tokens, output {out} tokens -> about ${usd:.3f}"


def save_b64(result, path: Path) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json)))
    img.save(path)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--cutout", type=Path, required=True)
    ap.add_argument("--glass-mask", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--variants", default="whole,windows")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    variants = args.variants.split(",")

    if "whole" in variants:
        t0 = time.time()
        r = client.images.edit(
            model=MODEL,
            image=[png_bytes(Image.open(args.photo).convert("RGB"), "photo.png"),
                   png_bytes(Image.open(args.reference).convert("RGBA"), "reference.png")],
            prompt=PROMPT_WHOLE, background="transparent", output_format="png", quality=args.quality,
        )
        save_b64(r, args.out / "whole_openai.png")
        print(f"whole:   {time.time() - t0:.0f}s, {cost(getattr(r, 'usage', None))}", flush=True)

    if "windows" in variants:
        cut = Image.open(args.cutout).convert("RGBA")
        glass = Image.open(args.glass_mask).convert("L").resize(cut.size, Image.NEAREST)
        g = np.asarray(glass) > 127
        # OpenAI mask convention: fully transparent pixels = the area to edit.
        mask = np.full((cut.size[1], cut.size[0], 4), 255, np.uint8)
        mask[g, 3] = 0
        t0 = time.time()
        r = client.images.edit(
            model=MODEL,
            image=png_bytes(cut, "cutout.png"),
            mask=png_bytes(Image.fromarray(mask, "RGBA"), "mask.png"),
            prompt=PROMPT_WINDOWS, background="transparent", output_format="png", quality=args.quality,
        )
        edited = save_b64(r, args.out / "windows_openai_raw.png").convert("RGBA").resize(cut.size, Image.LANCZOS)
        # Keep the original body: copy only the glass pixels (soft edge) from the edit into the cut-out.
        from PIL import ImageFilter
        soft = np.asarray(glass.filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)[..., None] / 255.0
        out = np.asarray(cut).astype(np.float32) * (1 - soft) + np.asarray(edited).astype(np.float32) * soft
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA").save(args.out / "windows_composited.png")
        print(f"windows: {time.time() - t0:.0f}s, {cost(getattr(r, 'usage', None))}", flush=True)


if __name__ == "__main__":
    main()
