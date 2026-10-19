"""One car photo -> OpenAI image edit -> transparent-background studio PNG with tinted glass."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# Errors worth retrying (matched by class name so the check does not depend on SDK internals).
RETRYABLE = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


class RenderError(RuntimeError):
    """A render failed. The message never contains request headers or the API key."""


@dataclass
class RenderResult:
    png: bytes                 # the transparent PNG returned by OpenAI
    size: tuple[int, int]
    seconds: float
    input_image_tokens: int
    input_text_tokens: int
    output_tokens: int
    cost_usd: float            # estimated from the token rates in config.yaml


def _png_file(img: Image.Image, name: str) -> tuple[str, bytes, str]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return (name, buf.getvalue(), "image/png")


def shrink(img: Image.Image, max_side: int) -> Image.Image:
    """Cap the long side (OpenAI's output is ~1.5 MP anyway; larger inputs only cost more tokens)."""
    k = min(1.0, max_side / max(img.size))
    return img if k == 1.0 else img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)


class OpenAIRenderer:
    """Renders photos for one service tier (low / medium / high, see config.yaml)."""

    def __init__(self, cfg: dict[str, Any], tier: str | None = None, client: Any = None, root: Path | None = None,
                 view: str = "exterior"):
        self.cfg = cfg
        self.tier = tier or cfg["default_tier"]
        self.view = view
        table = {"exterior": cfg["tiers"], "interior": cfg["interior_tiers"]}.get(view)
        if table is None:
            raise RenderError(f"unknown view {view!r}; choose exterior or interior")
        if self.tier not in table:
            raise RenderError(f"unknown tier {self.tier!r}; choose from {sorted(table)}")
        self.t = table[self.tier]
        o = cfg["openai"]
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=0, timeout=o["timeout_seconds"])  # retries are handled below
        self.client = client
        self.reference = None
        ref = self.t.get("style_reference")
        if ref:
            path = Path(ref) if Path(ref).is_absolute() else (root or Path.cwd()) / ref
            if not path.exists():
                raise RenderError(f"style_reference not found: {path}")
            self.reference = _png_file(shrink(Image.open(path).convert("RGBA"), o["send_max_side"]), "reference.png")

    def _prompt(self) -> str:
        p = self.t["prompt"].strip()
        return p + ("\n" + self.cfg["prompt_reference_suffix"].strip() if self.reference else "")

    def _cost(self, usage: Any) -> tuple[int, int, int, float]:
        det = getattr(usage, "input_tokens_details", None)
        img_in = getattr(det, "image_tokens", 0) or 0
        txt_in = getattr(det, "text_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        r = self.cfg["pricing_usd_per_million_tokens"]
        usd = (img_in * r["image_input"] + txt_in * r["text_input"] + out * r["image_output"]) / 1e6
        return img_in, txt_in, out, usd

    def render(self, photo: Image.Image) -> RenderResult:
        o = self.cfg["openai"]
        images = [_png_file(shrink(photo.convert("RGB"), o["send_max_side"]), "photo.png")]
        if self.reference:
            images.append(self.reference)
        t0 = time.perf_counter()
        for attempt in range(o["max_retries"] + 1):
            try:
                resp = self.client.images.edit(
                    model=o["model"], image=images, prompt=self._prompt(), quality=self.t["quality"],
                    background="transparent", output_format="png",
                )
                break
            except Exception as e:
                name = type(e).__name__
                if name not in RETRYABLE or attempt == o["max_retries"]:
                    raise RenderError(f"OpenAI request failed: {name}: {str(getattr(e, 'message', ''))[:200]}") from None
                time.sleep(o["retry_backoff_seconds"] * (attempt + 1))
        png = base64.b64decode(resp.data[0].b64_json)
        size = Image.open(io.BytesIO(png)).size
        img_in, txt_in, out, usd = self._cost(getattr(resp, "usage", None))
        return RenderResult(png, size, time.perf_counter() - t0, img_in, txt_in, out, usd)
