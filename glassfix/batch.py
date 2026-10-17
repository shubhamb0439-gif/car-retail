"""Render a folder of cars (<input>/<car_id>/*.jpg) in parallel and write PNGs, costs.csv and a labelled sheet."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .imageio import list_car_photos, load_image
from .render import OpenAIRenderer, RenderError

CSV_FIELDS = ["car_id", "photo", "output", "quality", "status", "seconds", "input_image_tokens",
              "input_text_tokens", "output_tokens", "cost_usd", "output_size", "error"]


def _one(renderer: OpenAIRenderer, car_id: str, path: Path, out_dir: Path, quality: str) -> dict[str, Any]:
    row = {"car_id": car_id, "photo": path.name, "quality": quality, "output": ""}
    try:
        res = renderer.render(load_image(path), quality)
    except (RenderError, OSError, ValueError) as e:
        return {**row, "status": "failed", "error": str(e)}
    out = out_dir / car_id / f"{path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(res.png)
    return {**row, "status": "ok", "output": str(out.relative_to(out_dir)), "seconds": f"{res.seconds:.1f}",
            "input_image_tokens": res.input_image_tokens, "input_text_tokens": res.input_text_tokens,
            "output_tokens": res.output_tokens, "cost_usd": f"{res.cost_usd:.4f}",
            "output_size": f"{res.size[0]}x{res.size[1]}", "error": ""}


def _sheet(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    cols, cw, ch, lab = 4, 420, 260, 20
    n_rows = (len(ok) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cw, 24 + n_rows * (ch + lab)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), title, fill="black")
    for i, r in enumerate(ok):
        x, y = (i % cols) * cw, 24 + (i // cols) * (ch + lab)
        im = Image.open(out_dir / r["output"]).convert("RGBA")
        im.thumbnail((cw - 8, ch - 4))
        canvas.paste(Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB"), (x + 4, y + lab))
        d.text((x + 6, y + 3), f"{r['car_id']}/{r['photo']}  |  ${float(r['cost_usd']):.3f}  |  {r['seconds']}s", fill="black")
    canvas.save(out_dir / "sheet.jpg", quality=88)


def run(cfg: dict[str, Any], input_dir: Path, output_dir: Path, run_name: str, quality: str | None,
        renderer: OpenAIRenderer | None = None) -> list[dict[str, Any]]:
    quality = quality or cfg["openai"]["quality"]
    cars = list_car_photos(input_dir)
    if not cars:
        raise SystemExit(f"No photos found under {input_dir}")
    out_dir = output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = renderer or OpenAIRenderer(cfg)
    jobs = [(car_id, p) for car_id, paths in cars.items() for p in paths]
    print(f"[glassfix] {len(jobs)} photos, quality={quality}, model={cfg['openai']['model']}, "
          f"style reference={'yes' if renderer.reference else 'no'}", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["openai"]["concurrency"]) as pool:
        rows = list(pool.map(lambda j: _one(renderer, j[0], j[1], out_dir, quality), jobs))
    for r in rows:
        tail = f"${r['cost_usd']}  {r['seconds']}s" if r["status"] == "ok" else f"FAILED: {r['error']}"
        print(f"  {r['car_id']}/{r['photo']}: {tail}", flush=True)

    with open(out_dir / "costs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    total = sum(float(r["cost_usd"]) for r in ok)
    _sheet(rows, out_dir, f"{run_name}  |  quality {quality}  |  {len(ok)}/{len(rows)} ok  |  total ${total:.3f}")
    avg = total / len(ok) if ok else 0.0
    print(f"[glassfix] done: {len(ok)}/{len(rows)} ok, total ${total:.3f}, average ${avg:.4f} per photo -> {out_dir}")
    return rows
