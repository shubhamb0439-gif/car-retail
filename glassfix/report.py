"""Self-contained HTML reports (no external requests): run report and method-comparison report."""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .imageio import load_rgb
from .pipeline import overlay_panel


class ImageStore:
    """Embeds each image once; markup refers to images by id."""

    def __init__(self, long_side: int, quality: int):
        self.long_side, self.quality = long_side, quality
        self.data: dict[str, str] = {}

    def add(self, key: str, rgb: np.ndarray) -> str:
        if key not in self.data:
            im = Image.fromarray(rgb)
            im.thumbnail((self.long_side, self.long_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.quality)
            self.data[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return key

    def add_file(self, key: str, path: Path) -> str:
        if key not in self.data:
            self.add(key, load_rgb(path))
        return key

    def script(self) -> str:
        return "const IMG=" + json.dumps(self.data) + ";"


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    return {
        "photos": n,
        "glass_pct": round(100 * sum(r["glass_found"] for r in records) / n, 1) if n else 0.0,
        "cracks": sum(r["crack_detected"] for r in records),
        "avg_s": round(sum(r["total_seconds"] for r in records) / n, 2) if n else 0.0,
    }


def _tiles(s: dict[str, Any]) -> str:
    items = [("Photos processed", s["photos"]), ("Glass found", f"{s['glass_pct']}%"),
             ("Cracks flagged", s["cracks"]), ("Avg time / photo", f"{s['avg_s']} s")]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tv">{_esc(v)}</div><div class="tl">{_esc(k)}</div></div>' for k, v in items
    ) + "</div>"


def _slider(before_id: str, after_id: str, alt: str) -> str:
    return f"""
<div class="ba" style="--pos:50%">
  <img data-img="{after_id}" alt="{_esc(alt)} after" class="ba-after">
  <div class="ba-before"><img data-img="{before_id}" alt="{_esc(alt)} before"></div>
  <div class="ba-handle"></div>
  <span class="ba-tag l">Before</span><span class="ba-tag r">After</span>
  <input type="range" min="0" max="100" value="50" aria-label="Before/after position for {_esc(alt)}">
</div>"""


def _metrics(r: dict[str, Any]) -> str:
    crack = '<span class="pill warn">crack flagged</span>' if r["crack_detected"] else '<span class="pill">no crack</span>'
    glass = '<span class="pill ok">glass found</span>' if r["glass_found"] else '<span class="pill bad">no glass found</span>'
    return (f'<div class="meta">{glass}{crack}<span>glass area {r["glass_area_pct"]}%</span>'
            f'<span>{r["glass_regions"]} region(s)</span><span>{r["total_seconds"]} s</span></div>')


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16191d;--mut:#5d6672;--line:#e3e6ea;--acc:#1f5fbf;--ok:#1d7a46;--warn:#a35a00;--bad:#b3261e;--chip:#eef1f5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121417;--card:#1b1e22;--ink:#e8eaed;--mut:#9aa3ad;--line:#2b3036;--acc:#7aa7ff;--ok:#5cc28a;--warn:#f0a948;--bad:#ff8a80;--chip:#252a30}}
:root[data-theme="dark"]{--bg:#121417;--card:#1b1e22;--ink:#e8eaed;--mut:#9aa3ad;--line:#2b3036;--acc:#7aa7ff;--ok:#5cc28a;--warn:#f0a948;--bad:#ff8a80;--chip:#252a30}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 4px}
.sub{color:var(--mut);margin:0 0 24px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:8px 0 8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tv{font-size:24px;font-weight:650}.tl{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:600;background:var(--chip)}
tr:last-child td{border-bottom:0}
.tablewrap{overflow-x:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0}
.card h3{font-size:15px;margin:0 0 10px;font-weight:600}
.meta{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;color:var(--mut);font-size:13px;margin-top:10px}
.pill{background:var(--chip);border-radius:99px;padding:2px 10px;color:var(--ink)}
.pill.ok{color:var(--ok)}.pill.warn{color:var(--warn)}.pill.bad{color:var(--bad)}
.carhead{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline}
.fp{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--mut)}
.ba{position:relative;overflow:hidden;border-radius:8px;user-select:none;background:#000;line-height:0}
.ba img{display:block;width:100%;height:auto}
.ba-before{position:absolute;inset:0;clip-path:inset(0 calc(100% - var(--pos)) 0 0)}
.ba-before img{width:100%;height:100%;object-fit:cover}
.ba-handle{position:absolute;top:0;bottom:0;left:var(--pos);width:2px;margin-left:-1px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.25);pointer-events:none}
.ba-handle::after{content:"";position:absolute;top:50%;left:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.35)}
.ba input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}
.ba input:focus-visible+*,.ba:focus-within{outline:2px solid var(--acc);outline-offset:2px}
.ba-tag{position:absolute;top:10px;font:600 12px/1 sans-serif;color:#fff;background:rgba(0,0,0,.55);padding:5px 8px;border-radius:5px;pointer-events:none}
.ba-tag.l{left:10px}.ba-tag.r{right:10px}
details{margin-top:10px}summary{cursor:pointer;color:var(--acc);font-size:13px}
details img{width:100%;border-radius:8px;margin-top:8px}
.variants{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.variants button{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:7px;padding:5px 9px;cursor:pointer}
.variants button[aria-pressed="true"]{background:var(--acc);border-color:var(--acc);color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:12px}
.grid figure{margin:0;cursor:pointer}.grid img{width:100%;border-radius:6px;display:block;border:2px solid transparent}
.grid figure.sel img{border-color:var(--acc)}
.grid figcaption{font-size:12px;color:var(--mut);margin-top:3px;line-height:1.3}
.legend{color:var(--mut);font-size:13px}
footer{margin-top:48px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px}
"""

JS = """
document.querySelectorAll('img[data-img]').forEach(i=>{i.src=IMG[i.dataset.img]});
document.querySelectorAll('.ba').forEach(b=>{const r=b.querySelector('input');const f=()=>b.style.setProperty('--pos',r.value+'%');r.addEventListener('input',f);f();});
document.querySelectorAll('[data-photo]').forEach(card=>{
  const after=card.querySelector('.ba-after'),meta=card.querySelector('.vmeta'),btns=card.querySelectorAll('.variants button'),figs=card.querySelectorAll('.grid figure[data-k]');
  const pick=k=>{after.src=IMG[k];btns.forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===k));figs.forEach(f=>f.classList.toggle('sel',f.dataset.k===k));
    const m=card.querySelector('template[data-k="'+k+'"]');if(m)meta.innerHTML=m.innerHTML;};
  btns.forEach(b=>b.addEventListener('click',()=>pick(b.dataset.k)));figs.forEach(f=>f.addEventListener('click',()=>pick(f.dataset.k)));
  if(btns.length)pick(btns[0].dataset.k);
});
"""


def _page(title: str, body: str, store: ImageStore) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)}</title>'
            f'<style>{CSS}</style></head><body><main>{body}'
            f'<footer>Generated by glassfix {dt.datetime.now():%Y-%m-%d %H:%M}. Proof of concept: results are '
            f'automatic and unretouched. Drag the slider (or focus it and use the arrow keys) to compare.</footer>'
            f'</main><script>{store.script()}{JS}</script></body></html>')


def _overlay_for(r: dict[str, Any], out_dir: Path, long_side: int) -> np.ndarray:
    """Mask overlay drawn at report size."""
    import cv2

    full = load_rgb(Path(r["source_path"]))
    k = min(1.0, long_side / max(full.shape[:2]))
    size = (round(full.shape[1] * k), round(full.shape[0] * k))
    small = cv2.resize(full, size, interpolation=cv2.INTER_AREA)
    g = cv2.resize(np.asarray(Image.open(out_dir / r["outputs"]["glass_mask"])), size, interpolation=cv2.INTER_NEAREST)
    c = cv2.resize(np.asarray(Image.open(out_dir / r["outputs"]["crack_mask"])), size, interpolation=cv2.INTER_AREA)
    return overlay_panel(small, g, c)


# ------------------------------------------------------------------ run report
def build_run_report(records: list[dict[str, Any]], run_dir: Path, cfg: dict[str, Any]) -> Path:
    rc = cfg["report"]
    store = ImageStore(rc["image_long_side"], rc["jpeg_quality"])
    first = records[0]
    parts = [f"<h1>{_esc(rc['title'])}</h1>",
             f'<p class="sub">Run <b>{_esc(run_dir.name)}</b> · detector {_esc(first["detector"])} · '
             f'style {_esc(first["style"])} · crack preservation {"on" if first["preserve_cracks"] else "off"} · '
             f'device {_esc(first["device"])}</p>',
             _tiles(_summary(records))]

    by_car: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_car[r["car_id"]].append(r)

    for car_id, recs in by_car.items():
        fps = {r["style_fingerprint"] for r in recs}
        consistent = "identical style on all photos" if len(fps) == 1 else "WARNING: style differs between photos"
        parts.append(f'<div class="carhead"><h2>{_esc(car_id)}</h2>'
                     f'<span class="fp">style {" / ".join(sorted(fps))}: {consistent}</span></div>')
        for r in recs:
            out_dir = run_dir / car_id
            key = f"{car_id}/{r['photo']}"
            b = store.add_file("o:" + key, Path(r["source_path"]))
            a = store.add_file("c:" + key, out_dir / r["outputs"]["clean"])
            m = store.add("m:" + key, _overlay_for(r, out_dir, store.long_side))
            parts.append(f'<div class="card"><h3>{_esc(r["photo"])}</h3>{_slider(b, a, key)}{_metrics(r)}'
                         f'<details><summary>Show detected glass (blue) and cracks (red)</summary>'
                         f'<img data-img="{m}" alt="mask overlay {_esc(key)}"></details></div>')

    out = run_dir / "report.html"
    out.write_text(_page(rc["title"], "".join(parts), store), encoding="utf-8")
    return out


# ------------------------------------------------------------------ compare report
def _variant_label(v: str) -> str:
    det, style, cr = v.split("__")
    det = {"groundingdino_sam2": "Grounding DINO", "florence2_sam2": "Florence-2"}.get(det, det)
    if style.startswith("gen-"):
        return f"{det} · generative: {style[4:]}"
    if style == "clear":
        return f"{det} · clear (original glass)"
    if style == "dereflect":
        return f"{det} · dereflect (DSIT)"
    if style == "cutout":
        return f"{det} · cut-out PNG · {cr.replace('-', ' ')}"
    if style == "studiowhite":
        return f"{det} · studio white · {cr.replace('-', ' ')}"
    if style == "cleanglass":
        return f"{det} · clean glass · {cr.replace('-', ' ')}"
    return f"{det} · {style} · {cr.replace('-', ' ')}"


def build_compare_report(records: list[dict[str, Any]], run_dir: Path, cfg: dict[str, Any]) -> Path:
    rc = cfg["report"]
    store = ImageStore(rc["compare_image_long_side"], rc["jpeg_quality"])
    variants = list(dict.fromkeys(r["variant"] for r in records))
    device = records[0]["device"] if records else "?"

    rows = []
    for v in variants:
        s = _summary([r for r in records if r["variant"] == v])
        rows.append(f"<tr><td>{_esc(_variant_label(v))}</td><td>{s['photos']}</td><td>{s['glass_pct']}%</td>"
                    f"<td>{s['cracks']}</td><td>{s['avg_s']} s</td></tr>")
    photos_total = len({(r["car_id"], r["photo"]) for r in records})
    parts = [f"<h1>{_esc(rc['title'])}: method comparison</h1>",
             f'<p class="sub">Run <b>{_esc(run_dir.name)}</b> · {photos_total} photos × {len(variants)} variants · '
             f'device {_esc(device)}. Avg time per photo = detection + restyling for one variant (detection is '
             f'shared by the variants of the same detector).</p>',
             _tiles(_summary(records) | {"photos": photos_total}),
             '<h2>Summary by variant</h2><div class="tablewrap"><table><thead><tr><th>Variant</th><th>Photos</th>'
             '<th>Glass found</th><th>Cracks flagged</th><th>Avg time / photo</th></tr></thead><tbody>'
             + "".join(rows) + "</tbody></table></div>",
             '<p class="legend">Pick a variant with the buttons or the thumbnails; the slider compares it with the '
             'original. Mask overlays (blue = glass, red = crack) are under each photo.</p>']

    by_photo: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_photo[(r["car_id"], r["photo"])][r["variant"]] = r

    current_car = None
    for (car_id, photo), vr in by_photo.items():
        if car_id != current_car:
            parts.append(f"<h2>{_esc(car_id)}</h2>")
            current_car = car_id
        key = f"{car_id}/{photo}"
        any_r = next(iter(vr.values()))
        orig = store.add_file("o:" + key, Path(any_r["source_path"]))
        buttons, figs, temps, overlays = [], [], [], []
        seen_det = set()
        for v in variants:
            r = vr.get(v)
            if r is None:
                continue
            out_dir = run_dir / v / car_id
            k = store.add_file(f"c:{v}:{key}", out_dir / r["outputs"]["clean"])
            label = _variant_label(v)
            buttons.append(f'<button type="button" data-k="{k}" aria-pressed="false">{_esc(label)}</button>')
            figs.append(f'<figure data-k="{k}"><img data-img="{k}" alt="{_esc(label)}" loading="lazy">'
                        f'<figcaption>{_esc(label)}<br>glass {r["glass_area_pct"]}% · '
                        f'{"crack flagged" if r["crack_detected"] else "no crack"} · {r["total_seconds"]} s</figcaption></figure>')
            temps.append(f'<template data-k="{k}">{_metrics(r)}</template>')
            if r["detector"] not in seen_det:
                seen_det.add(r["detector"])
                m = store.add(f"m:{r['detector']}:{key}", _overlay_for(r, out_dir, store.long_side))
                overlays.append(f'<figure><img data-img="{m}" alt="mask {_esc(r["detector"])}">'
                                f'<figcaption>{_esc(_variant_label(v).split(" · ")[0])} detection</figcaption></figure>')
        parts.append(
            f'<div class="card" data-photo><h3>{_esc(photo)}</h3><div class="variants">{"".join(buttons)}</div>'
            f'{_slider(orig, orig, key)}<div class="vmeta"></div>{"".join(temps)}'
            f'<div class="grid">{"".join(figs)}</div>'
            f'<details><summary>Show detected glass (blue) and cracks (red) per detector</summary>'
            f'<div class="grid">{"".join(overlays)}</div></details></div>')

    out = run_dir / "report.html"
    out.write_text(_page(rc["title"] + " (comparison)", "".join(parts), store), encoding="utf-8")
    return out
