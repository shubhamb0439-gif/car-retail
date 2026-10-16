"""Hand-off to external (generative) glass editors, which need their own Python environments.

  glassfix crops    detect glass, write one crop per photo (+ masks and metadata) to <run>/crops/
  <editor tool>     e.g. tools/edit_qwen.py or tools/edit_dai.py: <run>/crops -> <run>/edits_<name>/
  glassfix compose  paste each edited crop back inside the glass mask only, write outputs + report

Only pixels inside the glass mask can change; wipers and everything outside keep their original pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .detect import Box, Detection
from .imageio import load_rgb, save_mask_png, to_working
from .config import with_overrides
from .pipeline import Prepared, _log, _setup, _write_run_files, detect_all, finish, variant_name


def _boxes(bs: list[Box]) -> list[dict[str, Any]]:
    return [{"box": list(b.xyxy), "score": b.score, "label": b.label} for b in bs]


def _unboxes(items: list[dict[str, Any]]) -> list[Box]:
    return [Box(tuple(i["box"]), i["score"], i["label"]) for i in items]


def crop_box(mask_w: np.ndarray, cfg: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Padded bounding box (x0, y0, x1, y1) of the glass in working-resolution coordinates."""
    ys, xs = np.nonzero(mask_w)
    if len(ys) == 0:
        return None
    h, w = mask_w.shape
    pad = max(16, int(cfg["generative"]["crop_pad_frac"] * max(np.ptp(ys), np.ptp(xs))))
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))


def export_crops(cfg: dict[str, Any], input_dir: Path, output_dir: Path, run_name: str) -> Path:
    device, cars, run_dir = _setup(cfg, input_dir, output_dir, run_name)
    detector_name = cfg["detector"]["name"]
    preps, _ = detect_all(cars, detector_name, cfg, device)
    crops_dir = run_dir / "crops"
    g = cfg["generative"]
    n = 0
    for prep in preps:
        d = crops_dir / prep.car_id
        d.mkdir(parents=True, exist_ok=True)
        stem = prep.path.stem
        box = crop_box(prep.mask_w, cfg)
        meta = {
            "source_path": str(prep.path), "car_id": prep.car_id, "detector": detector_name,
            "image_size": list(prep.image_size), "working_size": [prep.work.shape[1], prep.work.shape[0]],
            "crop_box": [int(v) for v in box] if box else None,
            "glass_boxes": _boxes(prep.det.glass_boxes), "car_boxes": _boxes(prep.det.car_boxes),
            "rejected_boxes": _boxes(prep.det.rejected_boxes), "excluded_boxes": _boxes(prep.det.excluded_boxes),
            "timings": prep.timings,
        }
        save_mask_png(d / f"{stem}_glass_mask_work.png", prep.mask_w)
        excl = prep.det.exclude_mask if prep.det.exclude_mask is not None else np.zeros_like(prep.mask_w)
        save_mask_png(d / f"{stem}_exclude_mask_work.png", excl)
        if box:
            x0, y0, x1, y1 = box
            crop = prep.work[y0:y1, x0:x1]
            k = min(1.0, g["crop_max_side"] / max(crop.shape[:2]))
            if k < 1:
                crop = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
            Image.fromarray(crop).save(d / f"{stem}.png")
            # Glass mask for the crop too, so editors that inpaint can use it.
            mcrop = cv2.resize(prep.mask_w[y0:y1, x0:x1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            save_mask_png(d / f"{stem}_mask.png", mcrop)
            n += 1
        with open(d / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    with open(crops_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("Each <car>/<photo>.png is a glass crop to edit; <photo>_mask.png marks the glass in it.\n"
                "Write edited crops with the same relative path and size into a sibling folder, e.g.\n"
                "../edits_qwen/<car>/<photo>.png, then run: python -m glassfix compose ...\n")
    _log(f"[glassfix] {n} glass crops written to {crops_dir}")
    return crops_dir


def _load_prepared(meta_path: Path, cfg: dict[str, Any]) -> tuple[Prepared, dict[str, Any]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stem = meta_path.stem
    work, _ = to_working(load_rgb(meta["source_path"]), cfg["processing"]["max_side"])
    mask_w = (np.asarray(Image.open(meta_path.parent / f"{stem}_glass_mask_work.png")) > 127).astype(np.uint8)
    excl = (np.asarray(Image.open(meta_path.parent / f"{stem}_exclude_mask_work.png")) > 127).astype(np.uint8)
    det = Detection(mask_w, _unboxes(meta["glass_boxes"]), _unboxes(meta["car_boxes"]),
                    _unboxes(meta["rejected_boxes"]), {}, _unboxes(meta["excluded_boxes"]), excl)
    prep = Prepared(Path(meta["source_path"]), meta["car_id"], work, mask_w, det,
                    dict(meta["timings"]), tuple(meta["image_size"]))
    return prep, meta


def _edit_to_work(prep: Prepared, meta: dict[str, Any], edit_path: Path) -> np.ndarray:
    """Place an edited crop back into a working-resolution frame (float RGB 0..1, zero elsewhere)."""
    x0, y0, x1, y1 = meta["crop_box"]
    edited = np.asarray(Image.open(edit_path).convert("RGB")).astype(np.float32) / 255.0
    edited = cv2.resize(edited, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    out = prep.work.astype(np.float32) / 255.0
    out[y0:y1, x0:x1] = np.clip(edited, 0.0, 1.0)
    return out


def compose(cfg: dict[str, Any], run_dir: Path, edits: dict[str, Path], extra_styles: list[str]) -> Path:
    """Composite external edits (and optional built-in styles for reference) into a comparison report."""
    from .device import pick_device
    from .report import build_compare_report

    device = pick_device(cfg["device"])
    crops_dir = run_dir / "crops"
    metas = sorted(crops_dir.glob("*/*.json"))
    if not metas:
        raise SystemExit(f"No crops in {crops_dir}; run `glassfix crops` first")
    records = []
    for meta_path in metas:
        prep, meta = _load_prepared(meta_path, cfg)
        det_name = meta["detector"]
        for style in extra_styles:
            vcfg = with_overrides(cfg, **{"style.name": style, "style.preserve_cracks": True})
            v = variant_name(det_name, style, True)
            records.append(finish(prep, vcfg, run_dir / v / prep.car_id, det_name, device, v))
        for name, edit_dir in edits.items():
            edit_path = edit_dir / prep.car_id / f"{meta_path.stem}.png"
            v = f"{det_name}__gen-{name}__cracks-on"
            vcfg = with_overrides(cfg, **{"style.name": "external", "style.preserve_cracks": True})
            if meta["crop_box"] and not edit_path.exists():
                _log(f"  missing edit {edit_path}; skipping")
                continue
            ext = _edit_to_work(prep, meta, edit_path) if meta["crop_box"] else None
            records.append(finish(prep, vcfg, run_dir / v / prep.car_id, det_name, device, v, external_w=ext))
        _log(f"  composed {prep.car_id}/{prep.path.name}")
    _write_run_files(records, run_dir, cfg, device, {})
    report = build_compare_report(records, run_dir, cfg)
    _log(f"[glassfix] report: {report}")
    return report
