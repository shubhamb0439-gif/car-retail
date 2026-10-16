"""Per-photo pipeline, folder runs and the method comparison."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import style_fingerprint, with_overrides
from .detect import Detection, GlassDetector
from .device import pick_device, set_seeds
from .imageio import list_car_photos, load_rgb, save_jpg, save_mask_png, to_working
from .masks import clean_mask
from .restyle import restyle

CSV_FIELDS = [
    "variant", "car_id", "photo", "detector", "style", "preserve_cracks", "glass_found", "glass_area_pct",
    "glass_regions", "crack_detected", "crack_length_px", "total_seconds", "device", "style_fingerprint",
]


# Styles for which crack preservation changes nothing (they never remove real detail).
NO_CRACK_TOGGLE = {"clear", "dereflect"}  # cutout keeps the toggle (cracks are drawn on its glass)


@dataclass
class Prepared:
    """Everything that depends only on the photo and the detector (shared by all style variants).

    The full-resolution image is not kept (reloaded when writing outputs) so large sets fit in memory.
    """
    path: Path
    car_id: str
    work: np.ndarray
    mask_w: np.ndarray
    det: Detection
    timings: dict[str, float]
    image_size: tuple[int, int]                 # (width, height) of the original
    reflection_w: np.ndarray | None = None      # DSIT reflection layer (style 'dereflect' only)


def variant_name(detector: str, style: str, preserve: bool) -> str:
    return f"{detector}__{style}__cracks-{'on' if preserve else 'off'}"


def prepare(path: Path, car_id: str, detector: GlassDetector, cfg: dict[str, Any]) -> Prepared:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    full = load_rgb(path)
    work, _ = to_working(full, cfg["processing"]["max_side"])
    size = (full.shape[1], full.shape[0])
    del full
    t["load"] = time.perf_counter() - t0

    det = detector.detect(work)
    t.update(det.timings)

    t0 = time.perf_counter()
    cars = sorted(det.car_boxes, key=lambda b: -b.area)
    if cfg["mask"]["car_box_mode"] == "largest":
        cars = cars[:1]
    mask_w = clean_mask(det.raw_mask, [c.xyxy for c in cars], cfg["mask"])
    t["mask_clean"] = time.perf_counter() - t0
    return Prepared(path, car_id, work, mask_w, det, t, size)


def detect_all(cars: dict[str, list[Path]], detector_name: str, cfg: dict[str, Any],
               device: str) -> tuple[list[Prepared], float]:
    """Run one detector over every photo, then unload it (frees memory for DSIT)."""
    _log(f"[glassfix] device={device} loading {detector_name} ...")
    detector = GlassDetector(detector_name, cfg, device)
    preps = []
    for car_id, paths in cars.items():
        for p in paths:
            set_seeds(cfg["seed"])
            prep = prepare(p, car_id, detector, cfg)
            preps.append(prep)
            _log(f"  {detector_name} {car_id}/{p.name}: glass area "
                 f"{100 * prep.mask_w.mean():.2f}% detect {sum(prep.timings.values()):.1f}s")
    detector.close()
    return preps, detector.load_seconds


def add_reflections(preps: list[Prepared], cfg: dict[str, Any], device: str) -> float:
    """Run DSIT on every photo's glass (style 'dereflect'). Loaded only after the detector is gone."""
    from .dereflect import DsitDereflector, reflection_map

    _log("[glassfix] loading DSIT reflection-removal model ...")
    model = DsitDereflector(cfg, device)
    for prep in preps:
        if prep.mask_w.any():
            t0 = time.perf_counter()
            prep.reflection_w = reflection_map(prep.work, prep.mask_w, model, cfg)
            prep.timings["dereflect"] = time.perf_counter() - t0
            _log(f"  dsit {prep.car_id}/{prep.path.name}: {prep.timings['dereflect']:.1f}s")
    model.close()
    return model.load_seconds


def overlay_panel(rgb: np.ndarray, glass: np.ndarray, crack: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32)
    g = glass > 0
    out[g] = out[g] * 0.55 + np.array([0, 190, 255], np.float32) * 0.45
    out[crack > 0] = (255, 40, 40)
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 120, 255), max(2, rgb.shape[1] // 600))
    return out


def compare_strip(full: np.ndarray, glass: np.ndarray, crack: np.ndarray, clean: np.ndarray,
                  height: int) -> np.ndarray:
    """original | mask overlay | result, drawn at preview size (cheap on memory)."""
    size = (round(full.shape[1] * height / full.shape[0]), height)
    small = cv2.resize(full, size, interpolation=cv2.INTER_AREA)
    g = cv2.resize(glass, size, interpolation=cv2.INTER_NEAREST)
    c = cv2.dilate(cv2.resize(crack, size, interpolation=cv2.INTER_AREA), np.ones((2, 2), np.uint8))
    panels = [small, overlay_panel(small, g, c), cv2.resize(clean, size, interpolation=cv2.INTER_AREA)]
    gap = np.full((height, 8, 3), 255, np.uint8)
    return np.hstack([panels[0], gap, panels[1], gap, panels[2]])


def finish(prep: Prepared, cfg: dict[str, Any], out_dir: Path, detector: str, device: str,
           variant: str, external_w: np.ndarray | None = None) -> dict[str, Any]:
    """Restyle a prepared photo with cfg's style and write all per-photo outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path, name, style = prep.path, prep.path.stem, cfg["style"]["name"]
    timings = {k: v for k, v in prep.timings.items() if k != "dereflect" or style == "dereflect"}
    glass_found = bool(prep.mask_w.any())

    t0 = time.perf_counter()
    full = load_rgb(path)
    timings["load"] = timings.get("load", 0.0) + time.perf_counter() - t0
    fallback_pct = 0.0
    rgba = None
    if glass_found and style == "cutout":
        from .cutout import make_cutout

        car_w = prep.det.car_mask if prep.det.car_mask is not None else np.zeros_like(prep.mask_w)
        res, rgba = make_cutout(full, prep.work, prep.mask_w, cfg, car_w, prep.det.interior_mask,
                                prep.det.exclude_mask)
        timings.update(res.timings)
        clean, glass_m, crack_m, crack_info = res.clean, res.glass_mask, res.crack_mask, res.crack_info
    elif glass_found:
        res = restyle(full, prep.work, prep.mask_w, cfg,
                      reflection_w=prep.reflection_w if style == "dereflect" else None,
                      exclude_w=prep.det.exclude_mask, external_w=external_w,
                      interior_w=prep.det.interior_mask)
        timings.update(res.timings)
        clean, glass_m, crack_m, crack_info = res.clean, res.glass_mask, res.crack_mask, res.crack_info
        fallback_pct = res.fallback_pct
    else:  # nothing to do: output the original untouched
        clean = full
        glass_m = crack_m = np.zeros(full.shape[:2], np.uint8)
        crack_info = {"crack_detected": False, "crack_length_px": 0, "crack_components": 0}

    t0 = time.perf_counter()
    q = cfg["processing"]["jpeg_quality"]
    save_jpg(out_dir / f"{name}_clean.jpg", clean, q)
    if rgba is not None:
        Image.fromarray(rgba, "RGBA").save(out_dir / f"{name}_cutout.png", compress_level=6)
    save_mask_png(out_dir / f"{name}_glass_mask.png", glass_m)
    save_mask_png(out_dir / f"{name}_crack_mask.png", crack_m)
    if prep.det.interior_mask is not None and prep.det.interior_mask.any():
        save_mask_png(out_dir / f"{name}_interior_mask.png",
                      cv2.resize(prep.det.interior_mask * 255, (full.shape[1], full.shape[0]),
                                 interpolation=cv2.INTER_NEAREST))
    strip = compare_strip(full, glass_m, crack_m, clean, cfg["processing"]["compare_panel_height"])
    save_jpg(out_dir / f"{name}_compare.jpg", strip, 90)
    timings["save"] = time.perf_counter() - t0

    n_regions = cv2.connectedComponents((glass_m > 0).astype(np.uint8))[0] - 1
    record = {
        "variant": variant,
        "car_id": prep.car_id,
        "photo": path.name,
        "source_path": str(path),
        "detector": detector,
        "style": style,
        "preserve_cracks": bool(cfg["style"]["preserve_cracks"]) or style in NO_CRACK_TOGGLE,
        "device": device,
        "glass_found": glass_found,
        "glass_area_pct": round(100.0 * float((glass_m > 0).mean()), 3),
        "glass_regions": int(n_regions),
        **crack_info,
        "fallback_pct": fallback_pct,
        "glass_boxes": [{"box": [round(v, 1) for v in b.xyxy], "score": round(b.score, 3), "label": b.label}
                        for b in prep.det.glass_boxes],
        "rejected_boxes": [{"box": [round(v, 1) for v in b.xyxy], "label": b.label} for b in prep.det.rejected_boxes],
        "car_boxes": [[round(v, 1) for v in b.xyxy] for b in prep.det.car_boxes],
        "excluded_boxes": [{"box": [round(v, 1) for v in b.xyxy], "label": b.label} for b in prep.det.excluded_boxes],
        "image_size": list(prep.image_size),
        "working_size": [prep.work.shape[1], prep.work.shape[0]],
        "style_fingerprint": style_fingerprint(cfg),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "total_seconds": round(sum(timings.values()), 3),
        "outputs": {k: f"{name}_{k}.{ext}" for k, ext in
                    (("clean", "jpg"), ("glass_mask", "png"), ("crack_mask", "png"), ("compare", "jpg"))
                    + ((("cutout", "png"),) if rgba is not None else ())},
    }
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def write_summary(records: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _setup(cfg: dict[str, Any], input_dir: Path, output_dir: Path, run_name: str):
    set_seeds(cfg["seed"])
    device = pick_device(cfg["device"])
    cars = list_car_photos(input_dir)
    if not cars:
        raise SystemExit(f"No photos found under {input_dir}")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return device, cars, run_dir


def _write_run_files(records, run_dir: Path, cfg, device, load_times) -> None:
    write_summary(records, run_dir / "summary.csv")
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "device": device, "model_load_seconds": load_times}, f, indent=2)


def run(cfg: dict[str, Any], input_dir: Path, output_dir: Path, run_name: str) -> Path:
    from .report import build_run_report

    device, cars, run_dir = _setup(cfg, input_dir, output_dir, run_name)
    detector_name, style = cfg["detector"]["name"], cfg["style"]["name"]
    variant = variant_name(detector_name, style, cfg["style"]["preserve_cracks"])
    _log(f"[glassfix] style={style}")
    preps, det_load = detect_all(cars, detector_name, cfg, device)
    load_times = {detector_name: round(det_load, 2)}
    if style == "dereflect":
        load_times["dsit"] = round(add_reflections(preps, cfg, device), 2)
    records = []
    for prep in preps:
        rec = finish(prep, cfg, run_dir / prep.car_id, detector_name, device, variant)
        records.append(rec)
        _log(f"  {prep.car_id}/{prep.path.name}: glass={rec['glass_found']} area={rec['glass_area_pct']}% "
             f"cracks={rec['crack_detected']} {rec['total_seconds']:.1f}s")
    _write_run_files(records, run_dir, cfg, device, load_times)
    report = build_run_report(records, run_dir, cfg)
    _log(f"[glassfix] report: {report}")
    return report


def compare(cfg: dict[str, Any], input_dir: Path, output_dir: Path, run_name: str) -> Path:
    from .report import build_compare_report

    device, cars, run_dir = _setup(cfg, input_dir, output_dir, run_name)
    c = cfg["compare"]
    records: list[dict[str, Any]] = []
    load_times: dict[str, float] = {}
    for detector_name in c["detectors"]:
        dcfg = with_overrides(cfg, **{"detector.name": detector_name})
        preps, det_load = detect_all(cars, detector_name, dcfg, device)
        load_times[detector_name] = round(det_load, 2)
        if "dereflect" in c["styles"]:
            load_times[f"dsit ({detector_name})"] = round(add_reflections(preps, dcfg, device), 2)
        for prep in preps:
            for style in c["styles"]:
                for preserve in c["preserve_cracks"]:
                    if style in NO_CRACK_TOGGLE and not preserve:
                        continue  # these styles never remove cracks: the "off" variant would be identical
                    vcfg = with_overrides(dcfg, **{"style.name": style, "style.preserve_cracks": preserve})
                    variant = variant_name(detector_name, style, preserve)
                    records.append(finish(prep, vcfg, run_dir / variant / prep.car_id, detector_name,
                                          device, variant))
            _log(f"  styled {detector_name} {prep.car_id}/{prep.path.name}")
        del preps
    _write_run_files(records, run_dir, cfg, device, load_times)
    report = build_compare_report(records, run_dir, cfg)
    _log(f"[glassfix] report: {report}")
    return report
