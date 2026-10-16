"""Command line: python -m glassfix {run,compare,doctor}."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

from . import device as _device  # noqa: F401  (sets env vars before torch/HF are imported)
from .config import load_config, with_overrides
from .detect import DETECTORS, model_ids, weights_cached


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", type=Path, default=Path("test_photos"), help="folder of <car_id>/<photos>")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--config", type=Path, default=None, help="config.yaml to use (default: repo config.yaml)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    p.add_argument("--max-side", type=int, default=None, help="processing resolution (long side)")


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def doctor(cfg: dict, download: bool) -> int:
    import torch

    from .device import pick_device

    print(f"python        {platform.python_version()} ({sys.executable})")
    print(f"platform      {platform.platform()}")
    dev = pick_device(cfg["device"])
    print(f"device        {dev}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  cuda        {torch.version.cuda}, {p.name}, {p.total_memory / 2**30:.1f} GB")
    mps = getattr(torch.backends, "mps", None)
    print(f"  mps         {'available' if mps is not None and mps.is_available() else 'not available'}")
    for pkg in ("torch", "torchvision", "transformers", "huggingface-hub", "opencv-contrib-python",
                "scikit-image", "numpy", "scipy", "pillow", "pillow-heif", "pyyaml"):
        print(f"{pkg:<22}{_version(pkg)}")
    try:
        import cv2

        print(f"cv2.ximgproc          {'ok' if hasattr(cv2, 'ximgproc') else 'MISSING (guided filter unavailable)'}")
    except ImportError:
        print("cv2                   MISSING")

    models_dir = cfg["models_dir"]
    print(f"\nmodel weights in {Path(models_dir).resolve()}:")
    ids = list(dict.fromkeys(m for d in DETECTORS for m in model_ids(cfg, d)))
    missing = []
    for mid in ids:
        ok = weights_cached(mid, models_dir)
        print(f"  {'[x]' if ok else '[ ]'} {mid}")
        if not ok:
            missing.append(mid)
    if missing and download:
        from huggingface_hub import snapshot_download

        for mid in missing:
            print(f"downloading {mid} ...", flush=True)
            snapshot_download(mid, cache_dir=str(Path(models_dir).resolve()))
        print("done.")
    elif missing:
        print("  (missing weights download automatically on first run, or use `doctor --download`)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glassfix", description="Car glass reflection removal PoC")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a folder with one method")
    _common(r)
    r.add_argument("--style", choices=["cutout", "studiowhite", "cleanglass", "dereflect", "clear", "tint", "studio"], default=None)
    r.add_argument("--detector", choices=list(DETECTORS), default=None)
    r.add_argument("--preserve-cracks", action=argparse.BooleanOptionalAction, default=None)

    c = sub.add_parser("compare", help="run every detector x style x crack-preservation combination")
    _common(c)

    cr = sub.add_parser("crops", help="detect glass and export crops for an external (generative) editor")
    _common(cr)
    cr.add_argument("--detector", choices=list(DETECTORS), default=None)

    co = sub.add_parser("compose", help="paste externally edited crops back and build a comparison report")
    co.add_argument("--output", type=Path, default=Path("results"))
    co.add_argument("--run-name", required=True)
    co.add_argument("--config", type=Path, default=None)
    co.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    co.add_argument("--max-side", type=int, default=None)
    co.add_argument("--edits", nargs="+", default=[], metavar="NAME=DIR",
                    help="edited crop folders, e.g. qwen=results/gen1/edits_qwen dai=results/gen1/edits_dai")
    co.add_argument("--with-styles", nargs="*", default=["cleanglass"],
                    help="built-in styles to include for reference (default: cleanglass)")

    d = sub.add_parser("doctor", help="print device, versions and model-weight status")
    d.add_argument("--config", type=Path, default=None)
    d.add_argument("--download", action="store_true", help="download any missing model weights")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    # Must be set before torch is imported (it is imported lazily, so this is in time).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(cfg.get("cpu_threads", 4)))
    if args.cmd == "doctor":
        return doctor(cfg, args.download)

    cfg = with_overrides(cfg, **{"device": args.device, "processing.max_side": args.max_side})
    from . import pipeline

    if args.cmd == "crops":
        from .generative import export_crops

        cfg = with_overrides(cfg, **{"detector.name": args.detector})
        export_crops(cfg, args.input, args.output, args.run_name)
        return 0
    if args.cmd == "compose":
        from .generative import compose

        edits = {}
        for item in args.edits:
            name, _, folder = item.partition("=")
            if not folder:
                raise SystemExit(f"--edits expects NAME=DIR, got {item!r}")
            edits[name] = Path(folder)
        compose(cfg, args.output / args.run_name, edits, args.with_styles)
        return 0
    if args.cmd == "run":
        cfg = with_overrides(cfg, **{"style.name": args.style, "detector.name": args.detector,
                                     "style.preserve_cracks": args.preserve_cracks})
        pipeline.run(cfg, args.input, args.output, args.run_name)
    else:
        pipeline.compare(cfg, args.input, args.output, args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
