"""Command line: python -m glassfix {render,check}."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import ROOT, load_config, with_overrides


def check(cfg: dict) -> int:
    """Verify setup without spending money: key present, reference found, account can list models."""
    ok = True
    key = os.environ.get("OPENAI_API_KEY", "")
    print(f"OPENAI_API_KEY        {'set' if key else 'MISSING'}")
    if key != key.strip():
        print("  warning: the key has leading/trailing spaces; re-set it without them")
    tier = cfg["tiers"][cfg["default_tier"]]
    ref = tier.get("style_reference")
    if ref:
        p = Path(ref) if Path(ref).is_absolute() else ROOT / ref
        print(f"style reference       {'found' if p.exists() else 'MISSING'} ({p})")
        ok &= p.exists()
    print(f"model / default tier  {cfg['openai']['model']} / {cfg['default_tier']} (quality {tier['quality']})")
    if key:
        try:
            from openai import OpenAI

            ids = {m.id for m in OpenAI(max_retries=0).models.list().data}
            has = cfg["openai"]["model"] in ids
            print(f"API access            ok; model {'available' if has else 'NOT listed for this key'}")
            ok &= has
        except Exception as e:  # never print headers or the key
            print(f"API access            FAILED ({type(e).__name__})")
            ok = False
    return 0 if ok and key else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glassfix", description="Car photo -> transparent studio PNG (OpenAI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a folder of car photos (<input>/<car_id>/*.jpg)")
    r.add_argument("--input", type=Path, default=Path("test_photos"))
    r.add_argument("--output", type=Path, default=Path("results"))
    r.add_argument("--run-name", required=True)
    r.add_argument("--tier", default=None, help="service tier from config.yaml: low | medium | high")
    r.add_argument("--view", choices=["exterior", "interior"], default="exterior",
                   help="exterior photos (default) or interior / open-door photos")
    r.add_argument("--no-reference", action="store_true", help="don't send the style reference image")
    r.add_argument("--config", type=Path, default=None)
    c = sub.add_parser("check", help="check API key, style reference and model access (free)")
    c.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "check":
        return check(cfg)

    from .batch import run
    from .render import OpenAIRenderer

    tier = args.tier or cfg["default_tier"]
    if args.no_reference:
        table = "tiers" if args.view == "exterior" else "interior_tiers"
        cfg = with_overrides(cfg, **{f"{table}.{tier}.style_reference": ""})
    rows = run(cfg, args.input, args.output, args.run_name, tier, OpenAIRenderer(cfg, tier, root=ROOT, view=args.view))
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
