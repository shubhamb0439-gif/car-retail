"""COMPARISON ONLY: edit glass crops with Dereflection-Any-Image (AAAI 2026).

License caveat: the code is MIT and the weights are tagged Apache-2.0, but the weights are fine-tuned
from Stable Diffusion 2.1 (CreativeML OpenRAIL++-M), so this does NOT meet the project's
MIT/Apache/BSD-only rule. Use it to compare quality, not in a delivered product.

Runs in its own environment (it pins diffusers 0.32 / transformers 4.46) with the DAI repository:

    git clone https://github.com/Abuuu122/Dereflection-Any-Image /content/DAI-repo
    python -m venv /content/dai_env && /content/dai_env/bin/pip install -r /content/DAI-repo/requirements.txt pyyaml peft
    /content/dai_env/bin/python tools/edit_dai.py results/gen1/crops results/gen1/edits_dai --repo /content/DAI-repo

Model loading follows the repository's run.py. Not run on the 4 GB development laptop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crops", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--repo", type=Path, required=True, help="clone of github.com/Abuuu122/Dereflection-Any-Image")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    args = ap.parse_args()

    d = yaml.safe_load(args.config.read_text(encoding="utf-8"))["generative"]["dai"]
    sys.path.insert(0, str(args.repo.resolve()))
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel

    from DAI.controlnetvae import ControlNetVAEModel
    from DAI.decoder import CustomAutoencoderKL
    from DAI.pipeline_all import DAIPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mid = d["model_id"]
    t0 = time.time()
    controlnet = ControlNetVAEModel.from_pretrained(mid, subfolder="controlnet", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet", torch_dtype=dtype).to(device)
    vae_2 = CustomAutoencoderKL.from_pretrained(mid, subfolder="vae_2", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").to(device)
    tokenizer = AutoTokenizer.from_pretrained(mid, subfolder="tokenizer", use_fast=False)
    pipe = DAIPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
                       safety_checker=None, scheduler=None, feature_extractor=None, t_start=0).to(device)
    print(f"pipeline ready in {time.time() - t0:.0f}s on {device}", flush=True)

    crops = sorted(p for p in args.crops.glob("*/*.png") if not p.stem.endswith(("_mask", "_work")))
    for i, crop in enumerate(crops, 1):
        rel = crop.relative_to(args.crops)
        dst = args.out / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(crop).convert("RGB")
        resolution = None if max(img.size) < 768 else 0  # as in the repository's run.py
        t0 = time.time()
        with torch.inference_mode():
            pred = pipe(
                image=torch.tensor(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
                prompt=d["prompt"], vae_2=vae_2, processing_resolution=resolution,
            ).prediction[0]
        pred = np.clip((pred + 1) / 2, 0, 1)
        Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.LANCZOS).save(dst)
        print(f"[{i}/{len(crops)}] {rel}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
