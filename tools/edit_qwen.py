"""Edit glass crops with Qwen-Image-Edit-2509 (Apache-2.0) + the Lightning 8-step LoRA (Apache-2.0).

Runs in its own environment on a CUDA GPU (Colab/Kaggle); it does not import glassfix.

    pip install -r tools/requirements-qwen.txt
    python tools/edit_qwen.py results/gen1/crops results/gen1/edits_qwen

Needs about 16 GB of GPU memory with 4-bit loading and CPU offload (Colab T4 / Kaggle T4 class),
and several minutes per crop. Settings come from config.yaml -> generative.qwen.
NOTE: written against the diffusers documentation and model card; not run on the 4 GB development laptop.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crops", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    ap.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    ap.add_argument("--no-4bit", action="store_true", help="load in full precision (needs a large GPU)")
    args = ap.parse_args()

    q = yaml.safe_load(args.config.read_text(encoding="utf-8"))["generative"]["qwen"]
    if not torch.cuda.is_available():
        raise SystemExit("Qwen-Image-Edit needs a CUDA GPU (use the Colab/Kaggle notebook).")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = getattr(torch, args.dtype)

    from diffusers import QwenImageEditPlusPipeline

    kwargs = {"torch_dtype": dtype}
    if not args.no_4bit:
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype},
            components_to_quantize=["transformer", "text_encoder"],
        )
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(q["model_id"], **kwargs)
    if q.get("lightning_repo"):
        pipe.load_lora_weights(q["lightning_repo"], weight_name=q["lightning_file"])
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline ready in {time.time() - t0:.0f}s (dtype {dtype})", flush=True)

    crops = sorted(p for p in args.crops.glob("*/*.png") if not p.stem.endswith(("_mask", "_work")))
    for i, crop in enumerate(crops, 1):
        rel = crop.relative_to(args.crops)
        dst = args.out / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(crop).convert("RGB")
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                image=[img],
                prompt=q["prompt"],
                negative_prompt=" ",
                true_cfg_scale=float(q["true_cfg_scale"]),
                num_inference_steps=int(q["steps"]),
                guidance_scale=1.0,
                generator=torch.Generator(device="cpu").manual_seed(int(q["seed"])),  # same seed for every photo
                num_images_per_prompt=1,
            ).images[0]
        out.resize(img.size, Image.LANCZOS).save(dst)
        print(f"[{i}/{len(crops)}] {rel}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
