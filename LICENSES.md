# Licenses

Everything glassfix uses is open source with a license that allows commercial use.
Licenses were checked on 2026-10-16 (Hugging Face model-card tags and PyPI metadata).
Have your legal team review this before production use.

## Models (weights download to `./models` on first run)

| Model | Used for | Hugging Face repo | License |
|---|---|---|---|
| Grounding DINO (tiny; base optional) | text-prompted glass/car boxes | `IDEA-Research/grounding-dino-tiny`, `IDEA-Research/grounding-dino-base` | Apache-2.0 |
| SAM 2.1 (hiera-small; large optional) | pixel masks from boxes | `facebook/sam2.1-hiera-small`, `facebook/sam2.1-hiera-large` | Apache-2.0 |
| Florence-2 (base; large optional) | alternative box detector | `florence-community/Florence-2-base`, `florence-community/Florence-2-large` | MIT |
| DSIT reflection separation | style `dereflect` (optional) | weights `dsit-26.6959.ckpt` from the XReflection model zoo (checkpoints.mingjia.li), converted to `models/xreflection/*.safetensors` | Apache-2.0 per the XReflection README (see caveat) |

`florence-community/*` is the transformers-native conversion of Microsoft's `microsoft/Florence-2-*` (MIT).

**DSIT caveat:** the Apache-2.0 grant comes from the XReflection README, written by the DSIT authors. The
original `mingcv/DSIT` repository has no license file, and XReflection doesn't ship one either. DSIT's Swin
backbone was fine-tuned on Objects365, which has its own terms, and the model was trained on academic
reflection datasets. Get written confirmation before commercial use. The style is optional and off by default.

### Generative editors (optional; run on a GPU in their own environments, `tools/`)

| Model / package | Used for | License | Delivery? |
|---|---|---|---|
| `Qwen/Qwen-Image-Edit-2509` | see-through glass (`tools/edit_qwen.py`) | Apache-2.0 | ✅ |
| `lightx2v/Qwen-Image-Lightning` (8-step LoRA) | faster Qwen edits | Apache-2.0 | ✅ |
| diffusers 0.41.0 / bitsandbytes 0.50.2 / peft | running Qwen in 4-bit | Apache-2.0 / MIT / Apache-2.0 | ✅ |
| Dereflection-Any-Image (`sjtu-deepvision/dereflection-any-image-v0`, code MIT) | `tools/edit_dai.py` | Weights tagged Apache-2.0, but **fine-tuned from Stable Diffusion 2.1 (CreativeML OpenRAIL++-M)** | ❌ comparison only |

**Not used, by design:** Ultralytics YOLO (AGPL-3.0) and FLUX.1 [dev] / Fill [dev] (non-commercial).
The studio highlight is generated procedurally in NumPy. The built-in styles use no image assets and no
generative models. Generative editing is a separate, optional GPU step (table above).

## Python libraries (pinned in `requirements.txt`)

| Package | Version | License |
|---|---|---|
| torch | 2.14.1 | BSD-3-Clause (bundles Apache-2.0/BSD/MIT/BSL-1.0 components) |
| torchvision | 0.29.1 | BSD-3-Clause |
| transformers | 5.19.0 | Apache-2.0 |
| huggingface-hub | 2.2.0 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| tokenizers | 0.23.3 | Apache-2.0 |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 (see note) |
| opencv-contrib-python-headless (container) | 5.0.0.93 | Apache-2.0 (no Qt; still bundles FFmpeg, LGPL-2.1) |
| scikit-image | 0.26.0 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause |
| pillow | 12.3.0 | MIT-CMU (HPND, permissive) |
| pyyaml | 6.0.3 | MIT |
| timm | 1.0.30 | Apache-2.0 |
| pytest (tests only) | 9.1.1 | MIT |
| pillow-heif (optional, HEIC input) | 1.8.1 | BSD-3-Clause (see note) |

The report uses plain HTML/CSS/JS written for this project, with no third-party front-end code.
Vendored code: `glassfix/third_party/xreflection/` (DSIT + Swin backbone, Apache-2.0 / MIT; see the NOTICE file there).

## LGPL notes (permissive enough for commercial use, but not MIT/Apache/BSD)

- **pillow-heif** is optional, in `requirements-heic.txt`. Its code is BSD-3, but the prebuilt wheels
  bundle libheif and libde265 (LGPL-3.0) as dynamic libraries. Without it, HEIC files are rejected
  with a clear message and JPG/PNG work normally.
- **opencv-contrib-python** wheels bundle FFmpeg (LGPL-2.1), plus Qt on Linux, for video and GUI
  features. glassfix uses neither. For a production server, `opencv-contrib-python-headless` drops Qt.

Dynamic linking of LGPL libraries is generally compatible with commercial, closed-source use, as long
as users can swap the library. Confirm this with your legal team.

## Test photos

All photos in `test_photos/` are CC0 1.0 from Wikimedia Commons. See `test_photos/SOURCES.md`.
