"""Device auto-detection and determinism."""

from __future__ import annotations

import os
import random

import numpy as np

# Let unsupported ops fall back to CPU on Apple MPS instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Windows without Developer Mode cannot symlink in the HF cache; that is fine, just noisy.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
