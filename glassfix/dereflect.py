"""Learned reflection removal with DSIT (vendored from XReflection, Apache-2.0).

DSIT splits a photo of glass into a transmission layer (what is behind the glass) and a reflection
layer. glassfix runs it on the glass crop only, in overlapping tiles (memory grows steeply with tile
size), and returns the predicted reflection, which the pipeline subtracts at full resolution.
"""

from __future__ import annotations

import gc
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _weights_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    d = Path(cfg["models_dir"]).resolve() / "xreflection"
    c = cfg["dereflect"]
    return d / Path(c["checkpoint_url"]).name, d / c["weights_file"]


def dsit_weights_ready(cfg: dict[str, Any]) -> bool:
    return _weights_paths(cfg)[1].exists()


class _LazyTensor:
    """Where a tensor lives inside a torch zip checkpoint (read later, one tensor at a time)."""

    def __init__(self, key: str, dtype: Any, offset: int, size: tuple, stride: tuple):
        self.key, self.dtype, self.offset, self.size, self.stride = key, dtype, offset, size, stride


def _stream_state_dict(ckpt: Path, prefix: str):
    """Yield (name, fp16 tensor) for entries of ckpt['state_dict'] starting with `prefix`.

    A torch checkpoint is a zip of a pickle index plus one raw file per storage. The index is read with
    a restricted unpickler (only tensor-rebuild calls are allowed, so no arbitrary code runs) and each
    tensor is read on its own. This keeps memory low: on Windows a memory-mapped or fully loaded
    2.5 GB checkpoint reserves 2.5 GB of commit, which a busy laptop may not have.
    """
    import pickle
    import zipfile

    import torch

    storage_dtypes = {n: getattr(torch, d) for n, d in (
        ("FloatStorage", "float32"), ("HalfStorage", "float16"), ("BFloat16Storage", "bfloat16"),
        ("DoubleStorage", "float64"), ("LongStorage", "int64"), ("IntStorage", "int32"),
        ("ShortStorage", "int16"), ("CharStorage", "int8"), ("ByteStorage", "uint8"), ("BoolStorage", "bool"))}

    class _Opaque:  # stands in for any non-tensor object (hyper-parameters, callbacks, ...)
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    def rebuild(storage, offset, size, stride, *rest):
        key, dtype = storage
        return _LazyTensor(key, dtype, offset, tuple(size), tuple(stride))

    class Index(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if module == "torch" and name in storage_dtypes:
                return storage_dtypes[name]
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if module == "torch" and name in ("float32", "float16", "bfloat16", "int64", "bool"):
                return getattr(torch, name)
            return _Opaque

        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            return (pid[2], pid[1] if not isinstance(pid[1], type) else torch.float32)

    with zipfile.ZipFile(ckpt) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        root = pkl[: -len("data.pkl")]
        with z.open(pkl) as f:
            index = Index(f).load()
        for name, lt in index["state_dict"].items():
            if not name.startswith(prefix) or not isinstance(lt, _LazyTensor):
                continue
            raw = z.read(f"{root}data/{lt.key}")
            flat = torch.frombuffer(bytearray(raw), dtype=lt.dtype)
            t = torch.as_strided(flat, lt.size, lt.stride, lt.offset).clone()
            yield name[len(prefix):], (t.to(torch.float16) if t.is_floating_point() else t)


def prepare_dsit_weights(cfg: dict[str, Any]) -> Path:
    """Download the DSIT checkpoint once and convert its network weights to a pickle-free fp16 file."""
    ckpt, out = _weights_paths(cfg)
    if out.exists():
        return out
    from safetensors.torch import save_file

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        print(f"downloading {cfg['dereflect']['checkpoint_url']} (~2.5 GB) ...", flush=True)
        tmp = ckpt.with_suffix(".part")
        urllib.request.urlretrieve(cfg["dereflect"]["checkpoint_url"], tmp)
        tmp.rename(ckpt)
    weights = {}
    for name, t in _stream_state_dict(ckpt, "net_g."):
        for p in ("module.", "_orig_mod."):
            if name.startswith(p):
                name = name[len(p):]
        weights[name] = t.contiguous()
    if not weights:
        raise RuntimeError(f"no 'net_g.' weights found in {ckpt.name}")
    save_file(weights, str(out), metadata={"source": cfg["dereflect"]["checkpoint_url"]})
    del weights
    gc.collect()
    if not cfg["dereflect"]["keep_checkpoint"]:
        ckpt.unlink()
    return out


def _read_safetensors(path: Path, dtype):
    """Yield (name, tensor) from a .safetensors file using plain reads, one tensor at a time.

    safetensors' own loader memory-maps the file, which Windows charges against the commit limit; on a
    busy laptop that alone can fail ("paging file is too small").
    """
    import json
    import struct

    import torch

    st_dtypes = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
                 "I32": torch.int32, "BOOL": torch.bool, "U8": torch.uint8}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            t = torch.frombuffer(bytearray(f.read(end - start)), dtype=st_dtypes[info["dtype"]]).reshape(info["shape"])
            yield name, (t.to(dtype) if t.is_floating_point() else t)


def _ramp_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """Tile blending weight: 1 in the middle, linear fade over `overlap` px at each edge."""
    def ramp(n: int) -> np.ndarray:
        r = np.ones(n, np.float32)
        if overlap > 0:
            fade = np.linspace(0.05, 1.0, min(overlap, n // 2), dtype=np.float32)
            r[: len(fade)] = fade
            r[n - len(fade):] = fade[::-1]
        return r
    return np.outer(ramp(h), ramp(w))


def _tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


class DsitDereflector:
    def __init__(self, cfg: dict[str, Any], device: str):
        import torch

        from .third_party.xreflection.dsit_arch import DSIT

        t0 = time.perf_counter()
        c = cfg["dereflect"]
        self.cfg, self.device = c, device
        self.tile, self.overlap = int(c["tile_px"]), int(c["tile_overlap_px"])
        if self.tile % 32:
            raise ValueError("dereflect.tile_px must be a multiple of 32")
        self.model = DSIT(None, input_resolution=(384, 384), window_size=12,
                          enc_blk_nums=list(c["enc_blk_nums"]), dec_blk_nums=list(c["dec_blk_nums"]))
        # fp16 on CUDA halves memory; CPU stays float32 (DSIT's custom layers reject bfloat16).
        self.dtype = torch.float16 if (device == "cuda" and c["fp16_on_cuda"]) else torch.float32
        self.model.to(dtype=self.dtype)
        params = dict(self.model.state_dict())
        loaded = set()
        with torch.no_grad():
            for name, t in _read_safetensors(prepare_dsit_weights(cfg), self.dtype):
                if name not in params or params[name].shape != t.shape:
                    raise RuntimeError(f"DSIT weights mismatch at {name}")
                params[name].copy_(t)
                loaded.add(name)
        missing = [k for k in params if k not in loaded and "relative_position_index" not in k and "attn_mask" not in k]
        if missing:
            raise RuntimeError(f"DSIT weights missing: {missing[:5]}")
        self.model.to(device=device).eval()
        self.load_seconds = time.perf_counter() - t0

    def reflection(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: float32 (H, W, 3) in 0..1. Returns the predicted reflection layer, same shape, >= 0."""
        import torch

        h, w = rgb.shape[:2]
        # Pad small images up to a full tile (reflect), and both dims to multiples of 32.
        ph, pw = max(h, self.tile), max(w, self.tile)
        ph, pw = -(-ph // 32) * 32, -(-pw // 32) * 32
        img = cv2.copyMakeBorder(rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
        tile_h, tile_w = min(self.tile, ph), min(self.tile, pw)
        acc = np.zeros((ph, pw, 3), np.float32)
        wsum = np.zeros((ph, pw, 1), np.float32)
        weight = _ramp_weight(tile_h, tile_w, self.overlap)[..., None]
        for y in _tile_starts(ph, tile_h, self.overlap):
            for x in _tile_starts(pw, tile_w, self.overlap):
                patch = img[y:y + tile_h, x:x + tile_w]
                inp = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None]
                with torch.inference_mode():
                    t_out, _r_out, _ = self.model(inp.to(self.device, self.dtype))
                del _r_out
                t_np = t_out[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                # Reflection = input minus clean transmission (consistent with what we subtract later).
                acc[y:y + tile_h, x:x + tile_w] += np.clip(patch - t_np, 0.0, None) * weight
                wsum[y:y + tile_h, x:x + tile_w] += weight
        return (acc / np.maximum(wsum, 1e-6))[:h, :w]

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def reflection_map(work_rgb: np.ndarray, mask_w: np.ndarray, model: DsitDereflector,
                   cfg: dict[str, Any]) -> np.ndarray:
    """DSIT reflection layer for the glass of one photo, as a working-resolution (H, W, 3) map.

    Only the padded bounding box of the glass is processed, downscaled so its long side is at most
    dereflect.max_side_px; the (mostly smooth) reflection is upsampled back. Zero outside the glass.
    """
    out = np.zeros(work_rgb.shape, np.float32)
    ys, xs = np.nonzero(mask_w)
    if len(ys) == 0:
        return out
    pad = 16
    y0, y1 = max(0, ys.min() - pad), min(mask_w.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask_w.shape[1], xs.max() + pad + 1)
    crop = work_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    h, w = crop.shape[:2]
    k = min(1.0, cfg["dereflect"]["max_side_px"] / max(h, w))
    small = cv2.resize(crop, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else crop
    refl = model.reflection(small)
    if k < 1:
        refl = cv2.resize(refl, (w, h), interpolation=cv2.INTER_LINEAR)
    out[y0:y1, x0:x1] = refl * (mask_w[y0:y1, x0:x1, None] > 0)
    return out
