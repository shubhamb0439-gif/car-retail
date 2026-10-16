"""Glass detectors: Grounding DINO + SAM 2, and Florence-2 + SAM 2 (all via Hugging Face transformers)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from .masks import clip_to_box

DETECTORS = ("groundingdino_sam2", "florence2_sam2")


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class Detection:
    raw_mask: np.ndarray                 # uint8 0/1 union of SAM masks (before cleanup)
    glass_boxes: list[Box]
    car_boxes: list[Box]
    rejected_boxes: list[Box] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    excluded_boxes: list[Box] = field(default_factory=list)
    exclude_mask: np.ndarray | None = None  # uint8 0/1: objects on the glass (wipers), never repainted
    interior_mask: np.ndarray | None = None  # uint8 0/1: seats/headrests/steering wheel seen through glass
    car_mask: np.ndarray | None = None       # uint8 0/1: the main car (for the cut-out output)


def model_ids(cfg: dict[str, Any], detector: str) -> list[str]:
    d = cfg["detector"]
    first = d["groundingdino"]["model_id"] if detector == "groundingdino_sam2" else d["florence2"]["model_id"]
    return [first, d["sam2"]["model_id"]]


# --------------------------------------------------------------------- box detectors
class GroundingDino:
    def __init__(self, cfg: dict[str, Any], device: str, cache_dir: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        c = cfg["detector"]["groundingdino"]
        self.box_threshold, self.text_threshold = c["box_threshold"], c["text_threshold"]
        self.max_side = int(c.get("max_side", 1024))
        self.device = device
        self.processor = AutoProcessor.from_pretrained(c["model_id"], cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(c["model_id"], cache_dir=cache_dir, device_map=device)
        self.model.eval()

    def boxes(self, image: Image.Image, prompt: str) -> list[Box]:
        import torch

        # Boxes only need coarse resolution (SAM 2 refines the masks); a smaller input roughly halves
        # Grounding DINO's peak memory, which matters on low-memory machines.
        scale = min(1.0, self.max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
        inputs = self.processor(images=image, text=prompt.lower(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        return [Box(tuple(float(v) / scale for v in b.tolist()), float(s), str(l))
                for b, s, l in zip(res["boxes"], res["scores"], labels)]


class Florence2:
    def __init__(self, cfg: dict[str, Any], device: str, cache_dir: str):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        c = cfg["detector"]["florence2"]
        self.task, self.max_new_tokens, self.num_beams = c["task"], c["max_new_tokens"], c["num_beams"]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(c["model_id"], cache_dir=cache_dir)
        self.model = Florence2ForConditionalGeneration.from_pretrained(c["model_id"], cache_dir=cache_dir, device_map=device)
        self.model.eval()

    def boxes(self, image: Image.Image, prompt: str) -> list[Box]:
        import torch

        text = prompt.strip()
        inputs = self.processor(text=self.task + text, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      num_beams=self.num_beams, do_sample=False)
        decoded = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(decoded, task=self.task, image_size=image.size)
        d = parsed.get(self.task, {})
        bboxes = d.get("bboxes", [])
        labels = d.get("labels") or d.get("bboxes_labels") or [""] * len(bboxes)
        return [Box(tuple(float(v) for v in b), 1.0, str(l)) for b, l in zip(bboxes, labels)]  # no scores


# --------------------------------------------------------------------- SAM 2
class Sam2:
    def __init__(self, cfg: dict[str, Any], device: str, cache_dir: str):
        from transformers import Sam2Model, Sam2Processor

        c = cfg["detector"]["sam2"]
        self.multimask = bool(c["multimask"])
        self.batch = max(1, int(c.get("boxes_per_call", 4)))
        self.device = device
        self.processor = Sam2Processor.from_pretrained(c["model_id"], cache_dir=cache_dir)
        self.model = Sam2Model.from_pretrained(c["model_id"], cache_dir=cache_dir, device_map=device)
        self.model.eval()

    def masks(self, image: Image.Image, boxes: list[Box]) -> list[np.ndarray]:
        """One boolean mask per box. Boxes go through SAM in small batches: SAM upsamples every candidate
        mask to full size in float32, so one call with many boxes can need hundreds of MB at once."""
        out: list[np.ndarray] = []
        for i in range(0, len(boxes), self.batch):
            out.extend(self._masks(image, boxes[i:i + self.batch]))
        return out

    def _masks(self, image: Image.Image, boxes: list[Box]) -> list[np.ndarray]:
        import torch

        if not boxes:
            return []
        input_boxes = [[list(b.xyxy) for b in boxes]]
        inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=self.multimask)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = masks.numpy()                       # (n_boxes, n_candidates, H, W) bool
        if masks.shape[1] > 1:
            best = outputs.iou_scores[0].float().cpu().numpy().argmax(-1)   # (n_boxes,)
            masks = masks[np.arange(len(boxes)), best]
        else:
            masks = masks[:, 0]
        return [m.astype(bool) for m in masks]


# --------------------------------------------------------------------- combined
class GlassDetector:
    """Loads a box detector + SAM 2 once; call .detect() per image."""

    def __init__(self, name: str, cfg: dict[str, Any], device: str):
        if name not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {name!r}")
        cache_dir = str(Path(cfg["models_dir"]).resolve())
        self.name, self.cfg = name, cfg
        t0 = time.perf_counter()
        self.box_model = (GroundingDino if name == "groundingdino_sam2" else Florence2)(cfg, device, cache_dir)
        self.sam = Sam2(cfg, device, cache_dir)
        self.load_seconds = time.perf_counter() - t0

    def detect(self, rgb: np.ndarray) -> Detection:
        d = self.cfg["detector"]
        image = Image.fromarray(rgb)
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        glass = self.box_model.boxes(image, d["glass_prompt"])
        cars = self.box_model.boxes(image, d["car_prompt"])
        t["detect_boxes"] = time.perf_counter() - t0

        # Reject "glass" boxes that are really the whole car, or that sit low on the car (lights).
        main_car = max(cars, key=lambda c: c.area, default=None)
        kept, rejected = [], []
        for b in glass:
            bad = False
            if main_car is not None:
                cx0, cy0, cx1, cy1 = main_car.xyxy
                rel_cy = ((b.xyxy[1] + b.xyxy[3]) / 2 - cy0) / max(cy1 - cy0, 1.0)
                bad = b.area > d["max_glass_box_frac_of_car"] * main_car.area or rel_cy > d["max_glass_center_y_in_car"]
            (rejected if bad else kept).append(b)

        t0 = time.perf_counter()
        raw = np.zeros(rgb.shape[:2], np.uint8)
        for box, m in zip(kept, self.sam.masks(image, kept)):
            raw |= clip_to_box(m.astype(np.uint8), box.xyxy, self.cfg["mask"]["clip_to_box_expand"])
        t["segment"] = time.perf_counter() - t0

        # Objects lying on the glass (wipers) are cut out of the mask so they are never repainted.
        excluded: list[Box] = []
        cut = np.zeros_like(raw)
        if d.get("exclude_prompt") and raw.any():
            t0 = time.perf_counter()
            glass_zone = np.zeros_like(raw)
            for b in kept:
                x0, y0, x1, y1 = (int(v) for v in b.xyxy)
                glass_zone[y0:y1, x0:x1] = 1
            for box in self.box_model.boxes(image, d["exclude_prompt"]):
                x0, y0, x1, y1 = (int(v) for v in box.xyxy)
                if glass_zone[y0:y1, x0:x1].any() and box.area < 0.5 * max((g.area for g in kept), default=0):
                    excluded.append(box)
            if excluded:
                for box, m in zip(excluded, self.sam.masks(image, excluded)):
                    cut |= clip_to_box(m.astype(np.uint8), box.xyxy, 0.0)
                k = int(d["exclude_dilate_px"])
                if k > 0:
                    cut = cv2.dilate(cut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
                raw = raw & (1 - cut)
            t["exclude"] = time.perf_counter() - t0
        # Interior parts seen through the glass (for the studio look).
        interior = np.zeros_like(raw)
        if d.get("interior_prompt") and raw.any():
            t0 = time.perf_counter()
            biggest = max((g.area for g in kept), default=0)
            inner = [b for b in self.box_model.boxes(image, d["interior_prompt"])
                     if b.area < 0.8 * biggest and raw[int((b.xyxy[1] + b.xyxy[3]) / 2),
                                                         int((b.xyxy[0] + b.xyxy[2]) / 2)]]
            for box, m in zip(inner, self.sam.masks(image, inner)):
                interior |= clip_to_box(m.astype(np.uint8), box.xyxy, 0.0)
            interior &= raw
            t["interior"] = time.perf_counter() - t0
        # The whole main car, for the transparent cut-out.
        car_mask = None
        if d.get("car_mask") and main_car is not None:
            t0 = time.perf_counter()
            m = self.sam.masks(image, [main_car])[0].astype(np.uint8)
            car_mask = binary_fill_holes(clip_to_box(m, main_car.xyxy, 0.02)).astype(np.uint8)
            # Glass must lie on the car: drops building/other-car windows detected in the background.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            on_car = cv2.dilate(car_mask, k)
            raw = raw & on_car
            interior = interior & raw
            t["car_mask"] = time.perf_counter() - t0
        return Detection(raw, kept, cars, rejected, t, excluded, cut, interior, car_mask)

    def close(self) -> None:
        del self.box_model, self.sam
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def weights_cached(model_id: str, models_dir: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    cache_dir = str(Path(models_dir).resolve())
    for fname in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
        if isinstance(try_to_load_from_cache(model_id, fname, cache_dir=cache_dir), str):
            return True
    return False
