"""Tests with a fake OpenAI client: no network, no cost."""

from __future__ import annotations

import base64
import csv
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from glassfix.batch import run
from glassfix.config import ROOT, load_config, with_overrides
from glassfix.imageio import list_car_photos, load_image
from glassfix.render import OpenAIRenderer, RenderError


def _png(w=64, h=40, rgba=(10, 20, 30, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), rgba).save(buf, "PNG")
    return buf.getvalue()


class FakeImages:
    def __init__(self, fail_times=0, error_name="RateLimitError"):
        self.calls, self.fail_times, self.error_name = [], fail_times, error_name

    def edit(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_times:
            raise type(self.error_name, (Exception,), {})("simulated")
        usage = SimpleNamespace(input_tokens_details=SimpleNamespace(image_tokens=1000, text_tokens=100),
                                output_tokens=400)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(_png()).decode())], usage=usage)


def fake_client(**kw):
    return SimpleNamespace(images=FakeImages(**kw))


def cfg(**over):
    return with_overrides(load_config(), **{"openai.retry_backoff_seconds": 0, **over})


def test_medium_tier_studio_lighting_with_reference():
    client = fake_client()
    r = OpenAIRenderer(cfg(), "medium", client=client, root=ROOT)
    res = r.render(Image.new("RGB", (4000, 3000), "grey"))
    call = client.images.calls[0]
    assert call["model"] == load_config()["openai"]["model"]
    assert call["background"] == "transparent" and call["output_format"] == "png" and call["quality"] == "medium"
    assert len(call["image"]) == 2                                       # photo + style reference
    sent = Image.open(io.BytesIO(call["image"][0][1]))
    assert max(sent.size) == load_config()["openai"]["send_max_side"]     # downscaled before upload
    assert "studio lighting" in call["prompt"]
    assert call["prompt"].endswith(load_config()["prompt_reference_suffix"].strip())
    assert res.png.startswith(b"\x89PNG") and res.size == (64, 40)


def test_low_tier_keeps_original_lighting_without_reference():
    client = fake_client()
    OpenAIRenderer(cfg(), "low", client=client, root=ROOT).render(Image.new("RGB", (800, 600)))
    call = client.images.calls[0]
    assert call["quality"] == "low" and len(call["image"]) == 1
    assert "ORIGINAL LIGHTING" in call["prompt"] and "studio lighting" not in call["prompt"]
    assert call["prompt"] == load_config()["tiers"]["low"]["prompt"].strip()


def test_high_tier_is_medium_look_at_high_quality():
    client = fake_client()
    OpenAIRenderer(cfg(), "high", client=client, root=ROOT).render(Image.new("RGB", (800, 600)))
    tiers = load_config()["tiers"]
    assert client.images.calls[0]["quality"] == "high"
    assert tiers["high"]["prompt"] == tiers["medium"]["prompt"]


def test_unknown_tier_rejected():
    with pytest.raises(RenderError):
        OpenAIRenderer(cfg(), "platinum", client=fake_client())


def test_cost_from_token_usage():
    res = OpenAIRenderer(cfg(), "low", client=fake_client()).render(Image.new("RGB", (800, 600)))
    rates = load_config()["pricing_usd_per_million_tokens"]
    expected = (1000 * rates["image_input"] + 100 * rates["text_input"] + 400 * rates["image_output"]) / 1e6
    assert res.cost_usd == pytest.approx(expected)
    assert (res.input_image_tokens, res.input_text_tokens, res.output_tokens) == (1000, 100, 400)


def test_retries_rate_limits_then_succeeds():
    client = fake_client(fail_times=2)
    OpenAIRenderer(cfg(), "low", client=client).render(Image.new("RGB", (100, 100)))
    assert len(client.images.calls) == 3


def test_non_retryable_error_fails_without_leaking_details():
    client = fake_client(fail_times=99, error_name="AuthenticationError")
    with pytest.raises(RenderError) as e:
        OpenAIRenderer(cfg(), "low", client=client).render(Image.new("RGB", (100, 100)))
    assert len(client.images.calls) == 1 and "AuthenticationError" in str(e.value)


def test_exif_rotation_applied(tmp_path):
    img = Image.new("RGB", (60, 30), "red")
    exif = Image.Exif()
    exif[0x0112] = 6                                  # "rotate 90° CW to display"
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif)
    assert load_image(p).size == (30, 60)


def test_batch_writes_pngs_csv_and_sheet(tmp_path):
    for car, names in {"car_a": ["front", "side"], "car_b": ["rear"]}.items():
        (tmp_path / "in" / car).mkdir(parents=True)
        for n in names:
            Image.new("RGB", (300, 200), "blue").save(tmp_path / "in" / car / f"{n}.jpg")
    assert set(list_car_photos(tmp_path / "in")) == {"car_a", "car_b"}
    c = cfg()
    rows = run(c, tmp_path / "in", tmp_path / "out", "job1", "low", OpenAIRenderer(c, "low", client=fake_client()))
    assert all(r["status"] == "ok" for r in rows) and len(rows) == 3
    out = tmp_path / "out" / "job1"
    assert (out / "car_a" / "front.png").exists() and (out / "car_b" / "rear.png").exists()
    assert (out / "sheet.jpg").exists()
    with open(out / "costs.csv", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    assert {d["tier"] for d in data} == {"low"} and {d["quality"] for d in data} == {"low"}
    assert all(float(d["cost_usd"]) > 0 for d in data)


def test_batch_reports_failures_and_continues(tmp_path):
    (tmp_path / "in" / "car").mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(tmp_path / "in" / "car" / "a.jpg")
    c = cfg()
    rows = run(c, tmp_path / "in", tmp_path / "out", "job", "low",
               OpenAIRenderer(c, "low", client=fake_client(fail_times=99, error_name="BadRequestError")))
    assert rows[0]["status"] == "failed" and "BadRequestError" in rows[0]["error"]


def test_interior_view_uses_interior_prompts_without_reference():
    tiers = load_config()["interior_tiers"]
    for tier in ("low", "medium"):
        client = fake_client()
        OpenAIRenderer(cfg(), tier, client=client, root=ROOT, view="interior").render(Image.new("RGB", (800, 600)))
        call = client.images.calls[0]
        assert len(call["image"]) == 1 and call["prompt"] == tiers[tier]["prompt"].strip()
        assert call["quality"] == tier and "interior" in call["prompt"]
    assert "ORIGINAL LIGHTING" in tiers["low"]["prompt"] and "studio lighting" in tiers["medium"]["prompt"]


def test_unknown_view_rejected():
    with pytest.raises(RenderError):
        OpenAIRenderer(cfg(), "low", client=fake_client(), view="underbody")
