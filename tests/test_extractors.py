"""Tests for the extractors, using a fake backend (no network, no API key)."""
import json

from PIL import Image

from receipt.cache import DiskCache
from receipt.extractors.base import parse_output
from receipt.extractors.ocr_llm import OcrLlm
from receipt.extractors.vlm import Vlm

GOOD = json.dumps({"merchant": None, "date": None, "subtotal": None, "tax": None, "total": 1000,
                   "items": [{"name": "TEA", "qty": 1, "price": 1000}]})


class FakeBackend:
    model = "fake-model"

    def __init__(self, text=GOOD, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def generate(self, prompt, text=None, image_png=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("429 rate limit")
        return {"text": self.text, "input_tokens": 10, "output_tokens": 5}


def make_ocr_dir(tmp_path):
    d = tmp_path / "ocr"
    d.mkdir(exist_ok=True)
    (d / "r1.json").write_text(json.dumps({"id": "r1", "text": "TEA 1.000\nTOTAL 1.000", "seconds": 1}))
    return d


def test_parse_output():
    assert parse_output(GOOD)[1] is True
    assert parse_output("not json") == (None, False)
    assert parse_output('{"items": "oops"}') == (None, False)


def test_ocr_llm_and_cache(tmp_path):
    backend = FakeBackend()
    ex = OcrLlm(backend, DiskCache("t", root=tmp_path), make_ocr_dir(tmp_path))
    r1 = ex.extract("r1")
    r2 = ex.extract("r1")
    assert r1.json_valid and r1.receipt["total"] == 1000 and not r1.cached
    assert r2.cached and r2.input_tokens == 10
    assert backend.calls == 1


def test_failed_call_not_cached(tmp_path):
    ex = OcrLlm(FakeBackend(fail=True), DiskCache("t", root=tmp_path), make_ocr_dir(tmp_path))
    r = ex.extract("r1")
    assert r.error and not r.json_valid
    ok = OcrLlm(FakeBackend(), DiskCache("t", root=tmp_path), make_ocr_dir(tmp_path)).extract("r1")
    assert ok.json_valid and not ok.cached


def test_invalid_json_is_reported(tmp_path):
    ex = OcrLlm(FakeBackend(text="sorry"), DiskCache("t", root=tmp_path), make_ocr_dir(tmp_path))
    r = ex.extract("r1")
    assert r.json_valid is False and r.receipt is None and r.error is None


def test_missing_ocr_file(tmp_path):
    ex = OcrLlm(FakeBackend(), DiskCache("t", root=tmp_path), tmp_path)
    assert "FileNotFoundError" in ex.extract("nope").error


def test_vlm(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (50, 80), "white").save(img)
    backend = FakeBackend()
    ex = Vlm(backend, DiskCache("v", root=tmp_path))
    assert ex.extract("r1", img).json_valid
    assert ex.extract("r1", img).cached
    assert backend.calls == 1


def test_parse_output_strips_code_fence():
    assert parse_output("```json\n" + GOOD + "\n```")[1] is True
