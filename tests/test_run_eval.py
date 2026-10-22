"""Tests for eval/run_eval.py using a fake extractor (no model, no network)."""
import json

from eval.run_eval import run, summarize
from receipt.extractors.base import ExtractionResult

ROWS = [
    {"id": f"r{i}", "image": f"r{i}.png",
     "receipt": {"merchant": None, "date": None, "subtotal": None, "tax": None, "total": 1000.0,
                 "items": [{"name": "TEA", "qty": 1.0, "price": 1000.0}]}}
    for i in range(3)
]
GOOD = {"merchant": None, "date": None, "subtotal": None, "tax": None, "total": 1000.0,
        "items": [{"name": "TEA", "qty": 1.0, "price": 1000.0}]}


class Fake:
    def __init__(self, fail_ids=()):
        self.fail_ids, self.calls = set(fail_ids), []

    def extract(self, rid, path):
        self.calls.append(rid)
        if rid in self.fail_ids:
            return ExtractionResult(rid, None, False, "", 0.0, error="RuntimeError: 429")
        return ExtractionResult(rid, GOOD, True, "{}", 1.5, 10, 5)


def test_run_and_summarize(tmp_path):
    out = tmp_path / "o.jsonl"
    run(Fake(), ROWS, out, log=lambda *_: None)
    s = summarize(out, ROWS)
    assert s["n"] == 3 and s["json_valid_rate"] == 1.0 and s["total_acc"] == 1.0
    assert s["items_f1"] == 1.0 and s["mean_seconds"] == 1.5 and s["input_tokens_total"] == 30


def test_resume_skips_done(tmp_path):
    out = tmp_path / "o.jsonl"
    run(Fake(), ROWS[:2], out, log=lambda *_: None)
    f = Fake()
    run(f, ROWS, out, log=lambda *_: None)
    assert f.calls == ["r2"]
    assert len(out.read_text().splitlines()) == 3


def test_failures_not_written_and_retried(tmp_path):
    out = tmp_path / "o.jsonl"
    run(Fake(fail_ids={"r1"}), ROWS, out, log=lambda *_: None)
    assert [json.loads(l)["receipt_id"] for l in out.read_text().splitlines()] == ["r0", "r2"]
    f = Fake()
    run(f, ROWS, out, log=lambda *_: None)
    assert f.calls == ["r1"]


def test_stops_after_consecutive_failures(tmp_path):
    f = Fake(fail_ids={"r0", "r1", "r2"})
    run(f, ROWS, tmp_path / "o.jsonl", max_consecutive_errors=2, log=lambda *_: None)
    assert f.calls == ["r0", "r1"]


def test_summarize_adds_ocr_time(tmp_path):
    out = tmp_path / "o.jsonl"
    run(Fake(), ROWS, out, log=lambda *_: None)
    s = summarize(out, ROWS, {"r0": 10.0, "r1": 12.0})
    assert s["ocr_timed_n"] == 2 and s["mean_ocr_seconds"] == 11.0
    assert s["mean_end_to_end_seconds"] == 12.5 and s["mean_seconds"] == 1.5
    assert "mean_ocr_seconds" not in summarize(out, ROWS)
