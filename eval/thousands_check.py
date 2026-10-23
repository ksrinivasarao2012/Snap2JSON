"""Diagnostic: how many wrong amounts are exactly 1000x too small (e.g. "total": 60.000, which JSON reads as 60)?

Usage (from the repo root): python eval/thousands_check.py [validation test ...]   (default: validation test)
For every Qwen run of plain OCR+LLM, rows (A) and VLM: total accuracy, wrong totals, wrong totals that are the true
total / 1000, and (same first-match rule as REPORT.md 6.5) wrong item prices that are the true price / 1000.
Read only: it does not change any score. Found 2026-10-23 while looking at why 4B < 2B on text totals.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rapidfuzz import fuzz  # noqa: E402

from eval.metrics import NAME_THRESHOLD, _norm_text, _same  # noqa: E402

METHODS = ["ocr_llm", "ocr_llm_rows", "vlm"]
MODELS = ["2B", "4B", "8B"]


def is_thousandth(pred, true):
    return isinstance(pred, (int, float)) and true is not None and abs(pred * 1000 - true) < 0.5


def name_sim(a, b):
    return fuzz.ratio(_norm_text(a.get("name")) or "", _norm_text(b.get("name")) or "")


def check(split, method, model):
    gt = {r["id"]: r["receipt"] for r in map(json.loads, open(ROOT / "data" / "processed" / f"{split}.jsonl",
                                                              encoding="utf-8"))}
    path = ROOT / "results" / "runs" / f"{method}__Qwen_Qwen3-VL-{model}-Instruct__{split}.jsonl"
    n = ok = tot_k = price_wrong = price_k = 0
    for o in map(json.loads, open(path, encoding="utf-8")):
        g, p = gt[o["receipt_id"]], o["receipt"] or {}
        if g["total"] is not None:
            n += 1
            if _same("total", p.get("total"), g["total"]):
                ok += 1
            else:
                tot_k += is_thousandth(p.get("total"), g["total"])
        pred = p.get("items") or []
        for it in g["items"]:
            best = max(pred, key=lambda q: name_sim(q, it), default=None)
            if best is None or name_sim(best, it) < NAME_THRESHOLD or it["price"] is None:
                continue
            if best.get("price") != it["price"]:
                price_wrong += 1
                price_k += is_thousandth(best.get("price"), it["price"])
    return n, ok, tot_k, price_wrong, price_k


def main(splits):
    print("split       model method        total_acc  wrong  wrong_by_1000  item_price_wrong  by_1000")
    for split in splits:
        for model in MODELS:
            for method in METHODS:
                n, ok, k, pw, pk = check(split, method, model)
                print(f"{split:11} {model:5} {method:13} {ok / n:9.3f}  {n - ok:5}  {k:13}  {pw:16}  {pk:7}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test"])
