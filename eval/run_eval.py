"""Run one method over one split, save per-receipt outputs, and write a scored summary.

Usage (from the repo root):
  python eval/run_eval.py --backend qwen   --method vlm     --split validation --limit 3
  python eval/run_eval.py --backend qwen   --method ocr_llm --split test
  python eval/run_eval.py --backend gemini --method vlm     --split validation --limit 3
Splits: validation, test, train, test_degraded. Method ocr_llm needs cache/ocr (run_ocr.py);
ocr_llm_rows (layout A) and ocr_llm_boxes (layout B) need cache/ocr_boxes (run_ocr_boxes.py).

Outputs: results/runs/<method>__<model>__<split>.jsonl (one line per receipt) and
results/tables/<same name>.json (summary). Resumable: receipts already in the .jsonl are skipped,
and failed calls (e.g. rate limits) are not written, so a rerun retries them. A run stops after
3 failed calls in a row.
"""
import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.metrics import aggregate, score_receipt  # noqa: E402

IMAGE_ROOT = ROOT / "data" / "raw" / "cord"
OCR_TIMING = ROOT / "cache" / "ocr_timing"  # written by data/scripts/time_ocr.py
OCR_VARIANTS = ROOT / "cache" / "ocr_variants"  # written by data/scripts/compare_ocr.py (boxes + seconds per receipt)


def load_rows(split):
    with open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def run(extractor, rows, out_path, image_root=IMAGE_ROOT, max_consecutive_errors=3, log=print):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["receipt_id"] for l in open(out_path, encoding="utf-8")}
    errors = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for k, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            r = extractor.extract(row["id"], Path(image_root) / row["image"])
            if r.error:
                errors += 1
                log(f"{row['id']}: FAILED ({r.error[:200]})")
                if errors >= max_consecutive_errors:
                    log(f"stopping after {errors} failures in a row; rerun later to resume")
                    break
                continue
            errors = 0
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            f.flush()
            log(f"{k}/{len(rows)} {row['id']} valid={r.json_valid} {r.seconds}s cached={r.cached}")
    return out_path


def load_ocr_seconds(ids, timing_dir=OCR_TIMING):
    """{id: OCR seconds} for receipts re-timed by time_ocr.py (missing ones are left out)."""
    out = {}
    for i in ids:
        p = Path(timing_dir) / f"{i}.json"
        if p.exists():
            out[i] = json.loads(p.read_text(encoding="utf-8"))["seconds"]
    return out


def summarize(out_path, rows, ocr_seconds=None):
    """ocr_seconds ({id: s}, OCR+LLM only) adds end-to-end time = OCR + model call."""
    gt = {r["id"]: r["receipt"] for r in rows}
    subs = {r["id"]: r.get("sub_items", []) for r in rows}
    outs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    outs = [o for o in outs if o["receipt_id"] in gt]
    summary = aggregate([score_receipt(o["receipt"], gt[o["receipt_id"]], sub_items=subs[o["receipt_id"]])
                         for o in outs])
    secs = [o["seconds"] for o in outs]
    summary.update(
        n_receipts_in_split=len(rows),
        mean_seconds=statistics.mean(secs) if secs else None,
        median_seconds=statistics.median(secs) if secs else None,
        input_tokens_total=sum(o["input_tokens"] or 0 for o in outs),
        output_tokens_total=sum(o["output_tokens"] or 0 for o in outs),
    )
    if ocr_seconds:
        both = [(ocr_seconds[o["receipt_id"]], o["seconds"]) for o in outs if o["receipt_id"] in ocr_seconds]
        if both:
            summary.update(
                ocr_timed_n=len(both),
                mean_ocr_seconds=statistics.mean(a for a, _ in both),
                mean_end_to_end_seconds=statistics.mean(a + b for a, b in both),
                median_end_to_end_seconds=statistics.median(a + b for a, b in both),
            )
    return summary


def run_name(method, model, split, ocr_variant=None):
    """<method>__<model>__<split>; an OCR variant is added to the method part (e.g. ocr_llm_rows-ocrT)."""
    m = f"{method}-ocr{ocr_variant}" if ocr_variant else method
    return f"{m}__{model.replace('/', '_')}__{split}"


def build_extractor(backend_name, method, model, ocr_variant=None):
    if backend_name == "gemini":
        from receipt.extractors.gemini import GeminiBackend

        backend = GeminiBackend(model)
    else:
        from receipt.extractors.qwen import DEFAULT_MODEL, QwenBackend

        backend = QwenBackend(model or DEFAULT_MODEL)
    if method == "vlm":
        from receipt.extractors.vlm import Vlm

        return Vlm(backend), backend.model
    from receipt.extractors.ocr_llm import OcrLlm

    layout = {"ocr_llm": "plain", "ocr_llm_rows": "rows", "ocr_llm_boxes": "boxes"}[method]
    kwargs = {"boxes_dir": OCR_VARIANTS / ocr_variant} if ocr_variant else {}
    return OcrLlm(backend, layout=layout, **kwargs), backend.model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["gemini", "qwen"], required=True)
    p.add_argument("--method", choices=["ocr_llm", "ocr_llm_rows", "ocr_llm_boxes", "vlm"], required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test", "train", "test_degraded"])
    p.add_argument("--model", default=None, help="model name (Qwen default is in qwen.py; Gemini uses GEMINI_MODEL)")
    p.add_argument("--limit", type=int, default=None, help="only the first N receipts")
    p.add_argument("--ocr-variant", default=None,
                   help="read OCR boxes (and OCR seconds) from cache/ocr_variants/<name> instead of cache/ocr_boxes; "
                        "rows/boxes methods only, e.g. T = server detector (data/scripts/compare_ocr.py)")
    p.add_argument("--summarize-only", action="store_true",
                   help="rebuild the summary from the saved run without loading the model (needs --model for gemini)")
    a = p.parse_args()

    if a.ocr_variant and a.method not in ("ocr_llm_rows", "ocr_llm_boxes"):
        p.error("--ocr-variant needs --method ocr_llm_rows or ocr_llm_boxes")
    if a.ocr_variant and not (OCR_VARIANTS / a.ocr_variant).is_dir():
        p.error(f"no OCR variant folder {OCR_VARIANTS / a.ocr_variant}")
    rows = load_rows(a.split)[: a.limit]
    if a.summarize_only:
        from receipt.extractors.qwen import DEFAULT_MODEL

        model = a.model or (DEFAULT_MODEL if a.backend == "qwen" else None)
        if not model:
            p.error("--summarize-only with gemini needs --model")
        out = ROOT / "results" / "runs" / f"{run_name(a.method, model, a.split, a.ocr_variant)}.jsonl"
    else:
        extractor, model = build_extractor(a.backend, a.method, a.model, a.ocr_variant)
        out = run(extractor, rows, ROOT / "results" / "runs" / f"{run_name(a.method, model, a.split, a.ocr_variant)}.jsonl")
    name = out.stem
    timing_dir = OCR_VARIANTS / a.ocr_variant if a.ocr_variant else OCR_TIMING
    ocr_seconds = load_ocr_seconds([r["id"] for r in rows], timing_dir) if a.method.startswith("ocr_llm") else None
    summary = summarize(out, rows, ocr_seconds)
    summary.update(method=a.method, model=model, split=a.split)
    if a.ocr_variant:
        summary.update(ocr_variant=a.ocr_variant)
    table = ROOT / "results" / "tables" / f"{name}.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
