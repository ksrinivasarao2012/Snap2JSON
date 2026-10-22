"""Run one method over one split, save per-receipt outputs, and write a scored summary.

Usage (from the repo root):
  python eval/run_eval.py --backend qwen   --method vlm     --split validation --limit 3
  python eval/run_eval.py --backend qwen   --method ocr_llm --split test
  python eval/run_eval.py --backend gemini --method vlm     --split validation --limit 3
Splits: validation, test, train, test_degraded. Method ocr_llm needs cache/ocr (run_ocr.py).

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


def summarize(out_path, rows):
    gt = {r["id"]: r["receipt"] for r in rows}
    outs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    outs = [o for o in outs if o["receipt_id"] in gt]
    summary = aggregate([score_receipt(o["receipt"], gt[o["receipt_id"]]) for o in outs])
    secs = [o["seconds"] for o in outs]
    summary.update(
        n_receipts_in_split=len(rows),
        mean_seconds=statistics.mean(secs) if secs else None,
        median_seconds=statistics.median(secs) if secs else None,
        input_tokens_total=sum(o["input_tokens"] or 0 for o in outs),
        output_tokens_total=sum(o["output_tokens"] or 0 for o in outs),
    )
    return summary


def build_extractor(backend_name, method, model):
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

    return OcrLlm(backend), backend.model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["gemini", "qwen"], required=True)
    p.add_argument("--method", choices=["ocr_llm", "vlm"], required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test", "train", "test_degraded"])
    p.add_argument("--model", default=None, help="model name (Qwen default is in qwen.py; Gemini uses GEMINI_MODEL)")
    p.add_argument("--limit", type=int, default=None, help="only the first N receipts")
    a = p.parse_args()

    rows = load_rows(a.split)[: a.limit]
    extractor, model = build_extractor(a.backend, a.method, a.model)
    name = f"{a.method}__{model.replace('/', '_')}__{a.split}"
    out = run(extractor, rows, ROOT / "results" / "runs" / f"{name}.jsonl")
    summary = summarize(out, rows)
    summary.update(method=a.method, model=model, split=a.split)
    table = ROOT / "results" / "tables" / f"{name}.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
