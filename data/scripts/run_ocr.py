"""Run PaddleOCR over every receipt image and save the text (resumable).

Usage: python data/scripts/run_ocr.py [validation test train test_degraded]   (default: all four)
Writes cache/ocr/<id>.json = {"id", "text", "seconds"} (cache/ is gitignored). Receipts that
already have a file are skipped, so you can stop (Ctrl+C) and rerun at any time.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from receipt.ocr.paddle import ocr_image  # noqa: E402

OUT = ROOT / "cache" / "ocr"


def main(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8")]
        for k, row in enumerate(rows, 1):
            path = OUT / f"{row['id']}.json"
            if path.exists():
                continue
            t = time.time()
            text = ocr_image(ROOT / "data" / "raw" / "cord" / row["image"])
            sec = round(time.time() - t, 2)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"id": row["id"], "text": text, "seconds": sec}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
            print(f"{split} {k}/{len(rows)} {row['id']} {sec}s", flush=True)
        print(f"{split}: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test", "train", "test_degraded"])
