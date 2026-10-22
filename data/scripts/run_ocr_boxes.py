"""Run PaddleOCR again and save every text box with its position (resumable).

Usage: python data/scripts/run_ocr_boxes.py [validation test train test_degraded]   (default: all four)
Writes cache/ocr_boxes/<id>.json = {"id", "boxes": [{"text", "box": [x0, y0, x1, y1], "score"}],
"same_text"}. cache/ocr/ is never changed. same_text says whether these boxes, joined in the
original format, give exactly the cached OCR text the earlier OCR+LLM runs used.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from receipt.ocr.paddle import boxes_to_text, ocr_boxes  # noqa: E402

OCR = ROOT / "cache" / "ocr"
OUT = ROOT / "cache" / "ocr_boxes"


def main(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8")]
        for k, row in enumerate(rows, 1):
            path = OUT / f"{row['id']}.json"
            if path.exists():
                continue
            boxes = ocr_boxes(ROOT / "data" / "raw" / "cord" / row["image"])
            old = json.loads((OCR / f"{row['id']}.json").read_text(encoding="utf-8"))["text"]
            same = boxes_to_text(boxes) == old
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"id": row["id"], "boxes": boxes, "same_text": same}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
            print(f"{split} {k}/{len(rows)} {row['id']} boxes={len(boxes)} same_text={same}", flush=True)
        print(f"{split}: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test", "train", "test_degraded"])
