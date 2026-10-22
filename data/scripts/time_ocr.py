"""Re-time PaddleOCR on this machine, without touching the OCR text the LLM runs used (resumable).

Usage: python data/scripts/time_ocr.py [validation test train test_degraded]   (default: all four)
Writes cache/ocr_timing/<id>.json = {"id", "seconds", "same_text", "device"}. cache/ocr/ is never
changed, so OCR+LLM outputs and their cache stay valid. One warm-up image is run first so model
loading is not counted. same_text says whether this run reproduced the cached OCR text exactly.
"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from receipt.ocr.paddle import ocr_image  # noqa: E402

OCR = ROOT / "cache" / "ocr"
OUT = ROOT / "cache" / "ocr_timing"


def device():
    import paddle

    return paddle.device.get_device() + " | " + (platform.processor() or platform.machine())


def main(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    dev, warmed = None, False
    for split in splits:
        rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8")]
        for k, row in enumerate(rows, 1):
            path = OUT / f"{row['id']}.json"
            if path.exists():
                continue
            image = ROOT / "data" / "raw" / "cord" / row["image"]
            if not warmed:
                ocr_image(image)  # loads the models; not timed
                dev, warmed = device(), True
            t = time.time()
            text = ocr_image(image)
            sec = round(time.time() - t, 3)
            old = json.loads((OCR / f"{row['id']}.json").read_text(encoding="utf-8"))["text"]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"id": row["id"], "seconds": sec, "same_text": text == old, "device": dev}),
                           encoding="utf-8")
            tmp.replace(path)
            print(f"{split} {k}/{len(rows)} {row['id']} {sec}s same_text={text == old}", flush=True)
        print(f"{split}: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test", "train", "test_degraded"])
