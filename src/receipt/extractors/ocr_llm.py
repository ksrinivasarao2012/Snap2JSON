"""OCR text -> model extractor (any backend).

layout="plain": the OCR text saved by data/scripts/run_ocr.py (one box per line; all first runs).
layout="rows" / "boxes": text rebuilt from the boxes saved by data/scripts/run_ocr_boxes.py
(receipt.ocr.layout: A = rows, B = rows + x positions). Same OCR step, only the text format differs.
"""
import json
import time
from pathlib import Path

from receipt.cache import CACHE_DIR, DiskCache, make_key
from receipt.extractors.base import PROMPT, ExtractionResult, Extractor, parse_output


LAYOUTS = ("plain", "rows", "boxes")


def load_ocr_text(receipt_id: str, ocr_dir: Path = CACHE_DIR / "ocr") -> str:
    return json.loads((Path(ocr_dir) / f"{receipt_id}.json").read_text(encoding="utf-8"))["text"]


def load_layout_text(receipt_id: str, layout: str, boxes_dir: Path = CACHE_DIR / "ocr_boxes") -> str:
    from receipt.ocr.layout import boxes_text, rows_text

    boxes = json.loads((Path(boxes_dir) / f"{receipt_id}.json").read_text(encoding="utf-8"))["boxes"]
    return rows_text(boxes) if layout == "rows" else boxes_text(boxes)


class OcrLlm(Extractor):
    def __init__(self, backend, cache: DiskCache = None, ocr_dir: Path = CACHE_DIR / "ocr",
                 layout: str = "plain", boxes_dir: Path = CACHE_DIR / "ocr_boxes"):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        self.backend = backend
        self.cache = cache or DiskCache("ocr_llm")
        self.ocr_dir, self.layout, self.boxes_dir = ocr_dir, layout, boxes_dir
        self.name = f"ocr+{backend.model}" if layout == "plain" else f"ocr_{layout}+{backend.model}"

    def extract(self, receipt_id, image_path=None) -> ExtractionResult:
        try:
            if self.layout == "plain":
                text = load_ocr_text(receipt_id, self.ocr_dir)
            else:
                text = load_layout_text(receipt_id, self.layout, self.boxes_dir)
            key = make_key(self.backend.model, PROMPT, text)
            hit = self.cache.get(key)
            cached = hit is not None
            if not cached:
                t = time.time()
                hit = self.backend.generate(PROMPT, text=text)
                hit["seconds"] = round(time.time() - t, 3)
                self.cache.set(key, hit)  # only successful calls reach this line
        except Exception as e:  # rate limit, missing OCR file, network...
            return ExtractionResult(receipt_id, None, False, "", 0.0, error=f"{type(e).__name__}: {e}")
        receipt, valid = parse_output(hit["text"])
        return ExtractionResult(receipt_id, receipt, valid, hit["text"], hit["seconds"],
                                hit.get("input_tokens"), hit.get("output_tokens"), cached)
