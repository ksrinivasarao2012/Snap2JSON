"""OCR text -> model extractor (any backend). Reads the OCR text saved by data/scripts/run_ocr.py."""
import json
import time
from pathlib import Path

from receipt.cache import CACHE_DIR, DiskCache, make_key
from receipt.extractors.base import PROMPT, ExtractionResult, Extractor, parse_output


def load_ocr_text(receipt_id: str, ocr_dir: Path = CACHE_DIR / "ocr") -> str:
    return json.loads((Path(ocr_dir) / f"{receipt_id}.json").read_text(encoding="utf-8"))["text"]


class OcrLlm(Extractor):
    def __init__(self, backend, cache: DiskCache = None, ocr_dir: Path = CACHE_DIR / "ocr"):
        self.backend = backend
        self.cache = cache or DiskCache("ocr_llm")
        self.ocr_dir = ocr_dir
        self.name = f"ocr+{backend.model}"

    def extract(self, receipt_id, image_path=None) -> ExtractionResult:
        try:
            text = load_ocr_text(receipt_id, self.ocr_dir)
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
