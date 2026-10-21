"""Image -> model extractor (Gemini vision)."""
import time

from receipt.cache import DiskCache, make_key
from receipt.extractors.base import PROMPT, ExtractionResult, Extractor, parse_output
from receipt.extractors.gemini import prepare_image


class GeminiVlm(Extractor):
    def __init__(self, backend, cache: DiskCache = None):
        self.backend = backend
        self.cache = cache or DiskCache("gemini_vlm")
        self.name = f"image+{backend.model}"

    def extract(self, receipt_id, image_path) -> ExtractionResult:
        try:
            image = prepare_image(image_path)
            key = make_key(self.backend.model, PROMPT, image)
            hit = self.cache.get(key)
            cached = hit is not None
            if not cached:
                t = time.time()
                hit = self.backend.generate(PROMPT, image_png=image)
                hit["seconds"] = round(time.time() - t, 3)
                self.cache.set(key, hit)
        except Exception as e:
            return ExtractionResult(receipt_id, None, False, "", 0.0, error=f"{type(e).__name__}: {e}")
        receipt, valid = parse_output(hit["text"])
        return ExtractionResult(receipt_id, receipt, valid, hit["text"], hit["seconds"],
                                hit.get("input_tokens"), hit.get("output_tokens"), cached)
