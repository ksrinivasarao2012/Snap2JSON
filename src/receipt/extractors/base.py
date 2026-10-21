"""Common interface for all extraction methods, the shared prompt, and response parsing.

Every method (OCR text -> model, image -> model, for Gemini or Qwen) takes a receipt and returns
an ExtractionResult. The prompt is identical for all methods; only the input differs.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from receipt.schema import Receipt

PROMPT = """Extract the data from this receipt and return it as JSON with exactly these keys:
- merchant: store name, or null if not shown
- date: date as printed, or null if not shown
- items: list of {name, qty, price}; name as printed, qty as a number, price as the line total for that item
- subtotal: number, or null if the receipt does not show one
- tax: number, or null if the receipt does not show one
- total: number, or null if the receipt does not show one
All amounts are plain numbers with no currency symbol and no thousands separators (for example 16500, not 16,500).
Use null for anything not on the receipt. Do not guess. Return only the JSON."""


@dataclass
class ExtractionResult:
    receipt_id: str
    receipt: Optional[dict]  # None when the output was not valid JSON for our schema
    json_valid: bool
    raw_text: str
    seconds: float  # model/API latency (for cached calls: the original latency)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None  # set when the call itself failed (e.g. rate limit); not cached


class Extractor(ABC):
    name: str

    @abstractmethod
    def extract(self, receipt_id: str, image_path: Union[str, Path]) -> ExtractionResult:
        """Run the method on one receipt."""


def parse_output(raw_text: str):
    """Return (receipt dict, True) if raw_text is valid JSON matching our schema, else (None, False)."""
    try:
        return Receipt.model_validate(json.loads(raw_text)).model_dump(), True
    except (ValueError, ValidationError):  # json.JSONDecodeError is a ValueError
        return None, False
