"""Normalization helpers for dates, currency and numbers."""
import re
from typing import Optional


def parse_amount(value) -> Optional[float]:
    """'16,500' / '23.000' / 'Rp 1.000' -> 16500.0 / 23000.0 / 1000.0.

    CORD is Indonesian rupiah with no decimals, so every separator is a thousands separator.
    Returns None when there are no digits.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return float(digits) if digits else None


def parse_qty(value) -> Optional[float]:
    """'2' -> 2.0, '1.5' -> 1.5, junk -> None. Quantities are not thousands-separated."""
    if value is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(m.group().replace(",", ".")) if m else None
