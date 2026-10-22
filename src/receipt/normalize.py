"""Normalization helpers for dates, currency and numbers."""
import re
from typing import Optional


def parse_amount(value) -> Optional[float]:
    """'16,500' / '23.000' / 'Rp 1.000' -> 16500.0 / 23000.0 / 1000.0.

    Rupiah amounts normally have no decimals, so a separator followed by 3 digits is a thousands
    separator. A separator followed by exactly 2 final digits is a decimal part
    ('Rp 58000.00' -> 58000.0, '9.500,00' -> 9500.0, '5272.73' -> 5272.73).
    Returns None when there are no digits.
    """
    if value is None:
        return None
    s = str(value).strip()
    m = re.search(r"[.,](\d{2})$", s)
    cents = m.group(1) if m else ""
    if m:
        s = s[: m.start()]
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    return float(f"{digits}.{cents}") if cents else float(digits)


def parse_qty(value) -> Optional[float]:
    """'2' -> 2.0, '1.5' -> 1.5, junk -> None. Quantities are not thousands-separated."""
    if value is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(m.group().replace(",", ".")) if m else None
