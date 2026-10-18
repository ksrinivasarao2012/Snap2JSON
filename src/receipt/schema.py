"""Pydantic Receipt and Item models shared by all extraction methods.

Money values are plain numbers (CORD receipts are in IDR, no decimals).
Fields CORD does not label (merchant, date) stay in the schema so models can
still extract them, but ground truth is null and they are not scored.
"""
from typing import Optional

from pydantic import BaseModel


class Item(BaseModel):
    name: str
    qty: Optional[float] = None
    price: Optional[float] = None  # line total as printed on the receipt


class Receipt(BaseModel):
    merchant: Optional[str] = None  # not in CORD labels -> not scored
    date: Optional[str] = None  # not in CORD labels -> not scored
    items: list[Item] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
