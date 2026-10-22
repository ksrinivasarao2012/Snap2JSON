"""Tests for normalize.py."""
import pytest

from receipt.normalize import parse_amount, parse_qty


@pytest.mark.parametrize("raw,expected", [
    ("16,500", 16500.0),
    ("23.000", 23000.0),
    ("3,112,800", 3112800.0),
    ("50000", 50000.0),
    ("Rp 1.000", 1000.0),
    ("*110000", 110000.0),
    (45500, 45500.0),
    ("Rp 58000.00", 58000.0),
    ("9.500,00", 9500.0),
    ("130.000,00", 130000.0),
    ("47619.00", 47619.0),
    ("Rp 5272.73", 5272.73),
    ("1,234,567", 1234567.0),
])
def test_parse_amount_values(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "abc", "-"])
def test_parse_amount_none(raw):
    assert parse_amount(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("1", 1.0),
    ("2", 2.0),
    ("1.5", 1.5),
    ("1,5", 1.5),
    ("3 x", 3.0),
])
def test_parse_qty_values(raw, expected):
    assert parse_qty(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "x"])
def test_parse_qty_none(raw):
    assert parse_qty(raw) is None
