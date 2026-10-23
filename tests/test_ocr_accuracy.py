"""Tests for the OCR-vs-CORD word scorer (synthetic receipt, no files)."""
from eval.ocr_accuracy import score_receipt, summarize


def word(text, x0, y0, x1, y1):
    return {"text": text, "quad": {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}}


LINES = [
    {"category": "menu.cnt", "words": [word("2", 0, 0, 10, 20)]},
    {"category": "menu.nm", "words": [word("ICED", 20, 0, 60, 20), word("TEA", 65, 0, 95, 20)]},
    {"category": "menu.price", "words": [word("16,500", 200, 0, 260, 20)]},
]


def test_perfect_ocr():
    boxes = [{"text": "2 ICED TEA", "box": [0, 0, 100, 20]}, {"text": "16,500", "box": [195, 0, 265, 20]}]
    s = summarize(score_receipt(LINES, boxes)["all"])
    assert s["detected"] == 1.0 and s["cer"] == 0.0 and s["word_acc"] == 1.0 and s["numbers_acc"] == 1.0


def test_missed_count_and_misread_price():
    boxes = [{"text": "ICED TEA", "box": [18, 0, 100, 20]}, {"text": "16,600", "box": [195, 0, 265, 20]}]
    c = score_receipt(LINES, boxes)
    s = summarize(c["all"])
    assert s["words"] == 4 and s["detected"] == 0.75
    assert s["cer"] == round(2 / 14, 4)  # "2" deleted + one wrong digit, over 1+4+3+6 characters
    assert s["numbers_n"] == 2 and s["numbers_acc"] == 0.0
    assert summarize(c["item count"])["detected"] == 0.0
