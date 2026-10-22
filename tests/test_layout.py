"""Tests for receipt.ocr.layout (row grouping of OCR boxes)."""
from receipt.ocr.layout import BOXES_HEADER, boxes_text, group_rows, rows_text


def box(text, x0, y0, x1, y1):
    return {"text": text, "box": [x0, y0, x1, y1], "score": 1.0}


# One printed row, slightly tilted (price is 15 px lower), then a TOTAL row. Given out of order.
BOXES = [
    box("11,000", 400, 115, 460, 155),
    box("TOTAL", 0, 200, 50, 240),
    box("DONAT GULA", 0, 100, 100, 140),
    box("1X", 200, 105, 220, 145),
    box("11,000", 400, 200, 460, 240),
]


def test_group_rows_joins_tilted_row_and_orders():
    rows = group_rows(BOXES)
    assert [[b["text"] for b in r] for r in rows] == [["DONAT GULA", "1X", "11,000"], ["TOTAL", "11,000"]]


def test_rows_far_apart_stay_separate():
    rows = group_rows([box("A", 0, 0, 10, 40), box("B", 0, 30, 10, 70)])  # overlap 10 of 40 = 0.25
    assert len(rows) == 2


def test_rows_text_keeps_columns():
    lines = rows_text(BOXES).splitlines()
    assert lines[0].split() == ["DONAT", "GULA", "1X", "11,000"]
    assert lines[1].split() == ["TOTAL", "11,000"]
    assert lines[0].index("11,000") == lines[1].index("11,000")  # same x -> same column


def test_boxes_text_has_positions():
    lines = boxes_text(BOXES).splitlines()
    assert lines[0] == BOXES_HEADER
    assert lines[1] == "DONAT GULA [x=0] | 1X [x=200] | 11,000 [x=400]"


def test_empty():
    assert rows_text([]) == "" and group_rows([]) == []


def test_tall_box_does_not_merge_two_rows():
    rows = group_rows([box("ICE", 100, 539, 330, 590), box("82,000", 600, 554, 730, 604),
                       box("now", 15, 544, 104, 618), box("AVOCADO", 102, 579, 331, 627)])
    assert [[b["text"] for b in r] for r in rows] == [["now", "ICE", "82,000"], ["AVOCADO"]]
