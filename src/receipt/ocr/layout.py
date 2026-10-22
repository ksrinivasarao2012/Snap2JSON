"""Turn OCR boxes into text that keeps the receipt layout.

Boxes come from receipt.ocr.paddle.ocr_boxes: {"text", "box": [x0, y0, x1, y1], "score"}.
- rows_text:  boxes on the same printed line become one line, spaced out by their x position,
              so "DONAT GULA   1X   11,000" stays together (layout A).
- boxes_text: the same rows, with every piece followed by its x position in pixels (layout B).
A box joins the nearest row whose centre line is within half the median box height of the box's
centre (tolerates slightly tilted photos), unless it would sit on top of a box already in that row.
Using the median height stops tall junk boxes (e.g. at the photo edge) from merging two rows.
"""
import statistics

MAX_DIST = 0.5  # x median box height
MAX_X_OVERLAP = 0.5  # x the narrower box's width


def _cy(b):
    return (b["box"][1] + b["box"][3]) / 2


def _x_collides(b, row):
    x0, x1 = b["box"][0], b["box"][2]
    for m in row["boxes"]:
        w = min(x1 - x0, m["box"][2] - m["box"][0])
        if w > 0 and min(x1, m["box"][2]) - max(x0, m["box"][0]) > MAX_X_OVERLAP * w:
            return True
    return False


def group_rows(boxes: list) -> list:
    """List of rows (top to bottom); each row is a list of boxes (left to right)."""
    if not boxes:
        return []
    limit = MAX_DIST * statistics.median(b["box"][3] - b["box"][1] for b in boxes)
    rows = []  # each: {"cy": mean centre of its boxes, "boxes": [...]}
    for b in sorted(boxes, key=_cy):
        near = [r for r in rows if abs(_cy(b) - r["cy"]) <= limit and not _x_collides(b, r)]
        if near:
            r = min(near, key=lambda r: abs(_cy(b) - r["cy"]))
            r["boxes"].append(b)
            r["cy"] = statistics.mean(_cy(m) for m in r["boxes"])
        else:
            rows.append({"cy": _cy(b), "boxes": [b]})
    rows.sort(key=lambda r: r["cy"])
    return [sorted(r["boxes"], key=lambda b: b["box"][0]) for r in rows]


def _char_width(boxes):
    widths = [(b["box"][2] - b["box"][0]) / len(b["text"]) for b in boxes if b["text"]]
    return max(statistics.median(widths), 1.0) if widths else 10.0


def rows_text(boxes: list) -> str:
    """Layout A: one line per printed row, pieces placed at their x position (in characters)."""
    cw = _char_width(boxes)
    left = min((b["box"][0] for b in boxes), default=0)
    lines = []
    for row in group_rows(boxes):
        line = ""
        for b in row:
            col = round((b["box"][0] - left) / cw)
            line += " " * max(col - len(line), 1 if line else 0) + b["text"]
        lines.append(line.rstrip())
    return "\n".join(lines)


BOXES_HEADER = "(one line per printed row; each piece of text is followed by its x position in pixels)"


def boxes_text(boxes: list) -> str:
    """Layout B: one line per printed row, each piece as `text [x=..]`."""
    lines = [" | ".join(f"{b['text']} [x={round(b['box'][0])}]" for b in row) for row in group_rows(boxes)]
    return BOXES_HEADER + "\n" + "\n".join(lines)
