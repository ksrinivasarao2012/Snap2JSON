"""PaddleOCR (3.x) wrapper: image -> text.

Returns the recognized lines joined top to bottom, one line per text row.
"""
from pathlib import Path
from typing import Union

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        # Receipts are photos of flat paper: skip the page-orientation and unwarping models (faster).
        _ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr


def ocr_image(path: Union[str, Path]) -> str:
    result = _get_ocr().predict(str(path))
    if not result:
        return ""
    res = result[0]
    texts, boxes = res["rec_texts"], res["rec_boxes"]  # boxes: [x_min, y_min, x_max, y_max]
    order = sorted(range(len(texts)), key=lambda i: (round(float(boxes[i][1]) / 15), float(boxes[i][0])))
    return "\n".join(texts[i] for i in order)
