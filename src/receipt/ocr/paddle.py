"""PaddleOCR (3.x) wrapper: image -> text, or image -> text boxes with positions.

ocr_image returns the recognized boxes joined top to bottom, one box per line (the format all
OCR+LLM runs so far used). ocr_boxes returns the same boxes with their pixel positions, so other
text layouts (e.g. rows) can be built from them later.
"""
from pathlib import Path
from typing import Union

# Receipts are photos of flat paper: skip the page-orientation and unwarping models (faster).
# Mobile models: ~5x faster than the default medium ones on CPU (timed on 10 val receipts).
BASE_KWARGS = dict(
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

_ocr = None


def make_ocr(**overrides):
    """A PaddleOCR instance with the project settings, optionally changed (e.g. text_det_thresh)."""
    from paddleocr import PaddleOCR

    return PaddleOCR(**{**BASE_KWARGS, **overrides})


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = make_ocr()
    return _ocr


def ocr_boxes(path: Union[str, Path], ocr=None) -> list:
    """[{"text", "box": [x_min, y_min, x_max, y_max], "score"}, ...] in PaddleOCR's own order."""
    result = (ocr or _get_ocr()).predict(str(path))
    if not result:
        return []
    res = result[0]
    return [{"text": t, "box": [round(float(v), 1) for v in b], "score": round(float(s), 4)}
            for t, b, s in zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"])]


def boxes_to_text(boxes: list) -> str:
    """The original format: boxes sorted by (y in 15 px bands, x), one box per line."""
    order = sorted(boxes, key=lambda b: (round(b["box"][1] / 15), b["box"][0]))
    return "\n".join(b["text"] for b in order)


def ocr_image(path: Union[str, Path]) -> str:
    return boxes_to_text(ocr_boxes(path))
