"""Compare PaddleOCR settings on the validation split (OCR only, no LLM). Resumable.

Usage:
  python data/scripts/compare_ocr.py run B C F D     # OCR val with each option (skips receipts already done)
  python data/scripts/compare_ocr.py score            # print the comparison table (A = current OCR)

Options (each changes one thing vs the current setup A; defaults: thresh 0.3, box_thresh 0.6, unclip 1.5):
  A  current setup (boxes already in cache/ocr_boxes, seconds from cache/ocr)
  B  lower detection thresholds: text_det_thresh 0.2, text_det_box_thresh 0.4
  C  larger box margin: text_det_unclip_ratio 2.0
  F  B + C
  D  upscale the image x1.5 before OCR (boxes scaled back to original pixels)
Writes cache/ocr_variants/<option>/<id>.json = {"id", "boxes", "seconds"} (gitignored).

Scores (validation only, never test):
  qty>1 missing : labelled quantity > 1 not on the item's OCR row or the next row (main target)
  name+price    : item name (fuzzy) and price digits both somewhere in the OCR text
  total         : labelled total's digits somewhere in the OCR text
  boxes, seconds: mean per receipt (more boxes can mean more junk)
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rapidfuzz import fuzz  # noqa: E402

from receipt.ocr.layout import group_rows  # noqa: E402
from receipt.ocr.paddle import boxes_to_text, make_ocr, ocr_boxes  # noqa: E402

SPLIT = "validation"
OUT = ROOT / "cache" / "ocr_variants"
OPTIONS = {
    "B": {"text_det_thresh": 0.2, "text_det_box_thresh": 0.4},
    "C": {"text_det_unclip_ratio": 2.0},
    "F": {"text_det_thresh": 0.2, "text_det_box_thresh": 0.4, "text_det_unclip_ratio": 2.0},
    "D": {},
}
UPSCALE = {"D": 1.5}


def load_rows():
    with open(ROOT / "data" / "processed" / f"{SPLIT}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def run(option):
    from PIL import Image

    out = OUT / option
    out.mkdir(parents=True, exist_ok=True)
    print(f"option {option}: {OPTIONS[option] or ''} upscale={UPSCALE.get(option, 1)}", flush=True)
    ocr = make_ocr(**OPTIONS[option])
    rows = load_rows()
    for k, row in enumerate(rows, 1):
        path = out / f"{row['id']}.json"
        if path.exists():
            continue
        image = ROOT / "data" / "raw" / "cord" / row["image"]
        scale = UPSCALE.get(option, 1)
        t = time.time()
        if scale != 1:
            img = Image.open(image).convert("RGB")
            tmp_img = out / "_upscaled.png"
            img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS).save(tmp_img)
            boxes = ocr_boxes(tmp_img, ocr)
            for b in boxes:
                b["box"] = [round(v / scale, 1) for v in b["box"]]
        else:
            boxes = ocr_boxes(image, ocr)
        sec = round(time.time() - t, 2)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": row["id"], "boxes": boxes, "seconds": sec}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
        print(f"{option} {k}/{len(rows)} {row['id']} boxes={len(boxes)} {sec}s", flush=True)
    print(f"option {option}: done", flush=True)


def _digits(text):
    return {re.sub(r"\D", "", t) for t in re.findall(r"\d[\d.,]*", text)} - {""}


def _qty_in(text, q):
    s = str(int(q)) if q == int(q) else str(q)
    return re.search(rf"(?<![\d.,]){re.escape(s)}(?![\d.,])", text) is not None


def score_option(get_boxes, get_seconds, rows):
    qty_miss = qty_n = cov = items = tot_hit = tot_n = 0
    n_boxes, secs = [], []
    for row in rows:
        boxes = get_boxes(row["id"])
        if boxes is None:
            continue
        n_boxes.append(len(boxes))
        secs.append(get_seconds(row["id"]))
        text = boxes_to_text(boxes)
        low, digits = text.lower(), _digits(text)
        rec = row["receipt"]
        if rec["total"] is not None:
            tot_n += 1
            tot_hit += str(int(rec["total"])) in digits
        lines = group_rows(boxes)
        for it in rec["items"]:
            items += 1
            name_ok = fuzz.partial_ratio(it["name"].lower(), low) >= 80
            cov += name_ok and it["price"] is not None and str(int(it["price"])) in digits
            q = it["qty"]
            if q is None or q <= 1 or not lines:
                continue
            qty_n += 1
            sc = [max(fuzz.partial_ratio(it["name"].lower(), b["text"].lower()) for b in r) for r in lines]
            i = max(range(len(lines)), key=lambda j: sc[j])
            near = " ".join(b["text"] for b in lines[i])
            if i + 1 < len(lines):
                near += " " + " ".join(b["text"] for b in lines[i + 1])
            qty_miss += not (sc[i] >= 80 and _qty_in(near, q))
    pct = lambda a, b: f"{100 * a / b:5.1f}% ({a}/{b})" if b else "n/a"
    return {
        "receipts": len(n_boxes),
        "qty>1 missing": pct(qty_miss, qty_n),
        "name+price": pct(cov, items),
        "total": pct(tot_hit, tot_n),
        "boxes": f"{statistics.mean(n_boxes):.1f}" if n_boxes else "n/a",
        "seconds": f"{statistics.mean(secs):.1f}" if secs else "n/a",
    }


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def score():
    rows = load_rows()
    results = {
        "A": score_option(
            lambda i: (_read(ROOT / "cache" / "ocr_boxes" / f"{i}.json") or {}).get("boxes"),
            lambda i: _read(ROOT / "cache" / "ocr" / f"{i}.json")["seconds"],
            rows,
        )
    }
    for opt in OPTIONS:
        if (OUT / opt).exists():
            results[opt] = score_option(
                lambda i, o=opt: (_read(OUT / o / f"{i}.json") or {}).get("boxes"),
                lambda i, o=opt: _read(OUT / o / f"{i}.json")["seconds"],
                rows,
            )
    cols = ["receipts", "qty>1 missing", "name+price", "total", "boxes", "seconds"]
    print("option | " + " | ".join(cols))
    for opt, r in results.items():
        print(f"{opt:6s} | " + " | ".join(str(r[c]) for c in cols))
    print("Seconds for A are from the original laptop run; the others from this run (same laptop).")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "score"):
        sys.exit(__doc__)
    if sys.argv[1] == "run":
        for opt in sys.argv[2:] or list(OPTIONS):
            run(opt)
    else:
        score()
