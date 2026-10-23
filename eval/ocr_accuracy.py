"""OCR accuracy against CORD's word annotations (no model involved).

Usage (from the repo root):
  python eval/ocr_accuracy.py                      # main OCR (cache/ocr_boxes), all scorable splits
  python eval/ocr_accuracy.py --variant T --splits validation    # an OCR variant from cache/ocr_variants/<name>
Needs data/raw/cord/<split>_lines.jsonl (data/scripts/extract_cord_lines.py). Writes results/tables/ocr_accuracy.json
(main OCR) or results/tables/ocr_accuracy_<variant>.json.

CORD labels the words of the fields it annotates (items, counts, prices, totals...), with their positions in the same
pixels as our images. Store name, address and footer text are mostly NOT labelled, so this measures how well OCR reads
the labelled words, not everything on the receipt (no OCR precision).

Matching: a labelled word belongs to the OCR box that contains its centre (the smallest one if several).
- detected: share of labelled words inside some OCR box
- CER: character error rate over all labelled text = edits / labelled characters. For each OCR box, its text is
  compared with its labelled words (left to right); a word in no box counts as fully deleted. Case and spaces are
  ignored. Unlabelled text inside a box counts as errors, so CER is a slight overestimate.
- word_acc: share of labelled words that appear exactly (case-insensitive) in their box's text
- numbers: share of labelled amounts / counts whose value appears in their box's text (`parse_amount`)
Degraded receipts: blur, darken and lowres keep the geometry, so they are scored with the test labels; rotate is not.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rapidfuzz.distance import Levenshtein  # noqa: E402

from receipt.normalize import parse_amount  # noqa: E402

LINES = ROOT / "data" / "raw" / "cord"
SPLITS = ["validation", "test", "train", "test_degraded"]
SAME_GEOMETRY = ("blur", "darken", "lowres")
GROUPS = {"item name": ("menu.nm",), "item count": ("menu.cnt",), "item price": ("menu.price", "menu.unitprice"),
          "subtotal / tax / total": ("sub_total.", "total.total_price")}


def group_of(category):
    for g, prefixes in GROUPS.items():
        if category.startswith(prefixes):
            return g
    return "other"


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def word_box(quad):
    xs, ys = [quad[f"x{i}"] for i in range(1, 5)], [quad[f"y{i}"] for i in range(1, 5)]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(xs)


def numbers_in(text):
    return {parse_amount(t) for t in re.findall(r"-?\d[\d.,]*", text)} - {None}


def is_number_word(category, text):
    return group_of(category) in ("item count", "item price", "subtotal / tax / total") and \
        re.search(r"\d", text) is not None


def score_receipt(valid_line, boxes):
    """Counts for one receipt: overall and per field group."""
    words = []
    for ln in valid_line:
        for w in ln["words"]:
            if w["text"].strip():
                cx, cy, x0 = word_box(w["quad"])
                words.append(dict(text=w["text"], cat=ln["category"], cx=cx, cy=cy, x0=x0))
    in_box = defaultdict(list)
    for w in words:
        hits = [k for k, b in enumerate(boxes)
                if b["box"][0] <= w["cx"] <= b["box"][2] and b["box"][1] <= w["cy"] <= b["box"][3]]
        w["box"] = min(hits, key=lambda k: (boxes[k]["box"][2] - boxes[k]["box"][0])
                       * (boxes[k]["box"][3] - boxes[k]["box"][1])) if hits else None
        if w["box"] is not None:
            in_box[w["box"]].append(w)
    c = defaultdict(lambda: defaultdict(int))
    for k, ws in in_box.items():  # character errors of each box against its labelled words
        gt = norm("".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"])))
        edits = Levenshtein.distance(norm(boxes[k]["text"]), gt)
        for w in ws:  # share the box's edits among its words by length (for the per-group numbers)
            c[group_of(w["cat"])]["edits"] += edits * len(norm(w["text"])) / max(len(gt), 1)
    for w in words:
        g = group_of(w["cat"])
        for key in (g, "all"):
            c[key]["words"] += 1
            c[key]["chars"] += len(norm(w["text"]))
            if w["box"] is None:
                c[key]["missed"] += 1
                if key == g:
                    c[g]["edits"] += len(norm(w["text"]))
                continue
            box_text = boxes[w["box"]]["text"]
            c[key]["word_ok"] += norm(w["text"]) in {norm(t) for t in box_text.split()} or \
                norm(w["text"]) == norm(box_text)
            if is_number_word(w["cat"], w["text"]):
                c[key]["num_n"] += 1
                c[key]["num_ok"] += parse_amount(w["text"]) in numbers_in(box_text)
        if w["box"] is None and is_number_word(w["cat"], w["text"]):
            for key in (g, "all"):
                c[key]["num_n"] += 1
    c["all"]["edits"] = sum(v["edits"] for k, v in c.items() if k != "all")
    return c


def summarize(c):
    pct = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return dict(words=c["words"], detected=pct(c["words"] - c["missed"], c["words"]),
                cer=pct(c["edits"], c["chars"]), word_acc=pct(c["word_ok"], c["words"]),
                numbers_n=c["num_n"], numbers_acc=pct(c["num_ok"], c["num_n"]))


def evaluate(split, boxes_dir):
    lines_split = "test" if split == "test_degraded" else split
    lines = {r["id"]: r["valid_line"] for r in map(json.loads, open(LINES / f"{lines_split}_lines.jsonl",
                                                                    encoding="utf-8"))}
    rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8")]
    out = {}
    for subset in (SAME_GEOMETRY if split == "test_degraded" else (None,)):
        total = defaultdict(lambda: defaultdict(float))
        n = 0
        for r in rows:
            if subset and r.get("degradation") != subset:
                continue
            path = Path(boxes_dir) / f"{r['id']}.json"
            if not path.exists():
                continue
            boxes = json.loads(path.read_text(encoding="utf-8"))["boxes"]
            base_id = r["id"].rsplit("_", 1)[0] if subset else r["id"]
            for g, counts in score_receipt(lines[base_id], boxes).items():
                for k, v in counts.items():
                    total[g][k] += v
            n += 1
        key = f"{split}/{subset}" if subset else split
        out[key] = {"receipts": n, **{g: summarize(v) for g, v in total.items()}}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None, help="OCR variant folder in cache/ocr_variants (default: cache/ocr_boxes)")
    p.add_argument("--splits", nargs="+", default=None)
    a = p.parse_args()
    boxes_dir = ROOT / "cache" / ("ocr_variants/" + a.variant if a.variant else "ocr_boxes")
    splits = a.splits or (["validation"] if a.variant else SPLITS)
    res = {}
    for s in splits:
        res.update(evaluate(s, boxes_dir))
    name = f"ocr_accuracy_{a.variant}" if a.variant else "ocr_accuracy"
    (ROOT / "results" / "tables" / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'split':24} {'receipts':>8} {'words':>6} {'detected':>9} {'CER':>6} {'word acc':>9} {'numbers':>8}")
    for k, v in res.items():
        a_ = v["all"]
        print(f"{k:24} {v['receipts']:8} {a_['words']:6.0f} {a_['detected']:9.3f} {a_['cer']:6.3f} "
              f"{a_['word_acc']:9.3f} {a_['numbers_acc']:8.3f}")
    print("\nper field group (detected / CER / numbers acc):")
    for k, v in res.items():
        print(" ", k, " | ".join(f"{g}: {s['detected']:.3f} / {s['cer']:.3f} / "
                                 f"{s['numbers_acc'] if s['numbers_acc'] is not None else '-'}"
                                 for g, s in v.items() if g not in ("all", "receipts")))


if __name__ == "__main__":
    main()
