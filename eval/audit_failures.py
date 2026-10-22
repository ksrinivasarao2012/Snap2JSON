"""Manual failure audit on VALIDATION: was each error the model's fault, the OCR's, the scorer's, or the label's?

Usage (from the repo root):
  python eval/audit_failures.py build     # writes results/audit_validation.csv
  python eval/audit_failures.py summary   # after the 'cause' column is filled in by hand

'build' takes every error of a run and samples 20 (fixed seed): up to 7 wrong header fields and 13 item errors.
Item errors are counted once each:
- item_wrong:    a true item and the prediction meant for it (similar name, or same qty and price), but qty,
                 price or name differ. (The scorer counts this as one miss AND one extra.)
- item_missing:  a true item with no prediction meant for it.
- item_invented: a prediction that is not meant for any true item (predicted sub-items are ignored, as in scoring).
Each row shows the true and predicted values, the printed text from CORD's word annotations, and for OCR methods
the OCR text near it (what that method saw: plain lines or grouped rows). 'cause' is filled in by hand: model / ocr / scorer / gt.
"""
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rapidfuzz import fuzz  # noqa: E402

from eval.metrics import NAME_THRESHOLD, _is_sub_item, _item_match, _norm_text, _same  # noqa: E402
from receipt.normalize import parse_amount  # noqa: E402
from receipt.ocr.layout import group_rows  # noqa: E402

RUNS = ["ocr_llm__Qwen_Qwen3-VL-2B-Instruct", "ocr_llm_rows__Qwen_Qwen3-VL-2B-Instruct",
        "vlm__Qwen_Qwen3-VL-2B-Instruct", "ocr_llm__Qwen_Qwen3-VL-8B-Instruct", "vlm__Qwen_Qwen3-VL-8B-Instruct"]
N_HEADER, N_ITEMS = 7, 13  # 20 per run
SEED = 42
OUT = ROOT / "results" / "audit_validation.csv"
CAUSES = ("model", "ocr", "scorer", "gt")
HEADER_CAT = {"subtotal": "sub_total.subtotal_price", "tax": "sub_total.tax_price", "total": "total.total_price"}


def money(x):
    if x is None:
        return "none"
    return f"{x:,.0f}" if float(x) == int(x) else f"{x:,.2f}"


def item_str(it):
    return f"{it.get('name')} | qty {money(it.get('qty'))} | price {money(it.get('price'))}"


def name_score(a, b):
    return fuzz.ratio(_norm_text(a.get("name")) or "", _norm_text(b.get("name")) or "")


def load_lines(split="validation"):
    with open(ROOT / "data" / "raw" / "cord" / f"{split}_lines.jsonl", encoding="utf-8") as f:
        return {r["id"]: r["valid_line"] for r in map(json.loads, f)}


def printed_groups(lines):
    """{group_id: [(category, text), ...]} from CORD's word annotations, in printed order."""
    groups = {}
    for ln in lines:
        groups.setdefault(ln["group_id"], []).append((ln["category"], " ".join(w["text"] for w in ln["words"])))
    return groups


def printed_item(groups, it):
    """Printed words of the CORD group for a true item (same name; same price when the name repeats)."""
    best = None
    for parts in groups.values():
        nm = " ".join(t for c, t in parts if c == "menu.nm")
        if not nm or _norm_text(nm) != _norm_text(it.get("name")):
            continue
        price = next((parse_amount(t) for c, t in parts if c == "menu.price"), None)
        if best is None or price == it.get("price"):
            best = parts
    return " | ".join(f"{c.split('.')[-1]}: {t}" for c, t in best) if best else "(not found)"


def printed_with_amount(lines, value):
    """Categories of printed lines that contain this amount (e.g. what a wrong total really is)."""
    out = []
    for ln in lines:
        for w in ln["words"]:
            if value is not None and parse_amount(w["text"]) == value:
                out.append(ln["category"])
                break
    return sorted(set(out))


def ocr_rows(rid, plain):
    """The OCR text the method saw, as lines: plain = one box per line (cache/ocr), else grouped rows."""
    if plain:
        return json.loads((ROOT / "cache" / "ocr" / f"{rid}.json").read_text(encoding="utf-8"))["text"].splitlines()
    boxes = json.loads((ROOT / "cache" / "ocr_boxes" / f"{rid}.json").read_text(encoding="utf-8"))["boxes"]
    return ["   ".join(b["text"] for b in row) for row in group_rows(boxes)]


def ocr_near(rows, query):
    """The OCR line most like the query, with the line before and after it."""
    if not rows or not query:
        return ""
    k = max(range(len(rows)), key=lambda i: fuzz.partial_ratio(query.lower(), rows[i].lower()))
    return "  //  ".join(rows[max(k - 1, 0):k + 2])


def in_ocr(rows, value):
    return value is not None and any(parse_amount(t) == value for r in rows for t in re.findall(r"-?\d[\d.,]*", r))


def match(pred, gt, subs):
    """Scorer's matching, then pair leftovers that were meant for each other. Returns (wrong, missing, invented)."""
    pairs = sorted(((s, i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)
                    if (s := _item_match(p, g, NAME_THRESHOLD)) is not None), key=lambda t: -t[0])
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    missed = [j for j in range(len(gt)) if j not in used_g]
    extra = [i for i in range(len(pred)) if i not in used_p and not _is_sub_item(pred[i], subs, NAME_THRESHOLD)]
    cand = []
    for i in extra:
        for j in missed:
            s = name_score(pred[i], gt[j])
            same_line = pred[i].get("price") == gt[j].get("price") and pred[i].get("qty") == gt[j].get("qty")
            if s >= 50 or same_line:
                cand.append((s + (100 if same_line else 0), i, j))
    wrong, pi, gj = [], set(), set()
    for _, i, j in sorted(cand, key=lambda t: -t[0]):
        if i not in pi and j not in gj:
            pi.add(i)
            gj.add(j)
            wrong.append((i, j))
    return wrong, [j for j in missed if j not in gj], [i for i in extra if i not in pi]


def problem_of(p, g):
    bits = []
    if g.get("qty") is not None and p.get("qty") != g["qty"]:
        bits.append(f"qty wrong (true {money(g['qty'])}, model {money(p.get('qty'))})")
    if g.get("price") is not None and p.get("price") != g["price"]:
        bits.append(f"price wrong (true {money(g['price'])}, model {money(p.get('price'))})")
    s = name_score(p, g)
    if s < NAME_THRESHOLD:
        bits.append(f"name too different (similarity {s:.0f} < 80)")
    return "; ".join(bits) or "no difference found"


def build(out_path=OUT):
    gt_rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "processed" / "validation.jsonl",
                                                        encoding="utf-8"))}
    lines = load_lines()
    rng = random.Random(SEED)
    out = []
    for run in RUNS:
        is_ocr = run.startswith("ocr_llm")
        header, items = [], []
        for o in map(json.loads, open(ROOT / "results" / "runs" / f"{run}__validation.jsonl", encoding="utf-8")):
            rid = o["receipt_id"]
            gt, subs = gt_rows[rid]["receipt"], gt_rows[rid].get("sub_items", [])
            pred = o["receipt"] or {}
            groups = printed_groups(lines[rid])
            rows = ocr_rows(rid, plain=run.startswith("ocr_llm__")) if is_ocr else []
            base = dict(run=run, receipt_id=rid, image=f"data/raw/cord/validation/{rid}.png")
            ocr_cols = lambda q, v: (dict(ocr_rows_near=ocr_near(rows, q),  # noqa: E731
                                          true_value_in_ocr="yes" if in_ocr(rows, v) else "no") if is_ocr else {})
            for f in ("subtotal", "tax", "total"):
                if gt.get(f) is None or (pred and _same(f, pred.get(f), gt[f])):
                    continue
                p = pred.get(f)
                printed = " | ".join(t for c, t in (x for g in groups.values() for x in g) if c == HEADER_CAT[f])
                problem = f"{f} wrong (true {money(gt[f])}, model {money(p)})" if p is not None else f"{f} not given"
                if p is not None:
                    seen = [c for c in printed_with_amount(lines[rid], p) if c != HEADER_CAT[f]]
                    if seen:
                        problem += f"; the model's value is printed as: {', '.join(seen)}"
                header.append(dict(base, kind="header", item_or_field=f, true_value=money(gt[f]), model_value=money(p),
                                   problem=problem, printed_on_receipt=printed or "(not found)",
                                   **ocr_cols(printed.split(" ")[0] if printed else f, gt[f])))
            gi, pi = gt.get("items") or [], pred.get("items") or []
            wrong, missing, invented = match(pi, gi, subs)
            for i, j in wrong:
                pr = printed_item(groups, gi[j])
                items.append(dict(base, kind="item_wrong", item_or_field=gi[j].get("name"), true_value=item_str(gi[j]),
                                  model_value=item_str(pi[i]), problem=problem_of(pi[i], gi[j]),
                                  printed_on_receipt=pr, **ocr_cols(gi[j].get("name"), gi[j].get("price"))))
            for j in missing:
                items.append(dict(base, kind="item_missing", item_or_field=gi[j].get("name"),
                                  true_value=item_str(gi[j]), model_value="(nothing)",
                                  problem="item not extracted", printed_on_receipt=printed_item(groups, gi[j]),
                                  **ocr_cols(gi[j].get("name"), gi[j].get("price"))))
            for i in invented:
                seen = printed_with_amount(lines[rid], pi[i].get("price"))
                problem = "model added an item that is not in the true item list"
                if seen:
                    problem += f"; its price is printed as: {', '.join(seen)}"
                items.append(dict(base, kind="item_invented", item_or_field=pi[i].get("name"), true_value="(nothing)",
                                  model_value=item_str(pi[i]), problem=problem, printed_on_receipt="",
                                  **ocr_cols(pi[i].get("name"), None)))
        picked = rng.sample(header, min(N_HEADER, len(header)))
        picked += rng.sample(items, min(N_ITEMS + N_HEADER - len(picked), len(items)))
        kinds = Counter(r["kind"] for r in items)
        print(f"{run}: header errors {len(header)}, item errors {len(items)} {dict(kinds)} -> sampled {len(picked)}")
        out += sorted(picked, key=lambda r: (r["receipt_id"], r["kind"]))
    fields = ["run", "receipt_id", "kind", "item_or_field", "true_value", "model_value", "problem",
              "printed_on_receipt", "ocr_rows_near", "true_value_in_ocr", "cause", "note", "image"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {out_path}")


def summary(path=OUT):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))  # Excel adds a BOM
    bad = [r for r in rows if r["cause"].strip().lower() not in CAUSES]
    if bad:
        sys.exit(f"{len(bad)} rows without a cause in {CAUSES}")
    print("run".ljust(42), "  ".join(c.rjust(6) for c in CAUSES), "     n")
    for run in dict.fromkeys(r["run"] for r in rows):
        c = Counter(r["cause"].strip().lower() for r in rows if r["run"] == run)
        print(run.ljust(42), "  ".join(str(c[k]).rjust(6) for k in CAUSES), f"{sum(c.values()):6}")


if __name__ == "__main__":
    {"build": build, "summary": summary}[sys.argv[1]](*sys.argv[2:])
