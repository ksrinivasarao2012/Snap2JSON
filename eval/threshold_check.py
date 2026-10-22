"""Check the item name-match threshold (metrics.NAME_THRESHOLD) on VALIDATION runs only.

Usage (from the repo root):
  python eval/threshold_check.py list    # writes results/threshold_check_validation.csv
  python eval/threshold_check.py score [labelled.csv]   # after the 'same' column is filled in by hand
  python eval/threshold_check.py rescore 80 60   # item P/R/F1 of every saved run at each threshold

Labelled copies: results/threshold_check_validation_strict.csv (a trimmed or misread name is a different
item) and ..._lenient.csv (only a different receipt line is a different item).
'list' pairs predicted and true items one-to-one exactly like the scorer (same qty and price, best
name score first) but with no name threshold, and keeps pairs scoring LOW..HIGH-1. Pairs are merged
across all validation runs. 'same' is filled in by hand (y/n) from the names and, when unclear, the
receipt image. 'score' counts, for each threshold, same-item pairs it rejects and different-item pairs
it accepts. 'rescore' writes results/tables/name_threshold_sensitivity.json (header fields do not
depend on the name threshold, so only items are re-scored).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from rapidfuzz import fuzz  # noqa: E402

from eval.metrics import _norm_text, aggregate, score_receipt  # noqa: E402

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "threshold_check_validation.csv"
LOW, HIGH = 50, 100


def pairs_for(pred_items, gt_items):
    """One-to-one pairs (score, pred, gt) with equal qty/price, best name score first, no threshold."""
    cand = []
    for i, p in enumerate(pred_items):
        for j, g in enumerate(gt_items):
            if g.get("qty") is not None and p.get("qty") != g["qty"]:
                continue
            if g.get("price") is not None and p.get("price") != g["price"]:
                continue
            cand.append((fuzz.ratio(_norm_text(p.get("name")) or "", _norm_text(g.get("name")) or ""), i, j))
    cand.sort(key=lambda t: -t[0])
    used_p, used_g, out = set(), set(), []
    for s, i, j in cand:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((s, pred_items[i], gt_items[j]))
    return out


def list_pairs():
    gt = {}
    with open(ROOT / "data" / "processed" / "validation.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gt[r["id"]] = r["receipt"].get("items") or []
    found = {}
    for run in sorted(RUNS.glob("*__validation.jsonl")):
        name = run.stem.replace("__validation", "")
        for line in open(run, encoding="utf-8"):
            o = json.loads(line)
            pred = (o["receipt"] or {}).get("items") or []
            for s, p, g in pairs_for(pred, gt[o["receipt_id"]]):
                if not LOW <= s < HIGH:
                    continue
                key = (o["receipt_id"], _norm_text(p.get("name")), _norm_text(g.get("name")))
                row = found.setdefault(key, {"receipt_id": key[0], "score": round(s, 1),
                                             "pred_name": p.get("name"), "gt_name": g.get("name"),
                                             "qty": g.get("qty"), "price": g.get("price"), "runs": []})
                row["runs"].append(name)
    rows = sorted(found.values(), key=lambda r: (r["score"], r["receipt_id"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["receipt_id", "score", "pred_name", "gt_name", "qty", "price",
                                          "n_runs", "runs", "same"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_runs": len(r["runs"]), "runs": " ".join(r["runs"]), "same": ""})
    print(f"{len(rows)} pairs scoring {LOW}-{HIGH - 1} -> {OUT.relative_to(ROOT)}")


def score(path=OUT):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))  # Excel adds a BOM
    todo = [r for r in rows if r["same"] not in ("y", "n")]
    if todo:
        sys.exit(f"{len(todo)} pairs still unlabelled")
    print(f"{len(rows)} pairs: {sum(r['same'] == 'y' for r in rows)} same, {sum(r['same'] == 'n' for r in rows)} different")
    print("threshold  same_rejected  diff_accepted  errors   (pair counts; weighted by runs in brackets)")
    for t in range(LOW, HIGH, 5):
        rej = [r for r in rows if r["same"] == "y" and float(r["score"]) < t]
        acc = [r for r in rows if r["same"] == "n" and float(r["score"]) >= t]
        wr, wa = sum(int(r["n_runs"]) for r in rej), sum(int(r["n_runs"]) for r in acc)
        print(f"{t:9}  {len(rej):13}  {len(acc):13}  {len(rej) + len(acc):6}   [{wr} / {wa} / {wr + wa}]")


def rescore(*thresholds):
    thresholds = [int(t) for t in thresholds] or [80, 60]
    out = {}
    for run in sorted(RUNS.glob("*.jsonl")):
        split = run.stem.split("__")[-1]
        with open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8") as f:
            rows = {r["id"]: r for r in map(json.loads, f)}
        outs = [o for o in map(json.loads, open(run, encoding="utf-8")) if o["receipt_id"] in rows]
        out[run.stem] = {}
        for t in thresholds:
            agg = aggregate([score_receipt(o["receipt"], rows[o["receipt_id"]]["receipt"], threshold=t,
                                           sub_items=rows[o["receipt_id"]].get("sub_items", [])) for o in outs])
            out[run.stem][str(t)] = {k: agg[k] for k in ("items_precision", "items_recall", "items_f1")}
        print(run.stem, "  ".join(f"F1@{t}={out[run.stem][str(t)]['items_f1']:.3f}" for t in thresholds))
    dest = ROOT / "results" / "tables" / "name_threshold_sensitivity.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", dest.relative_to(ROOT))


if __name__ == "__main__":
    {"list": list_pairs, "score": score, "rescore": rescore}[sys.argv[1]](*sys.argv[2:])
