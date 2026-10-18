"""Map CORD gt_parse labels into our schema; writes ground truth to data/processed/<split>.jsonl.

Usage: python data/scripts/map_cord.py [validation test train]
Merchant and date are not labelled in CORD, so they are always null (and not scored).
Nested item modifiers (menu.sub) are ignored.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from receipt.normalize import parse_amount, parse_qty  # noqa: E402
from receipt.schema import Item, Receipt  # noqa: E402

RAW = ROOT / "data" / "raw" / "cord"
OUT = ROOT / "data" / "processed"


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def as_dict(x):
    return x if isinstance(x, dict) else {}


def map_gt_parse(gt: dict) -> Receipt:
    items = []
    for m in as_list(gt.get("menu")):
        m = as_dict(m)
        name = m.get("nm")
        if not isinstance(name, str) or not name.strip():
            continue
        items.append(Item(name=name.strip(), qty=parse_qty(m.get("cnt")), price=parse_amount(m.get("price"))))
    sub = as_dict(gt.get("sub_total"))
    total = as_dict(gt.get("total"))
    return Receipt(
        items=items,
        subtotal=parse_amount(sub.get("subtotal_price")),
        tax=parse_amount(sub.get("tax_price")),
        total=parse_amount(total.get("total_price")),
    )


def main(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        n = 0
        with open(RAW / f"{split}.jsonl", encoding="utf-8") as src, \
                open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as dst:
            for line in src:
                row = json.loads(line)
                rec = map_gt_parse(row["gt_parse"])
                dst.write(json.dumps({"id": row["id"], "image": row["image"], "receipt": rec.model_dump()},
                                     ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n} receipts -> {OUT / (split + '.jsonl')}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test"])
