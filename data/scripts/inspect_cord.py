"""Print which gt_parse keys appear in a few CORD v2 samples (run after installing `datasets`)."""
import json
from collections import Counter

from datasets import load_dataset


def walk(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield prefix + k
            yield from walk(v, prefix + k + ".")
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, prefix)


ds = load_dataset("naver-clova-ix/cord-v2", split="validation", streaming=True)
counts, n = Counter(), 0
for i, row in enumerate(ds):
    gt = json.loads(row["ground_truth"])["gt_parse"]
    if i < 2:
        print(json.dumps(gt, indent=1, ensure_ascii=False))
    counts.update(set(walk(gt)))
    n += 1
    if n >= 20:
        break
print(f"\nKey presence over {n} receipts:")
for k, c in sorted(counts.items()):
    print(f"{k:35s} {c}/{n}")
