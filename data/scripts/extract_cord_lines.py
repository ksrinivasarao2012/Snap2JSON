"""Save CORD's word-level annotations (`valid_line`) for our receipts, to check what is actually printed.

Usage: python data/scripts/extract_cord_lines.py [validation test train]   (default: all three)
Writes data/raw/cord/<split>_lines.jsonl: one line per receipt {"id", "valid_line"}. Each valid_line entry has a
category (e.g. menu.nm, menu.cnt, menu.price), a group_id tying the words of one menu item together, and the
words with their text and position. Uses the same ids as download_cord.py (train: the fixed-seed sample).
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

RAW = Path(__file__).resolve().parents[1] / "raw" / "cord"
PREFIX = {"validation": "val", "test": "test", "train": "train"}


def main(splits):
    for split in splits:
        ds = load_dataset("naver-clova-ix/cord-v2", split=split)
        indices = range(len(ds))
        if split == "train":
            spec = json.loads((RAW.parents[1] / "processed" / "train_sample_ids.json").read_text())
            indices = spec["indices"]
        with open(RAW / f"{split}_lines.jsonl", "w", encoding="utf-8") as f:
            for i in indices:
                gt = json.loads(ds[i]["ground_truth"])
                f.write(json.dumps({"id": f"{PREFIX[split]}_{i:04d}", "valid_line": gt.get("valid_line", [])},
                                   ensure_ascii=False) + "\n")
        print(f"{split}: {len(indices)} receipts")


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test", "train"])
