"""Download CORD v2 (naver-clova-ix/cord-v2) from Hugging Face into data/raw/cord.

Usage: python data/scripts/download_cord.py [validation test train]
Writes data/raw/cord/<split>/<split>_<index>.png and data/raw/cord/<split>.jsonl
(one line per receipt: id, image, raw gt_parse). Default splits: validation, test.
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

RAW = Path(__file__).resolve().parents[1] / "raw" / "cord"
PREFIX = {"validation": "val", "test": "test", "train": "train"}


def main(splits):
    for split in splits:
        out_dir = RAW / split
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("naver-clova-ix/cord-v2", split=split)
        indices = range(len(ds))
        if split == "train":  # only the fixed-seed sample from make_splits.py
            spec = json.loads((RAW.parents[1] / "processed" / "train_sample_ids.json").read_text())
            assert len(ds) == spec["n_train"], "train size changed; update make_splits.py"
            indices = spec["indices"]
        with open(RAW / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i in indices:
                row = ds[i]
                rid = f"{PREFIX[split]}_{i:04d}"
                row["image"].convert("RGB").save(out_dir / f"{rid}.png")
                gt_parse = json.loads(row["ground_truth"])["gt_parse"]
                f.write(json.dumps({"id": rid, "image": f"{split}/{rid}.png", "gt_parse": gt_parse},
                                   ensure_ascii=False) + "\n")
        print(f"{split}: {len(indices)} receipts")


if __name__ == "__main__":
    main(sys.argv[1:] or ["validation", "test"])
