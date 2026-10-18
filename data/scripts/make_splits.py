"""Pick the fixed-seed random sample of CORD train receipts used as the extra test set.

Usage: python data/scripts/make_splits.py [sample_size]   (default 150)
Writes data/processed/train_sample_ids.json (sorted train indices). The same seed always gives
the same sample. The ids are also what a future LoRA run must EXCLUDE from training.
"""
import json
import random
import sys
from pathlib import Path

SEED = 42
N_TRAIN = 800  # size of CORD v2 train split (download_cord.py checks this)
OUT = Path(__file__).resolve().parents[1] / "processed" / "train_sample_ids.json"


def main(size):
    indices = sorted(random.Random(SEED).sample(range(N_TRAIN), size))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "n_train": N_TRAIN, "indices": indices}), encoding="utf-8")
    print(f"{size} train indices (seed {SEED}) -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
