"""Create the degraded CORD test set: one degraded copy per test receipt (blur / rotate / darken / low-res).

Usage: python data/scripts/degrade.py
Receipt i gets degradation i % 4, so each type covers 25 of the 100 receipts. Rotation angles use a
fixed seed. Reads data/processed/test.jsonl, writes images to data/raw/cord/test_degraded/ and
ground truth (same labels as the clean test set, plus the degradation used) to
data/processed/test_degraded.jsonl.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "cord"
SEED = 42


def blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def rotate(img, rng):
    angle = rng.choice([-1, 1]) * rng.uniform(5, 15)
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def darken(img, rng):
    return ImageEnhance.Brightness(img).enhance(0.4)


def low_res(img, rng):
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


DEGRADATIONS = [("blur", blur), ("rotate", rotate), ("darken", darken), ("lowres", low_res)]


def main():
    rng = random.Random(SEED)
    out_dir = RAW / "test_degraded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    with open(ROOT / "data" / "processed" / "test.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            name, fn = DEGRADATIONS[i % len(DEGRADATIONS)]
            img = Image.open(RAW / row["image"]).convert("RGB")
            rid = f"{row['id']}_{name}"
            fn(img, rng).save(out_dir / f"{rid}.png")
            out_rows.append({"id": rid, "image": f"test_degraded/{rid}.png", "degradation": name,
                             "receipt": row["receipt"], "sub_items": row.get("sub_items", [])})
    with open(ROOT / "data" / "processed" / "test_degraded.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test_degraded: {len(out_rows)} receipts")


if __name__ == "__main__":
    main()
