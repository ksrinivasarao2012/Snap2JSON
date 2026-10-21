"""Disk cache for API responses so a cached receipt is never re-called.

Each response is one JSON file in cache/<namespace>/<sha256>.json. The key is a hash of the
model name, the prompt and the input (image bytes or OCR text). Failed calls are never cached.
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Union

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def make_key(model: str, prompt: str, payload: Union[str, bytes]) -> str:
    h = hashlib.sha256()
    for part in (model, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    return h.hexdigest()


class DiskCache:
    def __init__(self, namespace: str, root: Path = CACHE_DIR):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str):
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(key))

    def get_or_call(self, key: str, fn: Callable[[], dict]):
        """Return (value, from_cache). fn is called only on a miss; if it raises, nothing is saved."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        value = fn()
        self.set(key, value)
        return value, False
