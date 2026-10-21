"""Tests for cache.py."""
import pytest

from receipt.cache import DiskCache, make_key


def test_key_depends_on_all_parts():
    base = make_key("m", "p", b"img")
    assert base == make_key("m", "p", b"img")
    assert base != make_key("m2", "p", b"img")
    assert base != make_key("m", "p2", b"img")
    assert base != make_key("m", "p", b"img2")


def test_miss_then_hit_calls_once(tmp_path):
    cache = DiskCache("t", root=tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"text": "héllo", "tokens": 5}

    assert cache.get_or_call("k", fn) == ({"text": "héllo", "tokens": 5}, False)
    assert cache.get_or_call("k", fn) == ({"text": "héllo", "tokens": 5}, True)
    assert len(calls) == 1


def test_persists_across_instances(tmp_path):
    DiskCache("t", root=tmp_path).set("k", {"a": 1})
    assert DiskCache("t", root=tmp_path).get("k") == {"a": 1}


def test_failed_call_not_cached(tmp_path):
    cache = DiskCache("t", root=tmp_path)

    def boom():
        raise RuntimeError("rate limit")

    with pytest.raises(RuntimeError):
        cache.get_or_call("k", boom)
    assert cache.get("k") is None
