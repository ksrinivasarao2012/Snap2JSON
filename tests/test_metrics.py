"""Tests for eval/metrics.py."""
import pytest

from eval.metrics import aggregate, prf, score_header, score_items, score_receipt


def item(name, qty=1.0, price=1000.0):
    return {"name": name, "qty": qty, "price": price}


def test_header_exact_and_unscored():
    gt = {"merchant": None, "date": None, "subtotal": None, "tax": 100.0, "total": 1100.0}
    pred = {"tax": 100.0, "total": 1000.0, "merchant": "X"}
    assert score_header(pred, gt) == {"tax": True, "total": False}


def test_header_missing_prediction_is_wrong():
    assert score_header({"total": None}, {"total": 500.0}) == {"total": False}
    assert score_header(None, {"total": 500.0}) == {"total": False}


def test_items_fuzzy_name_match():
    m, p, g = score_items([item("PIZZA TOST")], [item("PIZZA TOAST")])
    assert (m, p, g) == (1, 1, 1)


def test_items_case_insensitive():
    assert score_items([item("KOPI SUSU")], [item("Kopi Susu")])[0] == 1


def test_items_different_name_or_price_not_matched():
    assert score_items([item("ES TEH")], [item("NASI GORENG")])[0] == 0
    assert score_items([item("ES TEH", price=9000.0)], [item("ES TEH")])[0] == 0
    assert score_items([item("ES TEH", qty=2.0)], [item("ES TEH")])[0] == 0


def test_items_one_to_one():
    m, p, g = score_items([item("ES TEH"), item("ES TEH")], [item("ES TEH")])
    assert (m, p, g) == (1, 2, 1)


def test_prf_example():
    p, r, f1 = prf(3, 5, 4)
    assert (p, r) == (0.6, 0.75)
    assert f1 == pytest.approx(2 * 0.6 * 0.75 / 1.35)


def test_prf_zero():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_invalid_json_counts_as_wrong():
    gt = {"total": 1000.0, "items": [item("A")]}
    r = score_receipt(None, gt)
    assert r["json_valid"] is False and r["header"] == {"total": False} and r["items_matched"] == 0


def test_aggregate():
    gt = {"total": 1000.0, "items": [item("A")]}
    good = score_receipt({"total": 1000.0, "items": [item("A")]}, gt)
    bad = score_receipt(None, gt)
    agg = aggregate([good, bad])
    assert agg["json_valid_rate"] == 0.5
    assert agg["total_acc"] == 0.5 and agg["total_n"] == 2
    assert agg["tax_acc"] is None
    assert agg["items_precision"] == 1.0 and agg["items_recall"] == 0.5


def test_sub_items_ignored_not_counted_wrong():
    gt = [item("Iced Latte")]
    pred = [item("Iced Latte"), item("Extra Shot", price=5000.0), item("Cookie")]
    assert score_items(pred, gt) == (1, 3, 1)
    assert score_items(pred, gt, sub_items=["+ Extra Shot", "Oat Milk"]) == (1, 2, 1)


def test_sub_item_name_can_still_match_a_real_item():
    gt = [item("Extra Shot")]
    assert score_items([item("Extra Shot")], gt, sub_items=["Extra Shot"]) == (1, 1, 1)


def test_score_receipt_passes_sub_items():
    gt = {"total": 1000.0, "items": [item("A")]}
    r = score_receipt({"total": 1000.0, "items": [item("A"), item("Oat Milk")]}, gt, sub_items=["Oat Milk"])
    assert (r["items_matched"], r["items_pred"], r["items_gt"]) == (1, 1, 1)
