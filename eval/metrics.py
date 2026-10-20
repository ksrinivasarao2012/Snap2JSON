"""Header exact-match, item precision/recall/F1, JSON-valid rate.

A prediction is a dict in our schema, or None when the model output was not valid JSON.
Header fields are scored only when the ground truth has a value (CORD has no merchant/date,
and sometimes no subtotal/tax/total). Items are matched one-to-one: same quantity and price
(when the ground truth has them) and fuzzy name similarity >= NAME_THRESHOLD.
"""
from rapidfuzz import fuzz

NAME_THRESHOLD = 80  # provisional; to be checked on the val set only
HEADER_FIELDS = ["merchant", "date", "subtotal", "tax", "total"]


def _norm_text(x):
    return " ".join(str(x).lower().split()) if x is not None else None


def _same(field, pred, gt):
    if field in ("merchant", "date"):
        return _norm_text(pred) == _norm_text(gt)
    return pred is not None and float(pred) == float(gt)


def score_header(pred, gt):
    """{field: True/False} for fields the ground truth has; others are left out (not scored)."""
    out = {}
    for f in HEADER_FIELDS:
        if gt.get(f) is None:
            continue
        out[f] = bool(pred) and _same(f, pred.get(f), gt[f])
    return out


def _item_match(p, g, threshold):
    if g.get("qty") is not None and p.get("qty") != g["qty"]:
        return None
    if g.get("price") is not None and p.get("price") != g["price"]:
        return None
    score = fuzz.ratio(_norm_text(p.get("name")) or "", _norm_text(g.get("name")) or "")
    return score if score >= threshold else None


def score_items(pred_items, gt_items, threshold=NAME_THRESHOLD):
    """One-to-one greedy matching, best name scores first. Returns (matched, n_pred, n_gt)."""
    pairs = []
    for i, p in enumerate(pred_items):
        for j, g in enumerate(gt_items):
            s = _item_match(p, g, threshold)
            if s is not None:
                pairs.append((s, i, j))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g, matched = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched += 1
    return matched, len(pred_items), len(gt_items)


def prf(matched, n_pred, n_gt):
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def score_receipt(pred, gt, threshold=NAME_THRESHOLD):
    """Per-receipt result. pred=None means invalid JSON (everything counts as wrong)."""
    matched, n_pred, n_gt = score_items((pred or {}).get("items") or [], gt.get("items") or [], threshold)
    return {
        "json_valid": pred is not None,
        "header": score_header(pred, gt),
        "items_matched": matched,
        "items_pred": n_pred,
        "items_gt": n_gt,
    }


def aggregate(results):
    """Combine per-receipt results: per-field accuracy over scored receipts, micro item P/R/F1, valid rate."""
    n = len(results)
    out = {"n": n, "json_valid_rate": sum(r["json_valid"] for r in results) / n if n else 0.0}
    for f in HEADER_FIELDS:
        scored = [r["header"][f] for r in results if f in r["header"]]
        out[f"{f}_acc"] = sum(scored) / len(scored) if scored else None
        out[f"{f}_n"] = len(scored)
    p, r, f1 = prf(sum(x["items_matched"] for x in results),
                   sum(x["items_pred"] for x in results),
                   sum(x["items_gt"] for x in results))
    out.update(items_precision=p, items_recall=r, items_f1=f1)
    return out
