# Method A (OCR text grouped into rows): results report — Qwen3-VL-2B

Saved 2026-10-22 17:33 IST. All numbers come from real runs:
`results/runs/{ocr_llm, ocr_llm_rows, vlm}__Qwen_Qwen3-VL-2B-Instruct__<split>.jsonl` and the matching
`results/tables/*.json`. Model: Qwen3-VL-2B-Instruct only. 4B and 8B, and Method B, have not been run yet.

**Corrected 2026-10-23:** a ground-truth bug read amounts like `Rp 58000.00` as 5,800,000 (2 validation, 4 test,
8 train receipts). It was fixed and every number below was recomputed from the saved model outputs; no model was
re-run. Scores rose or stayed equal and the conclusions hold.
**Also 2026-10-23:** a predicted item that matches one of the receipt's sub-item lines (modifiers, which our ground
truth leaves out and the prompt never asks about) is no longer counted as wrong. This raised item precision for
every method; header fields did not change.
**Also 2026-10-23:** the item name threshold (80) was checked by hand on validation (`results/REPORT.md` section 7.1)
and kept. At a looser threshold of 60, 2B test item F1 is plain 0.484, rows 0.537, VLM 0.900 (at 80: 0.436, 0.495,
0.881), so A's small item gain holds.

## 1. Summary

- **Question:** OCR+LLM was far behind the image method (VLM). Is that partly because of how the OCR text was arranged?
- **Change tested:** only the arrangement of the OCR text. Before, every OCR box was on its own line ("plain"). In Method A, boxes on the same printed line are joined into one row. The OCR, model, prompt and scoring are all unchanged.
- **Result:** rows improve **every field on every split**.
  - Total accuracy: +6 to +14 points.
  - Subtotal: +12 to +19 points, reaching VLM level on clean receipts.
  - Item F1: only +4 to +9 points.
- **Main remaining item problem: quantities.**
  - With rows, quantity errors went **up** (28 → 44 items on test).
  - The 2B model often fills a missing quantity with digits taken from the price, e.g. `ICE BLACKCOFFE 82,000` → qty 82.
  - In 27 of those 44 cases the quantity is not visible in the OCR row at all.
- **The VLM is still best on every field and split except test subtotal** (rows 0.954 vs VLM 0.923), and faster
  end to end (test 2.52 s vs 4.56 s).

## 2. What changed (recap)

Same receipt `val_0010`, as the model sees it:

```
Plain (before)            Method A (rows)
Serbu 1                       Serbu 1                    40.000
40.000                     2  Choco Peanut Bread         20.000
2                                        TOTAL      60.000
Choco Peanut Bread        CASH                           60.000
20.000                    CHANGED                            0
60.000
TOTAL
CASH
60.000
CHANGED
0
```

**Row rule:** a box joins the nearest row whose centre is within half the receipt's median box height of the box's centre. It never joins a row where it would sit on top of another box. Inside each row, pieces are spaced by their x position. Full details are in `results/experiment_log_2026-10-22_1732.md`, section 5. Code: `src/receipt/ocr/layout.py`.

**Setup:**
- OCR: PaddleOCR PP-OCRv5 mobile, CPU. The boxes were re-extracted, and they reproduce the old text on 450/450 receipts, so plain and A start from identical OCR.
- Model: Qwen3-VL-2B-Instruct on an RTX 4000 Ada, greedy decoding, same prompt as every other run, never tuned.

## 3. Results

### 3.1 All splits

| Split | Method | Total acc | Subtotal acc | Tax acc | Item F1 | Item P | Item R | JSON valid | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|
| validation | Plain | 0.684 | 0.791 | 0.532 | 0.502 | 0.514 | 0.491 | 0.99 | 4.31 |
| validation | **A: rows** | **0.816** | **0.970** | **0.638** | **0.546** | 0.557 | 0.536 | 1.00 | 4.36 |
| validation | VLM | 0.939 | 0.985 | 0.787 | 0.882 | 0.878 | 0.886 | 1.00 | 2.44 |
| test | Plain | 0.600 | 0.769 | 0.634 | 0.436 | 0.443 | 0.430 | 1.00 | 4.53 |
| test | **A: rows** | **0.737** | **0.954** | **0.707** | **0.495** | 0.504 | 0.486 | 1.00 | 4.56 |
| test | VLM | 0.863 | 0.923 | 0.805 | 0.881 | 0.875 | 0.888 | 1.00 | 2.52 |
| train (150) | Plain | 0.595 | 0.731 | 0.618 | 0.448 | 0.448 | 0.448 | 0.993 | 4.33 |
| train (150) | **A: rows** | **0.696** | **0.849** | **0.709** | **0.535** | 0.541 | 0.528 | 0.987 | 4.47 |
| train (150) | VLM | 0.872 | 0.935 | 0.836 | 0.845 | 0.849 | 0.842 | 1.00 | 2.42 |
| test_degraded | Plain | 0.484 | 0.431 | 0.341 | 0.268 | 0.283 | 0.255 | 0.99 | 4.38 |
| test_degraded | **A: rows** | **0.547** | **0.569** | **0.366** | **0.338** | 0.366 | 0.315 | 0.99 | 4.38 |
| test_degraded | VLM | 0.789 | 0.815 | 0.610 | 0.686 | 0.672 | 0.701 | 1.00 | 2.61 |

Scored receipts per field (test / validation / train / degraded):
- total: 95 / 98 / 148 / 95
- subtotal: 65 / 67 / 93 / 65
- tax: 41 / 47 / 55 / 41

### 3.2 Gain of A over plain (percentage points)

| Split | Total | Subtotal | Tax | Item F1 |
|---|---|---|---|---|
| validation | +13.3 | +17.9 | +10.6 | +4.4 |
| test | +13.7 | +18.5 | +7.3 | +5.9 |
| train | +10.1 | +11.8 | +9.1 | +8.7 |
| test_degraded | +6.3 | +13.8 | +2.4 | +7.0 |

### 3.3 How much of the gap to the VLM does A close? (test)

| Field | Plain → VLM gap | Closed by A | Share closed |
|---|---|---|---|
| Total | 26.3 pts | 13.7 pts | ~52% |
| Subtotal | 15.4 pts | 18.5 pts | all (A is 3.1 pts above VLM) |
| Tax | 17.1 pts | 7.3 pts | ~43% |
| Item F1 | 44.5 pts | 5.9 pts | ~13% |

The header fields get roughly half or more of the gap back. Items get very little.

### 3.4 Receipt by receipt: what A fixed and what it broke

These count receipts where the field was right with one method and wrong with the other.

| Split | Field | Fixed by A | Broken by A | Both right | Both wrong |
|---|---|---|---|---|---|
| validation | total | 16 | 3 | 64 | 15 |
| validation | subtotal | 12 | 0 | 53 | 2 |
| test | total | 15 | 2 | 55 | 23 |
| test | subtotal | 12 | 0 | 50 | 3 |
| train | total | 22 | 7 | 81 | 38 |
| train | subtotal | 16 | 5 | 63 | 9 |
| test_degraded | total | 14 | 8 | 38 | 35 |
| test_degraded | subtotal | 10 | 1 | 27 | 27 |

A fixes far more than it breaks: 67 fixed vs 20 broken for total, over all splits. The gain is not just noise moving in one direction. Still, nearly a quarter of test totals (23/95) are wrong with both methods.

### 3.5 Degraded receipts by effect (total accuracy, correct / scored)

| Effect | Plain | A: rows | VLM |
|---|---|---|---|
| blur | 9/24 | 11/24 | 18/24 |
| rotate | 10/22 | 8/22 | 17/22 |
| darken | 18/25 | 21/25 | 24/25 |
| lowres | 9/24 | 12/24 | 16/24 |

- **Rotate is the one effect where rows do not help** (8 vs 10). That fits a known weak spot: on tilted text, boxes from neighbouring lines end up in the wrong row.
- **Blur and low-res gain only 2–3 receipts.** OCR loses the text itself there, which no layout can fix.
- **Only 22–25 receipts per effect,** so differences of 1–3 receipts are noise.

## 4. Why items improve so little

For every ground-truth item on test, I looked at the predicted item with the most similar name and sorted the outcome. This is a simple first-match diagnostic, so the "matched" counts differ from the official recall by at most a few items.
(Recomputed after the ground-truth fix; the tie rule may differ from the first version by 1–2 items per cell.)

| Outcome (test, 251 items) | Plain | A: rows | VLM |
|---|---|---|---|
| Matched (correct) | 110 (44%) | 122 (49%) | 223 (89%) |
| No predicted item with a similar name (≥ 80) | 65 (26%) | **44 (18%)** | 10 (4%) |
| Name OK, **quantity wrong** | 28 (11%) | **44 (18%)** | 1 (0.4%) |
| Name OK, price wrong | 35 (14%) | 28 (11%) | 16 (6%) |
| Name OK, quantity and price wrong | 13 (5%) | 13 (5%) | 1 (0.4%) |

Validation shows the same pattern (220 items):
- name misses: 51 → 38
- quantity-only errors: 26 → 39
- VLM quantity-only errors: 3

**What this shows:**
1. **Rows fix item names.** Items whose name could not be found dropped from 65 to 44. With one box per line, names were split from their rows or mixed up. Rows keep them whole.
2. **But many of those recovered items then fail on quantity,** so the matched count only rises from 110 to 122.
3. **The VLM almost never gets quantity wrong** (1 of 251). Quantity is the biggest single difference between the text and image methods once names are fixed.

### 4.1 What goes wrong with quantities (A, test, 44 cases)

Most common errors (ground-truth qty → predicted qty):
- 1 → 20 (4 cases)
- 1 → 2 (3)
- 1 → 85, 1 → 40, 1 → 20000, 1 → 22, 1 → 32 (2 each)
- 2 → 82 (1)

Real examples (the OCR row the model saw is on the right):

| Receipt | Ground truth | Predicted | OCR row |
|---|---|---|---|
| test_0004 | ICE BLACKCOFFE, qty 2, 82,000 | qty **82** | `now ICE BLACKCOFFE 82,000` |
| test_0015 | Lemon Tea (L), qty 1, 25,000 | qty **25** | `Lemon Tea (L) 25.000` |
| test_0015 | Caramel Small, qty 1, 38,000 | qty **38** | `Caramel Small 38.000` |
| test_0020 | cashew nuts chkn, qty 1, 64,500 | qty **64** | `cashew nuts chkn 64,500` |
| test_0017 | Plastik Amook, qty 1, 0 | qty **0** | `Plastik Amook 0` |
| test_0011 | BANABERRY FRESH CREAM CAK, qty 1 | qty **2** | `1 BANABERRY FRESH CREAM CAK 230,000` |

**The two causes:**
- **No quantity in the text.** In **27 of 44** cases, the ground-truth quantity does not appear in the item's OCR row outside the name. It is printed on the receipt (CORD labels a quantity only when one is printed: each of the 767 quantity labels (validation, test, train) matches a printed quantity word in CORD's own word annotations), so OCR missed it, like the missing `2` in `val_0010`, or it sits on another row.
  - With nothing to read, the 2B model **invents a quantity from the price digits**: `82,000` → 82, `25.000` → 25.
  - The VLM sees the whole receipt and doesn't do this.
- **Quantity visible, but still wrong** (17 of 44), e.g. `1 BANABERRY…` → qty 2. That is a plain model mistake. A bigger model may make fewer of these; untested.

Plain OCR text showed fewer quantity errors only because many of those items were already lost at the name step.

## 5. Speed and cost

| Test, per receipt | Plain | A: rows | VLM |
|---|---|---|---|
| OCR (lab CPU) | 2.42 s | 2.42 s | — |
| Model call | 2.11 s | 2.15 s | 2.52 s |
| End to end | 4.53 s | 4.56 s | 2.52 s |
| Input tokens | 273 | 284 | 1,349 |

- **A costs almost nothing extra:** about 4% more input tokens, and row grouping time is negligible.
- **A uses about 5× fewer input tokens than the VLM.** That would matter for paid APIs. Dollar cost is not computed yet (`cost.py`).

## 6. Conclusions (2B only)

1. **Text formatting matters a lot** for OCR+LLM on receipts. Keeping printed rows together gives +10 to +14 points on totals and +12 to +19 on subtotals, on every clean split, at no real cost. The original OCR+LLM numbers (one box per line) **understate** what the OCR approach can do. `results/REPORT.md` should say so.
2. **Layout explains part of the gap, not all.**
   - Headers get ~43% to all of the gap back.
   - Items get ~13%, because quantities are often not in the OCR text, and the small model then makes them up.
   - The image method still wins clearly.
3. **Degraded photos remain hard for OCR,** especially rotation (rows don't help there) and blur or low-res (the text is lost).

## 7. Limitations

- **One model size so far (2B).** The effect may be bigger or smaller at 4B and 8B.
- **Small splits,** one run each, no confidence intervals. The receipt-level fixed/broken counts (3.4) are the most direct evidence.
- **Row grouping is a simple rule** with fixed settings (0.5 × median height, 50% x-overlap). They were chosen by looking at a few receipts during development, not tuned on any split's scores. The example receipts were from test and validation.
- **Quantity scoring is strict:** exact match, and a missing quantity counts as wrong. CORD labels a quantity only when one is printed: each of the 767 quantity labels (validation, test, train) matches a printed quantity word in CORD's own word annotations, so quantity errors are real (an earlier version said otherwise; that was wrong).
- **Section 4 is a simple first-match diagnostic,** not the official metric.
- **Name threshold 80** was checked by hand on validation (see the note at the top); a looser 60 changes no conclusion.
- **The prompt is unchanged.** A rule like "if no quantity is printed, use 1" was considered and rejected: every scored quantity is printed, so the fix belongs in OCR and layout, not the prompt.

## 8. Next steps

1. Run **Method B** (rows + x positions) on 2B and compare with A.
2. Run **A on 4B and 8B,** to see whether the gain holds at larger sizes and whether quantity mistakes drop.
3. **Catch more lone quantity digits in OCR** (e.g. a larger text detector), compared on validation only.
4. Update `results/REPORT.md`:
   - The plain OCR+LLM numbers come from naive formatting.
   - Add the layout result and the OCR-miss limitation.
