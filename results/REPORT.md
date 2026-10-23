# Snap2JSON: OCR+LLM vs VLM on CORD receipts (Qwen3-VL 2B / 4B / 8B)

All numbers below come from real runs saved in `results/tables/*.json` and `results/runs/*.jsonl`
(24 main runs: 3 model sizes x 2 methods x 4 splits, 450 receipts each; plus 16 runs of the OCR layout
experiment: Method A on 2B, 4B and 8B, and Method B on 2B, all 4 splits each). Nothing is estimated except where marked.

**Corrected 2026-10-23:** a ground-truth bug read amounts like `Rp 58000.00` as 5,800,000 (2 validation, 4 test,
8 train receipts). It was fixed and every table was re-scored from the saved model outputs; no model was re-run.
All scores rose or stayed equal, and no conclusion about image vs text changed. All numbers below are the corrected ones.
**Also 2026-10-23:** a predicted item that matches one of the receipt's sub-item lines (modifiers such as
`+ Extra Shot`, which our ground truth leaves out and the prompt never asks about) is no longer counted as wrong.
This raised item precision for every method, the VLM most; header fields did not change.
**Also 2026-10-23:** the item name-match threshold (80) was checked by hand on validation (section 7.1). It is kept;
a looser check at 60 raises every item F1 a little and changes no conclusion. 100 errors were audited by hand
(section 7.2): only 3 were the scorer's fault and 2 were label problems.

**Key update (section 6):** the OCR text in the main runs put every OCR box on its own line, which splits printed rows.
Keeping rows together (Method A) raises OCR+LLM test total accuracy at every model size: 2B 0.600 -> 0.737,
4B 0.558 -> 0.716, 8B 0.811 -> 0.895. So the main (plain) OCR+LLM numbers come from naive text formatting and
**understate** the OCR approach on header fields. Rows do **not** raise item F1 on test at 4B or 8B (they do on
validation and train). Adding x positions (Method B, 2B) is worse than rows alone. The image method still wins on
every field at every size, except subtotal at 2B on test (rows 0.954 vs VLM 0.923).

**Scope of this report:** the open-model (Qwen3-VL) pair only. The Gemini pair has not been run, and
cost per receipt has not been computed yet (`cost.py` is a stub).

## 1. Question and setup

Does a receipt-to-JSON system do better when the model **reads the image directly** (VLM) or when a separate
**OCR step reads the text first** and the model only sees that text (OCR+LLM)?

The model is held fixed inside each pair, so only the input changes:

| Method | Input to the model | Model |
|---|---|---|
| OCR+LLM | PaddleOCR text only (no image), one OCR box per line ("plain") | Qwen3-VL-{2B,4B,8B}-Instruct |
| OCR+LLM, Method A | the same OCR text with boxes grouped into printed rows (section 6) | Qwen3-VL-{2B,4B,8B}-Instruct |
| OCR+LLM, Method B | rows, with each piece's x position in pixels (section 6.8) | Qwen3-VL-2B-Instruct only |
| VLM | the receipt image | the same Qwen3-VL model |

- Same prompt for every run (`src/receipt/extractors/base.py`). Greedy decoding (no sampling), so runs are repeatable.
- Prompt was **not tuned** at any point after the first runs, so validation was not used for tuning and test was never touched for tuning.
- OCR: PaddleOCR 3.x, PP-OCRv5 mobile detection + English recognition, run once on a laptop CPU and cached (`cache/ocr/`).
- Hardware for the models: lab GPU, NVIDIA RTX 4000 Ada (20 GB). 8B in bf16 fit without out-of-memory errors.
- Image is downscaled to at most 2048 px on its longest side before being given to the model.

### Data (CORD v2, Indonesian receipts, CC-BY-4.0)

| Split | Receipts | Notes |
|---|---|---|
| validation | 100 | |
| test | 100 | main result |
| train (random sample, seed 42) | 150 | extra test set |
| test_degraded | 100 | the test receipts with one effect each (blur / rotate / darken / low-res, by index % 4) |

CORD labels have **no merchant and no date**, so those are `null` and not scored. Scored header fields are
**total** (95 of 100 test receipts have it), **subtotal** (65) and **tax** (41). Receipts without a label for a
field are skipped for that field.

### Metrics

- **Header fields** (total, subtotal, tax): exact match after number normalization.
- **Items:** one-to-one matching; an item is correct if qty and price match exactly and the name is fuzzy-similar (rapidfuzz ratio >= 80, checked on validation in section 7.1). Reported as micro precision / recall / F1.
  A predicted item whose name matches one of the receipt's sub-item lines (CORD `menu.sub`, e.g. modifiers) is left
  out of precision: neither right nor wrong. Extracting sub-items is a separate, untested question.
- **JSON-valid rate:** output parses into our schema.
- **Seconds per receipt:** end to end on the lab machine; for OCR+LLM this is OCR time + model call (section 5).

## 2. Main result: image beats OCR text at every model size

Test split (100 receipts). "Total" = total exact-match accuracy over the 95 labelled receipts.

| Model | Method | Total acc | Subtotal acc (n=65) | Tax acc (n=41) | Item F1 | Item P | Item R | JSON valid |
|---|---|---|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.600 | 0.769 | 0.634 | 0.436 | 0.443 | 0.430 | 1.00 |
| 2B | **VLM** | **0.863** | 0.923 | 0.805 | 0.881 | 0.875 | 0.888 | 1.00 |
| 4B | OCR+LLM | 0.558 | 0.631 | 0.537 | 0.607 | 0.621 | 0.594 | 1.00 |
| 4B | **VLM** | **0.895** | 0.954 | 0.805 | 0.910 | 0.912 | 0.908 | 1.00 |
| 8B | OCR+LLM | 0.811 | 0.831 | 0.634 | 0.700 | 0.703 | 0.697 | 1.00 |
| 8B | **VLM** | **0.968** | 0.985 | 0.878 | 0.954 | 0.952 | 0.956 | 1.00 |

Gap (VLM minus OCR+LLM) on test:

| Model | Total acc | Item F1 |
|---|---|---|
| 2B | +26.3 pts | +44.5 pts |
| 4B | +33.7 pts | +30.3 pts |
| 8B | +15.8 pts | +25.4 pts |

OCR+LLM here is the plain format (one OCR box per line). With rows (Method A) the total gap shrinks to +12.6 pts (2B),
+17.9 (4B) and +7.4 (8B); the item F1 gap stays large: +38.6, +31.0 and +27.0 pts (section 6.7).

The same direction holds on validation, train and degraded (full tables in section 7). Even the smallest VLM (2B, image)
beats the largest OCR+LLM (8B, text) on item F1 (0.881 vs 0.700) and on total accuracy (0.863 vs 0.811).

## 3. Does the model size matter?

Test split, VLM method: total acc 0.863 (2B) -> 0.895 (4B) -> 0.968 (8B); item F1 0.881 -> 0.910 -> 0.954.
Going 2B -> 4B gives a small gain; 4B -> 8B gives the larger one.

For OCR+LLM the pattern is **not monotonic**: on test, 4B has lower total accuracy than 2B (0.558 vs 0.600) but higher
item F1 (0.607 vs 0.436). With rows (Method A) the same pattern remains: 4B total is below 2B on every split (test
0.716 vs 0.737, validation 0.653 vs 0.816), while 4B item F1 is higher (test 0.600 vs 0.495). So it is not only the
text layout. **The main cause is a number-format habit of 4B (section 6.9):** it often copies Indonesian amounts
like `60.000` into the JSON as a number, which JSON reads as 60, so the total is 1000x too small. So this is not a
real effect of size on reading the receipt.

Sample sizes are small (test: 95 totals, 65 subtotals, 41 taxes), so a difference of a few points on a single split,
especially for tax, is within noise. The image-vs-text gaps above are far larger than that.

## 4. Robustness (degraded test)

Same 100 test receipts with one effect each. Drop from test to degraded:

| Model | Method | Total acc | Item F1 |
|---|---|---|---|
| 2B | VLM | 0.863 -> 0.789 (-7.4) | 0.881 -> 0.686 |
| 4B | VLM | 0.895 -> 0.768 (-12.6) | 0.910 -> 0.781 |
| 8B | VLM | 0.968 -> 0.863 (-10.5) | 0.954 -> 0.857 |
| 2B | OCR+LLM | 0.600 -> 0.484 (-11.6) | 0.436 -> 0.268 |
| 4B | OCR+LLM | 0.558 -> 0.400 (-15.8) | 0.607 -> 0.325 |
| 8B | OCR+LLM | 0.811 -> 0.600 (-21.1) | 0.700 -> 0.395 |

Both methods get worse, and OCR+LLM falls further at every size. The 2B VLM is the exception for total accuracy:
it dropped least (-7.4), although its item F1 dropped the most of the three VLMs (0.881 -> 0.686). The degraded set is only 100 receipts with 22–25 scored per effect; a per-effect
breakdown for 2B (plain, rows, VLM) is in section 6.5, and differences of 1–3 receipts there are noise.

## 5. Speed and tokens

All timings on the lab machine: models on the GPU, PaddleOCR on the CPU (Intel Core Ultra 9 285, CPU Paddle build).
OCR was re-timed on this machine (`data/scripts/time_ocr.py`, model loading excluded); it reproduced the cached OCR text
exactly on all 450 receipts. Test split, mean seconds per receipt (median in brackets):

| Model | OCR step | OCR+LLM model call | **OCR+LLM end-to-end** | **VLM end-to-end** |
|---|---|---|---|---|
| 2B | 2.42 | 2.11 | **4.53** (3.37) | **2.52** (2.07) |
| 4B | 2.42 | 4.05 | **6.46** (5.14) | **4.67** (3.88) |
| 8B | 2.42 | 7.18 | **9.60** (7.93) | **7.91** (6.54) |

- **End to end, the VLM is faster at every size** (by about 1.7 to 2 s per receipt), because it skips the OCR step. The
  model call alone is only slightly slower with an image (+0.4 to +0.7 s).
- Model size matters more than input type: each size step roughly doubles the model-call time.
- Input tokens per receipt (test): about **1,349 with the image vs 273 with OCR text** (about 5x), identical across model sizes. Output is about 144 tokens per receipt for both.
- OCR runs on CPU and the model on GPU. That is a realistic setup, but OCR on a GPU would be faster; not tested.
- The earlier OCR times stored in `cache/ocr/` (10 to 14 s) were from a laptop CPU and are not used here.
- Timings come from one pass; I have not repeated runs, so differences of a few tenths of a second are not meaningful.

## 6. Why does OCR+LLM do so much worse? (tested)

### 6.1 The number is usually in the OCR text

First guess: PaddleOCR misreads numbers. Check: does the correct value appear as a digit string anywhere in the OCR
text? (A loose upper bound; the number could appear for another reason.)

| Split | Correct total appears in OCR text | Correct item price appears in OCR text |
|---|---|---|
| validation | 96 / 98 (98%) | 98% |
| test | 88 / 95 (93%) | 96% |
| train | 142 / 148 (96%) | 98% |
| test_degraded | 76 / 95 (80%) | 76% |

On clean receipts OCR usually captures the right number (93% on test). Yet plain OCR+LLM total accuracy is only
56% to 81% on test. So on clean receipts the main loss is **not** OCR misreading digits.

### 6.2 The text format split printed rows

The OCR text in all main runs (`receipt.ocr.paddle.ocr_image`) put **every OCR box on its own line**, ordered by 15 px
height bands. The boxes are about 41 px tall (median). So one printed row like `DONAT GULA  1X  11,000` became
three lines, sometimes out of order. On `test_0003`, 4B OCR+LLM took `DONAT GULA` to be the merchant and named the item
`1.00xITEMs`; every number was right but the structure was wrong.

### 6.3 Method A: group OCR boxes into rows

Only the arrangement of the OCR text changes. The OCR, model, prompt and scoring are identical. Boxes are re-extracted
with positions (`cache/ocr_boxes/`); joined in the old format they reproduce the old text on 450/450 receipts.

**Row rule** (`src/receipt/ocr/layout.py`): a box joins the nearest row whose centre is within 0.5 × the receipt's
median box height of the box's centre. It never joins a row where it would sit on top of another box (> 50% x-overlap).
Each row is sorted left to right, and pieces are spaced by their x position. The settings were chosen by looking at a
few receipts, not tuned on scores.

Example, `val_0010`:

```
Plain (main runs)        Method A (rows)
Serbu 1                      Serbu 1                    40.000
40.000                    2  Choco Peanut Bread         20.000
2                                       TOTAL      60.000
Choco Peanut Bread       CASH                           60.000
20.000                   CHANGED                            0
60.000
TOTAL ...
```

### 6.4 Method A results (Qwen3-VL-2B, all splits)

| Split | Method | Total acc | Subtotal acc | Tax acc | Item F1 | JSON valid | End-to-end s |
|---|---|---|---|---|---|---|---|
| validation | Plain | 0.684 | 0.791 | 0.532 | 0.502 | 0.99 | 4.31 |
| validation | **A: rows** | **0.816** | **0.970** | **0.638** | **0.546** | 1.00 | 4.36 |
| validation | VLM | 0.939 | 0.985 | 0.787 | 0.882 | 1.00 | 2.44 |
| test | Plain | 0.600 | 0.769 | 0.634 | 0.436 | 1.00 | 4.53 |
| test | **A: rows** | **0.737** | **0.954** | **0.707** | **0.495** | 1.00 | 4.56 |
| test | VLM | 0.863 | 0.923 | 0.805 | 0.881 | 1.00 | 2.52 |
| train | Plain | 0.595 | 0.731 | 0.618 | 0.448 | 0.993 | 4.33 |
| train | **A: rows** | **0.696** | **0.849** | **0.709** | **0.535** | 0.987 | 4.47 |
| train | VLM | 0.872 | 0.935 | 0.836 | 0.845 | 1.00 | 2.42 |
| test_degraded | Plain | 0.484 | 0.431 | 0.341 | 0.268 | 0.99 | 4.38 |
| test_degraded | **A: rows** | **0.547** | **0.569** | **0.366** | **0.338** | 0.99 | 4.38 |
| test_degraded | VLM | 0.789 | 0.815 | 0.610 | 0.686 | 1.00 | 2.61 |

- **Rows improve every field on every split.** Gains over plain:
  - total: +6 to +14 pts
  - subtotal: +12 to +19 pts
  - tax: +2 to +11 pts
  - item F1: +4 to +9 pts
- **More fixed than broken.** Receipt by receipt, over all splits, A fixed 67 totals that plain got wrong and broke 20.
- **Share of the test gap to the VLM that A closes:**
  - total: ~52%
  - subtotal: all of it (0.954 vs VLM 0.923)
  - tax: ~43%
  - item F1: only ~13%
- **Almost free.** Input tokens rise about 4% (test 284 vs 273 per receipt), and speed is unchanged.

### 6.5 What layout does not fix

**Quantities.** For each test ground-truth item, I took the predicted item with the closest name and sorted the result.
This is a first-match diagnostic, not the official metric. (Recomputed after the ground-truth fix; the tie rule may
differ from the first version by 1–2 items per cell.)

| Outcome (test, 251 items) | Plain | A: rows | VLM |
|---|---|---|---|
| Matched | 110 | 122 | 223 |
| No predicted item with a similar name | 65 | **44** | 10 |
| Name OK, quantity wrong (price OK) | 28 | **44** | 1 |
| Name OK, price wrong | 35 | 28 | 16 |
| Name OK, quantity and price wrong | 13 | 13 | 1 |

- **Rows recover item names, but many of those items then fail on quantity.**
- **In 27 of the 44 quantity errors, the quantity is not in the item's OCR row at all.** It is printed on the receipt
  (CORD labels a quantity only when one is printed: each of the 767 quantity labels (validation, test, train)
  matches a printed quantity word in CORD's own word annotations), so OCR missed it or it sits on another row. The 2B model then **invents a quantity from the price digits**:
  - `ICE BLACKCOFFE 82,000` → qty 82
  - `Lemon Tea (L) 25.000` → qty 25
- **The VLM almost never gets quantity wrong** (1 of 251).

**OCR detection misses.** On `val_0010` the image clearly shows `2  Serbu 1`, but the mobile OCR detector produced no
box for that `2`. No text format can recover a missed box; only the image method can.

**Degraded photos.** Total accuracy by effect, test_degraded, 2B (correct / scored):

| Effect | Plain | A: rows | VLM |
|---|---|---|---|
| blur | 9/24 | 11/24 | 18/24 |
| rotate | 10/22 | 8/22 | 17/22 |
| darken | 18/25 | 21/25 | 24/25 |
| lowres | 9/24 | 12/24 | 16/24 |

- **Rotation is the one effect where rows don't help.** Tilted lines are the known weak spot of the row rule.
- **Blur and low-res gain only 2–3 receipts with rows;** the text is lost in OCR itself.

### 6.6 Conclusion

- **Text arrangement explains a large part of the header-field gap** between OCR+LLM and the VLM, at every model
  size (about half of the test total gap; section 6.7).
- **It explains only a small part of the item gap,** and at 4B and 8B none of it on test. That gap comes mostly from
  information missing from the OCR text (quantities, missed boxes), which the model then guesses.
- **Giving x positions as numbers (Method B) does not help** the 2B model; plain spacing (Method A) is better.
- **The image method is still best on every field, split and size, except subtotal at 2B on test (rows 0.954 vs
  VLM 0.923), and it is faster end to end.**

Full 2B analysis: `results/method_A_rows_report_2026-10-22_1733.md`.

### 6.7 Method A on 4B and 8B (all splits)

Same rows, same prompt; only the model size changes. Test split:

| Model | Method | Total acc | Subtotal acc | Tax acc | Item F1 | End-to-end s |
|---|---|---|---|---|---|---|
| 4B | Plain | 0.558 | 0.631 | 0.537 | 0.607 | 6.46 |
| 4B | **A: rows** | **0.716** | **0.815** | **0.659** | 0.600 | 6.38 |
| 4B | VLM | 0.895 | 0.954 | 0.805 | 0.910 | 4.67 |
| 8B | Plain | 0.811 | 0.831 | 0.634 | 0.700 | 9.60 |
| 8B | **A: rows** | **0.895** | **0.938** | **0.805** | 0.684 | 9.53 |
| 8B | VLM | 0.968 | 0.985 | 0.878 | 0.954 | 7.91 |

Gain of rows over plain, range over the 4 splits (percentage points):

| Model | Total | Subtotal | Tax | Item F1 |
|---|---|---|---|---|
| 2B | +6 to +14 | +12 to +19 | +2 to +11 | +4 to +9 |
| 4B | +7 to +16 | +6 to +18 | +2 to +12 | -1 to +9 |
| 8B | +4 to +8 | +6 to +11 | 0 to +17 | -2 to +9 |

- **Header fields improve at every size and split.** Receipt by receipt over all splits, rows fixed / broke totals:
  2B 67 / 20, 4B 57 / 13, 8B 50 / 25. The larger the model, the less it needs the layout help (8B plain is
  already decent), but the net gain is positive at every size.
- **Share of the test gap to the VLM that rows close:** total 52% (2B), 47% (4B), 53% (8B); subtotal all of it (2B),
  57% (4B), 70% (8B); tax 43%, 45%, 70%.
- **Items are mixed.** On test, rows change item F1 by -0.7 (4B) and -1.6 pts (8B), which is within noise; on
  validation and train they add +5 to +9 pts at both sizes. So rows do not reliably close the item gap, which stays
  27 to 39 pts on test.
- **Degraded, rotate:** rows are worse than plain at every size (correct totals 2B 10 -> 8, 4B 6 -> 4, 8B 12 -> 9 of 22).
  Tilted printed lines break the row rule. For the other effects rows help at every size:

| Effect (test_degraded, totals correct / scored) | 4B plain | 4B rows | 4B VLM | 8B plain | 8B rows | 8B VLM |
|---|---|---|---|---|---|---|
| blur | 6/24 | 7/24 | 16/24 | 10/24 | 13/24 | 18/24 |
| rotate | 6/22 | 4/22 | 15/22 | 12/22 | 9/22 | 21/22 |
| darken | 16/25 | 22/25 | 25/25 | 21/25 | 24/25 | 25/25 |
| lowres | 10/24 | 12/24 | 17/24 | 14/24 | 16/24 | 18/24 |

- **Why items do not improve at 4B and 8B** (same first-match diagnostic as section 6.5, test, 251 items):

| Outcome | 4B plain | 4B rows | 8B plain | 8B rows | 8B VLM |
|---|---|---|---|---|---|
| Matched | 148 | 147 | 176 | 169 | 238 |
| No predicted item with a similar name | 46 | 42 | 38 | 34 | 8 |
| Name OK, quantity wrong (price OK) | 2 | 7 | 5 | 12 | 0 |
| Name OK, price wrong | 44 | **50** | 23 | **35** | 3 |
| Name OK, quantity and price wrong | 11 | 5 | 9 | 1 | 2 |

  Unlike 2B, the larger models rarely invent quantities. With rows they find a few more names, but they get **more
  prices wrong** (8B: 23 -> 35), so the matches do not go up. For 4B about half of these are the 1000x format
  problem (section 6.9). The 8B cases were read by hand (next point).
- **8B rows price errors, read by hand (test).** Counting every wrong price of a name-matched item (with or without
  a qty error): plain 32, rows 36. Rows fixed 21 of plain's errors and made 25 new ones (net +4). The 25 new ones:
  - **13: a row-grouping slip.** On a slightly tilted photo each price sits a little lower than its name, so it
    joins the next item's row and every item gets its neighbour's price. Almost all from 3 receipts (`test_0085`
    alone: 10; `test_0098`, `test_0045`). Plain text did not mislead 8B here; the wrong rows look certain.
  - 6: the 1000x format problem (section 6.9), e.g. `22.000` -> 22.
  - 5: OCR misread the price (`25,,00`, `20,00` on `test_0014`); 1 other.

  Example, `test_0085` as Method A wrote it (`Ayam Pop` should be 40.000, `Samabal SP` 11.000):
  ```
   2 Ayam Pop         7.000
   1 Samabal SP      40.000
                     11.000
   1 Sambal Gabus Asin Cb
  ```
  On validation and train rows **lower** 8B's wrong prices (validation 28 -> 20, train 46 -> 27;
  `eval/thousands_check.py`), so the test rise is a few tilted receipts, the same weak spot as rotated photos, not a
  general effect of rows. A tilt-aware row rule might fix it (TODO.md).
- **Speed and tokens:** unchanged (rows: 284 input tokens per test receipt at every size, plain 273).

### 6.8 Method B: rows plus x positions (2B only)

Each row is written as `text [x=..]` pieces joined with `|`, after one header line explaining the format, so the
model sees column positions as numbers instead of spaces. Everything else is as in Method A.

| Split | Plain | A: rows | B: rows + x | VLM |
|---|---|---|---|---|
| validation, total / item F1 | 0.684 / 0.502 | **0.816 / 0.546** | 0.745 / 0.498 | 0.939 / 0.882 |
| test, total / item F1 | 0.600 / 0.436 | **0.737 / 0.495** | 0.684 / 0.414 | 0.863 / 0.881 |
| train, total / item F1 | 0.595 / 0.448 | **0.696 / 0.535** | **0.696** / 0.441 | 0.872 / 0.845 |
| test_degraded, total / item F1 | 0.484 / 0.268 | **0.547 / 0.338** | 0.495 / 0.229 | 0.789 / 0.686 |

- **B is worse than A** on total on 3 splits (tied on train) and on item F1 on all 4.
- **B's item F1 is no better than plain** (0.414 vs 0.436 on test): the x numbers seem to distract the 2B model
  from the items. Totals still beat plain (fixed 62, broke 32 over all splits; A: 67 / 20).
- **Cost:** 414 input tokens per test receipt (A 284, +46%), about 0.1 s slower end to end.
- **Decision:** B was not run on 4B or 8B (the rule set before the run: only if B beats A on 2B).

### 6.9 Why 4B is below 2B on text totals: amounts written 1000x too small

Indonesian receipts use a dot as the thousands separator (`60.000` = 60,000 rupiah). The prompt asks for plain
numbers ("16500, not 16,500"), but 4B often writes `"total": 60.000`. In JSON that is the number 60, so the score
counts it as wrong. Sometimes it also drops the zeros (`535.5` for 535,500). Script: `eval/thousands_check.py`
(read only; it changes no score).

| Test, total | Total acc | Wrong totals | of which exactly true / 1000 |
|---|---|---|---|
| 2B rows | 0.737 | 25 | 2 |
| **4B plain** | 0.558 | 42 | **16** |
| **4B rows** | 0.716 | 27 | **19** |
| 4B VLM | 0.895 | 10 | 7 |
| 8B plain | 0.811 | 18 | 1 |
| 8B rows | 0.895 | 10 | 5 |

- **This explains most of the 4B < 2B gap.** On test, 19 of 4B rows' 27 wrong totals are exactly 1000x too small;
  on validation 23 of 34. 2B and 8B rarely do it; their wrong totals are mostly another printed amount (e.g. cash
  paid, section 7.2). The same holds on train and degraded.
- **It also hits item prices:** of 4B's wrong item prices on test, 28 of 55 (rows) and 29 of 55 (plain) are 1000x too
  small. It explains a few of 8B's extra price errors with rows too (6 of 36).
- **The 4B VLM does it as well** (7 of 10 wrong test totals), but less often, so it shows mainly in the text methods.
- **The scores are not changed.** A parsing rule cannot fix this safely: `60.000` (3 digits after the dot) is almost
  always thousands, but `535.5` or `26.25` could be either, and a size rule ("under 1000 means thousands") would break
  7 real ground-truth amounts under 1000 (e.g. a tax of 727). Choosing such a rule after seeing the results would also
  fit the scorer to the data. A fair fix is a separate experiment: ask for amounts as strings exactly as printed and
  convert them with the same `parse_amount` rule as the ground truth (TODO.md).
- So the 4B text numbers measure "reading the receipt **and** writing the number in the right format". The format
  part is a real failure of the system as built, so it stays in the score.

### 6.10 Would better OCR fix the items? A larger text detector (validation only)

Sections 6.5 and 7.2 found that many item errors come from text missing from the OCR, especially lone quantity
digits that the mobile detector does not box. Test: swap only the OCR text detector, keep everything else (recognizer,
Method A rows, prompt, models, scoring). Validation only; test was not used.

**OCR alone** (`data/scripts/compare_ocr.py`, boxes in `cache/ocr_variants/`, lab CPU):

| Validation, 100 receipts | Mobile detector (main runs) | Server detector | **Server + 2048 px cap ("T")** |
|---|---|---|---|
| Labelled qty > 1 not on the item's OCR row (or the next) | 13 / 39 | 6 / 39 | **6 / 39** |
| Item name and price both in the OCR text | 205 / 220 | 201 / 220 | 207 / 220 |
| Total in the OCR text | 94 / 98 | 91 / 98 | 93 / 98 |
| Receipts with no text found | 0 | 3 | 0 |
| OCR seconds per receipt (mean / median) | 2.3 / 1.6 | 15.1 / 8.6 | 9.2 / 8.6 |

- The server detector (`PP-OCRv5_server_det`) found **no text at all** on 3 very large photos (about 2300 x 4200 px).
  Capping its input at 2048 px on the longest side (only 15 of 100 images are larger; the VLM uses the same cap)
  fixed that. Earlier attempts (lower detection thresholds, a larger box margin, 1.5x upscaling) did not help.

**With the model** (Method A rows on the T text, `run_eval.py --ocr-variant T`):

| Validation | OCR | Total | Subtotal | Tax | Item F1 | End-to-end s |
|---|---|---|---|---|---|---|
| 2B | mobile | **0.816** | **0.970** | 0.638 | 0.546 | 4.36 |
| 2B | T | 0.786 | 0.940 | **0.681** | **0.606** | 11.37 |
| 2B | VLM (image) | 0.939 | 0.985 | 0.787 | 0.882 | 2.44 |
| 8B | mobile | 0.908 | 0.955 | 0.660 | 0.747 | 8.91 |
| 8B | T | **0.918** | 0.955 | **0.723** | **0.763** | 16.12 |
| 8B | VLM (image) | 0.980 | 1.000 | 0.787 | 0.924 | 7.65 |

- **Items improve, mostly at 2B:** item F1 +5.9 pts (2B) and +1.7 (8B). 2B quantity errors (first-match
  diagnostic, as in 6.5) drop from 39 to 31 and matched items rise from 120 to 133.
- **That closes only a small part of the item gap to the VLM:** about 18% at 2B and 9% at 8B.
- **Totals are mixed and small:** T fixed 2 / broke 5 totals at 2B, fixed 3 / broke 2 at 8B (noise level).
- **Much slower:** OCR rises from 2.3 to 9.2 s per receipt on CPU, so OCR+LLM takes 11 to 16 s, while the VLM takes
  2.4 to 7.7 s.
- **Conclusion:** weak OCR is a real but small part of the item gap. Even with a better detector, the text method
  stays far behind the image method on items and becomes 2 to 5 times slower. T is **not** used for the main results;
  it was not run on test or on the other splits.

## 7. All results

Total acc / Item F1. "OCR+LLM" is the plain format; "OCR+LLM rows" is Method A. JSON-valid rate was 1.00 everywhere except:
- 4B VLM train: 0.993
- 2B OCR+LLM: validation 0.99, test_degraded 0.99, train 0.993
- 2B OCR+LLM rows: train 0.987, test_degraded 0.99
- 2B OCR+LLM rows + x (B): validation 0.98, train 0.987
- 8B OCR+LLM rows: test_degraded 0.99

| Model | Method | validation | test | train | test_degraded |
|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.684 / 0.502 | 0.600 / 0.436 | 0.595 / 0.448 | 0.484 / 0.268 |
| 2B | OCR+LLM rows | 0.816 / 0.546 | 0.737 / 0.495 | 0.696 / 0.535 | 0.547 / 0.338 |
| 2B | OCR+LLM rows + x | 0.745 / 0.498 | 0.684 / 0.414 | 0.696 / 0.441 | 0.495 / 0.229 |
| 2B | VLM | 0.939 / 0.882 | 0.863 / 0.881 | 0.872 / 0.845 | 0.789 / 0.686 |
| 4B | OCR+LLM | 0.561 / 0.532 | 0.558 / 0.607 | 0.534 / 0.567 | 0.400 / 0.325 |
| 4B | OCR+LLM rows | 0.653 / 0.580 | 0.716 / 0.600 | 0.622 / 0.658 | 0.474 / 0.364 |
| 4B | VLM | 0.908 / 0.855 | 0.895 / 0.910 | 0.858 / 0.851 | 0.768 / 0.781 |
| 8B | OCR+LLM | 0.847 / 0.653 | 0.811 / 0.700 | 0.784 / 0.653 | 0.600 / 0.395 |
| 8B | OCR+LLM rows | 0.908 / 0.747 | 0.895 / 0.684 | 0.824 / 0.740 | 0.653 / 0.433 |
| 8B | VLM | 0.980 / 0.924 | 0.968 / 0.954 | 0.966 / 0.903 | 0.863 / 0.857 |

Validation scores are higher than test for most rows. Validation was not used for tuning, so this is not leakage; the
splits just differ (and each has only 100 receipts).

### 7.1 Name-match threshold check (validation only)

An item counts as correct only if qty and price match exactly and the name similarity (rapidfuzz ratio) is at least 80.
To check 80, every validation run was matched again with no name limit (same qty and price, one-to-one), and the
122 distinct name pairs scoring 50-99 were labelled by hand under two rules:

- **Strict** (`results/threshold_check_validation_strict.csv`): the name must be complete and read correctly. A
  trimmed name (`EL KEUN` for `EL KEUN HWANGTAE`) or a misread word (`Mie Rokung` for `Mie Trsi Kgkung`) counts as
  a different item. 91 same, 31 different.
- **Lenient** (`..._lenient.csv`): the pair is the same receipt line and the name clearly refers to it. Only a
  different item (e.g. prices swapped between two lines) counts as different. 116 same, 6 different.

Errors at each threshold (same-item pairs rejected + different-item pairs accepted):

| Threshold | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 |
|---|---|---|---|---|---|---|---|---|
| Strict | 28 | 28 | 20 | 18 | 17 | **16** | 26 | 42 |
| Lenient | 5 | **5** | 11 | 17 | 20 | 25 | 45 | 67 |

- **80 is kept as the main threshold:** it is the best under the strict rule (8 rejected, 8 accepted).
- **60 is the lenient check** (tied with 55; the stricter of the two). The most similar different-item pair, two
  swapped lines, scores 58.8, so 60 sits close to the edge.
- Many names printed over two lines are returned as two items (e.g. `Gojek Chicken` / `Chilli Sauce H`); a
  half-name with the right price passes at 60 but not at 80, and the other half still counts as a wrong item.

Item F1 on test at both thresholds (all runs: `results/tables/name_threshold_sensitivity.json`; header fields do
not depend on this threshold):

| Model | OCR+LLM @80 / @60 | OCR+LLM rows @80 / @60 | VLM @80 / @60 | VLM - OCR+LLM gap @80 / @60 |
|---|---|---|---|---|
| 2B | 0.436 / 0.484 | 0.495 / 0.537 | 0.881 / 0.900 | 0.445 / 0.416 |
| 4B | 0.607 / 0.660 | 0.600 / 0.650 | 0.910 / 0.932 | 0.303 / 0.272 |
| 8B | 0.700 / 0.738 | 0.684 / 0.738 | 0.954 / 0.976 | 0.254 / 0.238 |

(2B rows + x, Method B: 0.414 / 0.438.)

The looser threshold helps OCR+LLM slightly more (its names are more often trimmed), but the image method stays far
ahead on every split and model size, so the conclusions do not depend on the threshold.

### 7.2 Failure audit: whose fault is each error? (validation only)

100 errors were read by hand: 20 per run from 2B OCR+LLM, 2B OCR+LLM rows, 2B VLM, 8B OCR+LLM and 8B VLM, sampled with a
fixed seed (up to 7 wrong header fields and 13 item errors each). A wrong qty or price on a real item is one error
here, although the scorer counts it as one miss and one extra. Each error was checked against CORD's printed-word
annotations, the model output and, for OCR methods, the OCR text that method saw. Sheet:
`results/audit_validation.csv` (cause and a one-line note per error); script: `eval/audit_failures.py`.

Causes:
- **model:** the information was in the model's input, but the model got it wrong.
- **ocr:** the OCR text was wrong (misread), lacked it (missing), or put it in a misleading order (e.g. a price
  before its name in plain text, or a row-grouping slip), so the model could not get it from its input.
- **scorer:** the answer was right but was counted wrong.
- **gt:** the CORD label is wrong or debatable.

| Run (20 errors each) | model | ocr (misread / missing / order) | scorer | gt |
|---|---|---|---|---|
| 2B OCR+LLM (plain) | 8 | 12 (3 / 1 / 8) | 0 | 0 |
| 2B OCR+LLM rows (A) | 13 | 7 (4 / 1 / 2) | 0 | 0 |
| 8B OCR+LLM (plain) | 14 | 6 (4 / 2 / 0) | 0 | 0 |
| 2B VLM | 18 | - | 1 | 1 |
| 8B VLM | 17 | - | 2 | 1 |

- **The scores are mostly honest:** only 3 of 100 errors were the scorer's (words reordered, or printed add-ons joined
  to the name) and 2 were label problems. In val_0037, "Gyro Platter Regular 50,000" is printed but missing from
  CORD's item labels. In val_0021, CORD counts a discount line as an item, which it does not do on other receipts.
  Neither was changed in the ground truth.
- **For 2B on plain OCR text, most errors (12 of 20) come from the input,** and 8 of those are order: the price
  comes before its name, so the model pairs it with the wrong item. Rows (Method A) cut the OCR share to 7 of 20.
  This supports section 6: layout loss, not the model, drives much of the plain OCR+LLM gap at 2B.
- **8B on plain text has no order errors in the sample;** its OCR errors are misreads and missing quantities.
- **Repeated model mistakes, all methods** (the counts below are from the notes):
  - Tax: the service charge is given as tax (9 errors).
  - Total: the cash paid, the change or a voucher amount is given as total (7).
  - Names printed over two lines are cut short or split into two items (13).
  - Discount lines are listed as items (3), or mixed up with a real item's name or price (2).
- One run, 20 errors per run, labelled by one person: the shares are rough. The sample is from validation only.

## 8. Limitations

1. **Gemini pair not run yet**, so there is no result for a closed API model; findings apply to Qwen3-VL only.
2. **No cost numbers yet.** Token counts are logged, but conversion to dollars (`cost.py`) is not done. Open-model cost will only be an estimate.
3. **Speed is from one pass on one machine**, with OCR on CPU only (see section 5).
4. **One run per configuration**, greedy decoding, no confidence intervals. Subtotal and tax rest on 41 to 93 receipts.
5. **Item matching is strict** (exact qty and price, fuzzy name >= 80, one-to-one). One wrong digit counts as both a miss and a false positive. The 80 threshold was checked by hand on validation (7.1); the labels are one person's judgement, and at 80 some trimmed names still pass while some reordered names fail.
6. **Merchant and date are not evaluated**, because CORD has no labels for them.
7. **CORD only**: Indonesian receipts in one format family; results may not transfer to other countries or layouts.
8. **Plain OCR+LLM uses naive text formatting** (one box per line). Method A shows this costs 8 to 16 points on test
   totals across 2B, 4B and 8B. The row rule was set by eye, not tuned, and fails on rotated photos; a better layout
   might close more of the gap. On tilted photos the rule can shift every price by one row (section 6.7).
   Method B (x positions) was tried on 2B only.
9. **Quantity errors are real, not a labelling artefact.** CORD labels a quantity only when one is printed: each of the 767 quantity labels (validation, test, train)
   matches a printed quantity word in CORD's own word annotations.
   (An earlier version of this report said CORD labels quantity 1 when none is printed; that was wrong.)
   Unprinted quantities have no label and are not scored.
10. **OCR uses the mobile detector,** which misses some small isolated characters (e.g. a lone quantity digit).
    A larger detector halves those misses on validation but adds only +2 to +6 pts item F1 and about 7 s per receipt
    (section 6.10); it was tested on validation only.

## 9. Suggested next steps

1. Optional output-format experiment (section 6.9): amounts as printed strings, converted by `parse_amount`;
   validation first, then test once, all methods; reported next to the main result.
2. Optional: tilt-aware row grouping (section 6.7). The larger OCR detector was tested (6.10) and is not adopted.
3. Run the Gemini pair on a few receipts per day (cache makes reruns free) and add it to the same tables.
4. Implement `cost.py` and add cost per receipt (API price from tokens; GPU-seconds estimate for Qwen, labelled as an estimate).

## 10. Reproduce

```bash
# data, once: python data/scripts/download_cord.py (and train), degrade.py, run_ocr.py, time_ocr.py
bash run_all_qwen.sh                                    # 4B (default)
bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct
bash run_all_qwen.sh --model Qwen/Qwen3-VL-8B-Instruct
# layout experiment: boxes once (CPU), then Method A / B
python data/scripts/run_ocr_boxes.py
METHODS="ocr_llm_rows" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct
METHODS="ocr_llm_rows" bash run_all_qwen.sh                                   # 4B
METHODS="ocr_llm_rows" bash run_all_qwen.sh --model Qwen/Qwen3-VL-8B-Instruct
METHODS="ocr_llm_boxes" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct
python eval/threshold_check.py rescore 80 60                                  # name-threshold sensitivity table
# OCR side experiment (validation only): detector options, then Method A on option T
python data/scripts/compare_ocr.py run S T && python data/scripts/compare_ocr.py score
.venv/bin/python eval/run_eval.py --backend qwen --method ocr_llm_rows --ocr-variant T --split validation --model Qwen/Qwen3-VL-2B-Instruct
```

Outputs: `results/runs/<method>__<model>__<split>.jsonl` (per receipt) and `results/tables/<same>.json` (summary).
