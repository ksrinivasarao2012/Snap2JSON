# Snap2JSON: OCR+LLM vs VLM on CORD receipts (Qwen3-VL 2B / 4B / 8B)

All numbers below come from real runs saved in `results/tables/*.json` and `results/runs/*.jsonl`
(24 main runs: 3 model sizes x 2 methods x 4 splits, 450 receipts each; plus 4 runs of the OCR row-layout
experiment, Method A, on 2B). Nothing is estimated except where marked.

**Corrected 2026-10-23:** a ground-truth bug read amounts like `Rp 58000.00` as 5,800,000 (2 validation, 4 test,
8 train receipts). It was fixed and every table was re-scored from the saved model outputs; no model was re-run.
All scores rose or stayed equal, and no conclusion about image vs text changed. All numbers below are the corrected ones.
**Also 2026-10-23:** a predicted item that matches one of the receipt's sub-item lines (modifiers such as
`+ Extra Shot`, which our ground truth leaves out and the prompt never asks about) is no longer counted as wrong.
This raised item precision for every method, the VLM most; header fields did not change.
**Also 2026-10-23:** the item name-match threshold (80) was checked by hand on validation (section 7.1). It is kept;
a looser check at 60 raises every item F1 a little and changes no conclusion.

**Key update (section 6):** the OCR text in the main runs put every OCR box on its own line, which splits printed rows.
Keeping rows together (Method A) raises 2B OCR+LLM test total accuracy from 0.600 to 0.737 and subtotal from 0.769 to
0.954. So the main OCR+LLM numbers come from naive text formatting and **understate** the OCR approach. The image
method still wins on every field except subtotal, where 2B rows (0.954) edges out the 2B VLM (0.923) on test.

**Scope of this report:** the open-model (Qwen3-VL) pair only. The Gemini pair has not been run, and
cost per receipt has not been computed yet (`cost.py` is a stub).

## 1. Question and setup

Does a receipt-to-JSON system do better when the model **reads the image directly** (VLM) or when a separate
**OCR step reads the text first** and the model only sees that text (OCR+LLM)?

The model is held fixed inside each pair, so only the input changes:

| Method | Input to the model | Model |
|---|---|---|
| OCR+LLM | PaddleOCR text only (no image), one OCR box per line ("plain") | Qwen3-VL-{2B,4B,8B}-Instruct |
| OCR+LLM, Method A | the same OCR text with boxes grouped into printed rows (section 6) | Qwen3-VL-2B-Instruct so far |
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

OCR+LLM here is the plain format (one OCR box per line). With rows (Method A, 2B only so far) the 2B gap shrinks to
+12.6 pts on total and +38.6 pts on item F1 (section 6).

The same direction holds on validation, train and degraded (full tables in section 7). Even the smallest VLM (2B, image)
beats the largest OCR+LLM (8B, text) on item F1 (0.881 vs 0.700) and on total accuracy (0.863 vs 0.811).

## 3. Does the model size matter?

Test split, VLM method: total acc 0.863 (2B) -> 0.895 (4B) -> 0.968 (8B); item F1 0.881 -> 0.910 -> 0.954.
Going 2B -> 4B gives a small gain; 4B -> 8B gives the larger one.

For OCR+LLM the pattern is **not monotonic**: on test, 4B has lower total accuracy than 2B (0.558 vs 0.600) but higher
item F1 (0.607 vs 0.436). I did not investigate why (it may be how each model handles the unstructured text, or noise), so
this should not be read as a real effect of size.

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

- **Text arrangement explains a large part of the header-field gap** between OCR+LLM and the VLM.
- **It explains only a small part of the item gap.** That gap comes mostly from information missing from the OCR
  text (quantities, missed boxes), which the small model then guesses.
- **The image method is still best on every field and split, except subtotal on test (rows 0.954 vs VLM 0.923),
  and it is faster end to end.**

Full analysis: `results/method_A_rows_report_2026-10-22_1733.md`. Not yet run:
- Method B (rows + x positions)
- Method A on 4B and 8B

## 7. All results

Total acc / Item F1. "OCR+LLM" is the plain format; "OCR+LLM rows" is Method A. JSON-valid rate was 1.00 everywhere except:
- 4B VLM train: 0.993
- 2B OCR+LLM: validation 0.99, test_degraded 0.99, train 0.993
- 2B OCR+LLM rows: train 0.987, test_degraded 0.99

| Model | Method | validation | test | train | test_degraded |
|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.684 / 0.502 | 0.600 / 0.436 | 0.595 / 0.448 | 0.484 / 0.268 |
| 2B | OCR+LLM rows | 0.816 / 0.546 | 0.737 / 0.495 | 0.696 / 0.535 | 0.547 / 0.338 |
| 2B | VLM | 0.939 / 0.882 | 0.863 / 0.881 | 0.872 / 0.845 | 0.789 / 0.686 |
| 4B | OCR+LLM | 0.561 / 0.532 | 0.558 / 0.607 | 0.534 / 0.567 | 0.400 / 0.325 |
| 4B | VLM | 0.908 / 0.855 | 0.895 / 0.910 | 0.858 / 0.851 | 0.768 / 0.781 |
| 8B | OCR+LLM | 0.847 / 0.653 | 0.811 / 0.700 | 0.784 / 0.653 | 0.600 / 0.395 |
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
| 4B | 0.607 / 0.660 | - | 0.910 / 0.932 | 0.303 / 0.272 |
| 8B | 0.700 / 0.738 | - | 0.954 / 0.976 | 0.254 / 0.238 |

The looser threshold helps OCR+LLM slightly more (its names are more often trimmed), but the image method stays far
ahead on every split and model size, so the conclusions do not depend on the threshold.

## 8. Limitations

1. **Gemini pair not run yet**, so there is no result for a closed API model; findings apply to Qwen3-VL only.
2. **No cost numbers yet.** Token counts are logged, but conversion to dollars (`cost.py`) is not done. Open-model cost will only be an estimate.
3. **Speed is from one pass on one machine**, with OCR on CPU only (see section 5).
4. **One run per configuration**, greedy decoding, no confidence intervals. Subtotal and tax rest on 41 to 93 receipts.
5. **Item matching is strict** (exact qty and price, fuzzy name >= 80, one-to-one). One wrong digit counts as both a miss and a false positive. The 80 threshold was checked by hand on validation (7.1); the labels are one person's judgement, and at 80 some trimmed names still pass while some reordered names fail.
6. **Merchant and date are not evaluated**, because CORD has no labels for them.
7. **CORD only**: Indonesian receipts in one format family; results may not transfer to other countries or layouts.
8. **Plain OCR+LLM uses naive text formatting** (one box per line). Method A shows this costs 10+ points on totals at
   2B; the 4B and 8B OCR+LLM numbers have not been re-run with rows, so they likely understate OCR+LLM too.
9. **Quantity errors are real, not a labelling artefact.** CORD labels a quantity only when one is printed: each of the 767 quantity labels (validation, test, train)
   matches a printed quantity word in CORD's own word annotations.
   (An earlier version of this report said CORD labels quantity 1 when none is printed; that was wrong.)
   Unprinted quantities have no label and are not scored.
10. **OCR uses the mobile detector,** which misses some small isolated characters (e.g. a lone quantity digit).

## 9. Suggested next steps

1. Run Method B (rows + x positions) on 2B, and Methods A/B on 4B and 8B; check whether the 4B < 2B oddity on plain
   OCR+LLM disappears with rows.
2. Catch more lone quantity digits in OCR (e.g. a larger text detector), compared on validation only. Lower
   detection thresholds, a larger box margin and 1.5x upscaling were tried on validation and did not fix it.
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
METHODS="ocr_llm_boxes" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct
```

Outputs: `results/runs/<method>__<model>__<split>.jsonl` (per receipt) and `results/tables/<same>.json` (summary).
