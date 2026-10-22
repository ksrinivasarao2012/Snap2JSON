# Snap2JSON: OCR+LLM vs VLM on CORD receipts (Qwen3-VL 2B / 4B / 8B)

All numbers below come from real runs saved in `results/tables/*.json` and `results/runs/*.jsonl`
(24 runs: 3 model sizes x 2 methods x 4 splits, 450 receipts each). Nothing is estimated except where marked.

**Scope of this report:** the open-model (Qwen3-VL) pair only. The Gemini pair has not been run, and
cost per receipt has not been computed yet (`cost.py` is a stub).

## 1. Question and setup

Does a receipt-to-JSON system do better when the model **reads the image directly** (VLM) or when a separate
**OCR step reads the text first** and the model only sees that text (OCR+LLM)?

The model is held fixed inside each pair, so only the input changes:

| Method | Input to the model | Model |
|---|---|---|
| OCR+LLM | PaddleOCR text only (no image) | Qwen3-VL-{2B,4B,8B}-Instruct |
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
- **Items:** one-to-one matching; an item is correct if qty and price match exactly and the name is fuzzy-similar (rapidfuzz ratio >= 80, a provisional threshold). Reported as micro precision / recall / F1.
- **JSON-valid rate:** output parses into our schema.
- **Seconds per receipt:** end to end on the lab machine; for OCR+LLM this is OCR time + model call (section 5).

## 2. Main result: image beats OCR text at every model size

Test split (100 receipts). "Total" = total exact-match accuracy over the 95 labelled receipts.

| Model | Method | Total acc | Subtotal acc (n=65) | Tax acc (n=41) | Item F1 | Item P | Item R | JSON valid |
|---|---|---|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.579 | 0.754 | 0.610 | 0.402 | 0.395 | 0.410 | 1.00 |
| 2B | **VLM** | **0.832** | 0.908 | 0.780 | 0.807 | 0.776 | 0.841 | 1.00 |
| 4B | OCR+LLM | 0.516 | 0.600 | 0.512 | 0.547 | 0.543 | 0.550 | 1.00 |
| 4B | **VLM** | **0.853** | 0.923 | 0.780 | 0.825 | 0.796 | 0.857 | 1.00 |
| 8B | OCR+LLM | 0.768 | 0.800 | 0.610 | 0.652 | 0.636 | 0.669 | 1.00 |
| 8B | **VLM** | **0.926** | 0.954 | 0.854 | 0.866 | 0.832 | 0.904 | 1.00 |

Gap (VLM minus OCR+LLM) on test:

| Model | Total acc | Item F1 |
|---|---|---|
| 2B | +25.3 pts | +40.5 pts |
| 4B | +33.7 pts | +27.8 pts |
| 8B | +15.8 pts | +21.4 pts |

The same direction holds on validation, train and degraded (full tables in section 7). Even the smallest VLM (2B, image)
beats the largest OCR+LLM (8B, text) on item F1 (0.807 vs 0.652) and is close on total accuracy (0.832 vs 0.768).

## 3. Does the model size matter?

Test split, VLM method: total acc 0.832 (2B) -> 0.853 (4B) -> 0.926 (8B); item F1 0.807 -> 0.825 -> 0.866.
Going 2B -> 4B gives a small gain; 4B -> 8B gives the larger one.

For OCR+LLM the pattern is **not monotonic**: on test, 4B has lower total accuracy than 2B (0.516 vs 0.579) but higher
item F1 (0.547 vs 0.402). I did not investigate why (it may be how each model handles the unstructured text, or noise), so
this should not be read as a real effect of size.

Sample sizes are small (test: 95 totals, 65 subtotals, 41 taxes), so a difference of a few points on a single split,
especially for tax, is within noise. The image-vs-text gaps above are far larger than that.

## 4. Robustness (degraded test)

Same 100 test receipts with one effect each. Drop from test to degraded:

| Model | Method | Total acc | Item F1 |
|---|---|---|---|
| 2B | VLM | 0.832 -> 0.768 (-6.4) | 0.807 -> 0.629 |
| 4B | VLM | 0.853 -> 0.737 (-11.6) | 0.825 -> 0.711 |
| 8B | VLM | 0.926 -> 0.832 (-9.4) | 0.866 -> 0.781 |
| 2B | OCR+LLM | 0.579 -> 0.474 (-10.5) | 0.402 -> 0.259 |
| 4B | OCR+LLM | 0.516 -> 0.368 (-14.8) | 0.547 -> 0.290 |
| 8B | OCR+LLM | 0.768 -> 0.568 (-20.0) | 0.652 -> 0.377 |

Both methods get worse, and OCR+LLM falls further at 4B and 8B. The 2B VLM is the exception for total accuracy:
it dropped least (-6.4) although its item F1 dropped by about the same as the others. The degraded set is only 100 receipts with 25 per effect, so I did not
break results down per effect.

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

## 6. Why does OCR+LLM do so much worse? (checked, with one correction)

My first guess was that PaddleOCR misreads numbers. I checked this by testing whether the correct total appears as a digit string
anywhere in the OCR text (a loose upper bound: the number could appear for another reason):

| Split | Correct total appears in OCR text | Correct item price appears in OCR text |
|---|---|---|
| validation | 96 / 98 (98%) | 98% |
| test | 88 / 95 (93%) | 96% |
| train | 141 / 148 (95%) | 97% |
| test_degraded | 76 / 95 (80%) | 75% |

On clean receipts OCR usually *does* capture the right number (93% on test), yet OCR+LLM total accuracy is only 52% to 77%.
So on clean receipts the main loss is **not OCR misreading digits**. The likely cause is that the OCR text loses layout
(columns, which number belongs to which row, which line is the total vs. cash paid vs. change), so the model picks the wrong number.
This is a hypothesis supported by the numbers above, not something I verified by reading failure cases one by one; the next step is to read
the wrong outputs in `results/runs/`. On degraded receipts OCR coverage does drop (93% -> 80%), so there OCR errors do contribute.

## 7. All results

Total acc / Item F1 (JSON-valid rate was 1.00 everywhere except: 4B VLM train 0.993, 2B OCR+LLM validation 0.99, test_degraded 0.99, train 0.993).

| Model | Method | validation | test | train | test_degraded |
|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.684 / 0.471 | 0.579 / 0.402 | 0.561 / 0.424 | 0.474 / 0.259 |
| 2B | VLM | 0.918 / 0.807 | 0.832 / 0.807 | 0.818 / 0.787 | 0.768 / 0.629 |
| 4B | OCR+LLM | 0.541 / 0.493 | 0.516 / 0.547 | 0.493 / 0.534 | 0.368 / 0.290 |
| 4B | VLM | 0.888 / 0.783 | 0.853 / 0.825 | 0.804 / 0.796 | 0.737 / 0.711 |
| 8B | OCR+LLM | 0.827 / 0.607 | 0.768 / 0.652 | 0.730 / 0.625 | 0.568 / 0.377 |
| 8B | VLM | 0.959 / 0.846 | 0.926 / 0.866 | 0.912 / 0.845 | 0.832 / 0.781 |

Validation scores are higher than test for most rows. Validation was not used for tuning, so this is not leakage; the
splits just differ (and each has only 100 receipts).

## 8. Limitations

1. **Gemini pair not run yet**, so there is no result for a closed API model; findings apply to Qwen3-VL only.
2. **No cost numbers yet.** Token counts are logged, but conversion to dollars (`cost.py`) is not done. Open-model cost will only be an estimate.
3. **Speed is from one pass on one machine**, with OCR on CPU only (see section 5).
4. **One run per configuration**, greedy decoding, no confidence intervals. Subtotal and tax rest on 41 to 93 receipts.
5. **Item matching is strict** (exact qty and price, fuzzy name >= 80 provisional, one-to-one). One wrong digit counts as both a miss and a false positive. The 80 threshold was not tuned.
6. **Merchant and date are not evaluated**, because CORD has no labels for them.
7. **CORD only**: Indonesian receipts in one format family; results may not transfer to other countries or layouts.
8. **No failure-case reading yet**: section 6 explanation is a hypothesis.

## 9. Suggested next steps

1. Read 20 wrong OCR+LLM outputs vs. ground truth to confirm the layout-loss explanation (and look at the 4B vs 2B oddity).
2. Run the Gemini pair on a few receipts per day (cache makes reruns free) and add it to the same tables.
3. Implement `cost.py` and add cost per receipt (API price from tokens; GPU-seconds estimate for Qwen, labelled as an estimate).
4. Optional: pass OCR text with line positions/columns preserved to see whether layout is what the text-only path is missing.

## 10. Reproduce

```bash
# data, once: python data/scripts/download_cord.py (and train), degrade.py, run_ocr.py, time_ocr.py
bash run_all_qwen.sh                                    # 4B (default)
bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct
bash run_all_qwen.sh --model Qwen/Qwen3-VL-8B-Instruct
```

Outputs: `results/runs/<method>__<model>__<split>.jsonl` (per receipt) and `results/tables/<same>.json` (summary).
