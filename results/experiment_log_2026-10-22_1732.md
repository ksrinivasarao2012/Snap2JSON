# Snap2JSON experiment log — saved 2026-10-22 17:32 IST (first version 17:23)

A record of everything tried so far in this working session, the problems hit, how they were fixed, and every
result obtained. All numbers come from real runs in `results/tables/` and `results/runs/`. Anything not finished
when this file was saved is marked **pending**. The polished write-up of the main result is `results/REPORT.md`.

---

## 1. Setup on the lab machine

Machine: Linux, NVIDIA RTX 4000 Ada (21 GB total, ~20.3 GB free), Intel Core Ultra 9 285 (24 cores), 125 GB RAM.
Venv at `.venv` (created by the user).

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `unrecognized arguments: --limit3` | missing space | `--limit 3` |
| 2 | `Qwen3VLVideoProcessor requires the Torchvision library` | `torchvision` not installed | user installed it; added `torchvision` to the `models` extra in `pyproject.toml` |
| 3 | `python: command not found` when the run script was started from Claude's shell | that shell had no venv active | `run_all_qwen.sh` now uses `.venv/bin/python` when it exists |
| 4 | No progress lines printed while running | Python buffers output when piped into `tee` | script runs Python with `-u` (unbuffered) |
| 5 | Download bar stuck at `0/4` for 8B | bar counts whole files, not bytes | none needed; checked folder size was growing |
| 6 | `git push` failed (no GitHub login on lab machine) | no credentials | user made a fine-grained token (this repo only, short expiry), git credential cache for 2 h |

Warnings seen but harmless: `Can't initialize NVML` (driver/library version mismatch for `nvidia-smi`; CUDA itself works),
unauthenticated Hugging Face downloads.

Model sizes checked: 2B ~4.5 GB, 4B ~9 GB, 8B ~17.5 GB (bf16). All three fit; 8B ran with no out-of-memory errors.

## 2. Main experiment: OCR+LLM (plain text) vs VLM, Qwen3-VL 2B / 4B / 8B

Same model in each pair, same prompt (never tuned), greedy decoding, 4 splits (450 receipts), 24 runs.
Script: `bash run_all_qwen.sh [--model ...]`.

**Test split** (total acc over 95 labelled receipts; item F1 micro):

| Model | Method | Total acc | Subtotal acc | Tax acc | Item F1 | JSON valid |
|---|---|---|---|---|---|---|
| 2B | OCR+LLM | 0.579 | 0.754 | 0.610 | 0.402 | 1.00 |
| 2B | VLM | 0.832 | 0.908 | 0.780 | 0.807 | 1.00 |
| 4B | OCR+LLM | 0.516 | 0.600 | 0.512 | 0.547 | 1.00 |
| 4B | VLM | 0.853 | 0.923 | 0.780 | 0.825 | 1.00 |
| 8B | OCR+LLM | 0.768 | 0.800 | 0.610 | 0.652 | 1.00 |
| 8B | VLM | 0.926 | 0.954 | 0.854 | 0.866 | 1.00 |

Image beat OCR text at every size and on every split (all splits in `results/REPORT.md` section 7).
Degraded test: both drop; OCR+LLM drops more at 4B and 8B.

## 3. Fair speed: OCR re-timed on the lab machine

**Problem:** OCR+LLM `seconds` was only the model call. The OCR time stored earlier (10–14 s/receipt) came from a laptop CPU.

**Done:** `data/scripts/time_ocr.py` re-ran PaddleOCR on the lab CPU (CPU Paddle build; model loading not timed) into
`cache/ocr_timing/`, without touching `cache/ocr/`. All **450/450** OCR texts were identical to the cached ones.
`run_eval.py` now adds `mean_ocr_seconds` and `mean_end_to_end_seconds` for OCR+LLM, and has `--summarize-only`
to rebuild tables without loading a model.

OCR on the lab CPU: ~2.2–2.4 s/receipt mean (test 2.42 s, median 1.51 s).

**Test, mean seconds per receipt, end to end:**

| Model | OCR+LLM (OCR + model call) | VLM |
|---|---|---|
| 2B | 4.53 (2.42 + 2.11) | 2.52 |
| 4B | 6.46 (2.42 + 4.05) | 4.67 |
| 8B | 9.60 (2.42 + 7.18) | 7.91 |

The VLM is faster end to end at every size. Committed and pushed (`4f99220`).

## 4. Why does OCR+LLM do worse? Checks

**Check 1: is the right number even in the OCR text?** (digit-string search; loose upper bound)

| Split | Correct total in OCR text | Correct item price in OCR text |
|---|---|---|
| validation | 96/98 (98%) | 98% |
| test | 88/95 (93%) | 96% |
| train | 141/148 (95%) | 97% |
| test_degraded | 76/95 (80%) | 75% |

On clean receipts OCR usually has the number, yet OCR+LLM total accuracy is only 52–77% (test). So most of the loss is
not OCR misreading digits.

**Check 2: looked at the OCR text format.** `paddle.py` wrote **every OCR box on its own line**, ordered by 15 px height
bands, while boxes are ~41 px tall (median over all 450 receipts). So one printed row such as
`DONAT GULA  1X  11,000` was split into separate lines, sometimes out of order. Example `test_0003`: 4B OCR+LLM
called `DONAT GULA` the merchant and named the item `1.00xITEMs`; all numbers were right but the structure was wrong.

**Check 3: an OCR detection miss** (`val_0010`). The image clearly shows `2  Serbu 1` and `2  Choco Peanut Bread`.
PaddleOCR (mobile detector) produced **no box at all** for the first `2` (the second `2` was found with 0.997
confidence). No text layout can recover this; only the image method can. The larger "server" detector might catch it
but is slower and would change every OCR+LLM result, so it is kept as a limitation.

## 5. Layout experiment (option 2)

Goal: keep OCR, model, prompt and scoring identical; change only how the OCR text is arranged.

| Method | `--method` | What the model gets |
|---|---|---|
| Plain (all earlier runs) | `ocr_llm` | one OCR box per line |
| **A: rows** | `ocr_llm_rows` | boxes on the same printed line joined into one line, spaced by x position |
| **B: rows + positions** | `ocr_llm_boxes` | same rows, each piece as `text [x=..]`, separated by `|`, plus a one-line header explaining the format |

Same receipt (`val_0010`) in each format:

```
Plain:                 A (rows):                                B (rows + positions):
Serbu 1                    Serbu 1                    40.000    Serbu 1 [x=163] | 40.000 [x=600]
40.000                  2  Choco Peanut Bread         20.000    2 [x=105] | Choco Peanut Bread [x=162] | 20.000 [x=599]
2                                     TOTAL      60.000         TOTAL [x=398] | 60.000 [x=594]
Choco Peanut Bread     CASH                           60.000    CASH [x=94] | 60.000 [x=598]
20.000                 CHANGED                            0     CHANGED [x=92] | 0 [x=672]
60.000
TOTAL
...
```

### Steps done
1. `paddle.py`: new `ocr_boxes()` returns each box with `[x0, y0, x1, y1]` and score; `ocr_image()` now builds the same
   old text from those boxes (output unchanged).
2. `data/scripts/run_ocr_boxes.py`: saved boxes for all 450 receipts to `cache/ocr_boxes/` (`cache/ocr/` untouched).
   Re-joined in the old format they gave the cached text exactly on **450/450**.
3. `src/receipt/ocr/layout.py`: row grouping + layouts A and B (6 tests in `tests/test_layout.py`).
4. `OcrLlm` got a `layout` option; `run_eval.py` got methods `ocr_llm_rows` and `ocr_llm_boxes` (OCR time is added to
   end-to-end seconds the same way); `run_all_qwen.sh` takes `METHODS` and `SPLITS` env vars. 52 tests pass.

### Row grouping: how it works, and a bug fixed on the way
- **First attempt (rejected):** join boxes that overlap vertically by ≥ half the smaller height. On `test_0004`, tall
  junk boxes at the photo edge (`now`, `©lbe`) overlapped two printed rows and glued them together:
  `now AVOCADO COFFEE ICE BLACKCOFFE 82,000`.
- **Final rule:** compute the receipt's **median box height**; a box joins the nearest row whose centre (mean of its
  boxes' centres) is within **0.5 × median height** of the box's centre, and never joins a row where it would sit on
  top of a box already there (> 50% horizontal overlap). Then each row is sorted left to right. The median keeps
  one unusually tall box from stretching the ruler.
- Worked example `val_0010`: heights `24, 30, 37, 39, 40, 40, 41, 41, 43, 44, 66` → median 40 → limit 20 px.
  `Serbu 1` (centre 593.5) and `40.000` (607, 14 px lower because of tilt) join; `2` (623) is 22.7 px from that row
  so starts the next row with `Choco Peanut Bread` and `20.000`.
- Known weak spot: strongly tilted photos (e.g. the rotated degraded receipts) can still put a box in the wrong row.

### Results: Method A (rows) vs plain, Qwen3-VL-2B, all 4 splits (complete)

| Split | Method | Total acc | Subtotal acc | Tax acc | Item F1 | Item P | Item R | JSON valid | Model call s | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|---|
| validation | Plain | 0.684 | 0.776 | 0.511 | 0.471 | 0.452 | 0.491 | 0.99 | 2.04 | 4.31 |
| validation | **A: rows** | **0.796** | **0.955** | **0.617** | **0.505** | 0.481 | 0.532 | 1.00 | 2.09 | 4.36 |
| validation | VLM | 0.918 | 0.970 | 0.766 | 0.807 | 0.743 | 0.882 | 1.00 | — | 2.44 |
| test | Plain | 0.579 | 0.754 | 0.610 | 0.402 | 0.395 | 0.410 | 1.00 | 2.11 | 4.53 |
| test | **A: rows** | **0.695** | **0.923** | **0.683** | **0.449** | 0.448 | 0.450 | 1.00 | 2.15 | 4.56 |
| test | VLM | 0.832 | 0.908 | 0.780 | 0.807 | 0.776 | 0.841 | 1.00 | — | 2.52 |
| train | Plain | 0.561 | 0.710 | 0.600 | 0.424 | 0.413 | 0.437 | 0.993 | 2.13 | 4.33 |
| train | **A: rows** | **0.662** | **0.817** | **0.673** | **0.507** | 0.499 | 0.515 | 0.987 | 2.27 | 4.47 |
| train | VLM | 0.818 | 0.903 | 0.800 | 0.787 | 0.759 | 0.818 | 1.00 | — | 2.42 |
| test_degraded | Plain | 0.474 | 0.415 | 0.317 | 0.259 | 0.268 | 0.251 | 0.99 | 1.96 | 4.38 |
| test_degraded | **A: rows** | **0.526** | **0.554** | **0.341** | **0.323** | 0.341 | 0.307 | 0.99 | 1.96 | 4.38 |
| test_degraded | VLM | 0.768 | 0.800 | 0.585 | 0.629 | 0.596 | 0.665 | 1.00 | — | 2.61 |

(VLM rows are the earlier 2B VLM runs, repeated for comparison. VLM has no OCR step, so its model call is its end-to-end time.)

Gain of A over plain (percentage points):

| Split | Total | Subtotal | Tax | Item F1 |
|---|---|---|---|---|
| validation | +11.2 | +17.9 | +10.6 | +3.4 |
| test | +11.6 | +16.9 | +7.3 | +4.7 |
| train | +10.1 | +10.7 | +7.3 | +8.3 |
| test_degraded | +5.2 | +13.9 | +2.4 | +6.4 |

Reading (2B only):
- **Rows improve every field on every split.** The gain is consistent, so it is not a one-split fluke, although each split is small.
- **Header fields gain most** (total +10 to +12 pts on clean splits; subtotal +11 to +18). Subtotal with rows reaches the VLM on test (0.923 vs 0.908) and validation (0.955 vs 0.970).
- **Items gain only a little** (F1 +3 to +8 pts). The item gap to the VLM stays large (test 0.449 vs 0.807), so most of the item gap is **not** layout. To check by reading failures: OCR misses (like the missing `2` in `val_0010`), strict qty/price matching, item-name differences.
- **Degraded receipts gain least** on total and tax, as expected: there OCR itself loses text (correct total in OCR text only 80%), and tilted photos make row grouping harder.
- The VLM is still best everywhere and also faster end to end (test 2.52 s vs 4.56 s).
- Cost of A: input about 4% longer (test 28,443 vs 27,294 tokens); speed about the same.
- JSON-valid: A failed on 2 train receipts and 1 degraded receipt (plain: 1 and 1).

### Not done yet
- **Method B (rows + positions), 2B:** stopped by the user after 1 validation receipt; to be run another day
  (`METHODS="ocr_llm_boxes" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct`, resumes). B's OCR text is about
  236 tokens vs 107 for A (test, measured with the Qwen tokenizer), so B is expected to be slightly slower.
- **4B and 8B** for A and B: not started.
- A 3-receipt smoke test of 4B Method A on validation and the 1-receipt partial B run exist locally but are **not**
  committed (checks only, not results).

## 6. Files added or changed in this session

| File | What |
|---|---|
| `run_all_qwen.sh` / `.bat` | run methods × splits for one model; `METHODS`, `SPLITS`, extra args pass through |
| `data/scripts/time_ocr.py` | re-time OCR into `cache/ocr_timing/` |
| `data/scripts/run_ocr_boxes.py` | save OCR boxes into `cache/ocr_boxes/` |
| `src/receipt/ocr/paddle.py` | `ocr_boxes()`, `boxes_to_text()` |
| `src/receipt/ocr/layout.py` | row grouping, layouts A and B |
| `src/receipt/extractors/ocr_llm.py` | `layout` option |
| `eval/run_eval.py` | OCR/end-to-end seconds, `--summarize-only`, new methods |
| `tests/test_layout.py`, `tests/test_run_eval.py`, `tests/test_extractors.py` | new tests (52 total, all pass) |
| `results/REPORT.md` | main write-up (speed section updated) |
| `pyproject.toml` | `torchvision` added |

## 7. Next steps
1. Run 2B Method B; compare A vs B vs plain on all splits.
2. Run A (and B if it beats A) on 4B and 8B, to see whether the layout gain holds at larger sizes.
3. Read wrong item outputs for Method A to see why items improve so little.
4. Update `results/REPORT.md` with the layout results and the OCR-miss limitation; commit and push.
