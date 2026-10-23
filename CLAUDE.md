# Snap2JSON — Receipt Extraction: OCR+LLM vs VLM

## Owner
Final-year B.Tech AI & Data Science student targeting AI Engineer roles. Windows, no local GPU. Has a university lab machine with a GPU for open models. The user creates the Python venv themselves; do not create or install into environments unless asked.

## Project
User uploads a receipt photo; system returns structured JSON (merchant, date, items with qty and price, subtotal, tax, total).

### Core experiment
Compare two APPROACHES on the same receipts, keeping the MODEL fixed within each pair (only the input changes):
- Gemini pair: OCR text -> Gemini Flash  vs  image -> Gemini Flash
- Open pair: OCR text -> Qwen3-VL  vs  image -> Qwen3-VL (lab GPU: NVIDIA RTX 4000 Ada, 20 GB VRAM; default Qwen/Qwen3-VL-4B-Instruct, 8B-Instruct optional via --model)

The main result compares methods within each pair. Gemini vs Qwen is only a side note.

### Metrics per method
- Accuracy: header fields (merchant, date, subtotal, tax, total) = exact match after normalization; items = item-level precision/recall/F1 with fuzzy name matching; JSON-valid rate.
- Speed: seconds per receipt (compare only within a pair; API and local GPU differ).
- Cost per receipt: measure tokens / GPU-seconds, convert with public API prices. For open models this is an estimate and must be labelled as such.

## Data (CORD only; SROIE and custom receipts dropped)
CORD v2, Hugging Face `naver-clova-ix/cord-v2` (CC-BY-4.0), Indonesian receipts. Labels are in `gt_parse` with CORD's own keys (`nm`=name, `cnt`=count, `price`, ...). `data/scripts/map_cord.py` converts them to OUR schema. Models never see CORD keys.

Splits:
- CORD val (100): prompt tuning ONLY
- CORD test (100): main result, reported separately
- CORD train random sample (100–200, fixed seed): extra test set
- Degraded CORD test (100): blur/rotate/darken/low-res copies, robustness test

No model training by default (optional later: LoRA on CORD train, excluding sampled test receipts).
Fields missing from CORD labels are `null` and not scored (see CORD findings below once inspected).

## CORD findings (inspected 20 val receipts)
- NO merchant and NO date in `gt_parse`. Both are `null` in ground truth and not scored.
- Items (`menu`): `nm`, `cnt`, `price` in 20/20. `menu` is a list, but a DICT when the receipt has one item: the mapper must handle both.
- Rare item keys: `unitprice` 8/20, `discountprice` 2/20, nested `sub` (modifiers) 2/20, `vatyn` 1/20.
- `total.total_price` 20/20; `sub_total.subtotal_price` 11/20; `sub_total.tax_price` 8/20. Missing -> `null`, not scored for that receipt.
- Payment keys (`cashprice`, `changeprice`, `creditcardprice`, `emoneyprice`) are outside our schema; ignored.
- Number formats vary: `"16,500"` and `"23.000"` both mean thousands (IDR, no decimals). Normalization must treat both as integers.
- Scored header fields in practice: total (always), subtotal and tax (about half the receipts).

## Tech stack (100% free, no paid APIs, no credit card)
PaddleOCR · Gemini API free tier · Qwen3-VL via Hugging Face on lab GPU · Pydantic schema shared by all methods · Gradio demo app on Hugging Face Spaces (free CPU) · optional SQLite request logging (latency, tokens, cost) · pandas evaluation · GitHub Actions CI.

Live demo (decided 2026-10-23): one Gradio app on HF Spaces, image -> Gemini only (the better and faster method; Vercel/Render free tiers are too small for PaddleOCR). OCR+LLM and open models run only in offline evaluation. Dropped: FastAPI, Streamlit, Azure, Tesseract, Dockerfile.

## Rules
- Gemini free tier may use submitted data: only send public dataset images.
- Gemini free tier has tight daily limits: cache every API response to disk (`cache/`) and never re-call for a cached receipt.
- No made-up metrics anywhere. Results come only from real runs.
- Git commits must NOT contain any Claude / AI co-author or attribution lines.
- Keep explanations simple and short.

## Layout
`src/receipt/` (schema, normalize, cache, cost, ocr/, extractors/) · `data/scripts/` · `eval/` · `results/` · `app/` (Gradio demo, not created yet) · `tests/` · `.github/workflows/ci.yml` (pytest on push, Python 3.10/3.12)

## Status
Repo: https://github.com/ksrinivasarao2012/Snap2JSON (branch main). The `.env` (GEMINI_API_KEY, GEMINI_MODEL) is never committed.

Done and tested (fake backends only, 68 tests):
- Data: CORD val 100, test 100, train sample 150 (seed 42), degraded test 100 (one effect per receipt: blur/rotate/darken/lowres by index % 4). Ground truth in `data/processed/*.jsonl`. Images are NOT in git: rebuild with `data/scripts/download_cord.py` (and `train`) and `degrade.py`.
- Full-split label counts: total present 98/100 val, 95/100 test; subtotal 67/65; tax 47/41. Merchant/date never labelled (null, not scored).
- OCR: PaddleOCR 3.x mobile models (PP-OCRv5 mobile det + en rec, mkldnn off) run on all 450 images; text committed in `cache/ocr/<id>.json` (with seconds). ~10 s/receipt on laptop CPU. OCR accuracy vs CORD word labels (`eval/ocr_accuracy.py`, REPORT.md 6.11): test 97% of labelled words detected, CER 3.2%; item counts only 83% detected; blur/lowres CER 33-42%.
- `eval/metrics.py`: header exact match, item P/R/F1 (one-to-one, name fuzzy >= 80 AND qty/price equal), JSON-valid rate. The 80 threshold was checked by hand on validation (2026-10-23, `eval/threshold_check.py`, labels in `results/threshold_check_validation_{strict,lenient}.csv`): best under strict labels, kept. Lenient check at 60 is in `results/tables/name_threshold_sensitivity.json` and REPORT.md 7.1.
- `src/receipt/extractors/`: shared prompt + parser (base.py), `OcrLlm` and `Vlm` work with any backend; backends `gemini.py` and `qwen.py`. Disk cache in `cache/`.
- `eval/run_eval.py`: resumable runner -> `results/runs/*.jsonl` + `results/tables/*.json`. `run_all_qwen.sh` (lab, Linux) / `.bat` runs both methods on all 4 splits; extra args pass through (e.g. `--model`).

Qwen pair DONE on the lab GPU (Linux, venv at `.venv`; needs `torchvision`): Qwen3-VL 2B, 4B and 8B (8B bf16 fits in 20 GB), both methods, all 4 splits = 24 runs, prompt never tuned. Write-up in `results/REPORT.md`. Test (after the 2026-10-23 scoring fixes), total acc VLM vs OCR+LLM plain: 2B 0.863/0.600, 4B 0.895/0.558, 8B 0.968/0.811; item F1 2B 0.881/0.436, 4B 0.910/0.607, 8B 0.954/0.700. Correct total appears in the OCR text for 93% of test receipts; the failure audit and Method A confirm the OCR+LLM loss is mostly layout (plus missed count digits and, at 4B, the 1000x number format), not general OCR misreads.

Speed: OCR re-timed on the lab CPU (`data/scripts/time_ocr.py` -> `cache/ocr_timing/`, committed; same text as `cache/ocr/` on 450/450), ~2.4 s/receipt. OCR+LLM tables now have `mean_end_to_end_seconds` (OCR + model call); rebuild tables without a model via `run_eval.py --summarize-only`. Test end-to-end: VLM faster at every size (2B 2.52 vs 4.53 s, 4B 4.67 vs 6.46, 8B 7.91 vs 9.60). The old seconds in `cache/ocr/` are laptop times, not used.

Ground-truth fix (2026-10-23): `parse_amount` now treats a final 2-digit part as decimals (`Rp 58000.00` -> 58000, was 5,800,000). data/processed regenerated (2 val, 4 test, 8 train receipts changed) and all results/tables re-scored from saved runs. REPORT.md and the Method A report are updated to the corrected numbers; the experiment log keeps the pre-fix ones as a dated record.
Scoring (2026-10-23): predicted items matching a receipt's sub-item lines (`sub_items` in data/processed, from CORD
menu.sub) are ignored in precision. `parse_amount` keeps minus signs. CORD labels quantity only when printed
(767/767 checked via `data/scripts/extract_cord_lines.py`), so quantity errors are real.

NOT verified: Gemini live calls (never succeeded; needs GEMINI_MODEL in .env; check free-tier limits in AI Studio).

Next steps:
1. Layout experiment DONE (2026-10-23): `src/receipt/ocr/layout.py` groups OCR boxes (`cache/ocr_boxes/`, 450/450 reproduce `cache/ocr` text) into rows. Method A (`ocr_llm_rows`) on 2B/4B/8B and Method B (`ocr_llm_boxes`, rows + x positions) on 2B, all splits (REPORT.md 6.4-6.8, Method A report section 9). Test total plain -> rows: 2B 0.600 -> 0.737, 4B 0.558 -> 0.716, 8B 0.811 -> 0.895 (VLM 0.863 / 0.895 / 0.968). Test item F1 rises only at 2B (0.436 -> 0.495); 4B/8B flat (0.607/0.600, 0.700/0.684), with more wrong prices. 2B invents qty from price digits; 4B/8B do not. Rows hurt on rotated photos. B is worse than A on 2B (not run on 4B/8B). 4B < 2B on text totals is explained (REPORT.md 6.9, `eval/thousands_check.py`): 4B writes `60.000`, which JSON reads as 60 (test, 4B rows: 19 of 27 wrong totals are 1000x too small). Scores NOT changed: no safe parsing rule (`535.5` is ambiguous). Fix = separate experiment (TODO).
   OCR side experiment DONE (2026-10-23, validation only, REPORT.md 6.10): PP-OCRv5 server det + 2048 px cap (`cache/ocr_variants/T`, `run_eval.py --ocr-variant T`) halves missing qty (13 -> 6 of 39); Method A item F1 2B 0.546 -> 0.606, 8B 0.747 -> 0.763; OCR 2.3 -> 9.2 s. Not adopted.
2. DONE (2026-10-23): failure audit, 20 validation errors per run for 2B plain/rows/VLM and 8B plain/VLM (`eval/audit_failures.py`, `results/audit_validation.csv`, REPORT.md 7.2). Scorer 3/100, label 2/100. 2B plain text: 12/20 errors from OCR input (8 order); rows: 7/20. Why 4B < 2B on total: explained in step 1 (1000x number format).
3. Gemini pair on a few receipts per day (daily limits), cache makes reruns free.
4. Tune prompt/threshold on val only; never on test.
5. Still to do: cost.py, Gradio demo app (`app/app.py`). CI DONE (GitHub Actions, base deps only). Old stubs removed 2026-10-23. README DONE (results, charts, findings, setup). Charts DONE: `eval/analysis.ipynb` -> `results/charts/` (needs the `analysis` extra: matplotlib, jupyter).
