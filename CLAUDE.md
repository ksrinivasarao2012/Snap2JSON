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
PaddleOCR (Tesseract backup, PaddleOCR-VL optional) · Gemini API free tier · Qwen3-VL via Hugging Face on lab GPU · Pydantic schema shared by all methods · FastAPI backend · Streamlit frontend · SQLite request logging (latency, tokens, cost) · pandas evaluation · Docker · GitHub Actions · Azure for Students (Container Apps, likely).

Live Azure demo: only PaddleOCR+Gemini and Gemini vision. Open models run only in offline evaluation.

## Rules
- Gemini free tier may use submitted data: only send public dataset images.
- Gemini free tier has tight daily limits: cache every API response to disk (`cache/`) and never re-call for a cached receipt.
- No made-up metrics anywhere. Results come only from real runs.
- Git commits must NOT contain any Claude / AI co-author or attribution lines.
- Keep explanations simple and short.

## Layout
`src/receipt/` (schema, normalize, cache, cost, ocr/, extractors/) · `data/scripts/` · `eval/` · `results/` · `app/` (api, ui) · `tests/` · `docker/` · `.github/workflows/`

## Status
Repo: https://github.com/ksrinivasarao2012/Snap2JSON (branch main). The `.env` (GEMINI_API_KEY, GEMINI_MODEL) is never committed.

Done and tested (fake backends only, 44 tests):
- Data: CORD val 100, test 100, train sample 150 (seed 42), degraded test 100 (one effect per receipt: blur/rotate/darken/lowres by index % 4). Ground truth in `data/processed/*.jsonl`. Images are NOT in git: rebuild with `data/scripts/download_cord.py` (and `train`) and `degrade.py`.
- Full-split label counts: total present 98/100 val, 95/100 test; subtotal 67/65; tax 47/41. Merchant/date never labelled (null, not scored).
- OCR: PaddleOCR 3.x mobile models (PP-OCRv5 mobile det + en rec, mkldnn off) run on all 450 images; text committed in `cache/ocr/<id>.json` (with seconds). ~10 s/receipt on laptop CPU. Only a 10-receipt spot check against ground truth; no real OCR accuracy number yet.
- `eval/metrics.py`: header exact match, item P/R/F1 (one-to-one, name fuzzy >= 80 AND qty/price equal), JSON-valid rate. The 80 threshold is PROVISIONAL: check on val only.
- `src/receipt/extractors/`: shared prompt + parser (base.py), `OcrLlm` and `Vlm` work with any backend; backends `gemini.py` and `qwen.py`. Disk cache in `cache/`.
- `eval/run_eval.py`: resumable runner -> `results/runs/*.jsonl` + `results/tables/*.json`. `run_all_qwen.sh` (lab, Linux) / `.bat` runs both methods on all 4 splits; extra args pass through (e.g. `--model`).

Qwen pair DONE on the lab GPU (Linux, venv at `.venv`; needs `torchvision`): Qwen3-VL 2B, 4B and 8B (8B bf16 fits in 20 GB), both methods, all 4 splits = 24 runs, prompt never tuned. Write-up in `results/REPORT.md`. Test, total acc VLM vs OCR+LLM: 2B 0.832/0.579, 4B 0.853/0.516, 8B 0.926/0.768; item F1 2B 0.807/0.402, 4B 0.825/0.547, 8B 0.866/0.652. Correct total appears in the OCR text for 93% of test receipts, so the OCR+LLM loss looks like layout loss, not digit errors (hypothesis; failure cases not read yet).

Speed: OCR re-timed on the lab CPU (`data/scripts/time_ocr.py` -> `cache/ocr_timing/`, committed; same text as `cache/ocr/` on 450/450), ~2.4 s/receipt. OCR+LLM tables now have `mean_end_to_end_seconds` (OCR + model call); rebuild tables without a model via `run_eval.py --summarize-only`. Test end-to-end: VLM faster at every size (2B 2.52 vs 4.53 s, 4B 4.67 vs 6.46, 8B 7.91 vs 9.60). The old seconds in `cache/ocr/` are laptop times, not used.

Ground-truth fix (2026-10-23): `parse_amount` now treats a final 2-digit part as decimals (`Rp 58000.00` -> 58000, was 5,800,000). data/processed regenerated (2 val, 4 test, 8 train receipts changed) and all results/tables re-scored from saved runs. REPORT.md and the Method A report are updated to the corrected numbers; the experiment log keeps the pre-fix ones as a dated record.
Scoring (2026-10-23): predicted items matching a receipt's sub-item lines (`sub_items` in data/processed, from CORD
menu.sub) are ignored in precision. `parse_amount` keeps minus signs. CORD labels quantity only when printed
(767/767 checked via `data/scripts/extract_cord_lines.py`), so quantity errors are real.

NOT verified: Gemini live calls (never succeeded; needs GEMINI_MODEL in .env; check free-tier limits in AI Studio).

Next steps:
1. Layout experiment: `src/receipt/ocr/layout.py` groups OCR boxes (`cache/ocr_boxes/`, 450/450 reproduce `cache/ocr` text) into rows. Methods `ocr_llm_rows` (A) and `ocr_llm_boxes` (B, rows + x positions). DONE: A on 2B, all splits (`results/method_A_rows_report_2026-10-22_1733.md`): test total 0.579 -> 0.695, subtotal 0.754 -> 0.923, item F1 0.402 -> 0.449 (VLM 0.832 / 0.908 / 0.807). Items barely improve: 2B invents qty from price digits when qty is not in the text. TODO: B on 2B (`METHODS="ocr_llm_boxes" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct`), then A/B on 4B and 8B; update REPORT.md (plain OCR+LLM numbers use naive formatting).
2. Read ~20 wrong OCR+LLM outputs to check the layout-loss hypothesis (and why 4B < 2B on total).
3. Gemini pair on a few receipts per day (daily limits), cache makes reruns free.
4. Tune prompt/threshold on val only; never on test.
5. Still stubs: cost.py, tesseract.py, FastAPI app, Streamlit UI, logging_db, Dockerfile, CI workflow, analysis notebook, charts, README results.
