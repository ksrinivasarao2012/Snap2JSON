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
- `eval/run_eval.py`: resumable runner -> `results/runs/*.jsonl` + `results/tables/*.json`.

NOT verified: Gemini live calls (never succeeded; was blocked on .env/GEMINI_MODEL, and the free-tier limits are unchecked: read them in AI Studio) and the Qwen backend (never run; needs the lab GPU). Both may have bugs on first real run.

Next steps:
1. At the lab: clone, `uv pip install -e ".[dev,models]"`, download data, run `python eval/run_eval.py --backend qwen --method vlm|ocr_llm --split validation --limit 3`, fix any errors, then run all splits for both methods.
2. Gemini pair on a few receipts per day (daily limits), cache makes reruns free.
3. Tune prompt/threshold on val only; never on test.
4. Still stubs: cost.py, tesseract.py, FastAPI app, Streamlit UI, logging_db, Dockerfile, CI workflow, analysis notebook, results tables/charts, README results.
