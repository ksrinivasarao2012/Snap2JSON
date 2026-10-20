# Snap2JSON — Receipt Extraction: OCR+LLM vs VLM

## Owner
Final-year B.Tech AI & Data Science student targeting AI Engineer roles. Windows, no local GPU. Has a university lab machine with a GPU for open models. The user creates the Python venv themselves; do not create or install into environments unless asked.

## Project
User uploads a receipt photo; system returns structured JSON (merchant, date, items with qty and price, subtotal, tax, total).

### Core experiment
Compare two APPROACHES on the same receipts, keeping the MODEL fixed within each pair (only the input changes):
- Gemini pair: OCR text -> Gemini Flash  vs  image -> Gemini Flash
- Open pair: OCR text -> Qwen3-VL  vs  image -> Qwen3-VL (lab GPU; size TBD)

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
Done: folder structure, schema, normalize (+tests), CORD download/mapping, splits (val 100, test 100, train sample 150 with seed 42, degraded test 100), ground truth in `data/processed/`.
CORD findings above were from 20 receipts; on the full splits total is present in 98/100 val and 95/100 test, subtotal 67/65, tax 47/41.
Not started: metrics, OCR, extractors, cache, cost, eval runner, API, UI, Docker, CI.
