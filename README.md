# Snap2JSON

Snap2JSON: snap a receipt, get structured JSON. Comparing OCR + LLM vs. vision-language models on accuracy, speed, and cost.

## Problem
Turn a receipt photo into structured JSON (merchant, date, items with qty and price, subtotal, tax, total).

## Method
Two approaches on the same CORD v2 receipts, with the model held constant inside each pair:
- Gemini Flash: OCR text vs image
- Qwen3-VL: OCR text vs image

Metrics: header-field exact match, item precision/recall/F1, JSON-valid rate, seconds per receipt, cost per receipt (open-model cost is an estimate).

## Results
TBD — only real runs will be reported here.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then add GEMINI_API_KEY
```
