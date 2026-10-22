# Snap2JSON — TODO

Last updated: 2026-10-22. Results so far: `results/REPORT.md`. Session record: `results/experiment_log_2026-10-22_1732.md`.

## Done

- [x] Data: CORD validation 100, test 100, train sample 150 (seed 42), degraded test 100
- [x] PaddleOCR text for all 450 receipts (`cache/ocr/`)
- [x] Metrics, extractors, resumable eval runner, run scripts (`run_all_qwen.sh` / `.bat`)
- [x] Qwen3-VL 2B / 4B / 8B: OCR+LLM (plain) vs VLM on all 4 splits (24 runs)
- [x] OCR re-timed on the lab machine; end-to-end speed in tables and report
- [x] OCR boxes saved (`cache/ocr_boxes/`); row grouping (`src/receipt/ocr/layout.py`)
- [x] Method A (rows) on 2B, all splits + detailed report (`results/method_A_rows_report_2026-10-22_1733.md`)
- [x] REPORT.md updated with Method A

## 1) Needs the lab GPU

- [ ] **Method B (rows + x positions) on 2B**, all splits. It resumes from the 1 receipt already done.
      `METHODS="ocr_llm_boxes" bash run_all_qwen.sh --model Qwen/Qwen3-VL-2B-Instruct`
- [ ] **Method A on 4B and 8B**, all splits. About 30 and 52 min.
      `METHODS="ocr_llm_rows" bash run_all_qwen.sh` (4B default) and `... --model Qwen/Qwen3-VL-8B-Instruct`
      A 3-receipt 4B smoke test exists locally; it is a check, not a result.
- [ ] **Method B on 4B and 8B**, only if B beats A on 2B.
- [ ] *Optional, validation only:* prompt rule for missing quantities, since 2B invents qty from price digits. If adopted,
      re-run **every** method with the new prompt so the comparison stays fair.
- [ ] *Optional, later:* LoRA fine-tune on CORD train (excluding the 150 sampled receipts).

## 2) No GPU needed (laptop is fine)

### Experiment
- [ ] Compare A vs B vs plain once B is done; update REPORT.md and the Method A report
- [ ] Look at why 4B < 2B on plain OCR+LLM totals (and whether rows fix it)
- [ ] **Gemini pair:**
  - set `GEMINI_MODEL` in `.env` and check the free-tier limits in AI Studio
  - run a few receipts per day (the cache makes reruns free)
  - live calls have never been tested
- [ ] **`cost.py`:** Gemini cost from tokens and public prices; Qwen estimate from GPU-seconds (label it as an estimate)
- [ ] Charts and `eval/analysis.ipynb` → `results/charts/`
- [ ] *Optional:* OCR accuracy number, e.g. character error rate against labels (only a spot check so far)

### App
- [ ] FastAPI backend (`app/api/main.py`): upload a receipt and get JSON back (PaddleOCR+Gemini, or Gemini vision)
- [ ] SQLite request logging (`app/api/logging_db.py`): latency, tokens, cost
- [ ] Streamlit UI (`app/ui/streamlit_app.py`)
- [ ] *Low priority:* Tesseract backup (`src/receipt/ocr/tesseract.py`)

### Engineering
- [ ] Dockerfile (`docker/Dockerfile`)
- [ ] GitHub Actions CI: run pytest on push (`.github/workflows/ci.yml`)
- [ ] Azure Container Apps deploy (Gemini methods only)

### Write-up and housekeeping
- [ ] **REPORT.md and the Method A report show pre-fix numbers.** On 2026-10-23 a ground-truth bug was fixed
      (amounts like `Rp 58000.00` were read as 5,800,000; 2 val, 4 test, 8 train receipts) and all 28 tables were
      re-scored from the saved runs (no model re-runs). Every score rose or stayed equal; update the report tables.
- [ ] README: results section (currently "TBD") with a link to REPORT.md; Linux setup steps
- [ ] `run_all_qwen.bat`: add the `METHODS` / `SPLITS` options that the `.sh` has
- [ ] When the lab work is done: delete the GitHub token
      (Settings → Developer settings → Fine-grained tokens) and run `git credential-cache exit`

## Rules to keep in mind
- Tune prompts or thresholds on **validation only**, never on test.
- Only real results in reports; no made-up numbers.
- Gemini free tier: only send public dataset images, and cache every response.
- No AI co-author lines in commits.
