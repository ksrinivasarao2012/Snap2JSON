# Snap2JSON — TODO

Last updated: 2026-10-23. Results so far: `results/REPORT.md`. Session record: `results/experiment_log_2026-10-22_1732.md`.

## Priority order

1. **Larger OCR detector** for lone quantity digits (CPU, validation only), then tilt-aware row grouping.
2. **Gemini pair** (laptop): start early, because the free-tier daily limits spread it over several days.
3. **`cost.py`:** cost per receipt is a required metric and still missing.
4. *Optional:* **prompt-fix experiment** (section 1), as a separate experiment next to the untuned main result.
5. **Write-up:** charts and the README results section.
6. **Portfolio app:** FastAPI, Streamlit, Docker, CI, Azure (after the results are final).

Lower priority: larger OCR detector, sub-item extraction (Option 2), LoRA.

## Done

- [x] Data: CORD validation 100, test 100, train sample 150 (seed 42), degraded test 100
- [x] PaddleOCR text for all 450 receipts (`cache/ocr/`)
- [x] Metrics, extractors, resumable eval runner, run scripts (`run_all_qwen.sh` / `.bat`)
- [x] Qwen3-VL 2B / 4B / 8B: OCR+LLM (plain) vs VLM on all 4 splits (24 runs)
- [x] OCR re-timed on the lab machine; end-to-end speed in tables and report
- [x] OCR boxes saved (`cache/ocr_boxes/`); row grouping (`src/receipt/ocr/layout.py`)
- [x] Method A (rows) on 2B, all splits + detailed report (`results/method_A_rows_report_2026-10-22_1733.md`)
- [x] REPORT.md updated with Method A
- [x] Name threshold 80 checked by hand on validation (strict and lenient labels); lenient check at 60 in both reports
- [x] Failure audit: 100 validation errors labelled model / ocr / scorer / gt (`results/audit_validation.csv`, REPORT.md 7.2)
- [x] Method B (rows + x positions) on 2B, all splits: worse than A (REPORT.md 6.8)
- [x] Method A on 4B and 8B, all splits: totals up at every size, test item F1 not (REPORT.md 6.7)
- [x] Compared A vs B vs plain; REPORT.md and the Method A report (section 9) updated
- [x] Why 4B < 2B on text totals: it writes `60.000`, read by JSON as 60 (REPORT.md 6.9, `eval/thousands_check.py`)
- [x] 8B rows price errors read by hand: mostly a row-grouping slip on 3 tilted test receipts (REPORT.md 6.7)

## 1) Needs the lab GPU

- [ ] *Optional:* **Prompt-fix experiment**, its own experiment. The failure audit (REPORT.md 7.2) found repeated
      model mistakes: the service charge given as tax (9 of 100 audited errors), cash paid / change / voucher given
      as total (7), names printed over two lines cut short or split (13). Add a few prompt rules (e.g. "total is the
      amount due, not cash paid"; "tax is the PB1 / TAX line, not service"). Tune on **validation only**, then run
      test once, for both methods with the same prompt. Report it next to the untuned main result, not instead of it.
- [ ] *Optional:* **Output-format experiment** (REPORT.md 6.9), its own experiment. 4B often writes amounts like
      `60.000` (read by JSON as 60) or `535.5` (for 535,500). Change the prompt and schema so the model copies each
      amount **as a string, exactly as printed** (`"total": "60.000"`), and convert it with `parse_amount`, the same
      rule used for the ground truth. Do NOT fix it with a parsing rule on numbers (`535.5` is ambiguous; 7 real
      amounts are under 1000). Check on **validation only**, then run test once, every method and model (a few hours
      GPU). Report next to the untuned main result. Can be combined with the prompt-fix experiment, but then report
      which change did what.
- [ ] **Sub-item extraction (Option 2)**, its own experiment: add sub-items (modifiers, CORD `menu.sub`) to the prompt
      and schema, check the prompt on validation, then re-run every method and model (about 4 h GPU for Qwen;
      Gemini too). Until then sub-items are ignored in scoring (Option 1, done 2026-10-23).
- [ ] **Larger OCR text detector** (e.g. PP-OCRv5 server det) for lone quantity digits, validation only
      (`data/scripts/compare_ocr.py`; thresholds, box margin and upscaling did not fix it).
- [ ] *Optional, later:* LoRA fine-tune on CORD train (excluding the 150 sampled receipts).

## 2) No GPU needed (laptop is fine)

### Experiment
- [ ] *Optional:* **Tilt-aware row grouping** (no GPU to build; GPU to re-run A). On slightly tilted photos the row rule
      shifts every price to the next item's row (`test_0085`, `test_0098`, `test_0045`; REPORT.md 6.7), and rows lose
      to plain on rotated photos. Idea: estimate the slant from the boxes (e.g. the median angle of each box's top
      edge, or the line through boxes on the same printed line) and group along it. Check the grouping on
      **validation** receipts only, then re-run Method A.
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
- [x] REPORT.md and the Method A report updated with the corrected ground truth (2026-10-23). The experiment log
      (`results/experiment_log_2026-10-22_1732.md`) is a dated record and keeps the pre-fix numbers.
- [ ] README: results section (currently "TBD") with a link to REPORT.md; Linux setup steps
- [ ] `run_all_qwen.bat`: add the `METHODS` / `SPLITS` options that the `.sh` has
- [ ] When the lab work is done: delete the GitHub token
      (Settings → Developer settings → Fine-grained tokens) and run `git credential-cache exit`

## Rules to keep in mind
- Tune prompts or thresholds on **validation only**, never on test.
- Only real results in reports; no made-up numbers.
- Gemini free tier: only send public dataset images, and cache every response.
- No AI co-author lines in commits.
