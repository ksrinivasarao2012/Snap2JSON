@echo off
REM Runs Qwen VLM first, then OCR+LLM, on all four splits (450 receipts each). Resumable.
REM Usage: run_all_qwen.bat [extra args, e.g. --model Qwen/Qwen3-VL-8B-Instruct]
cd /d "%~dp0"
if not exist results mkdir results
for %%M in (vlm ocr_llm) do (
  for %%S in (validation test train test_degraded) do (
    echo ===== %%M %%S =====
    python eval/run_eval.py --backend qwen --method %%M --split %%S %*
  )
) >> results\qwen_full_run.log 2>&1
echo Done. See results\qwen_full_run.log
