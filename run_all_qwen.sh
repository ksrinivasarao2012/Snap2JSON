#!/usr/bin/env bash
# Same as run_all_qwen.bat, for Linux (the lab machine). Resumable.
# Usage: bash run_all_qwen.sh [extra args, e.g. --model Qwen/Qwen3-VL-8B-Instruct]
# Pick methods/splits with env vars, e.g. METHODS="ocr_llm_rows" SPLITS="validation test" bash run_all_qwen.sh
cd "$(dirname "$0")" || exit 1
mkdir -p results
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
for m in ${METHODS:-vlm ocr_llm}; do
  for s in ${SPLITS:-validation test train test_degraded}; do
    echo "===== $m $s ====="
    "$PY" -u eval/run_eval.py --backend qwen --method "$m" --split "$s" "$@"
  done
done 2>&1 | tee -a results/qwen_full_run.log
