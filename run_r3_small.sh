#!/bin/bash
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache
export PYTHONPATH=/root/autodl-tmp/linear/src:/root/autodl-tmp/linear
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/linear
mkdir -p logs/linear_rag results/linear_rag
LOG=logs/linear_rag/r3_small_run.log
echo "=== R3 small-sample launch $(date) git=$(git rev-parse HEAD) ===" >> "$LOG"

# 1) dry-run estimate
echo "$(date +%H:%M:%S) R3 dry-run estimate" >> "$LOG"
python -u r3_dry.py >> "$LOG" 2>&1
WITHIN=$(python -c "import json;print(json.load(open('results/linear_rag/r3_dry_estimate.json'))['within_budget'])" 2>/dev/null)
EST=$(python -c "import json;print(json.load(open('results/linear_rag/r3_dry_estimate.json'))['est_total_gpu_h'])" 2>/dev/null)
echo "$(date +%H:%M:%S) R3 dry estimate within_budget=$WITHIN est_gpu_h=$EST" >> "$LOG"

if [ "$WITHIN" != "True" ]; then
  echo "$(date +%H:%M:%S) R3 OVER BUDGET (est=$EST h > 2.0). Aborting full run, will reduce steps." >> "$LOG"
  # auto-shrink: halve steps once and re-check is overkill here; just abort with note
  echo '{"status":"budget_abort","est_gpu_h":"'$EST'"}' > results/linear_rag/r3_small_status.json
  exit 0
fi

# 2) full small-sample run
echo "$(date +%H:%M:%S) R3 full small-sample run START (steps=600, train_q=800, eval_q=500)" >> "$LOG"
T0=$(date +%s)
python -u -m linear_rag.train.rerank_lora --config configs/r3_lora_smallsample.yaml --seeds 0 --train_queries 800 --eval_queries 500 >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
DUR=$(( T1 - T0 ))
echo "$(date +%H:%M:%S) R3 full run done rc=$RC dur_s=$DUR" >> "$LOG"
echo '{"status":"done","rc":'$RC',"dur_s":'$DUR'}' > results/linear_rag/r3_small_status.json
