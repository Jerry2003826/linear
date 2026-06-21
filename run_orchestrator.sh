#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2
cd /root/autodl-tmp/linear

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=/root/autodl-tmp/linear/src:/root/autodl-tmp/linear
export MAX_GPU_HOURS_TOTAL=${MAX_GPU_HOURS_TOTAL:-10}
export GPU_PRICE_PER_HOUR=${GPU_PRICE_PER_HOUR:-2.0}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/linear_rag results/linear_rag
git rev-parse HEAD > logs/linear_rag/git_sha.txt

LOG=logs/linear_rag/orchestrate_run.log
echo "=== launch $(date) git=$(cat logs/linear_rag/git_sha.txt) MAX_GPU_HOURS_TOTAL=$MAX_GPU_HOURS_TOTAL price=$GPU_PRICE_PER_HOUR ===" >> $LOG

nohup python -u -m linear_rag.orchestrate >> $LOG 2>&1 &
PID=$!
echo $PID > logs/linear_rag/orchestrate.pid
echo "started orchestrator pid=$PID log=$LOG"
