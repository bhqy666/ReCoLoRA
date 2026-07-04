#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/peft_rank_sweep

export CONDA_ENV=${CONDA_ENV:-control}
export SEEDS=${SEEDS:-42}
export RANKS=${RANKS:-64 128 256}
export RUN_MODELS=${RUN_MODELS:-mistral_7b_v03,internlm2_5_7b_chat,qwen3_8b,llama3_1_8b_instruct}
export METHODS=${METHODS:-lora pissa adalora qlora dora}

LOG_FILE=${LOG_FILE:-logs/peft_rank_sweep/master_seed42_rank64_128_256.log}
PID_FILE=${PID_FILE:-logs/peft_rank_sweep/master_seed42_rank64_128_256.pid}

setsid nohup bash scripts/run_peft_rank_sweep_llm.sh > "$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "$pid"
