#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export SEEDS="${SEEDS:-42}"
export RANKS="${RANKS:-256}"
export METHODS="${METHODS:-qlora dora}"

CONDA_ENV=open_manus RUN_MODELS=qwen3_8b bash scripts/run_peft_rank_sweep_llm.sh
CONDA_ENV=control RUN_MODELS=internlm2_5_7b_chat bash scripts/run_peft_rank_sweep_llm.sh
