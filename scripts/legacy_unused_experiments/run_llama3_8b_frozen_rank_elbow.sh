#!/usr/bin/env bash
set -euo pipefail
cd /home/bhqy/Documents/project/HiLoRA
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CONDA_ENV=${CONDA_ENV:-open_manus}
export MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061}
export MODEL_TAG=${MODEL_TAG:-llama3_8b_frozen_rank_elbow}
export METHODS=${METHODS:-hilora}
export SEEDS=${SEEDS:-42 43 44}
export MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
export MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export HILORA_MAX_RANK=${HILORA_MAX_RANK:-32}
export HILORA_RESIDUAL_RANK=${HILORA_RESIDUAL_RANK:-4}
if [[ -z "${HILORA_EXTRA_ARGS:-}" ]]; then
  export HILORA_EXTRA_ARGS="--hilora_taskwise_elbow --hilora_allocate_full_rank --hilora_freeze_old_ranks --hilora_min_new_ranks_per_task 2 --hilora_task_elbow_energy_threshold 0.0 --hilora_task_min_active_rank 8 --hilora_task_bonus_map sst2:2,mrpc:2,qnli:3,rte:3,qqp:2,mnli:4 --hilora_layer_bonus_map q_proj:0,v_proj:1 --hilora_depth_bonus_map early:0,mid:1,late:2"
fi
export LOG_DIR=${LOG_DIR:-logs/llama3_8b_frozen_rank_elbow}
mkdir -p "$LOG_DIR"
bash scripts/run_llama3_llm_experiments.sh
