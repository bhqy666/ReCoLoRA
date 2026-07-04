#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONDA_ENV=${CONDA_ENV:-control}
SEED=${SEED:-42}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
RUN_MODELS=${RUN_MODELS:-qwen3_8b,llama3_8b,llama3_1_8b_instruct}
RECURSIVE_MAX_RANK=${RECURSIVE_MAX_RANK:-256}
RECURSIVE_MIN_RANK=${RECURSIVE_MIN_RANK:-8}
RUN_SUFFIX=${RUN_SUFFIX:-rank${RECURSIVE_MAX_RANK}_min${RECURSIVE_MIN_RANK}_elbowonly}
FORCE_RERUN=${FORCE_RERUN:-0}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

run_one() {
  local model_tag="$1"
  local model_path="$2"
  local target_modules="${3:-q_proj,v_proj}"
  local log_dir="logs/${model_tag}_recursive_hilora_${RUN_SUFFIX}"
  local out_dir="outputs/continual_llm/recursive_hilora/${model_tag}_${RUN_SUFFIX}/seed_${SEED}"
  mkdir -p "$log_dir" "$out_dir"

  if [[ "$FORCE_RERUN" != "1" && -s "$out_dir/summary.json" ]]; then
    echo "[runner] skip existing model=${model_tag} seed=${SEED} out_dir=${out_dir}"
    return 0
  fi

  echo "[runner] start model=${model_tag} seed=${SEED} $(date --iso-8601=seconds)"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    HF_ENDPOINT="$HF_ENDPOINT" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
    --method recursive_hilora \
    --model_name_or_path "$model_path" \
    --tasks "$TASKS" \
    --output_dir "$out_dir" \
    --seed "$SEED" \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --target_modules "$target_modules" \
    --max_steps_per_task "$MAX_STEPS_PER_TASK" \
    --max_eval_samples_per_task "$MAX_EVAL_SAMPLES_PER_TASK" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 256 \
    --learning_rate 2e-4 \
    --residual_lr 2e-5 \
    --warmup_ratio 0.03 \
    --recursive_max_rank "$RECURSIVE_MAX_RANK" \
    --recursive_min_rank "$RECURSIVE_MIN_RANK" \
    --recursive_fast_rank_bonus 3 \
    --recursive_fast_rank_multiplier 1.2 \
    --recursive_min_fast_rank 4 \
    --recursive_energy_threshold 0.9 \
    --recursive_elbow_ratio 1.5 \
    --recursive_svd_device cpu \
    --recursive_svd_n_iter 1 \
    2>&1 | tee "$log_dir/seed${SEED}.log"
  local status=${PIPESTATUS[0]}
  echo "[runner] done model=${model_tag} seed=${SEED} status=${status} $(date --iso-8601=seconds)"
  return "$status"
}

if [[ ",${RUN_MODELS}," == *",qwen3_8b,"* ]]; then
  run_one \
    qwen3_8b \
    /home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",llama3_8b,"* ]]; then
  run_one \
    llama3_8b \
    /home/bhqy/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",llama3_1_8b_instruct,"* ]]; then
  run_one \
    llama3_1_8b_instruct \
    /home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",mistral_7b_v03,"* ]]; then
  run_one \
    mistral_7b_v03 \
    /home/bhqy/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",internlm2_5_7b_chat,"* ]]; then
  run_one \
    internlm2_5_7b_chat \
    /home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/9b8d9553846ecf6393f3408fa9d3ec9928fdab4d \
    auto
fi
