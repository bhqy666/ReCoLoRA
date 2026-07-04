#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/4434a5ffc2582f9d5ac45085043ed3e3264f0a9b}
MODEL_TAG=${MODEL_TAG:-internlm2_5_7b}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-hilora lora pissa adalora dora}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

for method in $METHODS; do
  for seed in $SEEDS; do
    out_dir="$OUT_ROOT/$method/$MODEL_TAG/seed_$seed"
    if [[ -f "$out_dir/continual_results.json" ]]; then
      echo "[skip] method=$method seed=$seed"
      continue
    fi
    echo "[run] method=$method seed=$seed out=$out_dir"
    args=(
      --method "$method"
      --model_name_or_path "$MODEL_PATH"
      --tasks "$TASKS"
      --output_dir "$out_dir"
      --seed "$seed"
      --max_length 256
      --per_device_train_batch_size 1
      --per_device_eval_batch_size 4
      --gradient_accumulation_steps 8
      --max_steps_per_task 200
      --max_eval_samples_per_task 1000
      --learning_rate 2e-4
      --residual_lr 1e-4
      --target_modules auto
      --gradient_checkpointing
      --lora_r 16
      --lora_alpha 16
    )
    if [[ "$method" == "hilora" ]]; then
      args+=(--hilora_max_rank 16 --hilora_residual_rank 4 --hilora_svd_device cuda --hilora_dynamic_recovery --hilora_target_rank_ratio 0.5)
    elif [[ "$method" == "pissa" ]]; then
      args+=(--pissa_niter 4)
    elif [[ "$method" == "adalora" ]]; then
      args+=(--adalora_init_r 24 --adalora_target_r 16 --adalora_tinit 0 --adalora_tfinal 50 --adalora_delta_t 10)
    fi
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
      "${args[@]}" 2>&1 | tee "$LOG_DIR/internlm_${method}_seed_${seed}.log"
  done
done
