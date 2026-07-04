#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/4434a5ffc2582f9d5ac45085043ed3e3264f0a9b}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-hilora lora}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

for method in $METHODS; do
  for seed in $SEEDS; do
    out_dir="$OUT_ROOT/$method/internlm2_5_7b/seed_$seed"
    echo "[run] method=$method seed=$seed out=$out_dir"
    common_args=(
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
      --target_modules wqkv,wo
      --gradient_checkpointing
    )
    if [[ "$method" == "hilora" ]]; then
      python train_continual_llm.py "${common_args[@]}" \
        --hilora_max_rank 16 \
        --hilora_residual_rank 4 \
        --hilora_svd_device cuda \
        --hilora_dynamic_recovery \
        --hilora_target_rank_ratio 0.5 \
        --hilora_mask_interval 1 2>&1 | tee "$LOG_DIR/llm_${method}_seed_${seed}.log"
    else
      python train_continual_llm.py "${common_args[@]}" \
        --lora_r 16 \
        --lora_alpha 16 2>&1 | tee "$LOG_DIR/llm_${method}_seed_${seed}.log"
    fi
  done
done
