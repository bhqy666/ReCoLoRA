#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-hilora lora}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm}
LOG_DIR=${LOG_DIR:-logs}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
MAX_LENGTH=${MAX_LENGTH:-256}
GRAD_ACCUM=${GRAD_ACCUM:-8}
mkdir -p "$LOG_DIR"

for method in $METHODS; do
  for seed in $SEEDS; do
    out_dir="$OUT_ROOT/$method/$MODEL_TAG/seed_$seed"
    done_file="$out_dir/continual_results.json"
    if [[ -f "$done_file" ]]; then
      echo "[skip] method=$method seed=$seed exists: $done_file"
      continue
    fi

    echo "[run] method=$method seed=$seed out=$out_dir"
    common_args=(
      --method "$method"
      --model_name_or_path "$MODEL_PATH"
      --tasks "$TASKS"
      --output_dir "$out_dir"
      --seed "$seed"
      --max_length "$MAX_LENGTH"
      --per_device_train_batch_size 1
      --per_device_eval_batch_size 4
      --gradient_accumulation_steps "$GRAD_ACCUM"
      --max_steps_per_task "$MAX_STEPS_PER_TASK"
      --max_eval_samples_per_task "$MAX_EVAL_SAMPLES_PER_TASK"
      --learning_rate 2e-4
      --residual_lr 1e-4
      --target_modules auto
      --gradient_checkpointing
    )

    if [[ "$method" == "hilora" ]]; then
      PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py "${common_args[@]}" \
        --hilora_max_rank 16 \
        --hilora_residual_rank 4 \
        --hilora_svd_device cuda \
        --hilora_dynamic_recovery \
        --hilora_target_rank_ratio 0.5 \
        --hilora_mask_interval 1 2>&1 | tee "$LOG_DIR/qwen3_${method}_seed_${seed}.log"
    else
      PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py "${common_args[@]}" \
        --lora_r 16 \
        --lora_alpha 16 2>&1 | tee "$LOG_DIR/qwen3_${method}_seed_${seed}.log"
    fi
  done
done
