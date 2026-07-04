#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
VARIANTS=${VARIANTS:-no_dynamic one_stage fixed_rank random_init}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm_ablation/qwen3_8b}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    out_dir="$OUT_ROOT/$variant/seed_$seed"
    if [[ -f "$out_dir/continual_results.json" ]]; then
      echo "[skip] variant=$variant seed=$seed"
      continue
    fi
    echo "[run] variant=$variant seed=$seed out=$out_dir"
    args=(
      --method hilora
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
      --hilora_max_rank 16
      --hilora_residual_rank 4
      --hilora_svd_device cuda
    )
    case "$variant" in
      no_dynamic)
        ;;
      one_stage)
        args+=(--hilora_dynamic_recovery --hilora_stage1_ratio 0.0 --hilora_target_rank_ratio 0.5)
        ;;
      fixed_rank)
        args+=(--hilora_dynamic_recovery --hilora_force_rank 16 --hilora_target_rank_ratio 0.5)
        ;;
      random_init)
        args+=(--hilora_dynamic_recovery --hilora_init_strategy random --hilora_target_rank_ratio 0.5)
        ;;
      dynamic_recovery)
        args+=(--hilora_dynamic_recovery)
        ;;
      *)
        echo "unknown variant: $variant" >&2; exit 2 ;;
    esac
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
      "${args[@]}" 2>&1 | tee "$LOG_DIR/qwen3_ablation_${variant}_seed_${seed}.log"
  done
done
