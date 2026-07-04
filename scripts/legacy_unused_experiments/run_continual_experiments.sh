#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-control}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/continual}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-hilora lora full}"
TASKS="${TASKS:-sst2,mrpc,qnli,rte}"
# Defaults were stress-tested on the local 22GB GPU with GLUE max_length=128.
# Override these env vars if you move to a different GPU.
PEFT_TRAIN_BATCH="${PEFT_TRAIN_BATCH:-160}"
FULL_TRAIN_BATCH="${FULL_TRAIN_BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-128}"

COMMON_ARGS=(
  --model_name_or_path microsoft/deberta-v3-base
  --tasks "${TASKS}"
  --epochs_per_task 1
  --per_device_eval_batch_size "${EVAL_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_length "${MAX_LENGTH}"
  --fp16
)

mkdir -p "${RESULTS_ROOT}" logs results

for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    output_dir="${RESULTS_ROOT}/${method}/seed_${seed}"
    if [[ -f "${output_dir}/continual_results.json" ]]; then
      echo "[skip] ${method} seed=${seed}"
      continue
    fi

    method_args=()
    case "${method}" in
      hilora)
        method_args=(
          --per_device_train_batch_size "${PEFT_TRAIN_BATCH}"
          --learning_rate 2e-4
          --residual_lr 1e-4
          --hilora_stage1_ratio 0.75
          --hilora_max_rank 16
          --hilora_residual_rank 4
          --hilora_svd_device cuda
          --hilora_dynamic_recovery
          --hilora_target_rank_ratio 0.5
          --hilora_recovery_ratio_mode kept
          --hilora_mask_interval 1
        )
        ;;
      lora)
        method_args=(
          --per_device_train_batch_size "${PEFT_TRAIN_BATCH}"
          --learning_rate 2e-4
          --lora_r 8
          --lora_alpha 16
        )
        ;;
      full)
        method_args=(
          --per_device_train_batch_size "${FULL_TRAIN_BATCH}"
          --full_learning_rate 2e-5
        )
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac

    echo "[run] continual method=${method} seed=${seed} output=${output_dir}"
    TOKENIZERS_PARALLELISM=false conda run -n "${ENV_NAME}" python train_continual.py \
      --method "${method}" \
      --seed "${seed}" \
      --output_dir "${output_dir}" \
      "${COMMON_ARGS[@]}" \
      "${method_args[@]}"
  done
done

conda run -n "${ENV_NAME}" python scripts/aggregate_continual_results.py \
  --results_root "${RESULTS_ROOT}" \
  --output_csv results/continual_summary.csv \
  --output_tex results/continual_summary.tex \
  --output_json results/continual_summary.json
