#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-control}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/experiments}"
SUMMARY_CSV="${SUMMARY_CSV:-results/runs.csv}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-hilora lora adalora qlora full}"

COMMON_ARGS=(
  --model_name_or_path microsoft/deberta-v3-base
  --dataset_name squad_v2
  --target_modules query_proj,value_proj
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 4
  --num_train_epochs 2
  --warmup_steps 500
  --logging_steps 100
  --eval_steps 1000
  --save_steps 1000
  --save_total_limit 2
  --max_seq_length 384
  --doc_stride 128
  --fp16
  --results_csv "${SUMMARY_CSV}"
)

rm -f "${SUMMARY_CSV}"
mkdir -p "${RESULTS_ROOT}" "$(dirname "${SUMMARY_CSV}")"

for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    output_dir="${RESULTS_ROOT}/${method}/seed_${seed}"
    metrics_file="${output_dir}/metrics.json"

    if [[ -f "${metrics_file}" ]]; then
      echo "[skip] ${method} seed=${seed} already has ${metrics_file}"
      continue
    fi

    method_args=()
    case "${method}" in
      hilora)
        method_args=(
          --learning_rate 2e-4
          --residual_lr 1e-4
          --hilora_stage1_ratio 0.75
          --hilora_max_rank 16
          --hilora_residual_rank 4
        )
        ;;
      lora|adalora)
        method_args=(
          --learning_rate 2e-4
          --lora_r 8
          --lora_alpha 16
        )
        ;;
      qlora)
        method_args=(
          --learning_rate 1e-4
          --lora_r 8
          --lora_alpha 16
        )
        ;;
      full)
        method_args=(
          --learning_rate 2e-5
          --gradient_checkpointing
        )
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac

    echo "[run] method=${method} seed=${seed} output=${output_dir}"
    TOKENIZERS_PARALLELISM=false conda run -n "${ENV_NAME}" python train_qa.py \
      --method "${method}" \
      --seed "${seed}" \
      --output_dir "${output_dir}" \
      "${COMMON_ARGS[@]}" \
      "${method_args[@]}"
  done
done

conda run -n "${ENV_NAME}" python scripts/aggregate_results.py \
  --results_root "${RESULTS_ROOT}" \
  --output_csv results/summary.csv \
  --output_tex results/summary.tex \
  --output_json results/summary.json
