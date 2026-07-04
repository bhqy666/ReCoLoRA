#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-}
MODEL_TAG=${MODEL_TAG:-llama3_1_8b_instruct}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-hilora lora pissa adalora dora qlora}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm}
LOG_DIR=${LOG_DIR:-logs}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
MAX_LENGTH=${MAX_LENGTH:-256}
GRAD_ACCUM=${GRAD_ACCUM:-8}
HILORA_MAX_RANK=${HILORA_MAX_RANK:-16}
HILORA_RESIDUAL_RANK=${HILORA_RESIDUAL_RANK:-4}
HILORA_EXTRA_ARGS=${HILORA_EXTRA_ARGS:-}
mkdir -p "$LOG_DIR"

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH=$(python - <<'PY'
from pathlib import Path
roots = [
    Path('/home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots'),
    Path('/home/bhqy/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots'),
]
for root in roots:
    if root.exists():
        snaps = sorted([p for p in root.iterdir() if (p / 'config.json').exists()], key=lambda p: p.stat().st_mtime, reverse=True)
        for p in snaps:
            if (p / 'model.safetensors.index.json').exists() or list(p.glob('model-*.safetensors')):
                print(p)
                raise SystemExit(0)
raise SystemExit('Could not find complete Llama3.1-8B-Instruct snapshot. Set MODEL_PATH manually.')
PY
  )
fi

echo "[model] $MODEL_PATH"

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
      --lora_r 16
      --lora_alpha 16
    )

    extra_args=()
    case "$method" in
      hilora)
        extra_args=(--hilora_max_rank "$HILORA_MAX_RANK" --hilora_residual_rank "$HILORA_RESIDUAL_RANK" --hilora_svd_device cuda)
        if [[ -n "$HILORA_EXTRA_ARGS" ]]; then
          read -r -a hilora_more_args <<< "$HILORA_EXTRA_ARGS"
          extra_args+=("${hilora_more_args[@]}")
        fi
        ;;
      pissa)
        extra_args=(--pissa_niter 4)
        ;;
      adalora)
        extra_args=(--adalora_init_r 24 --adalora_target_r 16 --adalora_tinit 0 --adalora_tfinal 50 --adalora_delta_t 10)
        ;;
      dora|qlora|lora)
        ;;
      *)
        echo "unknown method: $method" >&2; exit 2 ;;
    esac

    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
      "${common_args[@]}" "${extra_args[@]}" 2>&1 | tee "$LOG_DIR/llama3_${method}_seed_${seed}.log"
  done
done
