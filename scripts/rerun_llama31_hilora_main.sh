#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-control}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm/hilora/llama3_1_8b_instruct}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

for seed in $SEEDS; do
  out_dir="$OUT_ROOT/seed_$seed"
  if [[ -f "$out_dir/continual_results.json" ]]; then
    echo "[skip] seed=$seed"
    continue
  fi
  echo "[run] seed=$seed out=$out_dir"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
    --method hilora \
    --model_name_or_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --output_dir "$out_dir" \
    --seed "$seed" \
    --max_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_steps_per_task 200 \
    --max_eval_samples_per_task 1000 \
    --learning_rate 2e-4 \
    --residual_lr 1e-4 \
    --target_modules auto \
    --gradient_checkpointing \
    --hilora_max_rank 16 \
    --hilora_residual_rank 4 \
    --hilora_svd_device cuda \
    2>&1 | tee "$LOG_DIR/llama31_hilora_main_seed_${seed}.log"
done
