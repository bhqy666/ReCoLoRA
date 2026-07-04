#!/usr/bin/env bash
set -euo pipefail
cd /home/bhqy/Documents/project/HiLoRA
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm/hilora/qwen3_8b_taskbank_oracle}
LOG_DIR=${LOG_DIR:-logs/qwen3_8b_taskbank_oracle_energyfix}
mkdir -p "$LOG_DIR"

for seed in $SEEDS; do
  out_dir="$OUT_ROOT/seed_$seed"
  if [[ -f "$out_dir/continual_results.json" ]]; then
    echo "[skip] seed=$seed"
    continue
  fi
  echo "[run] seed=$seed out=$out_dir"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_taskbank_llm.py \
    --model_name_or_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --output_dir "$out_dir" \
    --seed "$seed" --max_length 256 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 --max_steps_per_task 200 --max_eval_samples_per_task 1000 \
    --learning_rate 2e-4 --residual_lr 1e-4 --target_modules auto --gradient_checkpointing \
    --lora_alpha 16 --hilora_max_rank 32 --hilora_residual_rank 4 --hilora_svd_device cuda \
    --hilora_task_elbow_energy_threshold 0.0 --hilora_task_min_active_rank 6 \
    --hilora_task_bonus_map sst2:1,mrpc:1,qnli:2,rte:2,qqp:1,mnli:2 \
    --hilora_layer_bonus_map q_proj:0,v_proj:1 --hilora_depth_bonus_map early:0,mid:0,late:1 \
    2>&1 | tee "$LOG_DIR/seed${seed}.log"
done
