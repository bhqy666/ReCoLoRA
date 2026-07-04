#!/usr/bin/env bash
# Extra robustness experiments on Qwen3-8B, run sequentially on a single GPU.
#
# Energy-threshold (rho) sensitivity downstream training
# (HiLoRA, rho in {0.6, 0.9}, original task order, seeds 42/43/44)
# complements scripts/rho_sensitivity_ranks.py (rank-distribution only).
#
# Skips any run whose continual_results.json already exists, so this script
# is safe to re-run/resume after interruption.
set -euo pipefail
cd /home/bhqy/Documents/project/HiLoRA
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_PATH=${MODEL_PATH:-/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
SEEDS=${SEEDS:-42 43 44}
LOG_DIR=${LOG_DIR:-logs/extra_robustness}
mkdir -p "$LOG_DIR"

ORIG_TASKS="sst2,mrpc,qnli,rte,qqp,mnli"

common_args=(
  --model_name_or_path "$MODEL_PATH"
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
)

run_hilora () {
  local tasks=$1 out_dir=$2 seed=$3 energy=$4 tag=$5
  if [[ -f "$out_dir/continual_results.json" ]]; then
    echo "[skip] $tag seed=$seed (exists: $out_dir/continual_results.json)"
    return
  fi
  echo "[run] $tag seed=$seed out=$out_dir"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
    "${common_args[@]}" \
    --method hilora \
    --tasks "$tasks" \
    --output_dir "$out_dir" \
    --seed "$seed" \
    --hilora_max_rank 16 \
    --hilora_residual_rank 4 \
    --hilora_svd_device cuda \
    --hilora_energy_threshold "$energy" \
    2>&1 | tee "$LOG_DIR/${tag}_seed${seed}.log"
}

echo "=== energy_threshold sensitivity (rho=0.6, original order) ==="
for seed in $SEEDS; do
  run_hilora "$ORIG_TASKS" "outputs/continual_llm/hilora/qwen3_8b_energy0.6/seed_$seed" "$seed" "0.6" "energy0.6"
done

echo "=== energy_threshold sensitivity (rho=0.9, original order) ==="
for seed in $SEEDS; do
  run_hilora "$ORIG_TASKS" "outputs/continual_llm/hilora/qwen3_8b_energy0.9/seed_$seed" "$seed" "0.9" "energy0.9"
done

echo "=== All done ==="
