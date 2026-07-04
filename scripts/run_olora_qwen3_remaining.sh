#!/usr/bin/env bash
# Run the remaining O-LoRA seeds (43, 44) on Qwen3-8B with the exact same
# continual-GLUE protocol as the existing HiLoRA/LoRA seed runs, then aggregate
# the 3-seed comparison. Designed to be launched detached (setsid nohup).
set -u
cd /home/bhqy/Documents/project/HiLoRA

PY=/home/bhqy/anaconda3/envs/open_manus/bin/python
SNAP=$(ls -d /home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

for SEED in 43 44; do
  OUT=outputs/continual_llm/olora/qwen3_8b/seed_${SEED}
  mkdir -p "$OUT"
  echo "===== O-LoRA Qwen3-8B seed ${SEED} START $(date +%F_%H:%M:%S) ====="
  $PY train_continual_llm.py \
    --method olora --model_name_or_path "$SNAP" \
    --tasks sst2,mrpc,qnli,rte,qqp,mnli \
    --output_dir "$OUT" \
    --seed ${SEED} --max_steps_per_task 200 --max_eval_samples_per_task 1000 \
    --learning_rate 2e-4 --lora_r 8 --lora_alpha 16 --max_length 256 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
    --olora_orth_weight 0.5 --torch_dtype float16
  echo "===== O-LoRA Qwen3-8B seed ${SEED} DONE $(date +%F_%H:%M:%S) exit=$? ====="
done

echo "===== AGGREGATING 3-SEED COMPARISON $(date +%F_%H:%M:%S) ====="
$PY - <<'PYEOF'
import json, statistics, os
def load(seed_dirs):
    fin, forg = [], []
    for d in seed_dirs:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            return None
        s = json.load(open(p))
        fin.append(s["average_final_score"]); forg.append(s["average_forgetting"])
    return fin, forg
def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"

methods = {
  "LoRA":   [f"outputs/continual_llm/lora/qwen3_8b/seed_{s}" for s in (42,43,44)],
  "O-LoRA": [f"outputs/continual_llm/olora/qwen3_8b/seed_{s}" for s in (42,43,44)],
  "HiLoRA(no_dynamic)": [f"outputs/continual_llm_ablation/qwen3_8b/no_dynamic/seed_{s}" for s in (42,43,44)],
}
print(f"{'Method':22} {'FinalAvg':18} {'AvgForget':18} seeds")
for name, dirs in methods.items():
    r = load(dirs)
    if r is None:
        print(f"{name:22} (incomplete)"); continue
    fin, forg = r
    print(f"{name:22} {fmt(fin):18} {fmt(forg):18} {[round(x,4) for x in fin]}")
PYEOF
echo "===== ALL DONE $(date +%F_%H:%M:%S) ====="
