#!/usr/bin/env bash
# Remaining O-LoRA comparison experiments, run sequentially and detached.
#   1) Mistral-7B-v0.3: ReCoLoRA + O-LoRA (3 seeds each); LoRA single-adapter baseline already exists.
#   2) Qwen3-8B: O-LoRA r=16 (3 seeds) capacity-fairness variant.
# All use the exact continual-GLUE protocol (200 steps/task, lr 2e-4, q/v, fp16, forward order)
# matching the existing ReCoLoRA/LoRA seed runs.
set -u
cd /home/bhqy/Documents/project/HiLoRA
PY=/home/bhqy/anaconda3/envs/open_manus/bin/python
QWEN=$(ls -d /home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
MISTRAL=$(ls -d /home/bhqy/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/*/ | head -1)
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
TASKS=sst2,mrpc,qnli,rte,qqp,mnli
COMMON="--tasks $TASKS --max_steps_per_task 200 --max_eval_samples_per_task 1000 \
  --learning_rate 2e-4 --lora_alpha 16 --max_length 256 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
  --torch_dtype float16"

run() {  # run <tag> <model> <outdir> <seed> <extra-args...>
  local tag=$1 model=$2 out=$3 seed=$4; shift 4
  mkdir -p "$out"
  echo "===== ${tag} seed ${seed} START $(date +%F_%H:%M:%S) ====="
  $PY train_continual_llm.py --model_name_or_path "$model" --output_dir "$out" --seed ${seed} $COMMON "$@"
  echo "===== ${tag} seed ${seed} DONE $(date +%F_%H:%M:%S) exit=$? ====="
}

# ---- 1) Mistral-7B ReCoLoRA (canonical static two-stage, defaults match Qwen canonical) ----
for S in 42 43 44; do
  run "Mistral-ReCoLoRA" "$MISTRAL" outputs/continual_llm/hilora/mistral_7b_v03/seed_$S $S \
    --method hilora --hilora_max_rank 16 --hilora_energy_threshold 0.8
done
# ---- 2) Mistral-7B O-LoRA (r=8) ----
for S in 42 43 44; do
  run "Mistral-OLoRA" "$MISTRAL" outputs/continual_llm/olora/mistral_7b_v03/seed_$S $S \
    --method olora --lora_r 8 --olora_orth_weight 0.5
done
# ---- 3) Qwen3-8B O-LoRA r=16 (capacity-fairness variant) ----
for S in 42 43 44; do
  run "Qwen-OLoRA-r16" "$QWEN" outputs/continual_llm/olora/qwen3_8b_r16/seed_$S $S \
    --method olora --lora_r 16 --olora_orth_weight 0.5
done

echo "===== AGGREGATING $(date +%F_%H:%M:%S) ====="
$PY - <<'PYEOF'
import json, statistics, os
def fmt(vals):
    m=statistics.mean(vals); sd=statistics.pstdev(vals) if len(vals)>1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"
def row(name, dirs):
    fin,forg=[],[]
    for d in dirs:
        p=os.path.join(d,"summary.json")
        if not os.path.exists(p): print(f"{name:26} (incomplete: {d})"); return
        s=json.load(open(p)); fin.append(s["average_final_score"]); forg.append(s["average_forgetting"])
    print(f"{name:26} {fmt(fin):18} {fmt(forg):18} {[round(x,4) for x in fin]}")

print("\n### Mistral-7B-v0.3 (3-seed) ###")
print(f"{'Method':26} {'FinalAvg':18} {'AvgForget':18} seeds")
row("LoRA",   [f"outputs/continual_llm/lora/mistral_7b_v03/seed_{s}" for s in (42,43,44)])
row("O-LoRA", [f"outputs/continual_llm/olora/mistral_7b_v03/seed_{s}" for s in (42,43,44)])
row("ReCoLoRA", [f"outputs/continual_llm/hilora/mistral_7b_v03/seed_{s}" for s in (42,43,44)])

print("\n### Qwen3-8B (3-seed) ###")
print(f"{'Method':26} {'FinalAvg':18} {'AvgForget':18} seeds")
row("LoRA",        [f"outputs/continual_llm/lora/qwen3_8b/seed_{s}" for s in (42,43,44)])
row("O-LoRA r=8",  [f"outputs/continual_llm/olora/qwen3_8b/seed_{s}" for s in (42,43,44)])
row("O-LoRA r=16", [f"outputs/continual_llm/olora/qwen3_8b_r16/seed_{s}" for s in (42,43,44)])
row("ReCoLoRA",      [f"outputs/continual_llm_ablation/qwen3_8b/no_dynamic/seed_{s}" for s in (42,43,44)])
PYEOF
echo "===== SUITE ALL DONE $(date +%F_%H:%M:%S) ====="
