#!/usr/bin/env bash
# E1 + E2: complete the main-table baselines to 3 seeds.
#
# E1 (32 runs): the 16 best-rank baseline configs reported in the multi-backbone
#   table, rerun with seeds 43/44 (seed 42 already exists). Each config is invoked
#   through run_peft_rank_sweep_llm.sh so hyperparameters, output paths, and the
#   skip-if-done logic are exactly the ones used for the original seed-42 sweep.
# E2 (6 runs): O-LoRA (r=8, alpha=16, orth 0.5) on InternLM2.5-7B-Chat and
#   Llama-3.1-8B-Instruct, seeds 42/43/44, mirroring the existing Qwen/Mistral
#   O-LoRA runs from run_olora_remaining_suite.sh.
#
# Failures are isolated per config: one crashed run does not stop the rest.
# Estimated total: ~38 runs x 1-2 h on a single RTX 2080 Ti.

cd /home/bhqy/Documents/project/HiLoRA

SWEEP=scripts/run_peft_rank_sweep_llm.sh

echo "########## E1: baseline best-rank configs, seeds 43/44 ##########"
# model_tag           method   rank
E1_CONFIGS=(
  "qwen3_8b            lora     64"
  "qwen3_8b            pissa    64"
  "qwen3_8b            adalora  64"
  "qwen3_8b            dora     128"
  "mistral_7b_v03      lora     64"
  "mistral_7b_v03      dora     64"
  "mistral_7b_v03      pissa    256"
  "mistral_7b_v03      adalora  256"
  "internlm2_5_7b_chat lora     64"
  "internlm2_5_7b_chat pissa    64"
  "internlm2_5_7b_chat adalora  64"
  "internlm2_5_7b_chat dora     128"
  "llama3_1_8b_instruct lora    64"
  "llama3_1_8b_instruct pissa   64"
  "llama3_1_8b_instruct adalora 64"
  "llama3_1_8b_instruct dora    256"
)

# CONDA_ENV must be open_manus (transformers 4.51.3): the control env's 4.45.0
# does not know the qwen3 architecture. The original seed-42 sweep ran in
# open_manus as well (verified from env paths in the seed-42 logs).
for cfg in "${E1_CONFIGS[@]}"; do
  read -r model method rank <<<"$cfg"
  echo "===== E1 config: $model $method r$rank seeds 43 44 $(date +%F_%H:%M:%S) ====="
  CONDA_ENV=open_manus SEEDS="43 44" METHODS="$method" RANKS="$rank" RUN_MODELS="$model" bash "$SWEEP" \
    || echo "[E1-FAIL] $model $method r$rank (continuing)"
done

echo "########## E2: O-LoRA on InternLM2.5 and Llama-3.1, seeds 42/43/44 ##########"
PY=/home/bhqy/anaconda3/envs/open_manus/bin/python
INTERNLM=$(ls -d /home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/*/ | head -1)
LLAMA=$(ls -d /home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TASKS=sst2,mrpc,qnli,rte,qqp,mnli
COMMON="--tasks $TASKS --max_steps_per_task 200 --max_eval_samples_per_task 1000 \
  --learning_rate 2e-4 --lora_alpha 16 --max_length 256 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
  --torch_dtype float16"

run_olora() {  # run_olora <tag> <model_path> <out_dir> <seed>
  local tag=$1 model=$2 out=$3 seed=$4
  if [[ -s "$out/continual_results.json" ]]; then
    echo "[skip] $tag seed $seed exists: $out/continual_results.json"
    return 0
  fi
  mkdir -p "$out"
  echo "===== $tag seed $seed START $(date +%F_%H:%M:%S) ====="
  $PY train_continual_llm.py --model_name_or_path "$model" --output_dir "$out" --seed "$seed" \
    $COMMON --method olora --lora_r 8 --olora_orth_weight 0.5 \
    2>&1 | tee "logs/olora_$(basename "$out" | tr '/' '_')_seed${seed}.log"
  echo "===== $tag seed $seed DONE $(date +%F_%H:%M:%S) ====="
}

for S in 42 43 44; do
  run_olora "InternLM-OLoRA" "$INTERNLM" outputs/continual_llm/olora/internlm2_5_7b_chat/seed_$S $S \
    || echo "[E2-FAIL] internlm seed $S (continuing)"
done
for S in 42 43 44; do
  run_olora "Llama31-OLoRA" "$LLAMA" outputs/continual_llm/olora/llama3_1_8b_instruct/seed_$S $S \
    || echo "[E2-FAIL] llama3.1 seed $S (continuing)"
done

echo "########## Aggregating 3-seed main table ##########"
$PY scripts/aggregate_main_table_3seed.py || python3 scripts/aggregate_main_table_3seed.py
echo "########## ALL DONE $(date +%F_%H:%M:%S) ##########"
