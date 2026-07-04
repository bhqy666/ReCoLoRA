#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONDA_ENV=${CONDA_ENV:-control}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42}
RANKS=${RANKS:-64 128 256}
METHODS=${METHODS:-lora pissa adalora qlora dora}
RUN_MODELS=${RUN_MODELS:-mistral_7b_v03,internlm2_5_7b_chat,qwen3_8b,llama3_1_8b_instruct}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm_rank_sweep}
LOG_DIR=${LOG_DIR:-logs/peft_rank_sweep}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
MAX_LENGTH=${MAX_LENGTH:-256}
GRAD_ACCUM=${GRAD_ACCUM:-8}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
FORCE_RERUN=${FORCE_RERUN:-0}

mkdir -p "$LOG_DIR"

run_one_model() {
  local model_tag="$1"
  local model_path="$2"
  local target_modules="$3"

  for rank in $RANKS; do
    for method in $METHODS; do
      for seed in $SEEDS; do
        local out_dir="$OUT_ROOT/$method/${model_tag}_r${rank}/seed_$seed"
        local done_file="$out_dir/continual_results.json"
        if [[ "$FORCE_RERUN" != "1" && -s "$done_file" ]]; then
          echo "[skip] model=$model_tag method=$method rank=$rank seed=$seed exists: $done_file"
          continue
        fi

        local lora_alpha="$rank"
        local adalora_init_r=$((rank + rank / 2))

        echo "[run] model=$model_tag method=$method rank=$rank seed=$seed out=$out_dir $(date --iso-8601=seconds)"
        common_args=(
          --method "$method"
          --model_name_or_path "$model_path"
          --tasks "$TASKS"
          --output_dir "$out_dir"
          --seed "$seed"
          --torch_dtype float16
          --max_length "$MAX_LENGTH"
          --per_device_train_batch_size 1
          --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
          --gradient_accumulation_steps "$GRAD_ACCUM"
          --max_steps_per_task "$MAX_STEPS_PER_TASK"
          --max_eval_samples_per_task "$MAX_EVAL_SAMPLES_PER_TASK"
          --learning_rate 2e-4
          --residual_lr 1e-4
          --target_modules "$target_modules"
          --gradient_checkpointing
          --lora_r "$rank"
          --lora_alpha "$lora_alpha"
        )

        extra_args=()
        case "$method" in
          pissa)
            extra_args=(--pissa_niter 4)
            ;;
          adalora)
            extra_args=(
              --adalora_init_r "$adalora_init_r"
              --adalora_target_r "$rank"
              --adalora_tinit 0
              --adalora_tfinal 50
              --adalora_delta_t 10
            )
            ;;
          lora|qlora|dora)
            ;;
          *)
            echo "unknown method: $method" >&2
            exit 2
            ;;
        esac

        env \
          -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
          -u ALL_PROXY -u all_proxy \
          HF_ENDPOINT="$HF_ENDPOINT" \
          HF_DATASETS_OFFLINE=1 \
          HF_HUB_ENABLE_HF_TRANSFER=0 \
          PYTHONUNBUFFERED=1 \
          conda run --no-capture-output -n "$CONDA_ENV" python -u train_continual_llm.py \
          "${common_args[@]}" "${extra_args[@]}" 2>&1 | tee "$LOG_DIR/${model_tag}_${method}_r${rank}_seed_${seed}.log"
      done
    done
  done
}

if [[ ",${RUN_MODELS}," == *",mistral_7b_v03,"* ]]; then
  run_one_model \
    mistral_7b_v03 \
    /home/bhqy/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",internlm2_5_7b_chat,"* ]]; then
  run_one_model \
    internlm2_5_7b_chat \
    /home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/9b8d9553846ecf6393f3408fa9d3ec9928fdab4d \
    auto
fi

if [[ ",${RUN_MODELS}," == *",qwen3_8b,"* ]]; then
  run_one_model \
    qwen3_8b \
    /home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",llama3_1_8b_instruct,"* ]]; then
  run_one_model \
    llama3_1_8b_instruct \
    /home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",llama3_8b,"* ]]; then
  run_one_model \
    llama3_8b \
    /home/bhqy/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061 \
    q_proj,v_proj
fi
