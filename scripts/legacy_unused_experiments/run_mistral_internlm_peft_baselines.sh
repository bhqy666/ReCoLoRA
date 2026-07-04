#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-open_manus}
TASKS=${TASKS:-sst2,mrpc,qnli,rte,qqp,mnli}
SEEDS=${SEEDS:-42 43 44}
METHODS=${METHODS:-lora pissa adalora qlora dora}
OUT_ROOT=${OUT_ROOT:-outputs/continual_llm}
LOG_DIR=${LOG_DIR:-logs/peft_baselines_mistral_internlm}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-200}
MAX_EVAL_SAMPLES_PER_TASK=${MAX_EVAL_SAMPLES_PER_TASK:-1000}
MAX_LENGTH=${MAX_LENGTH:-256}
GRAD_ACCUM=${GRAD_ACCUM:-8}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$LOG_DIR"

run_one() {
  local model_tag="$1"
  local model_path="$2"
  local target_modules="$3"

  for method in $METHODS; do
    for seed in $SEEDS; do
      local out_dir="$OUT_ROOT/$method/$model_tag/seed_$seed"
      local done_file="$out_dir/continual_results.json"
      if [[ -f "$done_file" ]]; then
        echo "[skip] model=$model_tag method=$method seed=$seed exists: $done_file"
        continue
      fi

      echo "[run] model=$model_tag method=$method seed=$seed out=$out_dir $(date --iso-8601=seconds)"
      common_args=(
        --method "$method"
        --model_name_or_path "$model_path"
        --tasks "$TASKS"
        --output_dir "$out_dir"
        --seed "$seed"
        --torch_dtype float16
        --max_length "$MAX_LENGTH"
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 4
        --gradient_accumulation_steps "$GRAD_ACCUM"
        --max_steps_per_task "$MAX_STEPS_PER_TASK"
        --max_eval_samples_per_task "$MAX_EVAL_SAMPLES_PER_TASK"
        --learning_rate 2e-4
        --residual_lr 1e-4
        --target_modules "$target_modules"
        --gradient_checkpointing
        --lora_r 16
        --lora_alpha 16
      )

      extra_args=()
      case "$method" in
        pissa)
          extra_args=(--pissa_niter 4)
          ;;
        adalora)
          extra_args=(--adalora_init_r 24 --adalora_target_r 16 --adalora_tinit 0 --adalora_tfinal 50 --adalora_delta_t 10)
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
        "${common_args[@]}" "${extra_args[@]}" 2>&1 | tee "$LOG_DIR/${model_tag}_${method}_seed_${seed}.log"
    done
  done
}

RUN_MODELS=${RUN_MODELS:-mistral_7b_v03,internlm2_5_7b_chat}

if [[ ",${RUN_MODELS}," == *",mistral_7b_v03,"* ]]; then
  run_one \
    mistral_7b_v03 \
    /home/bhqy/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    q_proj,v_proj
fi

if [[ ",${RUN_MODELS}," == *",internlm2_5_7b_chat,"* ]]; then
  run_one \
    internlm2_5_7b_chat \
    /home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/9b8d9553846ecf6393f3408fa9d3ec9928fdab4d \
    auto
fi
