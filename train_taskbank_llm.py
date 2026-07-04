#!/usr/bin/env python
"""Oracle task-bank continual experiment for low-rank adapters.

Each task gets an independent adapter branch. Older branches are never updated;
evaluation uses the oracle task id to route to the corresponding branch. This
is the minimal experiment that tests whether adapter isolation eliminates
catastrophic forgetting before adding a learned router.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_continual_llm as base
from hilora import inject_hilora, inject_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiLoRA oracle task-bank experiment.")
    parser.add_argument("--branch_method", choices=["hilora", "lora", "pissa"], default="hilora")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tasks", default="sst2,mrpc,qnli,rte,qqp,mnli")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--epochs_per_task", type=float, default=1.0)
    parser.add_argument("--max_steps_per_task", type=int, default=200)
    parser.add_argument("--max_train_samples_per_task", type=int, default=None)
    parser.add_argument("--max_eval_samples_per_task", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--residual_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--target_modules", default="auto")
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--pissa_niter", type=int, default=4)
    parser.add_argument("--hilora_max_rank", type=int, default=32)
    parser.add_argument("--hilora_residual_rank", type=int, default=4)
    parser.add_argument("--hilora_energy_threshold", type=float, default=0.9)
    parser.add_argument("--hilora_elbow_ratio", type=float, default=1.5)
    parser.add_argument("--hilora_min_rank", type=int, default=2)
    parser.add_argument("--hilora_stage1_ratio", type=float, default=0.75)
    parser.add_argument("--hilora_svd_oversample", type=int, default=5)
    parser.add_argument("--hilora_svd_n_iter", type=int, default=1)
    parser.add_argument("--hilora_svd_device", default="cuda")
    parser.add_argument("--hilora_force_rank", type=int, default=None)
    parser.add_argument("--hilora_init_strategy", choices=["svd", "random"], default="svd")
    parser.add_argument("--hilora_task_min_active_rank", type=int, default=6)
    parser.add_argument("--hilora_task_elbow_energy_threshold", type=float, default=0.0)
    parser.add_argument("--hilora_task_bonus_map", default="sst2:1,mrpc:1,qnli:2,rte:2,qqp:1,mnli:2")
    parser.add_argument("--hilora_task_bonus_default", type=int, default=1)
    parser.add_argument("--hilora_layer_bonus_map", default="q_proj:0,v_proj:1")
    parser.add_argument("--hilora_depth_bonus_map", default="early:0,mid:0,late:1")

    # Fields consumed by train_continual_llm helpers.
    parser.set_defaults(
        hilora_taskwise_elbow=True,
        hilora_allocate_full_rank=True,
        hilora_preserve_inactive_ranks=True,
        hilora_freeze_old_ranks=False,
        hilora_old_train_tail=0,
        hilora_old_rank_grad_scale=1.0,
        hilora_anchor_weight=0.0,
        hilora_orth_weight=0.0,
        hilora_min_new_ranks_per_task=0,
        hilora_dynamic_recovery=False,
        hilora_target_rank_ratio=0.5,
        hilora_recovery_ratio_mode="kept",
        hilora_mask_interval=1,
        hilora_rank_beta1=0.85,
        hilora_rank_beta2=0.85,
        task_max_steps_map="",
        task_learning_rate_map="",
        task_residual_lr_map="",
    )
    return parser.parse_args()


def build_hilora_model(args: argparse.Namespace, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=base.dtype_from_arg(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    target_modules = base.infer_target_modules(model, args.target_modules)
    print(f"[TaskBank] targets={target_modules}")
    inject_hilora(
        model,
        target_modules=target_modules,
        max_rank=args.hilora_max_rank,
        residual_rank=args.hilora_residual_rank,
        energy_threshold=args.hilora_energy_threshold,
        elbow_ratio=args.hilora_elbow_ratio,
        min_rank=args.hilora_min_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        svd_oversample=args.hilora_svd_oversample,
        svd_n_iter=args.hilora_svd_n_iter,
        svd_device=args.hilora_svd_device,
        seed=args.seed,
        force_rank=args.hilora_force_rank,
        init_strategy=args.hilora_init_strategy,
        allocate_full_rank=True,
        preserve_inactive_ranks=True,
        verbose=True,
    )
    base.cast_adapter_parameters_to_fp32(model)
    model.to(device)
    base.configure_trainable(model, "hilora")
    return model


def build_branch_model(args: argparse.Namespace, device: torch.device):
    if args.branch_method == "hilora":
        args.method = "hilora"
        return build_hilora_model(args, device)

    args.method = args.branch_method
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=base.dtype_from_arg(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    target_modules = base.infer_target_modules(model, args.target_modules)
    print(f"[TaskBank] targets={target_modules}")
    if args.branch_method == "lora":
        inject_lora(
            model,
            target_modules=target_modules,
            rank=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            verbose=True,
        )
    elif args.branch_method == "pissa":
        if get_peft_model is None:
            raise RuntimeError("PEFT is required for PiSSA TaskBank.")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=int(args.lora_alpha),
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=f"pissa_niter_{max(1, args.pissa_niter)}",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unsupported branch method: {args.branch_method}")
    base.cast_adapter_parameters_to_fp32(model)
    model.to(device)
    base.configure_trainable(model, args.method)
    return model


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for task in tasks:
        if task not in base.TASK_SPECS:
            raise ValueError(f"Unsupported task: {task}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_data = {task: base.load_task_data(args, task) for task in tasks}
    train_datasets = {task: base.PromptDataset(task_data[task].train, task) for task in tasks}
    eval_datasets = {task: base.PromptDataset(task_data[task].validation, task) for task in tasks}
    train_collator = base.TrainCollator(tokenizer, args.max_length)
    eval_collator = base.EvalCollator(tokenizer, args.max_length)
    eval_loaders = {
        task: DataLoader(
            eval_datasets[task],
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_collator,
        )
        for task in tasks
    }

    task_bonus_map = base.parse_task_bonus_map(args.hilora_task_bonus_map, args.hilora_task_bonus_default)
    layer_bonus_map = base.parse_bonus_map(args.hilora_layer_bonus_map)
    depth_bonus_map = base.parse_bonus_map(args.hilora_depth_bonus_map)

    branch_scores: Dict[str, Dict[str, float]] = {}
    train_records: List[Dict[str, Any]] = []
    parameter_counts = None

    for stage_index, task in enumerate(tasks, start=1):
        print(f"[TaskBank] train branch method={args.branch_method} stage={stage_index} task={task}")
        base.set_seed(args.seed + stage_index)
        model = build_branch_model(args, device)
        if parameter_counts is None:
            parameter_counts = base.count_parameters(model)
            print(json.dumps(parameter_counts, indent=2, sort_keys=True))

        args.current_task_index = 1
        args.current_rank_bonus = task_bonus_map[task]
        args.current_layer_bonus_map = layer_bonus_map
        args.current_depth_bonus_map = depth_bonus_map
        args.current_max_steps_per_task = -1
        args.current_learning_rate = args.learning_rate
        args.current_residual_lr = args.residual_lr

        train_loader = DataLoader(
            train_datasets[task],
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=train_collator,
        )
        train_metrics = base.train_one_task(model, train_loader, task, args, device)
        del train_loader
        base.release_cuda_cache()

        metrics = base.evaluate_task(model, tokenizer, eval_loaders[task], task, args, device)
        branch_scores[task] = metrics
        train_records.append({"stage_index": stage_index, "task": task, **train_metrics})
        print(f"[TaskBank] branch_eval task={task} acc={metrics.get('accuracy'):.4f} score={metrics.get('score'):.4f}")

        del model
        base.release_cuda_cache()

    records: List[Dict[str, Any]] = []
    for stage_index, stage_task in enumerate(tasks, start=1):
        for eval_task in tasks[:stage_index]:
            metrics = branch_scores[eval_task]
            records.append(
                {
                    "stage_index": stage_index,
                    "stage_task": stage_task,
                    "eval_task": eval_task,
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "loss": metrics.get("loss"),
                    "score": metrics.get("score"),
                }
            )

    summary = base.compute_forgetting(records, tasks)
    payload_counts = parameter_counts or {"total_params": 0, "trainable_params": 0, "adapter_params": 0}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_outputs(output_dir, args, records, train_records, summary, payload_counts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
