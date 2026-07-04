#!/usr/bin/env python
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

try:
    from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover - PEFT is only required for baseline methods.
    AdaLoraConfig = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hilora import (  # noqa: E402
    HiLoRARecoveryAllocator,
    consolidate_recursive_hilora,
    hilora_cl_regularization,
    hilora_param_groups,
    inject_hilora,
    inject_lora,
    inject_olora,
    inject_recursive_hilora,
    iter_lora_modules,
    olora_advance_task,
    olora_orthogonality_loss,
    olora_param_groups,
    recursive_hilora_param_groups,
    recursive_hilora_slow_anchor_loss,
    reset_hilora_masks,
    set_hilora_stage,
    snapshot_hilora_old_subspace,
    update_hilora_taskwise_elbow,
)


PEFT_METHODS = {"pissa", "adalora", "dora", "qlora"}


TASK_SPECS = {
    "sst2": {
        "fields": ("sentence",),
        "labels": {0: "negative", 1: "positive"},
        "metric": "accuracy",
    },
    "mrpc": {
        "fields": ("sentence1", "sentence2"),
        "labels": {0: "not_equivalent", 1: "equivalent"},
        "metric": "f1",
    },
    "qnli": {
        "fields": ("question", "sentence"),
        "labels": {0: "entailment", 1: "not_entailment"},
        "metric": "accuracy",
    },
    "rte": {
        "fields": ("sentence1", "sentence2"),
        "labels": {0: "entailment", 1: "not_entailment"},
        "metric": "accuracy",
    },
    "qqp": {
        "fields": ("question1", "question2"),
        "labels": {0: "not_duplicate", 1: "duplicate"},
        "metric": "f1",
    },
    "mnli": {
        "fields": ("premise", "hypothesis"),
        "labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "metric": "accuracy",
        "validation_split": "validation_matched",
    },
}


def parse_task_bonus_map(spec: str, default: int = 1) -> Dict[str, int]:
    bonuses = {task: int(default) for task in TASK_SPECS}
    if not spec:
        return bonuses
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid task bonus item: {item}")
        task, value = item.split(":", 1)
        task = task.strip()
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task in bonus map: {task}")
        bonuses[task] = int(value)
    return bonuses


def parse_bonus_map(spec: str) -> Dict[str, int]:
    bonuses: Dict[str, int] = {}
    if not spec:
        return bonuses
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid bonus item: {item}")
        key, value = item.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty bonus key in: {item}")
        bonuses[key] = int(value)
    return bonuses


def parse_int_map(spec: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    if not spec:
        return values
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid int-map item: {item}")
        key, value = item.split(":", 1)
        key = key.strip()
        if key not in TASK_SPECS:
            raise ValueError(f"Unknown task in int-map: {key}")
        values[key] = int(value)
    return values


def parse_float_map(spec: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if not spec:
        return values
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid float-map item: {item}")
        key, value = item.split(":", 1)
        key = key.strip()
        if key not in TASK_SPECS:
            raise ValueError(f"Unknown task in float-map: {key}")
        values[key] = float(value)
    return values


@dataclass
class RawTaskData:
    train: Any
    validation: Any


class PromptDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], task: str) -> None:
        self.rows = list(rows)
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        label_id = int(row["label"])
        return {
            "task": self.task,
            "prompt": build_prompt(self.task, row),
            "answer": TASK_SPECS[self.task]["labels"][label_id],
            "label": label_id,
        }


class TrainCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_masks = []
        labels = []
        for ex in examples:
            prompt_ids = self.tokenizer(ex["prompt"], add_special_tokens=True, truncation=True, max_length=self.max_length)[
                "input_ids"
            ]
            answer_text = " " + ex["answer"]
            answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            ids = prompt_ids + answer_ids
            lab = [-100] * len(prompt_ids) + answer_ids
            if len(ids) > self.max_length:
                overflow = len(ids) - self.max_length
                prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
                prompt_ids = prompt_ids[prompt_trim:]
                ids = prompt_ids + answer_ids
                lab = [-100] * len(prompt_ids) + answer_ids
                ids = ids[-self.max_length :]
                lab = lab[-self.max_length :]
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(ids), dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))

        return pad_batch(input_ids, attention_masks, labels, self.tokenizer.pad_token_id)


class EvalCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return examples


def pad_batch(input_ids, attention_masks, labels, pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x.numel() for x in input_ids)
    batch_ids = []
    batch_masks = []
    batch_labels = []
    for ids, mask, lab in zip(input_ids, attention_masks, labels):
        pad = max_len - ids.numel()
        batch_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        batch_masks.append(torch.cat([mask, torch.zeros(pad, dtype=torch.long)]))
        batch_labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(batch_ids),
        "attention_mask": torch.stack(batch_masks),
        "labels": torch.stack(batch_labels),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential GLUE continual-learning experiment for causal LLMs.")
    parser.add_argument(
        "--method",
        choices=["hilora", "recursive_hilora", "lora", "olora", "pissa", "adalora", "dora", "qlora"],
        default="hilora",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/bhqy/.cache/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/4434a5ffc2582f9d5ac45085043ed3e3264f0a9b",
    )
    parser.add_argument("--tasks", default="sst2,mrpc,qnli,rte")
    parser.add_argument("--output_dir", default="outputs/continual_llm/hilora_internlm_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--epochs_per_task", type=float, default=1.0)
    parser.add_argument("--max_steps_per_task", type=int, default=-1)
    parser.add_argument("--max_train_samples_per_task", type=int, default=None)
    parser.add_argument("--max_eval_samples_per_task", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--residual_lr", type=float, default=1e-4)
    parser.add_argument("--task_max_steps_map", default="")
    parser.add_argument("--task_learning_rate_map", default="")
    parser.add_argument("--task_residual_lr_map", default="")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=True)

    parser.add_argument(
        "--target_modules",
        default="auto",
        help="Comma-separated Linear module name fragments. Use 'auto' for InternLM/Qwen defaults.",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--pissa_niter", type=int, default=4)
    parser.add_argument("--adalora_init_r", type=int, default=24)
    parser.add_argument("--adalora_target_r", type=int, default=16)
    parser.add_argument("--adalora_tinit", type=int, default=0)
    parser.add_argument("--adalora_tfinal", type=int, default=50)
    parser.add_argument("--adalora_delta_t", type=int, default=10)
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5)

    # O-LoRA (Wang et al., EMNLP 2023): one frozen LoRA branch per task + inter-task
    # orthogonality regularization on the down-projection subspaces.
    parser.add_argument("--olora_orth_weight", type=float, default=0.5)

    parser.add_argument("--hilora_max_rank", type=int, default=16)
    parser.add_argument("--hilora_residual_rank", type=int, default=4)
    parser.add_argument("--hilora_energy_threshold", type=float, default=0.8)
    parser.add_argument("--hilora_elbow_ratio", type=float, default=1.5)
    parser.add_argument("--hilora_min_rank", type=int, default=2)
    parser.add_argument("--hilora_stage1_ratio", type=float, default=0.75)
    parser.add_argument("--hilora_svd_oversample", type=int, default=5)
    parser.add_argument("--hilora_svd_n_iter", type=int, default=1)
    parser.add_argument("--hilora_svd_device", default="cuda")
    parser.add_argument("--hilora_force_rank", type=int, default=None)
    parser.add_argument("--hilora_init_strategy", choices=["svd", "random"], default="svd")
    parser.add_argument("--hilora_allocate_full_rank", action="store_true")
    parser.add_argument("--hilora_preserve_inactive_ranks", action="store_true")
    parser.add_argument("--hilora_taskwise_elbow", action="store_true")
    parser.add_argument("--hilora_task_elbow_energy_threshold", type=float, default=0.0)
    parser.add_argument("--hilora_task_min_active_rank", type=int, default=0)
    parser.add_argument("--hilora_task_bonus_map", default="sst2:0,mrpc:1,qnli:1,rte:2,qqp:1,mnli:2")
    parser.add_argument("--hilora_task_bonus_default", type=int, default=1)
    parser.add_argument("--hilora_layer_bonus_map", default="")
    parser.add_argument("--hilora_depth_bonus_map", default="")
    parser.add_argument("--hilora_freeze_old_ranks", action="store_true")
    parser.add_argument("--hilora_old_train_tail", type=int, default=0)
    parser.add_argument("--hilora_old_rank_grad_scale", type=float, default=1.0)
    parser.add_argument("--hilora_anchor_weight", type=float, default=0.0)
    parser.add_argument("--hilora_orth_weight", type=float, default=0.0)
    parser.add_argument("--hilora_min_new_ranks_per_task", type=int, default=0)
    parser.add_argument("--hilora_dynamic_recovery", action="store_true")
    parser.add_argument("--hilora_target_rank_ratio", type=float, default=0.5)
    parser.add_argument("--hilora_recovery_ratio_mode", choices=["kept", "dropped"], default="kept")
    parser.add_argument("--hilora_mask_interval", type=int, default=1)
    parser.add_argument("--hilora_rank_beta1", type=float, default=0.85)
    parser.add_argument("--hilora_rank_beta2", type=float, default=0.85)
    parser.add_argument("--recursive_max_rank", type=int, default=32)
    parser.add_argument("--recursive_min_rank", type=int, default=2)
    parser.add_argument("--recursive_fast_rank_bonus", type=int, default=3)
    parser.add_argument("--recursive_fast_rank_multiplier", type=float, default=1.2)
    parser.add_argument("--recursive_min_fast_rank", type=int, default=4)
    parser.add_argument("--recursive_max_fast_rank", type=int, default=None)
    parser.add_argument("--recursive_energy_threshold", type=float, default=0.9)
    parser.add_argument("--recursive_elbow_ratio", type=float, default=1.5)
    parser.add_argument("--recursive_svd_oversample", type=int, default=5)
    parser.add_argument("--recursive_svd_n_iter", type=int, default=1)
    parser.add_argument("--recursive_svd_device", default="cuda")
    parser.add_argument("--recursive_slow_ramp_start", type=float, default=0.5)
    parser.add_argument("--recursive_slow_ramp_min_scale", type=float, default=0.0)
    parser.add_argument("--recursive_slow_anchor_weight", type=float, default=0.0)
    parser.add_argument("--recursive_consolidate_verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(task: str, row: Dict[str, Any]) -> str:
    if task == "sst2":
        return (
            "Classify the sentiment. Answer with one label from: negative, positive.\n"
            f"Sentence: {row['sentence']}\nAnswer:"
        )
    if task == "mrpc":
        return (
            "Decide whether the two sentences are semantically equivalent. "
            "Answer with one label from: not_equivalent, equivalent.\n"
            f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\nAnswer:"
        )
    if task == "qnli":
        return (
            "Decide whether the sentence answers the question. "
            "Answer with one label from: entailment, not_entailment.\n"
            f"Question: {row['question']}\nSentence: {row['sentence']}\nAnswer:"
        )
    if task == "rte":
        return (
            "Decide whether sentence 1 entails sentence 2. "
            "Answer with one label from: entailment, not_entailment.\n"
            f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\nAnswer:"
        )
    if task == "qqp":
        return (
            "Decide whether the two questions are duplicates. "
            "Answer with one label from: not_duplicate, duplicate.\n"
            f"Question 1: {row['question1']}\nQuestion 2: {row['question2']}\nAnswer:"
        )
    if task == "mnli":
        return (
            "Classify the relation between premise and hypothesis. "
            "Answer with one label from: entailment, neutral, contradiction.\n"
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nAnswer:"
        )
    raise ValueError(f"Unsupported task: {task}")


def load_task_data(args: argparse.Namespace, task: str) -> RawTaskData:
    dataset = load_dataset("glue", task)
    spec = TASK_SPECS[task]
    validation_split = spec.get("validation_split", "validation")
    train = dataset["train"]
    validation = dataset[validation_split]
    if args.max_train_samples_per_task is not None:
        train = train.select(range(min(args.max_train_samples_per_task, len(train))))
    if args.max_eval_samples_per_task is not None:
        validation = validation.select(range(min(args.max_eval_samples_per_task, len(validation))))
    return RawTaskData(train=train, validation=validation)


def f1_score_binary(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def configure_trainable(model: nn.Module, method: str) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if method == "hilora":
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name or "res_middle" in name)
    elif method == "recursive_hilora":
        for name, param in model.named_parameters():
            param.requires_grad_(any(token in name for token in ("slow_", "fast_")))
    elif method == "lora":
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name)
    elif method == "olora":
        # Only the current task branch is trainable; frozen prior-task branches
        # (named with the "frozen" guard) stay in the forward pass but get no grad.
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name and "frozen" not in name)
    elif method in PEFT_METHODS:
        for name, param in model.named_parameters():
            param.requires_grad_(
                any(token in name for token in ("lora_", "ranknum", "lora_magnitude_vector"))
            )


def cast_adapter_parameters_to_fp32(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(token in name for token in ("lora_", "res_middle", "slow_", "fast_", "ranknum", "lora_magnitude_vector")):
            param.data = param.data.float()


def adapter_param_groups(model: nn.Module, args: argparse.Namespace) -> List[Dict[str, Any]]:
    learning_rate = float(getattr(args, "current_learning_rate", args.learning_rate))
    residual_lr = float(getattr(args, "current_residual_lr", args.residual_lr))
    if args.method == "hilora":
        return hilora_param_groups(model, learning_rate, residual_lr)
    if args.method == "recursive_hilora":
        return recursive_hilora_param_groups(model, learning_rate, residual_lr)
    if args.method == "olora":
        return olora_param_groups(model, learning_rate)
    if args.method in PEFT_METHODS:
        params = [param for param in model.parameters() if param.requires_grad]
        return [{"params": params, "lr": learning_rate}] if params else []
    params = []
    for module in iter_lora_modules(model):
        params.extend([module.lora_A, module.lora_B])
    return [{"params": params, "lr": learning_rate}] if params else []


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    adapter = sum(
        param.numel()
        for name, param in model.named_parameters()
        if any(token in name for token in ("lora_", "res_middle", "slow_", "fast_", "ranknum", "lora_magnitude_vector"))
    )
    return {"total_params": total, "trainable_params": trainable, "adapter_params": adapter}


def recursive_slow_lr_scale(completed_steps: int, total_steps: int, args: argparse.Namespace) -> float:
    progress = max(0.0, min(1.0, completed_steps / max(total_steps - 1, 1)))
    start = max(0.0, min(1.0, float(args.recursive_slow_ramp_start)))
    min_scale = max(0.0, min(1.0, float(args.recursive_slow_ramp_min_scale)))
    if progress <= start:
        return min_scale
    local = (progress - start) / max(1e-12, 1.0 - start)
    return min_scale + (1.0 - min_scale) * (math.exp(3.0 * local) - 1.0) / (math.exp(3.0) - 1.0)


def update_recursive_slow_lr(optimizer, completed_steps: int, total_steps: int, args: argparse.Namespace) -> None:
    scale = recursive_slow_lr_scale(completed_steps, total_steps, args)
    for group in optimizer.param_groups:
        if group.get("role") == "recursive_slow":
            group["lr"] = float(group.get("base_lr", group["lr"])) * scale


def optimizer_step(optimizer, scheduler, scaler, args, model) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def maybe_update_adalora(model: nn.Module, method: str, global_step: int) -> None:
    if method != "adalora":
        return
    if hasattr(model, "update_and_allocate"):
        model.update_and_allocate(global_step)


def train_one_task(model, dataloader, task: str, args, device: torch.device) -> Dict[str, Any]:
    model.train()
    configure_trainable(model, args.method)
    if args.method == "hilora":
        set_hilora_stage(model, 1)
        if args.hilora_taskwise_elbow:
            task_rank_stats = update_hilora_taskwise_elbow(
                model,
                task_name=task,
                task_index=getattr(args, "current_task_index", 1),
                rank_bonus=getattr(args, "current_rank_bonus", 0),
                active_rank_floor=args.hilora_task_min_active_rank,
                layer_bonus_map=getattr(args, "current_layer_bonus_map", {}),
                depth_bonus_map=getattr(args, "current_depth_bonus_map", {}),
                freeze_old_ranks=args.hilora_freeze_old_ranks,
                old_train_tail=args.hilora_old_train_tail,
                old_rank_grad_scale=args.hilora_old_rank_grad_scale,
                min_new_ranks=args.hilora_min_new_ranks_per_task,
                energy_threshold=args.hilora_task_elbow_energy_threshold,
                elbow_ratio=args.hilora_elbow_ratio,
                min_rank=args.hilora_min_rank,
                svd_oversample=args.hilora_svd_oversample,
                svd_n_iter=args.hilora_svd_n_iter,
                svd_device=args.hilora_svd_device,
                seed=args.seed,
                verbose=True,
            )
            print(f"[HiLoRA-task-elbow] summary={json.dumps(task_rank_stats, sort_keys=True)}")
        else:
            reset_hilora_masks(model)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(args.epochs_per_task * updates_per_epoch)))
    if args.max_steps_per_task > 0:
        total_steps = min(total_steps, args.max_steps_per_task)
    task_max_steps = int(getattr(args, "current_max_steps_per_task", -1) or -1)
    if task_max_steps > 0:
        total_steps = min(total_steps, task_max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    stage2_step = int(total_steps * args.hilora_stage1_ratio)
    stage2_steps = max(1, total_steps - stage2_step)

    recovery_allocator = None
    recovery_stats: Dict[str, Any] = {}
    if args.method == "hilora" and args.hilora_dynamic_recovery:
        recovery_allocator = HiLoRARecoveryAllocator(
            model,
            target_rank_ratio=args.hilora_target_rank_ratio,
            min_rank=args.hilora_min_rank,
            beta1=args.hilora_rank_beta1,
            beta2=args.hilora_rank_beta2,
            mask_interval=args.hilora_mask_interval,
            recovery_ratio_mode=args.hilora_recovery_ratio_mode,
        )
        recovery_allocator.reset()

    optimizer = torch.optim.AdamW(adapter_param_groups(model, args), weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    if args.method == "recursive_hilora":
        update_recursive_slow_lr(optimizer, 0, total_steps, args)

    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    seen_batches = 0
    running_loss = 0.0
    progress = tqdm(total=total_steps, desc=f"train:{task}", leave=False)

    while completed_steps < total_steps:
        for batch in dataloader:
            if args.method == "hilora" and completed_steps >= stage2_step:
                set_hilora_stage(model, 2)
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                if (
                    args.method == "hilora"
                    and (args.hilora_anchor_weight > 0.0 or args.hilora_orth_weight > 0.0)
                    and int(getattr(args, "current_task_index", 1)) > 1
                ):
                    raw_loss = raw_loss + hilora_cl_regularization(
                        model,
                        anchor_weight=args.hilora_anchor_weight,
                        orth_weight=args.hilora_orth_weight,
                    )
                if args.method == "olora" and args.olora_orth_weight > 0.0:
                    raw_loss = raw_loss + olora_orthogonality_loss(model, args.olora_orth_weight)
                if args.method == "recursive_hilora" and args.recursive_slow_anchor_weight > 0.0:
                    raw_loss = raw_loss + float(args.recursive_slow_anchor_weight) * recursive_hilora_slow_anchor_loss(model)
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            seen_batches += 1
            if seen_batches % args.gradient_accumulation_steps == 0:
                if recovery_allocator is not None and completed_steps >= stage2_step:
                    stats = recovery_allocator.step(completed_steps - stage2_step, stage2_steps)
                    if stats:
                        recovery_stats = stats
                maybe_update_adalora(model, args.method, completed_steps)
                if args.method == "recursive_hilora":
                    update_recursive_slow_lr(optimizer, completed_steps, total_steps, args)
                optimizer_step(optimizer, scheduler, scaler, args, model)
                completed_steps += 1
                if args.method == "recursive_hilora":
                    update_recursive_slow_lr(optimizer, completed_steps, total_steps, args)
                progress.update(1)
                if completed_steps >= total_steps:
                    break
    progress.close()
    if args.method == "hilora":
        set_hilora_stage(model, 2)
    metrics = {"train_loss": running_loss / max(seen_batches, 1), "train_steps": completed_steps}
    if args.method == "hilora" and args.hilora_taskwise_elbow:
        metrics.update({f"task_elbow_{k}": v for k, v in task_rank_stats.items() if isinstance(v, (int, float))})
    metrics.update({f"recovery_{k}": v for k, v in recovery_stats.items()})
    return metrics


@torch.no_grad()
def sequence_loss(model, tokenizer, prompt: str, answer: str, max_length: int, device, fp16: bool) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(ids) > max_length:
        overflow = len(ids) - max_length
        prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
        prompt_ids = prompt_ids[prompt_trim:]
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        ids = ids[-max_length:]
        labels = labels[-max_length:]
    batch = {
        "input_ids": torch.tensor([ids], device=device),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], device=device),
    }
    with torch.cuda.amp.autocast(enabled=fp16):
        loss = model(**batch).loss
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate_task(model, tokenizer, dataloader, task: str, args, device) -> Dict[str, float]:
    model.eval()
    labels_map = TASK_SPECS[task]["labels"]
    preds = []
    labels = []
    total_loss = 0.0
    total_examples = 0
    for examples in tqdm(dataloader, desc=f"eval:{task}", leave=False):
        for ex in examples:
            losses = {
                label_id: sequence_loss(model, tokenizer, ex["prompt"], label_text, args.max_length, device, args.fp16)
                for label_id, label_text in labels_map.items()
            }
            pred = min(losses, key=losses.get)
            preds.append(pred)
            labels.append(int(ex["label"]))
            total_loss += losses[int(ex["label"])]
            total_examples += 1
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    accuracy = float((preds_np == labels_np).mean()) if total_examples else 0.0
    metrics = {"accuracy": accuracy, "loss": total_loss / max(total_examples, 1), "score": accuracy}
    if TASK_SPECS[task]["metric"] == "f1":
        metrics["f1"] = f1_score_binary(preds_np, labels_np)
        metrics["score"] = metrics["f1"]
    return metrics


def compute_forgetting(records: List[Dict[str, Any]], tasks: List[str]) -> Dict[str, Any]:
    by_eval_task = {task: [] for task in tasks}
    for record in records:
        by_eval_task[record["eval_task"]].append(record)
    final_stage = tasks[-1]
    per_task = {}
    for task in tasks:
        task_records = by_eval_task[task]
        final_records = [r for r in task_records if r["stage_task"] == final_stage]
        if not task_records or not final_records:
            continue
        best_score = max(r["score"] for r in task_records)
        final_score = final_records[-1]["score"]
        per_task[task] = {
            "best_score": best_score,
            "final_score": final_score,
            "forgetting": best_score - final_score,
        }
    forgettings = [v["forgetting"] for t, v in per_task.items() if t != final_stage]
    final_scores = [v["final_score"] for v in per_task.values()]
    return {
        "per_task": per_task,
        "average_final_score": float(np.mean(final_scores)) if final_scores else None,
        "average_forgetting": float(np.mean(forgettings)) if forgettings else 0.0,
    }


def write_outputs(output_dir: Path, args, records, train_records, summary, parameter_counts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "records": records,
        "train_records": train_records,
        "summary": summary,
        "parameter_counts": parameter_counts,
    }
    (output_dir / "continual_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (output_dir / "eval_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage_index", "stage_task", "eval_task", "accuracy", "f1", "loss", "score"],
        )
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_from_arg(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def low_cpu_mem_usage_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def infer_target_modules(model: nn.Module, requested: str) -> List[str]:
    if requested != "auto":
        return [name.strip() for name in requested.split(",") if name.strip()]
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if "internlm" in model_type:
        return ["wqkv", "wo"]
    if "qwen" in model_type or "llama" in model_type:
        return ["q_proj", "v_proj"]
    return ["q_proj", "v_proj", "wqkv"]


def build_peft_adapter(model: nn.Module, method: str, target_modules: List[str], args: argparse.Namespace) -> nn.Module:
    if get_peft_model is None or LoraConfig is None:
        raise RuntimeError("PEFT is required for pissa/adalora/dora/qlora baselines.")

    if method == "adalora":
        if AdaLoraConfig is None:
            raise RuntimeError("AdaLoraConfig is unavailable in this PEFT installation.")
        adalora_total_step = args.max_steps_per_task if args.max_steps_per_task > 0 else None
        if adalora_total_step is not None:
            adalora_total_step = max(
                adalora_total_step,
                args.adalora_tfinal + args.adalora_delta_t + 1,
            )
        config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            init_r=args.adalora_init_r,
            target_r=args.adalora_target_r,
            r=args.adalora_init_r,
            lora_alpha=int(args.lora_alpha),
            lora_dropout=args.lora_dropout,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_delta_t,
            total_step=adalora_total_step,
            orth_reg_weight=args.adalora_orth_reg_weight,
            bias="none",
        )
    else:
        init_lora_weights = True
        use_dora = False
        if method == "pissa":
            init_lora_weights = f"pissa_niter_{max(1, args.pissa_niter)}"
        elif method == "dora":
            use_dora = True
        elif method == "qlora":
            init_lora_weights = True
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=int(args.lora_alpha),
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=init_lora_weights,
            use_dora=use_dora,
        )

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for task in tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"Unsupported task: {task}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_data = {task: load_task_data(args, task) for task in tasks}
    train_datasets = {task: PromptDataset(task_data[task].train, task) for task in tasks}
    eval_datasets = {task: PromptDataset(task_data[task].validation, task) for task in tasks}

    quantization_config = None
    if args.method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_from_arg(args.torch_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_arg(args.torch_dtype) if quantization_config is None else None,
        low_cpu_mem_usage=low_cpu_mem_usage_available(),
        quantization_config=quantization_config,
    )
    if args.method == "qlora":
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("PEFT prepare_model_for_kbit_training is required for QLoRA.")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad_(False)

    target_modules = infer_target_modules(model, args.target_modules)
    print(f"[targets] {target_modules}")

    if args.method == "hilora":
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
            allocate_full_rank=args.hilora_allocate_full_rank or args.hilora_taskwise_elbow,
            preserve_inactive_ranks=args.hilora_preserve_inactive_ranks,
            verbose=True,
        )
    elif args.method == "recursive_hilora":
        inject_recursive_hilora(
            model,
            target_modules=target_modules,
            max_rank=args.recursive_max_rank,
            min_rank=args.recursive_min_rank,
            fast_rank_bonus=args.recursive_fast_rank_bonus,
            fast_rank_multiplier=args.recursive_fast_rank_multiplier,
            min_fast_rank=args.recursive_min_fast_rank,
            max_fast_rank=args.recursive_max_fast_rank,
            energy_threshold=args.recursive_energy_threshold,
            elbow_ratio=args.recursive_elbow_ratio,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            svd_oversample=args.recursive_svd_oversample,
            svd_n_iter=args.recursive_svd_n_iter,
            svd_device=args.recursive_svd_device,
            seed=args.seed,
            verbose=True,
        )
    elif args.method == "olora":
        inject_olora(
            model,
            target_modules=target_modules,
            rank=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            verbose=True,
        )
    else:
        if args.method in PEFT_METHODS:
            model = build_peft_adapter(model, args.method, target_modules, args)
        else:
            inject_lora(
                model,
                target_modules=target_modules,
                rank=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                verbose=True,
            )

    cast_adapter_parameters_to_fp32(model)
    if args.method != "qlora":
        model.to(device)
    configure_trainable(model, args.method)
    parameter_counts = count_parameters(model)
    print(json.dumps(parameter_counts, indent=2, sort_keys=True))

    train_collator = TrainCollator(tokenizer, args.max_length)
    eval_collator = EvalCollator(tokenizer, args.max_length)
    eval_loaders = {
        task: DataLoader(
            eval_datasets[task],
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_collator,
        )
        for task in tasks
    }

    records = []
    train_records = []
    task_bonus_map = parse_task_bonus_map(args.hilora_task_bonus_map, args.hilora_task_bonus_default)
    layer_bonus_map = parse_bonus_map(args.hilora_layer_bonus_map)
    depth_bonus_map = parse_bonus_map(args.hilora_depth_bonus_map)
    task_max_steps_map = parse_int_map(args.task_max_steps_map)
    task_lr_map = parse_float_map(args.task_learning_rate_map)
    task_residual_lr_map = parse_float_map(args.task_residual_lr_map)
    for stage_index, task in enumerate(tasks, start=1):
        args.current_max_steps_per_task = task_max_steps_map.get(task, -1)
        args.current_learning_rate = task_lr_map.get(task, args.learning_rate)
        args.current_residual_lr = task_residual_lr_map.get(task, args.residual_lr)
        if (
            args.current_max_steps_per_task > 0
            or args.current_learning_rate != args.learning_rate
            or args.current_residual_lr != args.residual_lr
        ):
            print(
                f"[task-override] task={task} "
                f"max_steps={args.current_max_steps_per_task} "
                f"lr={args.current_learning_rate} residual_lr={args.current_residual_lr}"
            )
        if args.method == "hilora" and args.hilora_taskwise_elbow:
            args.current_task_index = stage_index
            args.current_rank_bonus = sum(task_bonus_map[t] for t in tasks[:stage_index])
            args.current_layer_bonus_map = layer_bonus_map
            args.current_depth_bonus_map = depth_bonus_map
            print(
                f"[HiLoRA-task-elbow] task={task} stage={stage_index} "
                f"cumulative_rank_bonus={args.current_rank_bonus} "
                f"rank_floor={args.hilora_task_min_active_rank} "
                f"freeze_old_ranks={args.hilora_freeze_old_ranks} "
                f"old_train_tail={args.hilora_old_train_tail} "
                f"old_rank_grad_scale={args.hilora_old_rank_grad_scale} "
                f"anchor_weight={args.hilora_anchor_weight} "
                f"orth_weight={args.hilora_orth_weight} "
                f"min_new_ranks={args.hilora_min_new_ranks_per_task} "
                f"layer_bonus={json.dumps(layer_bonus_map, sort_keys=True)} "
                f"depth_bonus={json.dumps(depth_bonus_map, sort_keys=True)}"
            )
        if args.method == "olora" and stage_index > 1:
            advance_stats = olora_advance_task(model)
            cast_adapter_parameters_to_fp32(model)
            configure_trainable(model, args.method)
            print(f"[O-LoRA] advance task={task} stage={stage_index} {json.dumps(advance_stats, sort_keys=True)}")
        release_cuda_cache()
        train_loader = DataLoader(
            train_datasets[task],
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=train_collator,
        )
        train_metrics = train_one_task(model, train_loader, task, args, device)
        del train_loader
        release_cuda_cache()

        if args.method == "recursive_hilora":
            consolidation_stats = consolidate_recursive_hilora(
                model,
                max_rank=args.recursive_max_rank,
                min_rank=args.recursive_min_rank,
                fast_rank_bonus=args.recursive_fast_rank_bonus,
                fast_rank_multiplier=args.recursive_fast_rank_multiplier,
                min_fast_rank=args.recursive_min_fast_rank,
                max_fast_rank=args.recursive_max_fast_rank,
                energy_threshold=args.recursive_energy_threshold,
                elbow_ratio=args.recursive_elbow_ratio,
                svd_oversample=args.recursive_svd_oversample,
                svd_n_iter=args.recursive_svd_n_iter,
                svd_device=args.recursive_svd_device,
                seed=args.seed + stage_index,
                verbose=args.recursive_consolidate_verbose,
            )
            train_metrics.update({f"consolidate_{k}": v for k, v in consolidation_stats.items()})
            print(f"[RecursiveHiLoRA] consolidate={json.dumps(consolidation_stats, sort_keys=True)}")
            cast_adapter_parameters_to_fp32(model)
            configure_trainable(model, args.method)
            release_cuda_cache()

        train_records.append({"stage_index": stage_index, "task": task, **train_metrics})

        for eval_task in tasks[:stage_index]:
            metrics = evaluate_task(model, tokenizer, eval_loaders[eval_task], eval_task, args, device)
            record = {
                "stage_index": stage_index,
                "stage_task": task,
                "eval_task": eval_task,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "loss": metrics.get("loss"),
                "score": metrics.get("score"),
            }
            records.append(record)
            print(f"[eval] stage={task} eval={eval_task} acc={record['accuracy']:.4f} score={record['score']:.4f}")
        release_cuda_cache()
        if args.method == "hilora" and args.hilora_taskwise_elbow and (
            args.hilora_anchor_weight > 0.0 or args.hilora_orth_weight > 0.0
        ):
            snapshot_stats = snapshot_hilora_old_subspace(model)
            print(f"[HiLoRA-CL] snapshot={json.dumps(snapshot_stats, sort_keys=True)}")

    summary = compute_forgetting(records, tasks)
    write_outputs(Path(args.output_dir), args, records, train_records, summary, parameter_counts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
