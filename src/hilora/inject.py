import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .lora import HiLoRALinear
from .svd import randomized_svd, select_rank


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    for t in targets:
        if not t:
            continue
        if t in name:
            return True
    return False


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_hilora(
    model: nn.Module,
    target_modules: Iterable[str],
    max_rank: int = 16,
    residual_rank: int = 4,
    energy_threshold: float = 0.9,
    elbow_ratio: float = 1.5,
    min_rank: int = 2,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    residual_alpha: Optional[float] = None,
    svd_oversample: int = 5,
    svd_n_iter: int = 1,
    svd_device: str = "cpu",
    seed: Optional[int] = None,
    force_rank: Optional[int] = None,
    init_strategy: str = "svd",
    allocate_full_rank: bool = False,
    preserve_inactive_ranks: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Replace target Linear modules with HiLoRA adapters.

    Returns list of replaced module names.
    """

    replaced: List[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, target_modules):
            continue

        weight = module.weight.data
        w_device = weight.device
        w_dtype = weight.dtype

        w_cpu = weight.detach().to(device=svd_device, dtype=torch.float32)

        rank_svd = min(max_rank + max(0, residual_rank), min(w_cpu.shape))
        u, s, vh = randomized_svd(
            w_cpu,
            rank=rank_svd,
            oversample=svd_oversample,
            n_iter=svd_n_iter,
            seed=seed,
        )
        # Energy budget = energy within the truncated spectrum actually
        # computed (top `rank_svd` singular values), not the full matrix:
        # for large LLM weight matrices the top-`max_rank` energy is a
        # vanishing fraction of the full-matrix energy, so `energy_threshold`
        # would never bind against a full-matrix denominator.
        fro_norm_sq = float(torch.sum(s * s))

        if force_rank is not None:
            r_main = max(min_rank, min(int(force_rank), max_rank))
        else:
            r_main = select_rank(
                s[: max_rank],
                fro_norm_sq=fro_norm_sq,
                energy_threshold=energy_threshold,
                elbow_ratio=elbow_ratio,
                min_rank=min_rank,
                max_rank=max_rank,
            )
        r_res = int(residual_rank)
        r_alloc = max_rank if allocate_full_rank else r_main
        r_alloc = max(r_main, min(int(r_alloc), min(w_cpu.shape)))

        # Principal component kept out of the frozen base. In
        # preserve-inactive mode, all allocated SVD directions remain in the
        # forward pass, so the base must remove the full allocated subspace to
        # keep the injected layer function-preserving at initialization.
        base_rank = r_alloc if preserve_inactive_ranks else r_main
        u_main = u[:, :base_rank]
        s_main = s[:base_rank]
        vh_main = vh[:base_rank, :]
        w_p = (u_main * s_main) @ vh_main
        w_res = w_cpu - w_p

        new_module = HiLoRALinear(
            module,
            r_main=r_alloc,
            r_res=r_res,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            residual_alpha=residual_alpha,
        )
        new_module.preserve_inactive_ranks = bool(preserve_inactive_ranks)

        new_module = new_module.to(device=w_device, dtype=w_dtype)

        if init_strategy not in {"svd", "random"}:
            raise ValueError("init_strategy must be 'svd' or 'random'")

        with torch.no_grad():
            if init_strategy == "svd":
                new_module.base_layer.weight.copy_(w_res.to(device=w_device, dtype=w_dtype))
            else:
                new_module.base_layer.weight.copy_(weight.to(device=w_device, dtype=w_dtype))
            if module.bias is not None and new_module.base_layer.bias is not None:
                new_module.base_layer.bias.copy_(module.bias.data)

        if init_strategy == "svd":
            new_module.init_main_from_svd(
                u[:, :r_alloc].to(device=w_device, dtype=w_dtype),
                s[:r_alloc].to(device=w_device, dtype=w_dtype),
                vh[:r_alloc, :].to(device=w_device, dtype=w_dtype),
                rank_main=r_alloc,
            )
            new_module.set_active_rank(r_main)
        elif new_module.lora_A is not None and new_module.lora_B is not None:
            with torch.no_grad():
                nn.init.kaiming_uniform_(new_module.lora_A, a=math.sqrt(5))
                new_module.lora_B.zero_()
            new_module.set_active_rank(r_main)

        if new_module.res_middle is not None:
            start = r_alloc
            end = min(start + r_res, u.shape[1])
            left_basis = u[:, start:end]
            right_basis = vh[start:end, :]
            if left_basis.shape[1] == r_res:
                new_module.set_residual_basis(
                    left_basis.to(device=w_device, dtype=w_dtype),
                    right_basis.to(device=w_device, dtype=w_dtype),
                )
            else:
                new_module.set_residual_basis(
                    torch.zeros(new_module.out_features, r_res, device=w_device, dtype=w_dtype),
                    torch.zeros(r_res, new_module.in_features, device=w_device, dtype=w_dtype),
                )

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, new_module)
        replaced.append(name)

        if verbose:
            print(
                f"[HiLoRA] Replaced {name} with r_main={r_main}, "
                f"r_capacity={r_alloc}, r_res={r_res}, "
                f"preserve_inactive={new_module.preserve_inactive_ranks}"
            )

    if verbose and not replaced:
        print("[HiLoRA] No target modules matched.")

    return replaced


def iter_hilora_modules(model: nn.Module) -> List[HiLoRALinear]:
    modules: List[HiLoRALinear] = []
    for module in model.modules():
        if isinstance(module, HiLoRALinear):
            modules.append(module)
    return modules


def iter_named_hilora_modules(model: nn.Module) -> List[Tuple[str, HiLoRALinear]]:
    modules: List[Tuple[str, HiLoRALinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, HiLoRALinear):
            modules.append((name, module))
    return modules


def set_hilora_stage(model: nn.Module, stage: int) -> None:
    for module in iter_hilora_modules(model):
        module.set_stage(stage)


def reset_hilora_masks(model: nn.Module) -> None:
    for module in iter_hilora_modules(model):
        module.reset_rank_mask()
        module.set_recovery_ratio(1.0)


def snapshot_hilora_old_subspace(model: nn.Module) -> dict:
    """Save the currently active principal ranks as anchors for later tasks."""
    snapshots = 0
    active_ranks: List[int] = []
    for module in iter_hilora_modules(model):
        if module.lora_A is None or module.lora_B is None or module.r_main <= 0:
            continue
        rank = max(0, min(int(getattr(module, "current_active_rank", module.r_main)), module.r_main))
        if rank <= 0:
            continue
        module.hilora_anchor_rank = rank
        module.hilora_anchor_A = module.lora_A.detach()[:rank].float().cpu().clone()
        module.hilora_anchor_B = module.lora_B.detach()[:, :rank].float().cpu().clone()
        snapshots += 1
        active_ranks.append(rank)
    return {
        "snapshots": snapshots,
        "mean_anchor_rank": float(sum(active_ranks) / len(active_ranks)) if active_ranks else 0.0,
        "max_anchor_rank": int(max(active_ranks)) if active_ranks else 0,
    }


def hilora_cl_regularization(
    model: nn.Module,
    anchor_weight: float = 0.0,
    orth_weight: float = 0.0,
) -> torch.Tensor:
    """Anchor old ranks and keep newly opened ranks away from old subspaces."""
    device = None
    dtype = None
    anchor_terms: List[torch.Tensor] = []
    orth_terms: List[torch.Tensor] = []
    for module in iter_hilora_modules(model):
        if module.lora_A is None or module.lora_B is None or module.r_main <= 0:
            continue
        if device is None:
            device = module.lora_A.device
            dtype = module.lora_A.dtype
        anchor_rank = int(getattr(module, "hilora_anchor_rank", 0) or 0)
        anchor_a = getattr(module, "hilora_anchor_A", None)
        anchor_b = getattr(module, "hilora_anchor_B", None)
        if anchor_rank <= 0 or anchor_a is None or anchor_b is None:
            continue
        anchor_rank = max(0, min(anchor_rank, module.r_main, module.lora_A.shape[0], module.lora_B.shape[1]))
        active_rank = max(0, min(int(getattr(module, "current_active_rank", module.r_main)), module.r_main))
        old_rank = min(anchor_rank, active_rank)
        if old_rank <= 0:
            continue
        if anchor_weight > 0:
            ref_a = anchor_a[:old_rank].to(device=module.lora_A.device, dtype=module.lora_A.dtype)
            ref_b = anchor_b[:, :old_rank].to(device=module.lora_B.device, dtype=module.lora_B.dtype)
            anchor_terms.append((module.lora_A[:old_rank] - ref_a).pow(2).mean())
            anchor_terms.append((module.lora_B[:, :old_rank] - ref_b).pow(2).mean())
        if orth_weight > 0 and active_rank > old_rank:
            old_a = module.lora_A[:old_rank]
            new_a = module.lora_A[old_rank:active_rank]
            old_b = module.lora_B[:, :old_rank]
            new_b = module.lora_B[:, old_rank:active_rank]
            if new_a.numel() > 0 and new_b.numel() > 0:
                old_a_n = torch.nn.functional.normalize(old_a.float(), dim=1)
                new_a_n = torch.nn.functional.normalize(new_a.float(), dim=1)
                old_b_n = torch.nn.functional.normalize(old_b.float(), dim=0)
                new_b_n = torch.nn.functional.normalize(new_b.float(), dim=0)
                orth_terms.append((new_a_n @ old_a_n.T).pow(2).mean().to(module.lora_A.dtype))
                orth_terms.append((old_b_n.T @ new_b_n).pow(2).mean().to(module.lora_B.dtype))
    if device is None:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    reg = torch.zeros((), device=device, dtype=dtype)
    if anchor_terms:
        reg = reg + float(anchor_weight) * torch.stack(anchor_terms).mean()
    if orth_terms:
        reg = reg + float(orth_weight) * torch.stack(orth_terms).mean()
    return reg


def update_hilora_taskwise_elbow(
    model: nn.Module,
    task_name: str,
    task_index: int,
    rank_bonus: int = 0,
    active_rank_floor: int = 0,
    layer_bonus_map: Optional[Dict[str, int]] = None,
    depth_bonus_map: Optional[Dict[str, int]] = None,
    freeze_old_ranks: bool = False,
    old_train_tail: int = 0,
    old_rank_grad_scale: float = 1.0,
    min_new_ranks: int = 0,
    energy_threshold: float = 0.0,
    elbow_ratio: float = 1.5,
    min_rank: int = 2,
    svd_oversample: int = 5,
    svd_n_iter: int = 1,
    svd_device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Recompute an elbow rank at a task boundary and grow active ranks.

    The module capacity is fixed at injection time. This function only changes
    the active-rank mask, so old adapter slots are preserved and new slots can
    be opened for later tasks.
    """
    named_modules = list(iter_named_hilora_modules(model))
    layer_indices = []
    for name, _ in named_modules:
        match = re.search(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)", name)
        if match:
            layer_indices.append(int(match.group(1)))
    max_layer_index = max(layer_indices) if layer_indices else None

    def name_bonus(module_name: str) -> int:
        if not layer_bonus_map:
            return 0
        bonus = 0
        for pattern, value in layer_bonus_map.items():
            if pattern and pattern in module_name:
                bonus += int(value)
        return bonus

    def depth_bonus(module_name: str) -> int:
        if not depth_bonus_map or max_layer_index is None:
            return 0
        match = re.search(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)", module_name)
        if not match:
            return 0
        idx = int(match.group(1))
        denom = max(1, max_layer_index + 1)
        frac = (idx + 1) / denom
        if frac <= 1.0 / 3.0:
            bucket = "early"
        elif frac <= 2.0 / 3.0:
            bucket = "mid"
        else:
            bucket = "late"
        return int(depth_bonus_map.get(bucket, 0))

    active_ranks: List[int] = []
    elbow_ranks: List[int] = []
    capacities: List[int] = []
    layer_bonuses: List[int] = []
    trainable_ranks: List[int] = []

    for name, module in named_modules:
        capacity = module.r_main
        if capacity <= 0:
            continue
        with torch.no_grad():
            weight = module.merged_weight().to(device=svd_device, dtype=torch.float32)
            rank_svd = min(capacity, min(weight.shape))
            u, s, vh = randomized_svd(
                weight,
                rank=rank_svd,
                oversample=svd_oversample,
                n_iter=svd_n_iter,
                seed=None if seed is None else int(seed) + int(task_index),
            )
            del u, vh, weight
            r_elbow = select_rank(
                s,
                fro_norm_sq=None,
                energy_threshold=energy_threshold,
                elbow_ratio=elbow_ratio,
                min_rank=min_rank,
                max_rank=capacity,
            )
        previous = getattr(module, "current_active_rank", capacity)
        per_layer_bonus = int(rank_bonus) + name_bonus(name) + depth_bonus(name)
        proposed = int(r_elbow) + per_layer_bonus
        floor = max(int(min_rank), int(active_rank_floor))
        if int(task_index) <= 1:
            target = min(capacity, max(floor, proposed))
        else:
            min_growth_target = int(previous) + max(0, int(min_new_ranks))
            target = min(capacity, max(int(previous), min_growth_target, floor, proposed))
        module.set_active_rank(target)
        if freeze_old_ranks:
            if int(task_index) <= 1:
                train_start = 0
            else:
                train_start = max(0, min(int(previous), target) - max(0, int(old_train_tail)))
            module.set_trainable_rank_range(train_start, target)
        elif int(task_index) > 1 and float(old_rank_grad_scale) < 1.0:
            weights = torch.zeros(capacity, device=module.lora_train_mask.device, dtype=module.lora_train_mask.dtype)
            old_end = max(0, min(int(previous), target))
            if old_end > 0:
                weights[:old_end] = max(0.0, float(old_rank_grad_scale))
            if target > old_end:
                weights[old_end:target] = 1.0
            module.set_trainable_rank_weights(weights)
        else:
            module.set_all_active_ranks_trainable()
        active_ranks.append(target)
        elbow_ranks.append(int(r_elbow))
        capacities.append(capacity)
        layer_bonuses.append(per_layer_bonus)
        trainable_ranks.append(int(getattr(module, "current_trainable_rank", target)))
        if verbose:
            print(
                f"[HiLoRA-task-elbow] task={task_name} layer={name} "
                f"elbow={r_elbow} bonus={per_layer_bonus} floor={active_rank_floor} "
                f"active={target}/{capacity} trainable={trainable_ranks[-1]}"
            )

    if not active_ranks:
        return {}
    return {
        "task": task_name,
        "task_index": int(task_index),
        "rank_bonus": int(rank_bonus),
        "mean_elbow_rank": float(sum(elbow_ranks) / len(elbow_ranks)),
        "min_elbow_rank": int(min(elbow_ranks)),
        "max_elbow_rank": int(max(elbow_ranks)),
        "mean_layer_bonus": float(sum(layer_bonuses) / len(layer_bonuses)),
        "mean_active_rank": float(sum(active_ranks) / len(active_ranks)),
        "min_active_rank": int(min(active_ranks)),
        "max_active_rank": int(max(active_ranks)),
        "mean_trainable_rank": float(sum(trainable_ranks) / len(trainable_ranks)),
        "min_trainable_rank": int(min(trainable_ranks)),
        "max_trainable_rank": int(max(trainable_ranks)),
        "mean_capacity": float(sum(capacities) / len(capacities)),
    }


class HiLoRARecoveryAllocator:
    """AdaLoRA-style rank masking plus dynamic residual recovery.

    The allocator uses per-rank importance scores from LoRA gradients:
    |A * grad_A| + |B * grad_B|. It keeps a scheduled global rank budget,
    masks the remaining main-subspace ranks, and sets each layer's residual
    recovery ratio from the surviving-rank fraction.
    """

    def __init__(
        self,
        model: nn.Module,
        target_rank_ratio: float = 0.5,
        min_rank: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85,
        mask_interval: int = 1,
        recovery_ratio_mode: str = "kept",
    ) -> None:
        self.modules = iter_hilora_modules(model)
        self.target_rank_ratio = float(max(0.0, min(1.0, target_rank_ratio)))
        self.min_rank = max(1, int(min_rank))
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.mask_interval = max(1, int(mask_interval))
        if recovery_ratio_mode not in {"kept", "dropped"}:
            raise ValueError("recovery_ratio_mode must be 'kept' or 'dropped'")
        self.recovery_ratio_mode = recovery_ratio_mode
        self.exp_avg = [torch.zeros(module.r_main) for module in self.modules]
        self.exp_unc = [torch.zeros(module.r_main) for module in self.modules]

    def reset(self) -> None:
        for module in self.modules:
            module.reset_rank_mask()
            module.set_recovery_ratio(1.0)
        self.exp_avg = [torch.zeros(module.r_main) for module in self.modules]
        self.exp_unc = [torch.zeros(module.r_main) for module in self.modules]

    @staticmethod
    def _rank_importance(module: HiLoRALinear) -> Optional[torch.Tensor]:
        if module.lora_A is None or module.lora_B is None:
            return None
        if module.lora_A.grad is None or module.lora_B.grad is None:
            return None
        a_score = (module.lora_A.detach() * module.lora_A.grad.detach()).abs().mean(dim=1)
        b_score = (module.lora_B.detach() * module.lora_B.grad.detach()).abs().mean(dim=0)
        return (a_score + b_score).float().cpu()

    def _scheduled_rank_budget(self, completed_steps: int, total_steps: int) -> int:
        total_rank = sum(module.r_main for module in self.modules)
        target_rank = max(len(self.modules) * self.min_rank, math.ceil(total_rank * self.target_rank_ratio))
        target_rank = min(total_rank, target_rank)
        if total_steps <= 1:
            return target_rank
        progress = max(0.0, min(1.0, completed_steps / max(total_steps - 1, 1)))
        budget = target_rank + (total_rank - target_rank) * ((1.0 - progress) ** 3)
        return max(target_rank, min(total_rank, int(round(budget))))

    def step(self, completed_steps: int, total_steps: int) -> dict:
        if completed_steps % self.mask_interval != 0:
            return {}

        raw_scores: List[torch.Tensor] = []
        for idx, module in enumerate(self.modules):
            score = self._rank_importance(module)
            if score is None:
                score = torch.ones(module.r_main)
            if self.exp_avg[idx].numel() != score.numel():
                self.exp_avg[idx] = torch.zeros_like(score)
                self.exp_unc[idx] = torch.zeros_like(score)
            self.exp_avg[idx] = self.beta1 * self.exp_avg[idx] + (1.0 - self.beta1) * score
            self.exp_unc[idx] = self.beta2 * self.exp_unc[idx] + (1.0 - self.beta2) * (score - self.exp_avg[idx]).abs()
            raw_scores.append(self.exp_avg[idx] * self.exp_unc[idx])

        if not raw_scores:
            return {}

        rank_budget = self._scheduled_rank_budget(completed_steps, total_steps)
        flat_scores = torch.cat(raw_scores)
        rank_budget = max(1, min(rank_budget, flat_scores.numel()))
        threshold = torch.topk(flat_scores, k=rank_budget, largest=True).values.min()

        kept_total = 0
        offset = 0
        for module, score in zip(self.modules, raw_scores):
            keep = (score >= threshold).float()
            if int(keep.sum().item()) < min(self.min_rank, module.r_main):
                topk = min(self.min_rank, module.r_main)
                keep.zero_()
                keep[torch.topk(score, k=topk, largest=True).indices] = 1.0
            module.set_rank_mask(keep.to(module.lora_rank_mask.device))
            kept_ratio = float(keep.mean().item())
            recovery_ratio = kept_ratio if self.recovery_ratio_mode == "kept" else 1.0 - kept_ratio
            module.set_recovery_ratio(recovery_ratio)
            kept_total += int(keep.sum().item())
            offset += module.r_main

        return {
            "rank_budget": rank_budget,
            "kept_rank": kept_total,
            "mean_recovery_ratio": float(
                sum(module.recovery_ratio for module in self.modules) / max(len(self.modules), 1)
            ),
        }


def hilora_param_groups(
    model: nn.Module,
    main_lr: float,
    residual_lr: Optional[float] = None,
) -> List[dict]:
    main_params = []
    res_params = []
    for module in iter_hilora_modules(model):
        main_params.extend(module.lora_parameters())
        res_params.extend(module.residual_parameters())

    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": main_lr})
    if res_params:
        groups.append({"params": res_params, "lr": residual_lr if residual_lr is not None else main_lr})
    return groups
