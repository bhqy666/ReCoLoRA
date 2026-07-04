"""O-LoRA: Orthogonal subspace continual learning for LoRA.

Faithful re-implementation of the core mechanism from
Wang et al., "Orthogonal Subspace Learning for Language Model Continual
Learning" (Findings of EMNLP 2023; https://github.com/cmnfriend/O-LoRA),
adapted to this repository's custom LoRA injection harness so that it shares
the exact same continual-GLUE protocol, task order, training budget, and
evaluation code as the ReCoLoRA / LoRA / PiSSA / AdaLoRA / DoRA baselines.

Mechanism
---------
* Each task gets a fresh trainable LoRA branch ``(lora_A, lora_B)``.
* When a new task starts, the previous task's branch is detached, frozen, and
  pushed onto a stack of previous branches. Frozen branches remain in the
  forward pass (their contribution is summed in) but receive no gradient, so a
  later task cannot overwrite an earlier task's parameters.
* An orthogonality regularizer pushes the current task's down-projection
  ``lora_A`` to be orthogonal to all frozen ``lora_A`` subspaces, i.e. it
  penalizes ``||A_t A_i^T||_F^2`` for every previous task ``i < t``. This makes
  successive tasks occupy (approximately) orthogonal low-rank subspaces, which
  is the central idea of O-LoRA for reducing inter-task interference.

The effective weight after ``t`` tasks is
    W_eff = W0 + scaling * ( B_t A_t + sum_{i<t} B_i A_i ).
"""

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OLoRALinear(nn.Module):
    """O-LoRA wrapper for ``nn.Linear`` with a stack of frozen prior-task branches."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(lora_alpha) / max(1, self.rank)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        # Current (trainable) task branch.
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Frozen branches from previous tasks. Named with the "lora_" token so
        # that count_parameters / cast_adapter_parameters_to_fp32 treat them as
        # adapter parameters, while configure_trainable keeps them non-trainable
        # via the "frozen" name guard.
        self.frozen_lora_A = nn.ParameterList()
        self.frozen_lora_B = nn.ParameterList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        dropped = self.lora_dropout(x)
        # Current task branch.
        lora_out = F.linear(F.linear(dropped, self.lora_A), self.lora_B)
        out = result + lora_out * self.scaling
        # Frozen prior-task branches (no grad path needed; they are non-trainable).
        for a, b in zip(self.frozen_lora_A, self.frozen_lora_B):
            frozen_out = F.linear(F.linear(dropped, a), b)
            out = out + frozen_out * self.scaling
        return out

    @torch.no_grad()
    def advance_task(self) -> None:
        """Freeze the current branch and start a fresh trainable branch."""
        frozen_a = nn.Parameter(self.lora_A.detach().clone(), requires_grad=False)
        frozen_b = nn.Parameter(self.lora_B.detach().clone(), requires_grad=False)
        self.frozen_lora_A.append(frozen_a)
        self.frozen_lora_B.append(frozen_b)

        device = self.lora_A.device
        dtype = self.lora_A.dtype
        new_a = torch.empty(self.rank, self.base_layer.in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_a, a=math.sqrt(5))
        new_b = torch.zeros(self.base_layer.out_features, self.rank, device=device, dtype=dtype)
        self.lora_A = nn.Parameter(new_a)
        self.lora_B = nn.Parameter(new_b)

    def orthogonality_loss(self) -> torch.Tensor:
        """Sum_{i<t} ||A_t A_i^T||_F^2, computed in fp32."""
        if len(self.frozen_lora_A) == 0:
            return self.lora_A.new_zeros(())
        a_cur = self.lora_A.float()
        loss = a_cur.new_zeros(())
        for a_old in self.frozen_lora_A:
            prod = torch.matmul(a_cur, a_old.float().t())
            loss = loss + prod.pow(2).sum()
        return loss

    def num_frozen_tasks(self) -> int:
        return len(self.frozen_lora_A)


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    return any(target and target in name for target in targets)


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_olora(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    verbose: bool = False,
) -> List[str]:
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, target_modules):
            continue

        new_module = OLoRALinear(
            module,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, new_module)
        replaced.append(name)
        if verbose:
            print(f"[O-LoRA] Replaced {name} with r={rank}")
    return replaced


def iter_olora_modules(model: nn.Module) -> List[OLoRALinear]:
    return [module for module in model.modules() if isinstance(module, OLoRALinear)]


def olora_advance_task(model: nn.Module) -> dict:
    """Freeze current branches and open fresh ones across all O-LoRA layers."""
    modules = iter_olora_modules(model)
    for module in modules:
        module.advance_task()
    frozen = modules[0].num_frozen_tasks() if modules else 0
    return {"num_layers": len(modules), "frozen_tasks_per_layer": frozen}


def olora_orthogonality_loss(model: nn.Module, weight: float) -> torch.Tensor:
    """Weighted sum of per-layer orthogonality penalties."""
    modules = iter_olora_modules(model)
    if not modules:
        return torch.zeros((), device=next(model.parameters()).device)
    total = modules[0].lora_A.new_zeros(())
    for module in modules:
        total = total + module.orthogonality_loss()
    return weight * total


def olora_param_groups(model: nn.Module, learning_rate: float) -> List[dict]:
    params = []
    for module in iter_olora_modules(model):
        params.extend([module.lora_A, module.lora_B])
    return [{"params": params, "lr": learning_rate}] if params else []
