import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Standard LoRA wrapper for nn.Linear."""

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

        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + lora_out * self.scaling


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    return any(target and target in name for target in targets)


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
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

        new_module = LoRALinear(
            module,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, new_module)
        replaced.append(name)
        if verbose:
            print(f"[LoRA] Replaced {name} with r={rank}")
    return replaced


def iter_lora_modules(model: nn.Module) -> List[LoRALinear]:
    return [module for module in model.modules() if isinstance(module, LoRALinear)]
