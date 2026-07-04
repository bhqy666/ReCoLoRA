import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd import randomized_svd


class RecursiveReCoLoRALinear(nn.Module):
    """Recursive slow/fast low-rank linear layer.

    The effective weight is:
        W_eff = W_res_frozen + W_slow + W_fast

    At consolidation time W_eff is materialized, recompressed with SVD, and
    the fast branch is reset for the next task.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        slow_rank: int,
        fast_rank: int,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        self.slow_rank = int(slow_rank)
        self.fast_rank = int(fast_rank)
        self.lora_alpha = float(lora_alpha)
        self.fast_scaling = self.lora_alpha / max(1, self.fast_rank)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.slow_A = nn.Parameter(torch.zeros(self.slow_rank, self.in_features))
        self.slow_B = nn.Parameter(torch.zeros(self.out_features, self.slow_rank))
        self.fast_A = nn.Parameter(torch.empty(self.fast_rank, self.in_features))
        self.fast_B = nn.Parameter(torch.zeros(self.out_features, self.fast_rank))
        self.reset_fast()
        self.snapshot_slow_anchor()

    @staticmethod
    def _factor_from_svd(
        u: torch.Tensor,
        s: torch.Tensor,
        vh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_sqrt = torch.sqrt(s.clamp_min(0.0))
        a = torch.diag(s_sqrt) @ vh
        b = u @ torch.diag(s_sqrt)
        return a, b

    def set_slow_from_svd(self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> None:
        if s.numel() != self.slow_rank:
            raise ValueError(f"slow rank mismatch: expected {self.slow_rank}, got {s.numel()}")
        a, b = self._factor_from_svd(u, s, vh)
        with torch.no_grad():
            self.slow_A.copy_(a.to(device=self.slow_A.device, dtype=self.slow_A.dtype))
            self.slow_B.copy_(b.to(device=self.slow_B.device, dtype=self.slow_B.dtype))
        self.snapshot_slow_anchor()

    def reset_fast(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fast_A, a=math.sqrt(5))
            self.fast_B.zero_()

    def snapshot_slow_anchor(self) -> None:
        self.register_buffer("slow_A_anchor", self.slow_A.detach().float().cpu().clone(), persistent=False)
        self.register_buffer("slow_B_anchor", self.slow_B.detach().float().cpu().clone(), persistent=False)

    def slow_weight(self) -> torch.Tensor:
        return self.slow_B.float() @ self.slow_A.float()

    def fast_weight(self) -> torch.Tensor:
        return (self.fast_B.float() @ self.fast_A.float()) * self.fast_scaling

    def merged_weight(self) -> torch.Tensor:
        weight = self.base_layer.weight.detach().float().clone()
        weight = weight + self.slow_weight().to(weight.device)
        weight = weight + self.fast_weight().to(weight.device)
        return weight

    def slow_anchor_loss(self) -> torch.Tensor:
        if self.slow_A_anchor.numel() == 0 or self.slow_B_anchor.numel() == 0:
            return torch.zeros((), device=self.slow_A.device, dtype=self.slow_A.dtype)
        anchor_a = self.slow_A_anchor.to(device=self.slow_A.device, dtype=self.slow_A.dtype)
        anchor_b = self.slow_B_anchor.to(device=self.slow_B.device, dtype=self.slow_B.dtype)
        return (self.slow_A - anchor_a).pow(2).mean() + (self.slow_B - anchor_b).pow(2).mean()

    def replace_factors(
        self,
        slow_rank: int,
        fast_rank: int,
        u: torch.Tensor,
        s: torch.Tensor,
        vh: torch.Tensor,
        residual_weight: torch.Tensor,
    ) -> None:
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        slow_rank = int(slow_rank)
        fast_rank = int(fast_rank)
        if slow_rank <= 0 or fast_rank <= 0:
            raise ValueError("slow_rank and fast_rank must be positive")

        self.slow_rank = slow_rank
        self.fast_rank = fast_rank
        self.fast_scaling = self.lora_alpha / max(1, self.fast_rank)
        self.slow_A = nn.Parameter(torch.zeros(slow_rank, self.in_features, device=device, dtype=torch.float32))
        self.slow_B = nn.Parameter(torch.zeros(self.out_features, slow_rank, device=device, dtype=torch.float32))
        self.fast_A = nn.Parameter(torch.empty(fast_rank, self.in_features, device=device, dtype=torch.float32))
        self.fast_B = nn.Parameter(torch.zeros(self.out_features, fast_rank, device=device, dtype=torch.float32))
        with torch.no_grad():
            self.base_layer.weight.copy_(residual_weight.to(device=device, dtype=dtype))
        self.set_slow_from_svd(
            u[:, :slow_rank].to(device=device, dtype=torch.float32),
            s[:slow_rank].to(device=device, dtype=torch.float32),
            vh[:slow_rank, :].to(device=device, dtype=torch.float32),
        )
        self.reset_fast()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        slow_out = F.linear(self.lora_dropout(x), self.slow_A.to(dtype=x.dtype))
        slow_out = F.linear(slow_out, self.slow_B.to(dtype=x.dtype))
        result = result + slow_out
        fast_out = F.linear(self.lora_dropout(x), self.fast_A.to(dtype=x.dtype))
        fast_out = F.linear(fast_out, self.fast_B.to(dtype=x.dtype))
        return result + fast_out * self.fast_scaling


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    return any(target and target in name for target in targets)


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _fast_rank_from_slow(slow_rank: int, bonus: int, multiplier: float, min_rank: int, max_rank: int) -> int:
    rank = max(int(slow_rank) + int(bonus), int(math.ceil(int(slow_rank) * float(multiplier))))
    rank = max(int(min_rank), rank)
    rank = min(int(max_rank), rank)
    return max(1, rank)


def _select_elbow_rank(
    singular_values: torch.Tensor,
    min_rank: int,
    max_rank: int,
    elbow_ratio: float,
) -> int:
    """Select the principal rank by elbow only; max_rank is only a cap.

    The ratio rule catches sharp spectral drops. If there is no sharp drop,
    a normalized knee heuristic picks the point farthest from the endpoint
    chord, which avoids falling back to an energy threshold.
    """
    if singular_values.numel() == 0:
        return max(1, int(min_rank))

    max_rank = max(1, min(int(max_rank), int(singular_values.numel())))
    min_rank = max(1, min(int(min_rank), max_rank))
    s = singular_values[:max_rank].detach().float()
    if s.numel() <= min_rank:
        return int(s.numel())

    if s.numel() >= 2:
        ratios = s[:-1] / (s[1:] + 1e-12)
        max_ratio, idx = torch.max(ratios, dim=0)
        if float(max_ratio) >= float(elbow_ratio):
            return max(min_rank, min(int(idx.item() + 1), max_rank))

    if s.numel() <= 2:
        return min_rank

    x = torch.linspace(0.0, 1.0, steps=s.numel(), device=s.device, dtype=s.dtype)
    denom = (s[0] - s[-1]).abs().clamp_min(1e-12)
    y = (s - s[-1]) / denom
    chord = 1.0 - x
    knee_scores = (y - chord).abs()
    knee_scores[: min_rank - 1] = -1.0
    idx = int(torch.argmax(knee_scores).item())
    return max(min_rank, min(idx + 1, max_rank))


def inject_recursive_recolora(
    model: nn.Module,
    target_modules: Iterable[str],
    max_rank: int = 32,
    min_rank: int = 2,
    fast_rank_bonus: int = 3,
    fast_rank_multiplier: float = 1.2,
    min_fast_rank: int = 4,
    max_fast_rank: Optional[int] = None,
    energy_threshold: float = 0.9,
    elbow_ratio: float = 1.5,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    svd_oversample: int = 5,
    svd_n_iter: int = 1,
    svd_device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[str]:
    replaced: List[str] = []
    if max_fast_rank is None:
        max_fast_rank = max_rank + max(1, int(fast_rank_bonus))

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not _matches_target(name, target_modules):
            continue

        weight = module.weight.detach().to(device=svd_device, dtype=torch.float32)
        rank_svd = min(int(max_rank), min(weight.shape))
        u, s, vh = randomized_svd(
            weight,
            rank=rank_svd,
            oversample=svd_oversample,
            n_iter=svd_n_iter,
            seed=seed,
        )
        slow_rank = _select_elbow_rank(s, min_rank=min_rank, max_rank=rank_svd, elbow_ratio=elbow_ratio)
        fast_rank = _fast_rank_from_slow(slow_rank, fast_rank_bonus, fast_rank_multiplier, min_fast_rank, max_fast_rank)
        u_s = u[:, :slow_rank]
        s_s = s[:slow_rank]
        vh_s = vh[:slow_rank, :]
        principal = (u_s * s_s) @ vh_s
        residual = weight - principal

        new_module = RecursiveReCoLoRALinear(
            module,
            slow_rank=slow_rank,
            fast_rank=fast_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        new_module = new_module.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            new_module.base_layer.weight.copy_(residual.to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and new_module.base_layer.bias is not None:
                new_module.base_layer.bias.copy_(module.bias.data)
        new_module.slow_A.data = new_module.slow_A.data.float()
        new_module.slow_B.data = new_module.slow_B.data.float()
        new_module.fast_A.data = new_module.fast_A.data.float()
        new_module.fast_B.data = new_module.fast_B.data.float()
        new_module.set_slow_from_svd(
            u_s.to(device=module.weight.device, dtype=torch.float32),
            s_s.to(device=module.weight.device, dtype=torch.float32),
            vh_s.to(device=module.weight.device, dtype=torch.float32),
        )
        new_module.reset_fast()

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, new_module)
        replaced.append(name)
        if verbose:
            print(f"[RecursiveReCoLoRA] Replaced {name} slow_rank={slow_rank} fast_rank={fast_rank}")

    if verbose and not replaced:
        print("[RecursiveReCoLoRA] No target modules matched.")
    return replaced


def iter_recursive_recolora_modules(model: nn.Module) -> List[RecursiveReCoLoRALinear]:
    return [module for module in model.modules() if isinstance(module, RecursiveReCoLoRALinear)]


def iter_named_recursive_recolora_modules(model: nn.Module) -> List[Tuple[str, RecursiveReCoLoRALinear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, RecursiveReCoLoRALinear)
    ]


@torch.no_grad()
def consolidate_recursive_recolora(
    model: nn.Module,
    max_rank: int = 32,
    min_rank: int = 2,
    fast_rank_bonus: int = 3,
    fast_rank_multiplier: float = 1.2,
    min_fast_rank: int = 4,
    max_fast_rank: Optional[int] = None,
    energy_threshold: float = 0.9,
    elbow_ratio: float = 1.5,
    svd_oversample: int = 5,
    svd_n_iter: int = 1,
    svd_device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    named_modules = iter_named_recursive_recolora_modules(model)
    if max_fast_rank is None:
        max_fast_rank = max_rank + max(1, int(fast_rank_bonus))

    old_slow_ranks: List[int] = []
    new_slow_ranks: List[int] = []
    fast_ranks: List[int] = []
    errors: List[float] = []
    layer_ranks = []

    for idx, (name, module) in enumerate(named_modules):
        old_slow_ranks.append(int(module.slow_rank))
        weight = module.merged_weight().to(device=svd_device, dtype=torch.float32)
        rank_svd = min(int(max_rank), min(weight.shape))
        u, s, vh = randomized_svd(
            weight,
            rank=rank_svd,
            oversample=svd_oversample,
            n_iter=svd_n_iter,
            seed=None if seed is None else int(seed) + idx,
        )
        slow_rank = _select_elbow_rank(s, min_rank=min_rank, max_rank=rank_svd, elbow_ratio=elbow_ratio)
        fast_rank = _fast_rank_from_slow(slow_rank, fast_rank_bonus, fast_rank_multiplier, min_fast_rank, max_fast_rank)
        principal = (u[:, :slow_rank] * s[:slow_rank]) @ vh[:slow_rank, :]
        residual = weight - principal
        recon = residual + principal
        error = float(torch.linalg.norm(weight - recon) / torch.linalg.norm(weight).clamp_min(1e-12))
        module.replace_factors(
            slow_rank=slow_rank,
            fast_rank=fast_rank,
            u=u[:, :slow_rank].to(device=module.base_layer.weight.device),
            s=s[:slow_rank].to(device=module.base_layer.weight.device),
            vh=vh[:slow_rank, :].to(device=module.base_layer.weight.device),
            residual_weight=residual.to(device=module.base_layer.weight.device),
        )
        new_slow_ranks.append(int(slow_rank))
        fast_ranks.append(int(fast_rank))
        errors.append(error)
        direction = "unchanged"
        if slow_rank > old_slow_ranks[-1]:
            direction = "increased"
        elif slow_rank < old_slow_ranks[-1]:
            direction = "decreased"
        layer_ranks.append(
            {
                "layer": name,
                "old_slow_rank": int(old_slow_ranks[-1]),
                "new_slow_rank": int(slow_rank),
                "new_fast_rank": int(fast_rank),
                "rank_svd_cap": int(rank_svd),
                "direction": direction,
                "reconstruction_error": float(error),
            }
        )
        if verbose:
            print(
                f"[RecursiveReCoLoRA-consolidate] module={idx} layer={name} "
                f"slow_rank={old_slow_ranks[-1]}->{slow_rank} fast_rank={fast_rank} error={error:.6e}"
            )

    if not named_modules:
        return {}
    increased = sum(1 for old, new in zip(old_slow_ranks, new_slow_ranks) if new > old)
    decreased = sum(1 for old, new in zip(old_slow_ranks, new_slow_ranks) if new < old)
    unchanged = len(new_slow_ranks) - increased - decreased
    rank_histogram = {str(rank): int(new_slow_ranks.count(rank)) for rank in sorted(set(new_slow_ranks))}
    return {
        "modules": len(new_slow_ranks),
        "mean_slow_rank": float(sum(new_slow_ranks) / len(new_slow_ranks)),
        "min_slow_rank": int(min(new_slow_ranks)),
        "max_slow_rank": int(max(new_slow_ranks)),
        "mean_fast_rank": float(sum(fast_ranks) / len(fast_ranks)),
        "rank_increased_layers": int(increased),
        "rank_decreased_layers": int(decreased),
        "rank_unchanged_layers": int(unchanged),
        "mean_reconstruction_error": float(sum(errors) / len(errors)),
        "max_reconstruction_error": float(max(errors)),
        "rank_histogram": rank_histogram,
        "layer_ranks": layer_ranks,
    }


def recursive_recolora_param_groups(model: nn.Module, fast_lr: float, slow_lr: float) -> List[dict]:
    fast_params = []
    slow_params = []
    for module in iter_recursive_recolora_modules(model):
        fast_params.extend([module.fast_A, module.fast_B])
        slow_params.extend([module.slow_A, module.slow_B])
    groups = []
    if fast_params:
        groups.append({"params": fast_params, "lr": fast_lr, "role": "recursive_fast", "base_lr": fast_lr})
    if slow_params:
        groups.append({"params": slow_params, "lr": slow_lr, "role": "recursive_slow", "base_lr": slow_lr})
    return groups


def recursive_recolora_slow_anchor_loss(model: nn.Module) -> torch.Tensor:
    terms = [module.slow_anchor_loss() for module in iter_recursive_recolora_modules(model)]
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).mean()
