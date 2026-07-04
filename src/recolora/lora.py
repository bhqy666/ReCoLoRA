import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReCoLoRALinear(nn.Module):
    """Linear layer with ReCoLoRA adapters (principal + residual)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r_main: int,
        r_res: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        residual_alpha: Optional[float] = None,
    ) -> None:
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.r_main = int(r_main)
        self.r_res = int(r_res)

        self.lora_alpha = float(lora_alpha)
        self.residual_alpha = float(residual_alpha) if residual_alpha is not None else float(lora_alpha)

        self.scaling = self.lora_alpha / max(1, self.r_main)
        self.residual_scaling = self.residual_alpha / max(1, self.r_res) if self.r_res > 0 else 1.0

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Main adapter
        if self.r_main > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r_main, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r_main))
            self.register_buffer("lora_rank_mask", torch.ones(self.r_main))
            self.register_buffer("lora_train_mask", torch.ones(self.r_main))
        else:
            self.lora_A = None
            self.lora_B = None
            self.register_buffer("lora_rank_mask", torch.empty(0))
            self.register_buffer("lora_train_mask", torch.empty(0))

        # Residual adapter
        if self.r_res > 0:
            self.res_middle = nn.Parameter(torch.zeros(self.r_res, self.r_res))
            self.register_buffer("res_left_basis", torch.empty(self.out_features, self.r_res))
            self.register_buffer("res_right_basis", torch.empty(self.r_res, self.in_features))
        else:
            self.res_middle = None
            self.register_buffer("res_left_basis", torch.empty(0, 0))
            self.register_buffer("res_right_basis", torch.empty(0, 0))

        self.res_enabled = False
        self.recovery_ratio = 1.0
        self.preserve_inactive_ranks = False
        self.current_active_rank = self.r_main
        self.current_trainable_rank = self.r_main
        self._register_rank_gradient_hooks()
        self.set_stage(1)

    def _register_rank_gradient_hooks(self) -> None:
        if self.lora_A is None or self.lora_B is None:
            return

        def mask_a_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self.lora_train_mask.to(device=grad.device, dtype=grad.dtype)
            return grad * mask[:, None]

        def mask_b_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self.lora_train_mask.to(device=grad.device, dtype=grad.dtype)
            return grad * mask[None, :]

        self.lora_A.register_hook(mask_a_grad)
        self.lora_B.register_hook(mask_b_grad)

    def set_stage(self, stage: int) -> None:
        """Stage 1: only main adapter. Stage 2: enable residual adapter."""
        if self.res_middle is None:
            return
        requires = bool(stage >= 2)
        self.res_middle.requires_grad_(requires)
        self.res_enabled = requires

    def set_rank_mask(self, mask: torch.Tensor) -> None:
        if self.r_main <= 0:
            return
        if mask.numel() != self.r_main:
            raise ValueError(f"rank mask size mismatch: expected {self.r_main}, got {mask.numel()}")
        with torch.no_grad():
            new_mask = mask.to(device=self.lora_rank_mask.device, dtype=self.lora_rank_mask.dtype)
            if self.preserve_inactive_ranks:
                self.lora_rank_mask.fill_(1.0)
            else:
                self.lora_rank_mask.copy_(new_mask)
            self.current_active_rank = int(new_mask.sum().item())

    def reset_rank_mask(self) -> None:
        if self.r_main > 0:
            self.lora_rank_mask.fill_(1.0)
            self.current_active_rank = self.r_main
            self.lora_train_mask.fill_(1.0)
            self.current_trainable_rank = self.r_main

    def set_active_rank(self, rank: int) -> None:
        """Activate the first ``rank`` principal dimensions and mask the rest."""
        if self.r_main <= 0:
            self.current_active_rank = 0
            return
        rank = max(1, min(int(rank), self.r_main))
        mask = torch.zeros(self.r_main, device=self.lora_rank_mask.device, dtype=self.lora_rank_mask.dtype)
        mask[:rank] = 1.0
        self.set_rank_mask(mask)
        self.current_active_rank = rank

    def set_trainable_rank_range(self, start: int, end: int) -> None:
        """Allow gradients only for ranks in [start, end). Active masking is separate."""
        if self.r_main <= 0:
            self.current_trainable_rank = 0
            return
        start = max(0, min(int(start), self.r_main))
        end = max(start, min(int(end), self.r_main))
        mask = torch.zeros(self.r_main, device=self.lora_train_mask.device, dtype=self.lora_train_mask.dtype)
        if end > start:
            mask[start:end] = 1.0
        with torch.no_grad():
            self.lora_train_mask.copy_(mask)
            self.current_trainable_rank = int(mask.sum().item())

    def set_trainable_rank_weights(self, weights: torch.Tensor) -> None:
        """Set per-rank gradient multipliers for soft-freezing old ranks."""
        if self.r_main <= 0:
            self.current_trainable_rank = 0
            return
        if weights.numel() != self.r_main:
            raise ValueError(f"rank weight size mismatch: expected {self.r_main}, got {weights.numel()}")
        with torch.no_grad():
            new_weights = weights.to(device=self.lora_train_mask.device, dtype=self.lora_train_mask.dtype)
            self.lora_train_mask.copy_(new_weights.clamp(min=0.0, max=1.0))
            self.current_trainable_rank = int((self.lora_train_mask > 0).sum().item())

    def set_all_active_ranks_trainable(self) -> None:
        if self.r_main <= 0:
            self.current_trainable_rank = 0
            return
        end = int(getattr(self, "current_active_rank", self.r_main))
        self.set_trainable_rank_range(0, end)

    def set_recovery_ratio(self, ratio: float) -> None:
        self.recovery_ratio = float(max(0.0, min(1.0, ratio)))

    def merged_weight(self) -> torch.Tensor:
        """Return the current effective weight for task-wise spectral analysis."""
        weight = self.base_layer.weight.detach().float().clone()
        if self.lora_A is not None and self.lora_B is not None:
            mask = self.lora_rank_mask.detach().float().to(self.lora_A.device)
            a = self.lora_A.detach().float() * mask[:, None]
            b = self.lora_B.detach().float()
            weight = weight + (b @ a).to(weight.device) * self.scaling
        if self.res_middle is not None:
            left = self.res_left_basis.detach().float()
            middle = self.res_middle.detach().float()
            right = self.res_right_basis.detach().float()
            weight = weight + (left @ middle @ right).to(weight.device) * self.residual_scaling * self.recovery_ratio
        return weight

    def lora_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.lora_A is not None:
            params.append(self.lora_A)
        if self.lora_B is not None:
            params.append(self.lora_B)
        return params

    def residual_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.res_middle is not None:
            params.append(self.res_middle)
        return params

    @staticmethod
    def _init_adapter(
        A: Optional[nn.Parameter],
        B: Optional[nn.Parameter],
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        scaling: float,
    ) -> None:
        if A is None or B is None:
            return

        with torch.no_grad():
            s_sqrt = torch.sqrt(S)
            # A: (r, in), B: (out, r)
            A_init = torch.diag(s_sqrt) @ Vh
            B_init = U @ torch.diag(s_sqrt)

            if scaling != 1.0:
                scale = math.sqrt(scaling)
                A_init = A_init / scale
                B_init = B_init / scale

            A.copy_(A_init.to(dtype=A.dtype, device=A.device))
            B.copy_(B_init.to(dtype=B.dtype, device=B.device))

    def init_from_svd(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        rank_main: int,
        rank_res: int,
    ) -> None:
        if self.r_main != rank_main or self.r_res != rank_res:
            raise ValueError("Rank mismatch in ReCoLoRALinear.init_from_svd")

        self._init_adapter(
            self.lora_A,
            self.lora_B,
            U[:, :rank_main],
            S[:rank_main],
            Vh[:rank_main, :],
            self.scaling,
        )

        if rank_res > 0:
            start = rank_main
            end = rank_main + rank_res
            self.set_residual_basis(
                U[:, start:end],
                Vh[start:end, :],
            )

    def init_main_from_svd(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        rank_main: int,
    ) -> None:
        if self.r_main != rank_main:
            raise ValueError("Rank mismatch in ReCoLoRALinear.init_main_from_svd")

        self._init_adapter(
            self.lora_A,
            self.lora_B,
            U[:, :rank_main],
            S[:rank_main],
            Vh[:rank_main, :],
            self.scaling,
        )

    def set_residual_basis(
        self,
        left_basis: torch.Tensor,
        right_basis: torch.Tensor,
    ) -> None:
        if self.res_middle is None:
            return

        with torch.no_grad():
            self.res_left_basis.copy_(left_basis.to(dtype=self.res_left_basis.dtype, device=self.res_left_basis.device))
            self.res_right_basis.copy_(right_basis.to(dtype=self.res_right_basis.dtype, device=self.res_right_basis.device))
            self.res_middle.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        if self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)
            lora_out = lora_out * self.lora_rank_mask.to(dtype=lora_out.dtype, device=lora_out.device)
            lora_out = F.linear(lora_out, self.lora_B)
            result = result + lora_out * self.scaling

        if self.res_middle is not None and self.res_enabled:
            res_out = F.linear(self.lora_dropout(x), self.res_right_basis)
            res_out = F.linear(res_out, self.res_middle)
            res_out = F.linear(res_out, self.res_left_basis)
            result = result + res_out * self.residual_scaling * self.recovery_ratio

        return result
