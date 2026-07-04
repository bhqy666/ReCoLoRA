import math
from typing import Optional, Tuple

import torch


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    oversample: int = 5,
    n_iter: int = 1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a randomized truncated SVD.

    Returns U, S, Vh where matrix ≈ U @ diag(S) @ Vh.
    """
    if rank <= 0:
        raise ValueError("rank must be > 0")

    device = matrix.device
    dtype = matrix.dtype
    m, n = matrix.shape
    q = rank + max(0, oversample)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Random test matrix
    omega = torch.randn(n, q, device=device, dtype=dtype, generator=generator)
    y = matrix @ omega  # (m, q)

    for _ in range(max(0, n_iter)):
        y = matrix @ (matrix.transpose(-1, -2) @ y)

    # Orthonormal basis
    q_mat, _ = torch.linalg.qr(y, mode="reduced")
    b = q_mat.transpose(-1, -2) @ matrix  # (q, n)

    # SVD on the smaller matrix
    u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q_mat @ u_hat

    return u[:, :rank], s[:rank], vh[:rank, :]


def select_rank(
    singular_values: torch.Tensor,
    fro_norm_sq: Optional[float] = None,
    energy_threshold: float = 0.9,
    elbow_ratio: float = 1.5,
    min_rank: int = 2,
    max_rank: Optional[int] = None,
) -> int:
    """Select rank using energy threshold and an elbow heuristic."""
    if singular_values.numel() == 0:
        return min_rank

    s = singular_values.detach().float()
    if max_rank is None:
        max_rank = s.numel()

    # Energy-based rank
    if fro_norm_sq is None:
        fro_norm_sq = float(torch.sum(s * s))

    cum_energy = torch.cumsum(s * s, dim=0)
    energy_ratio = cum_energy / max(fro_norm_sq, 1e-12)
    r_energy = int(torch.searchsorted(energy_ratio, energy_threshold).item() + 1)
    r_energy = max(min_rank, min(r_energy, max_rank))

    # Elbow heuristic: largest ratio between consecutive singular values
    r_elbow = None
    if s.numel() >= 2:
        ratios = s[:-1] / (s[1:] + 1e-12)
        max_ratio, idx = torch.max(ratios, dim=0)
        if float(max_ratio) >= elbow_ratio:
            r_elbow = int(idx.item() + 1)

    r = r_energy
    if r_elbow is not None:
        r = max(r, r_elbow)

    r = max(min_rank, min(r, max_rank))
    return r
