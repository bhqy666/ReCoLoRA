#!/usr/bin/env python
"""Generate an empirical elbow/energy rank-selection figure from real Qwen3-8B weights.

Reproduces the exact rank-selection computation used by ReCoLoRA's injection
code (src/recolora/inject.py): randomized SVD with rank = max_rank + residual_rank,
energy-threshold + elbow-ratio rank selection on the truncated spectrum.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recolora.svd import randomized_svd, select_rank  # noqa: E402

MODEL = "/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

MAX_RANK = 16
RESIDUAL_RANK = 4
ENERGY_THRESHOLD = 0.8
ELBOW_RATIO = 1.5
MIN_RANK = 2
RANK_SVD = MAX_RANK + RESIDUAL_RANK  # 20
SEED = 42
EXAMPLE_KEY = "model.layers.0.self_attn.q_proj.weight"

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
weight_map = index["weight_map"]
target_re = re.compile(r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.weight$")
targets = [k for k in weight_map if target_re.match(k)]

by_shard = defaultdict(list)
for k in targets:
    by_shard[weight_map[k]].append(k)

device = "cuda" if torch.cuda.is_available() else "cpu"

r_mains = {}
example = None
full_spectrum = None

for shard, keys in sorted(by_shard.items()):
    path = os.path.join(MODEL, shard)
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in keys:
            w = f.get_tensor(k).float()
            u, s, vh = randomized_svd(w, rank=RANK_SVD, oversample=5, n_iter=1, seed=SEED)
            fro_norm_sq = float(torch.sum(s * s))
            r_main = select_rank(
                s[:MAX_RANK],
                fro_norm_sq=fro_norm_sq,
                energy_threshold=ENERGY_THRESHOLD,
                elbow_ratio=ELBOW_RATIO,
                min_rank=MIN_RANK,
                max_rank=MAX_RANK,
            )
            r_mains[k] = r_main
            if k == EXAMPLE_KEY:
                example = (s.numpy(), fro_norm_sq, r_main)
                full_spectrum = torch.linalg.svdvals(w.to(device)).cpu().numpy()
            print(f"{k}: r_main={r_main}")

vals = list(r_mains.values())
print(f"\nn={len(vals)} min={min(vals)} max={max(vals)} mean={sum(vals)/len(vals):.2f}")

s, fro_norm_sq, r_main = example
cum_energy = np.cumsum(s ** 2) / fro_norm_sq
idx = np.arange(1, RANK_SVD + 1)
d = full_spectrum.shape[0]
full_idx = np.arange(1, d + 1)
print(f"\nfull spectrum dim d={d}, sigma_1={full_spectrum[0]:.4f}, sigma_d={full_spectrum[-1]:.6f}")

print(f"first 20 raw singular values: {np.array2string(full_spectrum[:20], precision=3)}")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))

ax = axes[0]
ax.plot(full_idx, full_spectrum, color="#2a78d6", linewidth=1.2)
ax.axvspan(1, RANK_SVD, color="#eb6834", alpha=0.22, label=f"truncated budget ($r_{{\\max}}+r_{{res}}={RANK_SVD}$)")
ax.axvline(r_main, color="#e34948", linestyle="--", label=f"$r^*={r_main}$")
ax.set_xscale("log")
ax.set_xlabel("Singular value index $i$ (log scale)", fontsize=9)
ax.set_ylabel(r"Singular value $\sigma_i$", fontsize=9)
ax.set_title(f"(a) Raw spectrum, layer-0 q_proj ($d={d}$)", fontsize=9.5)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3, which="both", linewidth=0.5)
ax.tick_params(labelsize=8)

ax = axes[1]
ax.plot(idx, cum_energy, "o-", color="#008300", markersize=4, linewidth=1.5)
ax.axhline(ENERGY_THRESHOLD, color="black", linestyle=":", label=r"$\rho=0.8$")
ax.axvline(r_main, color="#e34948", linestyle="--", label=f"$r^*={r_main}$")
ax.set_xlabel("Singular value index $i$", fontsize=9)
ax.set_ylabel("Cumulative energy ratio", fontsize=9)
ax.set_title("(b) Energy within truncated budget", fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=8)

ax = axes[2]
bins = np.arange(MIN_RANK - 0.5, MAX_RANK + 1.5, 1)
ax.hist(vals, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.8)
ax.axvline(sum(vals) / len(vals), color="#e34948", linestyle="--",
           label=f"mean $r^*={sum(vals)/len(vals):.2f}$")
ax.set_xlabel(r"Selected principal rank $r^*$", fontsize=9)
ax.set_ylabel("Number of matrices", fontsize=9)
ax.set_title(f"(c) $r^*$ across {len(vals)} q/v matrices", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, axis="y", linewidth=0.5)
ax.tick_params(labelsize=8)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "figures", "elbow_rank_selection.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved {out_path}")
