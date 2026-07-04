#!/usr/bin/env python
"""Sensitivity of the elbow/energy rank selector to the energy threshold rho.

Reuses the exact randomized-SVD computation from make_elbow_figure.py (same
72 q_proj/v_proj matrices in Qwen3-8B, rank=20, oversample=5, seed=42), but
sweeps the energy_threshold (rho) used by select_rank while keeping the SVD
itself fixed. This isolates the effect of rho on the selected principal rank
r* and on the resulting principal-adapter parameter count, independent of any
downstream training.
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
ELBOW_RATIO = 1.5
MIN_RANK = 2
RANK_SVD = MAX_RANK + RESIDUAL_RANK  # 20
SEED = 42
RHOS = [0.6, 0.7, 0.8, 0.9, 0.95]
LORA_R16_PARAMS = 7_667_712  # for reference (q_proj+v_proj, r=16, 36 layers)

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
weight_map = index["weight_map"]
target_re = re.compile(r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.weight$")
targets = [k for k in weight_map if target_re.match(k)]

by_shard = defaultdict(list)
for k in targets:
    by_shard[weight_map[k]].append(k)

# Per-matrix: singular values (truncated) + fro_norm_sq + dim (in+out)
svd_cache = {}

for shard, keys in sorted(by_shard.items()):
    path = os.path.join(MODEL, shard)
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in keys:
            w = f.get_tensor(k).float()
            dim = w.shape[0] + w.shape[1]
            u, s, vh = randomized_svd(w, rank=RANK_SVD, oversample=5, n_iter=1, seed=SEED)
            fro_norm_sq = float(torch.sum(s * s))
            svd_cache[k] = (s, fro_norm_sq, dim)
            print(f"cached SVD: {k} dim={dim}")

results = {}
for rho in RHOS:
    r_list = []
    param_total = 0
    for k, (s, fro_norm_sq, dim) in svd_cache.items():
        r_main = select_rank(
            s[:MAX_RANK],
            fro_norm_sq=fro_norm_sq,
            energy_threshold=rho,
            elbow_ratio=ELBOW_RATIO,
            min_rank=MIN_RANK,
            max_rank=MAX_RANK,
        )
        r_list.append(r_main)
        param_total += r_main * dim
    results[rho] = {
        "min": int(min(r_list)),
        "mean": float(np.mean(r_list)),
        "max": int(max(r_list)),
        "principal_params": int(param_total),
    }

print("\nrho  | min r* | mean r* | max r* | principal params (~)")
print("-----+--------+---------+--------+----------------------")
for rho in RHOS:
    r = results[rho]
    print(f"{rho:.2f} | {r['min']:6d} | {r['mean']:7.2f} | {r['max']:6d} | {r['principal_params']:>10,}")
print(f"\n(reference: LoRA r=16 principal params = {LORA_R16_PARAMS:,})")

out_json = os.path.join(os.path.dirname(__file__), "..", "outputs", "rho_sensitivity_ranks.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print(f"\nSaved {out_json}")

# Figure: mean r* (with min/max range) and principal params vs rho
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

rhos_arr = np.array(RHOS)
means = np.array([results[r]["mean"] for r in RHOS])
mins = np.array([results[r]["min"] for r in RHOS])
maxs = np.array([results[r]["max"] for r in RHOS])

ax = axes[0]
ax.plot(rhos_arr, means, "o-", color="tab:blue", label="mean $r^*$")
ax.fill_between(rhos_arr, mins, maxs, color="tab:blue", alpha=0.2, label="min/max range")
ax.axvline(0.8, color="tab:red", linestyle="--", label=r"default $\rho=0.8$")
ax.axhline(MAX_RANK, color="gray", linestyle=":", label=f"$r_{{\\max}}={MAX_RANK}$")
ax.set_xlabel(r"Energy threshold $\rho$")
ax.set_ylabel(r"Selected principal rank $r^*$")
ax.set_title("(a) Rank selection vs. $\\rho$")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

params = np.array([results[r]["principal_params"] for r in RHOS])
ax = axes[1]
ax.plot(rhos_arr, params, "o-", color="tab:green", label="ReCoLoRA principal params")
ax.axhline(LORA_R16_PARAMS, color="tab:purple", linestyle="--", label="LoRA $r=16$ params")
ax.axvline(0.8, color="tab:red", linestyle="--", label=r"default $\rho=0.8$")
ax.set_xlabel(r"Energy threshold $\rho$")
ax.set_ylabel("Principal adapter parameters")
ax.set_title("(b) Adapter size vs. $\\rho$")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.tight_layout()
out_fig = os.path.join(os.path.dirname(__file__), "..", "figures", "rho_sensitivity.pdf")
fig.savefig(out_fig, bbox_inches="tight")
print(f"Saved {out_fig}")
