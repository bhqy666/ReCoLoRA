#!/usr/bin/env python
"""Compare q_proj/v_proj singular-value spectra across backbones.

Quick diagnostic (not part of the paper pipeline): reuses the same
randomized SVD + elbow/energy rank selector as make_elbow_figure.py /
rho_sensitivity_ranks.py (rank=20, oversample=5, n_iter=1, seed=42,
energy_threshold=0.8, elbow_ratio=1.5, min_rank=2, max_rank=16) on
Qwen3-8B vs Llama-3.1-8B-Instruct vs Llama-3-8B (NousResearch) to see
whether the spectra differ in ways that would explain why the HiLoRA
schedule tuned on Qwen3 transfers poorly to Llama.

For each matrix we report:
  - r* (selected principal rank under the Qwen3-tuned hyperparameters)
  - energy20/full: fraction of the matrix's true Frobenius energy
    captured by the top-20 singular values (low-rank-ness)
  - sigma1/sigma16: spectral concentration within the truncated budget
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hilora.svd import randomized_svd, select_rank  # noqa: E402

MAX_RANK = 16
RESIDUAL_RANK = 4
ENERGY_THRESHOLD = 0.8
ELBOW_RATIO = 1.5
MIN_RANK = 2
RANK_SVD = MAX_RANK + RESIDUAL_RANK  # 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "Qwen3-8B": "/home/bhqy/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    "Llama-3.1-8B-Instruct": "/home/bhqy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "Llama-3-8B": "/home/bhqy/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061",
}

target_re = re.compile(r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.weight$")


def analyze(model_path):
    index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    targets = [k for k in weight_map if target_re.match(k)]
    by_shard = defaultdict(list)
    for k in targets:
        by_shard[weight_map[k]].append(k)

    r_list, energy20_list, concentration_list = [], [], []
    for shard, keys in sorted(by_shard.items()):
        path = os.path.join(model_path, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                w = f.get_tensor(k).float()
                true_fro_sq = float(torch.sum(w * w))
                u, s, vh = randomized_svd(w.to(DEVICE), rank=RANK_SVD, oversample=5, n_iter=1, seed=SEED)
                s = s.cpu()
                fro_norm_sq = float(torch.sum(s * s))  # truncated-budget energy (as used by select_rank)
                r_main = select_rank(
                    s[:MAX_RANK],
                    fro_norm_sq=fro_norm_sq,
                    energy_threshold=ENERGY_THRESHOLD,
                    elbow_ratio=ELBOW_RATIO,
                    min_rank=MIN_RANK,
                    max_rank=MAX_RANK,
                )
                r_list.append(r_main)
                energy20_list.append(fro_norm_sq / true_fro_sq)
                concentration_list.append(float(s[0] / s[MAX_RANK - 1]))
    return np.array(r_list), np.array(energy20_list), np.array(concentration_list), len(targets)


for name, path in MODELS.items():
    r, e20, conc, n = analyze(path)
    n_saturated = int(np.sum(r == MAX_RANK))
    print(f"\n=== {name} (n={n} matrices) ===")
    print(f"  r*           : mean={r.mean():.2f} min={r.min()} max={r.max()}  "
          f"saturated@16={n_saturated}/{n} ({100*n_saturated/n:.0f}%)")
    print(f"  energy20/full: mean={e20.mean():.4f} min={e20.min():.4f} max={e20.max():.4f}")
    print(f"  sigma1/sigma16: mean={conc.mean():.2f} min={conc.min():.2f} max={conc.max():.2f}")
