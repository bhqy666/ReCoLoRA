"""Qwen3-8B stage-wise results figure for the paper.

(a) average score over all tasks seen so far, after each training stage;
(b) score on the first task (SST-2) after each stage (retention trajectory).

Series: ReCoLoRA (recursive), LoRA (r=64), O-LoRA (r=8). Lines show the mean
over available seeds; the shaded band is the min-max range when more than one
seed exists. Rerun after adding seeds; the figure updates automatically.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["sst2", "mrpc", "qnli", "rte", "qqp", "mnli"]
TASK_LABELS = ["SST-2", "MRPC", "QNLI", "RTE", "QQP", "MNLI"]

SERIES = [
    # (label, glob pattern, color, marker, linestyle)
    ("ReCoLoRA", "outputs/continual_llm/recursive_hilora/qwen3_8b_rank256_min8_elbowonly/seed_*", "#2a78d6", "o", "-"),
    ("LoRA (r=64)", "outputs/continual_llm_rank_sweep/lora/qwen3_8b_r64/seed_*", "#1baf7a", "s", "--"),
    ("O-LoRA (r=8)", "outputs/continual_llm/olora/qwen3_8b/seed_*", "#eda100", "^", ":"),
]


def stage_matrix(run_dir):
    """Return dict {stage_index: {eval_task: score}} for one run."""
    with open(os.path.join(run_dir, "continual_results.json")) as f:
        d = json.load(f)
    m = {}
    for r in d["records"]:
        m.setdefault(r["stage_index"], {})[r["eval_task"]] = r["score"]
    return m


def curves(pattern):
    """Per-seed avg-seen curves and per-task final forgetting."""
    avg_seen, forget = [], []
    for run_dir in sorted(glob.glob(os.path.join(ROOT, pattern))):
        m = stage_matrix(run_dir)
        stages = sorted(m)
        T = stages[-1]
        avg_seen.append([np.mean([m[s][t] for t in TASKS[:s]]) for s in stages])
        forget.append([max(m[s][t] for s in stages if t in m[s]) - m[T][t] for t in TASKS])
    return np.array(avg_seen), np.array(forget)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
x = np.arange(1, len(TASKS) + 1)

# (a) trajectory of average score on seen tasks
ax = axes[0]
for label, pattern, color, marker, ls in SERIES:
    avg_seen, _ = curves(pattern)
    if avg_seen.size == 0:
        continue
    mean = avg_seen.mean(axis=0)
    ax.plot(x, mean, ls, color=color, marker=marker, markersize=5,
            linewidth=1.8, label=label + ("" if len(avg_seen) > 1 else " (seed 42)"))
    if len(avg_seen) > 1:
        ax.fill_between(x, avg_seen.min(axis=0), avg_seen.max(axis=0), color=color, alpha=0.15, linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=8)
ax.set_xlabel("After training on task", fontsize=9)
ax.set_ylabel("Average score (seen tasks)", fontsize=9)
ax.set_title("(a) Average score on tasks seen so far", fontsize=10)
ax.grid(alpha=0.25, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")

# (b) where forgetting happens: per-task final forgetting
ax = axes[1]
width = 0.26
for k, (label, pattern, color, marker, ls) in enumerate(SERIES):
    _, forget = curves(pattern)
    if forget.size == 0:
        continue
    mean = forget.mean(axis=0)
    err = (forget.max(axis=0) - forget.min(axis=0)) / 2 if len(forget) > 1 else None
    ax.bar(x + (k - 1) * width, mean, width * 0.92, color=color,
           yerr=err, error_kw={"linewidth": 0.8, "capsize": 2},
           label=label + ("" if len(forget) > 1 else " (seed 42)"))
ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=8)
ax.set_xlabel("Task", fontsize=9)
ax.set_ylabel("Final forgetting $F_j$", fontsize=9)
ax.set_title("(b) Per-task forgetting at the end of the stream", fontsize=10)
ax.grid(alpha=0.25, linewidth=0.6, axis="y")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
ax.margins(y=0.12)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

fig.tight_layout()
out = os.path.join(ROOT, "figures", "results_curves.pdf")
fig.savefig(out, bbox_inches="tight")
print("Saved", out)
