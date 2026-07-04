"""Aggregate the 3-seed multi-backbone main table (ReCoLoRA vs rank-swept baselines + O-LoRA).

Reads raw continual_results.json files and prints mean +/- population std over
seeds 42/43/44 for final average score and average forgetting. Also writes
results/main_table_3seed.csv (local, not committed).

Sources:
  baselines : outputs/continual_llm_rank_sweep/{method}/{model}_r{rank}/seed_*
  O-LoRA    : outputs/continual_llm/olora/{model}/seed_*
  ReCoLoRA  : outputs/continual_llm/recursive_hilora/{model}_rank256_min8_elbowonly/seed_*
"""
import csv
import json
import os
import statistics

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEDS = (42, 43, 44)

# (model_tag, display_name)
MODELS = [
    ("qwen3_8b", "Qwen3-8B"),
    ("mistral_7b_v03", "Mistral-7B-v0.3"),
    ("internlm2_5_7b_chat", "InternLM2.5-7B-Chat"),
    ("llama3_1_8b_instruct", "Llama-3.1-8B-Instruct"),
]

# Best-rank baseline configs matching the paper's multi-backbone table.
BEST_RANK = {
    "qwen3_8b": {"lora": 64, "pissa": 64, "adalora": 64, "dora": 128},
    "mistral_7b_v03": {"lora": 64, "pissa": 256, "adalora": 256, "dora": 64},
    "internlm2_5_7b_chat": {"lora": 64, "pissa": 64, "adalora": 64, "dora": 128},
    "llama3_1_8b_instruct": {"lora": 64, "pissa": 64, "adalora": 64, "dora": 256},
}

METHOD_LABEL = {"lora": "LoRA", "pissa": "PiSSA", "adalora": "AdaLoRA", "dora": "DoRA"}


def load_run(path):
    fn = os.path.join(path, "continual_results.json")
    if not os.path.isfile(fn):
        return None
    with open(fn) as f:
        d = json.load(f)
    s = d["summary"]
    pc = d.get("parameter_counts", {})
    params = pc.get("trainable_params") or pc.get("trainable")
    return s["average_final_score"], s["average_forgetting"], params


def collect(dirs):
    runs = [load_run(d) for d in dirs]
    missing = [d for d, r in zip(dirs, runs) if r is None]
    runs = [r for r in runs if r is not None]
    return runs, missing


def fmt(vals):
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"


def main():
    rows = []
    for tag, name in MODELS:
        print(f"\n### {name} ###")
        print(f"{'Method':22} {'FinalAvg':20} {'AvgForget':20} {'#seeds':7} params")
        entries = []
        for method, rank in BEST_RANK[tag].items():
            dirs = [os.path.join(ROOT, f"outputs/continual_llm_rank_sweep/{method}/{tag}_r{rank}/seed_{s}") for s in SEEDS]
            entries.append((f"{METHOD_LABEL[method]} (r={rank})", dirs))
        entries.append(("O-LoRA (r=8)", [os.path.join(ROOT, f"outputs/continual_llm/olora/{tag}/seed_{s}") for s in SEEDS]))
        entries.append(("ReCoLoRA", [os.path.join(ROOT, f"outputs/continual_llm/recursive_hilora/{tag}_rank256_min8_elbowonly/seed_{s}") for s in SEEDS]))

        for label, dirs in entries:
            runs, missing = collect(dirs)
            if not runs:
                print(f"{label:22} (no runs found)")
                continue
            fin = [r[0] for r in runs]
            forg = [r[1] for r in runs]
            params = runs[0][2]
            note = "" if not missing else f"  [missing: {len(missing)} seed(s)]"
            print(f"{label:22} {fmt(fin):20} {fmt(forg):20} {len(runs):<7} {params}{note}")
            rows.append({
                "model": name, "method": label, "n_seeds": len(runs),
                "final_mean": round(statistics.mean(fin), 6),
                "final_std": round(statistics.pstdev(fin), 6) if len(fin) > 1 else 0.0,
                "forget_mean": round(statistics.mean(forg), 6),
                "forget_std": round(statistics.pstdev(forg), 6) if len(forg) > 1 else 0.0,
                "trainable_params": params,
            })

    out_csv = os.path.join(ROOT, "results", "main_table_3seed.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
