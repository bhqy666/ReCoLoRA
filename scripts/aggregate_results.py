#!/usr/bin/env python
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics.")
    parser.add_argument("--results_root", type=str, default="outputs/experiments")
    parser.add_argument("--output_csv", type=str, default="results/summary.csv")
    parser.add_argument("--output_tex", type=str, default="results/summary.tex")
    parser.add_argument("--output_json", type=str, default="results/summary.json")
    return parser.parse_args()


def load_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in sorted(results_root.glob("*/*/metrics.json")):
        with metrics_path.open() as handle:
            payload = json.load(handle)

        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        best_metrics = payload.get("best_metrics", {})
        params = payload.get("parameter_counts", {})

        runs.append(
            {
                "method": config.get("method", metrics_path.parents[1].name),
                "seed": config.get("seed", metrics_path.parent.name.replace("seed_", "")),
                "exact": metrics.get("exact"),
                "f1": metrics.get("f1"),
                "best_exact": best_metrics.get("best_eval_exact"),
                "best_f1": best_metrics.get("best_eval_f1"),
                "trainable_params": params.get("trainable_params"),
                "adapter_params": params.get("adapter_params"),
                "total_params": params.get("total_params"),
                "metrics_path": str(metrics_path),
            }
        )
    return runs


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["method"], []).append(run)

    rows = []
    for method, method_runs in sorted(grouped.items()):
        exacts = [run["exact"] for run in method_runs if run["exact"] is not None]
        f1s = [run["f1"] for run in method_runs if run["f1"] is not None]
        best_exacts = [run["best_exact"] for run in method_runs if run["best_exact"] is not None]
        best_f1s = [run["best_f1"] for run in method_runs if run["best_f1"] is not None]
        trainable = [run["trainable_params"] for run in method_runs if run["trainable_params"] is not None]
        adapter = [run["adapter_params"] for run in method_runs if run["adapter_params"] is not None]

        rows.append(
            {
                "method": method,
                "runs": len(method_runs),
                "exact_mean": mean(exacts),
                "exact_std": stdev(exacts),
                "f1_mean": mean(f1s),
                "f1_std": stdev(f1s),
                "best_exact_mean": mean(best_exacts),
                "best_exact_std": stdev(best_exacts),
                "best_f1_mean": mean(best_f1s),
                "best_f1_std": stdev(best_f1s),
                "trainable_params_mean": mean(trainable),
                "adapter_params_mean": mean(adapter),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & EM & F1 & Best EM & Best F1 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & "
            f"{row['exact_mean']:.2f}$\\pm${row['exact_std']:.2f} & "
            f"{row['f1_mean']:.2f}$\\pm${row['f1_std']:.2f} & "
            f"{row['best_exact_mean']:.2f}$\\pm${row['best_exact_std']:.2f} & "
            f"{row['best_f1_mean']:.2f}$\\pm${row['best_f1_std']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    runs = load_runs(Path(args.results_root))
    summary = aggregate(runs)

    write_csv(Path(args.output_csv), summary)
    write_tex(Path(args.output_tex), summary)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"runs": runs, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
