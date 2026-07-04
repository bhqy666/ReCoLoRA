#!/usr/bin/env python
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate continual-learning results.")
    parser.add_argument("--results_root", default="outputs/continual")
    parser.add_argument("--output_csv", default="results/continual_summary.csv")
    parser.add_argument("--output_tex", default="results/continual_summary.tex")
    parser.add_argument("--output_json", default="results/continual_summary.json")
    return parser.parse_args()


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(results_root.glob("*/*/continual_results.json")):
        with path.open() as handle:
            payload = json.load(handle)
        config = payload["config"]
        summary = payload["summary"]
        params = payload.get("parameter_counts", {})
        runs.append(
            {
                "method": config["method"],
                "seed": config["seed"],
                "average_final_score": summary["average_final_score"],
                "average_forgetting": summary["average_forgetting"],
                "trainable_params": params.get("trainable_params"),
                "adapter_params": params.get("adapter_params"),
                "path": str(path),
            }
        )
    return runs


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["method"], []).append(run)

    rows = []
    for method, method_runs in sorted(grouped.items()):
        final_scores = [run["average_final_score"] for run in method_runs]
        forgettings = [run["average_forgetting"] for run in method_runs]
        trainable = [run["trainable_params"] for run in method_runs if run["trainable_params"] is not None]
        adapter = [run["adapter_params"] for run in method_runs if run["adapter_params"] is not None]
        rows.append(
            {
                "method": method,
                "runs": len(method_runs),
                "average_final_score_mean": mean(final_scores),
                "average_final_score_std": stdev(final_scores),
                "average_forgetting_mean": mean(forgettings),
                "average_forgetting_std": stdev(forgettings),
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
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Avg. Final Score & Avg. Forgetting \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & "
            f"{100 * row['average_final_score_mean']:.2f}$\\pm${100 * row['average_final_score_std']:.2f} & "
            f"{100 * row['average_forgetting_mean']:.2f}$\\pm${100 * row['average_forgetting_std']:.2f} \\\\"
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
