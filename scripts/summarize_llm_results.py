#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def load_result(path: Path):
    data = json.loads(path.read_text())
    cfg = data.get('config', {})
    return {
        'method': cfg.get('method', path.parts[-3] if len(path.parts) >= 3 else ''),
        'model': Path(cfg.get('model_name_or_path', '')).name or path.parts[-2],
        'seed': cfg.get('seed', path.parent.name),
        'tasks': cfg.get('tasks', ''),
        'avg_final': data['summary'].get('average_final_score'),
        'avg_forgetting': data['summary'].get('average_forgetting'),
        'trainable_params': data.get('parameter_counts', {}).get('trainable_params'),
        'adapter_params': data.get('parameter_counts', {}).get('adapter_params'),
        'path': str(path),
    }


def fmt_mu_sigma(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ''
    if len(vals) == 1:
        return f'{vals[0]:.4f}'
    return f'{mean(vals):.4f} $\\pm$ {pstdev(vals):.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/continual_llm')
    ap.add_argument('--model-tag', default='qwen3_8b')
    ap.add_argument('--out-dir', default='results')
    args = ap.parse_args()

    rows = []
    for path in Path(args.root).glob(f'*/{args.model_tag}/seed_*/continual_results.json'):
        rows.append(load_result(path))
    rows.sort(key=lambda r: (r['method'], str(r['seed'])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'llm_{args.model_tag}_summary.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['method'])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row['method'], []).append(row)

    tex_lines = [
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Method & Final Avg. $\uparrow$ & Avg. Forgetting $\downarrow$ & Trainable Params \\',
        r'\midrule',
    ]
    for method, group in sorted(grouped.items()):
        final = fmt_mu_sigma([g['avg_final'] for g in group])
        forgetting = fmt_mu_sigma([g['avg_forgetting'] for g in group])
        params = group[0]['trainable_params'] if group else ''
        tex_lines.append(f'{method} & {final} & {forgetting} & {params} \\\\')
    tex_lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    tex_path = out_dir / f'llm_{args.model_tag}_table.tex'
    tex_path.write_text('\n'.join(tex_lines))

    print(f'wrote {csv_path}')
    print(f'wrote {tex_path}')
    print('\n'.join(tex_lines))


if __name__ == '__main__':
    main()
