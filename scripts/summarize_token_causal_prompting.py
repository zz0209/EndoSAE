import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_token_causal_prompting import methods
from evaluate_temporal_shared_sae import METRICS
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def summarize(run, smoke):
    config = read_json(run / 'config.json')
    names = ['reference_supcon'] + methods(config)
    seeds = config['seeds'][:1] if smoke else config['seeds']
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    rows, inputs = [], {}
    for seed in seeds:
        folder = root / f'seed{seed}'
        complete = read_json(folder / 'summary.json')
        assert complete['status'] == 'COMPLETE' and complete['smoke'] == smoke
        inputs[str(folder / 'summary.json')] = digest(folder / 'summary.json')
        for population in ('development', 'extension'):
            for method in names:
                path = folder / population / (method + '__score_procedure_curve.npz')
                inputs[str(path)] = digest(path)
                with np.load(path, allow_pickle=False) as curve:
                    threshold = complete['operating_points'][method]['threshold']
                    index = int(np.argmin(np.abs(curve['threshold'] - threshold))) if population == 'development' else 0
                    if population == 'development':
                        assert np.isclose(curve['threshold'][index], threshold, atol=1e-10)
                    for number, video in enumerate(curve['videos'].tolist()):
                        rows.append(dict(seed=seed, population=population, method=method, video=video, threshold=threshold,
                                         **{key: float(curve[value][number, index]) for key, value in METRICS.items()}))
    table = pd.DataFrame(rows)
    per_seed = table.groupby(['population', 'method', 'seed'])[list(METRICS)].mean().reset_index()
    averages = per_seed.groupby(['population', 'method'])[list(METRICS)].mean().reset_index()
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'procedures.csv', index=False)
    per_seed.to_csv(output / 'seeds.csv', index=False)
    averages.to_csv(output / 'summary.csv', index=False)
    differences = []
    contrasts = config.get('contrasts', [('direct_token_sparse', 'reference_supcon'), ('direct_token_sparse', 'direct_token_dense'),
                        ('direct_token_sparse', 'projected_token_sparse'), ('direct_token_dense', 'projected_token_dense')])
    for left, right in contrasts:
        paired = table[table.method == left].merge(table[table.method == right], on=['population', 'seed', 'video'],
                                                  suffixes=('_left', '_right'), validate='one_to_one')
        for row in paired.to_dict('records'):
            differences.append(dict(comparison=left + '_minus_' + right, population=row['population'],
                seed=row['seed'], video=row['video'], **{key: row[key + '_left'] - row[key + '_right'] for key in METRICS}))
    pd.DataFrame(differences).to_csv(output / 'paired_differences.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9) if len(names) > 5 else (15, 7), sharey=True)
    colors = ['#0072B2', '#D55E00', '#009E73']
    labels = ([config['method_labels'][name] for name in names] if 'method_labels' in config else
              ['Ordinary SupCon', 'Projected sparse', 'Projected dense', 'Direct sparse', 'Direct dense'])
    for i, population in enumerate(('development', 'extension')):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for number, name in enumerate(names):
                values = per_seed[(per_seed.population == population) & (per_seed.method == name)]
                for k, seed in enumerate(seeds):
                    value = float(values[values.seed == seed][metric].iloc[0])
                    ax.scatter(value * 100, number + (k - (len(seeds) - 1) / 2) * .12, color=colors[k],
                               label=str(seed) if i == j == number == 0 else None)
                ax.scatter(values[metric].mean() * 100, number, color='black', marker='|')
            ax.set_yticks(range(len(names)), labels)
            ax.set_title(('Development' if i == 0 else 'Examined extension') + ' | ' + metric.replace('_', ' '))
            ax.set_xlabel('Percent')
            ax.grid(axis='x', alpha=.2)
            ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle(config.get('figure_title', 'Token identity in causal prompting') + (' | real interface smoke' if smoke else ''))
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(.62, .94), ncol=len(seeds), frameon=False)
    fig.text(.02, .025, 'Equal procedure weighting; all seeds retained. Black marks show means. Panels use separate scales.\n'
             'Thresholds selected on development only. Both populations were previously examined.', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .91))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('application_outcomes.' + suffix), dpi=170)
    plt.close(fig)
    lines = ['# Causal prompting outcomes', '', 'Real interface smoke; no scientific application conclusion.' if smoke else
             'Exploratory application evidence on previously examined procedures. All three seeds are retained.', '',
             '| Population | Method | Repeat removal (%) | Other-prompt retention (%) | First-prompt retention (%) |',
             '|---|---|---:|---:|---:|']
    for row in averages.to_dict('records'):
        lines.append(f"| {row['population']} | {row['method']} | {row['removal'] * 100:.4f} | {row['retention'] * 100:.4f} | {row['first_prompt'] * 100:.4f} |")
    lines += ['', 'Procedure and seed tables retain variation and undefined values. These results do not establish independent or clinical validity.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', smoke=smoke, methods=names, seeds=seeds,
        results=averages.to_dict('records'), input_sha256=inputs, source_sha256=digest(__file__)))
    atomic_write_json(run / ('smoke_summary_progress.json' if smoke else 'summary_progress.json'), dict(completed=1, total=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.run, args.smoke)
