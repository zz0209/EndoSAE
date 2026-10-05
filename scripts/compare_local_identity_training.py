import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def compare(run, reference):
    current = read_json(run / 'training_summary.json')
    previous = read_json(reference / 'training_summary.json')
    assert current['status'] == previous['status'] == 'COMPLETE'
    assert read_json(run / 'folds.json') == read_json(reference / 'folds.json')
    assert read_json(run / 'records.json') == read_json(reference / 'records.json')
    lookup = {(r['method'], r['seed'], r['fold']): r for r in previous['outputs']}
    checks = []
    for item in current['outputs']:
        original = lookup[item['method'], item['seed'], item['fold']]
        left, right = Path(item['directory']), Path(original['directory'])
        a, b = read_json(left / 'sequence.json'), read_json(right / 'sequence.json')
        count = min(len(a), len(b))
        assert a[:count] == b[:count]
        with np.load(left / 'normalization.npz') as x, np.load(right / 'normalization.npz') as y:
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
        current_step = item['summary']['steps']
        old_step = original['summary']['steps']
        with np.load(left / f'held_{current_step:04d}.npz') as x, np.load(right / f'held_{old_step:04d}.npz') as y:
            for key in ['indices', 'source', 'query', 'same_identity', 'cross_interval']:
                np.testing.assert_array_equal(x[key], y[key])
        checks.append(dict(method=item['method'], seed=item['seed'], fold=item['fold'],
            common_training_steps=count, current_steps=current_step, original_steps=old_step,
            identical_pairs_normalization_and_sampling=True))
    rows = []
    for label, folder in [('pooled_training', reference), ('local_training', run)]:
        table = pd.read_csv(folder / 'analysis/procedures.csv')
        table = table[table.method.isin(['token_sparse', 'token_dense'])].copy()
        table['objective'] = label
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    metrics = ['recall', 'negative_retention', 'auroc', 'cross_interval_recall']
    seeds = table.groupby(['objective', 'method', 'partition', 'seed'])[metrics].mean().reset_index()
    procedures = table.groupby(['objective', 'method', 'partition', 'video_id'])[metrics].mean().reset_index()
    paired = table[table.objective == 'local_training'].merge(table[table.objective == 'pooled_training'],
        on=['method', 'seed', 'partition', 'video_id', 'fold'], suffixes=('_local', '_pooled'), validate='one_to_one')
    for metric in metrics:
        paired[metric + '_difference'] = paired[metric + '_local'] - paired[metric + '_pooled']
    output = run / 'comparison'
    output.mkdir(exist_ok=True)
    seeds.to_csv(output / 'seeds.csv', index=False)
    procedures.to_csv(output / 'procedures.csv', index=False)
    paired.to_csv(output / 'paired.csv', index=False)
    differences = paired.groupby(['method', 'partition', 'video_id'])[[m + '_difference' for m in metrics]].mean().reset_index()
    differences.to_csv(output / 'procedure_differences.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    combinations = [('pooled_training', 'token_sparse'), ('local_training', 'token_sparse'),
                    ('pooled_training', 'token_dense'), ('local_training', 'token_dense')]
    for axis, metric in zip(axes, ['recall', 'auroc'], strict=True):
        for i, (objective, method) in enumerate(combinations):
            values = seeds[(seeds.objective == objective) & (seeds.method == method) & (seeds.partition == 'validation')][metric]
            axis.bar(i, values.mean(), color='#176B87' if method == 'token_sparse' else '#BAC6CE')
            axis.scatter(i + np.linspace(-.12, .12, len(values)), values, color='black', s=20)
        axis.set_xticks(range(4), ['Pooled\nSAE', 'Local\nSAE', 'Pooled\ndense', 'Local\ndense'])
        axis.set_ylim(0, 1)
        axis.set_title('Same-lesion recall' if metric == 'recall' else 'Within-procedure AUROC')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Identity training and matched controls | exposed validation\nBars: mean across seeds; dots: every seed. Each uses training-selected checkpoints.')
    fig.savefig(output / 'training_comparison.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'verification.json', dict(status='PASS', checks=checks,
        input_hashes={str(p): digest(p) for p in [run / 'training_summary.json', reference / 'training_summary.json']},
        source_sha256=digest(__file__)))
    print(seeds.to_csv(index=False), flush=True)
    print(differences.to_csv(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    compare(args.run, args.reference)
