import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checkpoint_io import read_json, atomic_write_json
from src.evaluation.realcolon_task import digest


def analyze(run):
    config = read_json(run / 'config.json')
    assert read_json(run / 'analysis/summary.json')['status'] == 'COMPLETE'
    output = run / 'comparison'
    output.mkdir(exist_ok=True)
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    keys = ['video', 'method', 'seed', 'variant']
    tables = [pd.read_csv(run / 'analysis/procedures.csv')]
    for name, field in [('detector', 'detector_run'), ('annotated', 'annotated_run')]:
        table = pd.read_csv(Path(config[field]) / 'analysis/procedures.csv')
        table = table[table.variant == 'budget1024'].copy()
        table['variant'] = name
        tables.append(table[keys + metrics])
    table = pd.concat(tables, ignore_index=True)
    assert len(table) == 240 and not table.duplicated(keys).any()
    table.to_csv(output / 'procedures.csv', index=False)
    table.groupby(['method', 'seed', 'variant'])[metrics].mean().to_csv(output / 'seeds.csv')
    summary = table.groupby(['method', 'variant'])[metrics].mean()
    summary.to_csv(output / 'summary.csv')
    averages = table.groupby(['video', 'method', 'variant'])[metrics].mean()
    averages.to_csv(output / 'procedure_means.csv')
    rng = np.random.default_rng(20261007)
    bootstrap = rng.integers(0, 5, size=(10000, 5))
    rows = []
    for method in sorted(table.method.unique()):
        variants = {name: averages.xs((method, name), level=['method', 'variant']).sort_index()
                    for name in ['detector', 'source_annotated', 'query_annotated', 'annotated']}
        assert all(v.index.equals(variants['detector'].index) for v in variants.values())
        differences = {name: variants[name] - variants['detector']
                       for name in ['source_annotated', 'query_annotated', 'annotated']}
        differences['interaction'] = variants['annotated'] - variants['source_annotated'] - variants['query_annotated'] + variants['detector']
        for name, values in differences.items():
            for metric in metrics:
                delta = values[metric].to_numpy()
                lower, upper = np.quantile(delta[bootstrap].mean(1), [.025, .975])
                rows.append(dict(method=method, contrast=name, metric=metric, mean=delta.mean(), lower=lower,
                    upper=upper, positive=int((delta > 0).sum()), negative=int((delta < 0).sum())))
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(output / 'contrasts.csv', index=False)
    order = ['detector', 'source_annotated', 'query_annotated', 'annotated']
    labels = ['Detector', 'Source only', 'Query only', 'Both']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    for method, color, line, label in [('frozen_token_sparse', '#15678A', '--', 'Frozen SAE'),
            ('adaptive_token_sparse', '#15678A', '-', 'Adapted SAE'),
            ('frozen_token_dense', '#CA653F', '--', 'Frozen Dense'),
            ('adaptive_token_dense', '#CA653F', '-', 'Adapted Dense')]:
        for ax, metric, title in zip(axes, metrics, ['Repeat removal', 'Other-lesion retention', 'First-prompt retention'], strict=True):
            values = summary.loc[method].loc[order, metric] * 100
            ax.plot(range(4), values, color=color, linestyle=line, marker='o', label=label)
            ax.set_xticks(range(4), labels, rotation=20, ha='right')
            ax.set_title(title)
            ax.set_ylabel('Procedure mean (%)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Source/query region intervention | Fixed models | 5 development procedures | 3 seeds')
    fig.savefig(output / 'factorial.png', dpi=180)
    plt.close(fig)
    report = '\n\n'.join(['# Source/query support factorial',
        'All twelve models on the same5394events in five exposed development procedures. '
        'Source-only and query-only conditions reuse saved codes; neither and both endpoints reuse1605and1735. '
        'Every condition recalibrates with other procedures and repeats actual detection matching. '
        'Annotation-defined support supplies extra labels and is a mechanism intervention.',
        summary.to_markdown(floatfmt='.6f'),
        'All seeds are averaged within procedure before paired10000-draw bootstrap intervals. '
        'Interaction equals both minus source-only minus query-only plus detector. '
        'Threshold recalibration is part of each condition, so effects include the recalibrated comparator.',
        contrasts.to_markdown(index=False, floatfmt='.6f'), '![Source/query effects](factorial.png)'])
    (output / 'report.md').write_text(report + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', procedure_rows=len(table),
        smoke_replay_checks=read_json(run / 'smoke/summary.json')['replay_checks'],
        source_sha256=digest(__file__), bootstrap_seed=20261007, bootstrap_draws=10000))
    print(summary.to_string(), flush=True)
    print(contrasts.to_string(index=False), flush=True)
    print(averages.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
