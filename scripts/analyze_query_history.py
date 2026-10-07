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


def analyze(run, smoke):
    directory = run / ('smoke' if smoke else 'analysis')
    result = read_json(directory / 'summary.json')
    assert result['status'] == 'COMPLETE'
    events = pd.read_csv(directory / 'events.csv')
    assert not events.duplicated(['method', 'seed', 'variant', 'video', 'event_id']).any()
    output = run / ('smoke_comparison' if smoke else 'comparison')
    output.mkdir(exist_ok=True)
    rows, checks = [], []
    for keys, group in events.groupby(['method', 'seed', 'variant', 'video']):
        same, other = group[group.same_identity], group[~group.same_identity]
        first = other[other.first_prompt]
        row = dict(zip(['method', 'seed', 'variant', 'video'], keys),
            repeat_removal=(~same.retained).mean(), other_retention=other.retained.mean(),
            first_retention=first.retained.mean(), repeat_events=len(same), other_events=len(other),
            first_events=len(first), removed_repeats=int((~same.retained).sum()),
            lost_other=int((~other.retained).sum()), lost_first=int((~first.retained).sum()))
        assert np.isclose(row['repeat_removal'], row['removed_repeats'] / len(same))
        assert np.isclose(row['other_retention'], 1 - row['lost_other'] / len(other))
        assert np.isclose(row['first_retention'], 1 - row['lost_first'] / len(first))
        rows.append(row)
    procedures = pd.DataFrame(rows)
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    seeds = procedures.groupby(['method', 'seed', 'variant'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'variant'])[metrics].mean().reset_index()
    procedures.to_csv(output / 'procedures.csv', index=False)
    seeds.to_csv(output / 'seeds.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    for row in read_json(directory / 'calibration.json'):
        assert row['video'] not in row['fit_procedures'] and len(row['fit_procedures']) == 4
        checks.append(dict(key=row['key'], video=row['video'], variant=row['variant'], exclusion=True))
    availability = pd.read_csv(directory / 'availability.csv')
    availability = availability[availability.key == availability.key.iloc[0]]
    coverage = availability.groupby(['video', 'same_identity', 'first_prompt']).history_available.agg(['count', 'sum', 'mean'])
    coverage.to_csv(output / 'coverage.csv')
    contrasts = []
    if not smoke:
        rng = np.random.default_rng(20261007)
        mean = procedures.groupby(['method', 'variant', 'video'])[metrics].mean()
        for method in summary.method.unique():
            for variant, control in [('history_available_current', 'current'),
                    ('history_mean', 'history_available_current'), ('history_minimum', 'history_available_current')]:
                difference = mean.loc[(method, variant)] - mean.loc[(method, control)]
                assert len(difference) == 5
                indices = rng.integers(0, 5, size=(10000, 5))
                for metric in metrics:
                    values = difference[metric].to_numpy()
                    interval = np.quantile(values[indices].mean(1), [.025, .975])
                    contrasts.append(dict(method=method, variant=variant, control=control, metric=metric,
                        difference=float(values.mean()), low=float(interval[0]), high=float(interval[1]),
                        positive_procedures=int((values > 0).sum()), negative_procedures=int((values < 0).sum())))
        pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
        ids = ['method', 'seed', 'video', 'event_id']
        first = events[(~events.same_identity) & events.first_prompt]
        first.pivot(index=ids, columns='variant', values='retained').to_csv(output / 'first_prompts.csv')
    order = result['variants']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), layout='constrained')
    for method, group in summary.groupby('method'):
        group = group.set_index('variant').loc[order]
        for ax, metric in zip(axes, metrics, strict=True):
            ax.plot(range(len(order)), 100 * group[metric], marker='o', label=method.replace('_token_', ' '))
            ax.set_xticks(range(len(order)), [v.replace('history_available_current', 'History-ready current').replace('history_', '').replace('_', ' ') for v in order], rotation=25, ha='right')
            ax.set_title(metric.replace('_', ' ').capitalize())
            ax.set_ylabel('Procedure mean (%)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    for ax, metric in zip(axes, metrics, strict=True):
        values = 100 * summary[metric]
        margin = max(1., float(values.max() - values.min()) * .1)
        ax.set_ylim(max(0., float(values.min()) - margin), min(100., float(values.max()) + margin))
    fig.suptitle('Causal query history | Fixed models and source | Five development procedures')
    fig.savefig(output / 'query_history.png', dpi=180)
    plt.close(fig)
    report = '\n\n'.join(['# Causal query history',
        'Current: original scores. History-ready current: current score only when both historical observations are available. Mean and minimum: the current and two historical scores under exactly the same availability. Unavailable detections are retained. Each variant has other-procedure calibration.',
        summary.to_markdown(index=False, floatfmt='.6f'),
        '## Paired procedure effects', pd.DataFrame(contrasts).to_markdown(index=False, floatfmt='.6f') if contrasts else 'Smoke baseline only.',
        '## History coverage', coverage.to_markdown(floatfmt='.6f'),
        'Intervals resample five procedures after averaging three initializations. All procedures retain development exposure. Selected events do not estimate continuous prompt time. History availability, additional image information and recalibration jointly determine operational outcomes; the matched waiting control isolates the added score-history comparison. Temporal aggregation has no algorithmic novelty claim.',
        'All initialization/procedure rows and first-prompt decisions are saved beside this report.'])
    (output / 'report.md').write_text(report + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', events=len(events),
        aggregate_checks=3 * len(procedures), calibration_checks=checks,
        inputs={str(directory / name): digest(directory / name) for name in ['events.csv', 'summary.json', 'calibration.json']},
        source_sha256=digest(__file__)))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(args.run, args.smoke)
