import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.causal_identity_memory import METHODS
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest

METRICS = {
    'removal': 'source_suppression__procedure_values',
    'retention': 'all_other__baseline_qualified_retention__procedure_values',
    'first_prompt': 'all_other__first_baseline_frame_retention__procedure_values',
}
LABELS = ['Fixed memory', 'Track updates', 'Confidence updates', 'SAE guarded updates', 'Dense guarded updates']


def summarize(run, smoke, destination=None):
    config = read_json(run / 'config.json')
    output = destination or (run / 'smoke_analysis' if smoke else run / 'analysis')
    output.mkdir(exist_ok=False)
    seeds = config['seeds'][:1] if smoke else config['seeds']
    rows, source_rows, writes, inputs, times = [], [], [], {}, []
    for seed in seeds:
        base = run / 'smoke_verified' if smoke else run / 'evaluation' / f'seed{seed}'
        receipt = read_json(base / 'summary.json')
        assert receipt['status'] == 'COMPLETE'
        times.append(dict(seed=seed, complete_replay_seconds=receipt['seconds']))
        points = read_json(base / 'operating_points.json')['methods']
        inputs[str(base / 'identity.json')] = digest(base / 'identity.json')
        for population in ('development', 'extension'):
            for method in METHODS:
                for mute in config['mute_seconds']:
                    name = f'{method}_mute{mute}'
                    path = base / population / (name + '__score_procedure_curve.npz')
                    inputs[str(path)] = digest(path)
                    with np.load(path, allow_pickle=False) as curve:
                        threshold = points[name]['threshold']
                        index = int(np.searchsorted(curve['threshold'], threshold, side='right') - 1) if population == 'development' else 0
                        assert index >= 0
                        if population == 'development':
                            assert curve['threshold'][index] == threshold
                        for number, video in enumerate(curve['videos'].tolist()):
                            rows.append(dict(seed=seed, population=population, method=method, mute=mute, video=video,
                                threshold=threshold, **{key: float(curve[value][number, index]) for key, value in METRICS.items()}))
                    for folder in sorted((base / population).glob('*/sources/*')):
                        episode = read_json(folder / 'summary.json')
                        if not episode['click']['available']:
                            continue
                        with np.load(folder / (name + '__score_curve.npz')) as curve:
                            i = int(np.searchsorted(curve['threshold'], threshold, side='right') - 1) if population == 'development' else 0
                            assert i >= 0
                            source = list(curve['lesion_ids']).index(episode['source_lesion_id'])
                            others = episode['groups']['all_other']
                            source_rows.append(dict(seed=seed, population=population, method=method, mute=mute,
                                video=episode['video'], episode_id=episode['episode_id'],
                                source_baseline_seconds=float(curve['baseline_seconds'][source]),
                                source_removed_seconds=float(curve['acknowledged_removed_seconds'][i]),
                                other_baseline_seconds=float(curve['baseline_seconds'][others].sum()),
                                other_retained_seconds=float(curve['retained_baseline_seconds'][i, others].sum())))
            for path in sorted((base / population).glob('*/sources/*/writes.json')):
                for method, events in read_json(path).items():
                    counts = Counter(event['annotation_status'] for event in events)
                    writes.append(dict(seed=seed, population=population, video=path.parents[2].name,
                        episode_id=path.parent.name, method=method, writes=len(events),
                        **{status: counts[status] for status in ('source', 'other', 'unmatched', 'unknown')}))
    table = pd.DataFrame(rows)
    for method in ('static', 'track_memory', 'reference_memory'):
        counts = table[table.method == method].groupby(['population', 'mute', 'video'])[
            ['threshold'] + list(METRICS)].nunique(dropna=False)
        assert (counts == 1).all().all(), method
    table.to_csv(output / 'procedure_metrics.csv', index=False)
    pd.DataFrame(source_rows).to_csv(output / 'episode_seconds.csv', index=False)
    write_table = pd.DataFrame(writes)
    write_table.to_csv(output / 'write_annotations.csv', index=False)
    write_aggregates = write_table.groupby(['population', 'method', 'seed'])[
        ['writes', 'source', 'other', 'unmatched', 'unknown']].sum().reset_index()
    write_aggregates.to_csv(output / 'write_counts_by_seed.csv', index=False)
    per_seed = table.groupby(['population', 'method', 'mute', 'seed'])[list(METRICS)].mean().reset_index()
    per_seed.to_csv(output / 'seed_metrics.csv', index=False)
    aggregate = per_seed.groupby(['population', 'method', 'mute'])[list(METRICS)].agg(['mean', 'min', 'max'])
    aggregate.columns = ['_'.join(column) for column in aggregate.columns]
    aggregates = aggregate.reset_index().to_dict('records')
    contrasts, differences = [], []
    for comparator in ('static', 'reference_memory', 'dense_guard'):
        for candidate in METHODS:
            if candidate == comparator:
                continue
            merged = table[table.method == candidate].merge(table[table.method == comparator],
                on=['population', 'mute', 'seed', 'video'], suffixes=('_candidate', '_comparator'), validate='one_to_one')
            for row in merged.to_dict('records'):
                differences.append(dict(candidate=candidate, comparator=comparator,
                    **{key: row[key] for key in ('population', 'mute', 'seed', 'video')},
                    **{key: row[key + '_candidate'] - row[key + '_comparator'] for key in METRICS}))
    difference_table = pd.DataFrame(differences)
    difference_table.to_csv(output / 'paired_procedure_differences.csv', index=False)
    for keys, group in difference_table.groupby(['population', 'mute', 'candidate', 'comparator']):
        procedures = group.groupby('video')[list(METRICS)].mean()
        rng = np.random.default_rng(20261003)
        indices = rng.integers(len(procedures), size=(10000, len(procedures)))
        for metric in METRICS:
            values = procedures[metric].to_numpy()
            sampled = values[indices].mean(axis=1)
            low, high = np.quantile(sampled, [.025, .975])
            contrasts.append(dict(population=keys[0], mute=int(keys[1]), candidate=keys[2], comparator=keys[3], metric=metric,
                mean=float(values.mean()), ci_low=float(low), ci_high=float(high), procedures=len(values),
                positive_procedures=int((values > 1e-12).sum()), negative_procedures=int((values < -1e-12).sum())))
    pd.DataFrame(contrasts).to_csv(output / 'paired_intervals.csv', index=False)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    colors = ['#0072B2', '#D55E00', '#009E73']
    for i, population in enumerate(('development', 'extension')):
        for j, metric in enumerate(('removal', 'retention', 'first_prompt')):
            ax = axes[j, i]
            for number, method in enumerate(METHODS):
                for mute, marker, shift in ((0, 'o', -.13), (30, '^', .13)):
                    selected = per_seed[(per_seed.population == population) & (per_seed.method == method) & (per_seed.mute == mute)]
                    for k, row in enumerate(selected.to_dict('records')):
                        ax.scatter(row[metric] * 100, number + shift, marker=marker, color=colors[k], s=35)
            ax.set_yticks(range(len(METHODS)), LABELS)
            ax.invert_yaxis()
            ax.set_title(('Development' if i == 0 else 'Examined extension') + ' | ' + metric)
            ax.set_xlabel('Percent')
            ax.grid(axis='x', alpha=.2)
            ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Causal memory updates' + (' | real smoke' if smoke else ''))
    fig.legend([plt.Line2D([], [], color=colors[i], marker='o', linestyle='') for i in range(len(seeds))],
               [str(seed) for seed in seeds], loc='upper center', bbox_to_anchor=(.5, .963), ncol=len(seeds), frameon=False)
    fig.text(.03, .025, 'Circles: memory only. Triangles: memory + 30 s mute. Colors: three fixed component seeds.\n'
        'Equal procedure weights; fixed extension thresholds. Both populations were previously examined. Separate panel scales.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .925))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('application_comparison.' + suffix), dpi=180)
    plt.close(fig)
    report = ['# Causal memory updates', '',
        'Complete replay of previously examined development and extension procedures. Each population contains five procedures in the complete batch. '
        'Each method uses one development-selected threshold under the same retention floor and first-prompt requirement; extension thresholds are unchanged.', '',
        '| Population | Method | Mute (s) | Repeat removal | Other retention | First prompt |',
        '|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        report.append(f"| {row['population']} | {row['method']} | {row['mute']} | " +
            ' | '.join(f"{row[metric + '_mean'] * 100:.4f}%" for metric in METRICS) + ' |')
    report += ['', '## Memory admission', '',
        '| Population | Method | Writes | Source matched | Other lesion matched | Unmatched | Unknown |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in write_aggregates.groupby(['population', 'method'])[['writes', 'source', 'other', 'unmatched', 'unknown']].mean().reset_index().to_dict('records'):
        report.append(f"| {row['population']} | {row['method']} | " +
            ' | '.join(f"{row[key]:.2f}" for key in ('writes', 'source', 'other', 'unmatched', 'unknown')) + ' |')
    report += ['', 'Write counts sum observations within each population, then average the specified seeds. They are descriptive counts of correlated observations.', '',
        'Numbers are means across all specified seeds after equal procedure weighting. Seed ranges and every procedure are in analysis/. '
        'Seeds repeat the same patients and provide no additional independent sample size. Paired intervals resample procedures after averaging seeds; '
        'they describe exploratory variation with five procedures, without selection-adjusted inference.', '',
        'Episode seconds retain their episode identities; the same unacknowledged lesion can occur in several acknowledgement episodes. '
        'Do not sum these into unique patient duration. Write annotations are assigned only after decisions. An unmatched write does not establish absence of source tissue.', '',
        'Complete replay times include data loading, all ten conditions, curve construction and output writing; they do not measure online end-to-end latency. '
        'Template updates and immutable anchors are established mechanisms; a generic memory benefit alone does not establish an SAE-specific contribution.']
    report_path = output / 'report.md' if smoke else run / 'report.md'
    report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', smoke=smoke, aggregates=aggregates,
        paired_intervals=contrasts, write_counts_by_seed=write_aggregates.to_dict('records'),
        timings=times, input_sha256=inputs, source_sha256=digest(__file__)))
    atomic_write_json(output / 'manifest.json', dict(files={p.name: digest(p) for p in output.iterdir() if p.suffix in ('.csv', '.png', '.pdf')}))
    print('SUMMARY_COMPLETE', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summarize(args.run, args.smoke, args.output)
