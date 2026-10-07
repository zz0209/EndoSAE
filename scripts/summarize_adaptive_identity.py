import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_query_conditioned_components import boundary, measurements
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def predictions(folder):
    records = read_json(folder / 'held_records.json')
    with np.load(folder / 'held.npz', allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    unit = data['pooled_codes'].astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    np.testing.assert_allclose(unit, data['embeddings'], atol=3e-7, rtol=1e-6)
    direct = np.sum(unit[data['source']] * unit[data['query']], 1)
    np.testing.assert_allclose(direct, data['scores'], atol=3e-7, rtol=1e-6)
    for i, j, same in zip(data['source'], data['query'], data['same_identity'], strict=True):
        a, b = records[int(i)], records[int(j)]
        assert a['video_id'] == b['video_id'] and a['end_frame'] < b['start_frame']
        assert bool(same) == (a['lesion_id'] == b['lesion_id'])
        assert a['original_observation'] and b['original_observation']
    data['videos'] = np.array([records[int(i)]['video_id'] for i in data['source']])
    return records, data, float(np.abs(direct - data['scores']).max())


def bootstrap(values, seed=20261007):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    lower, upper = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), lower=float(lower), upper=float(upper), procedures=len(values))


def summarize(run, additional_runs=(), output=None):
    config = read_json(run / 'config.json')
    root = Path(config['storage_root'])
    training = read_json(root / 'training_summary.json')
    assert training['status'] == 'COMPLETE' and not training['smoke']
    output = output or run / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    rows, checks, fidelity, learning = [], [], [], []
    for item in training['outputs']:
        folder = Path(item['directory'])
        reference = root / 'frozen' / item['method'] / f"seed{item['seed']}" / f"fold{item['fold']}"
        records, data, error = predictions(folder)
        reference_records, original, _ = predictions(reference)
        assert records == reference_records
        for key in ['source', 'query', 'same_identity']:
            np.testing.assert_array_equal(data[key], original[key])
        assert not set(item['fit_videos']) & set(item['held_videos'])
        assert set(data['videos']) <= set(item['held_videos'])
        labels, videos = data['same_identity'].astype(bool), data['videos']
        eligible = [video for video in np.unique(videos) if len(set(labels[videos == video])) == 2]
        metadata = dict(condition='adaptive' if item['adaptive'] else 'frozen', method=item['method'],
                        seed=item['seed'], fold=item['fold'])
        for video in eligible:
            calibration = np.isin(videos, [v for v in eligible if v != video])
            selected = videos == video
            assert calibration.any() and not np.any(calibration & selected)
            threshold = boundary(data['scores'][calibration], videos[calibration], labels[calibration], config['negative_quantile'])
            reference_threshold = boundary(original['scores'][calibration], videos[calibration], labels[calibration], config['negative_quantile'])
            result = measurements(original['scores'][selected], data['scores'][selected], labels[selected], threshold, reference_threshold)
            rows.append(dict(metadata, video=video, **result))
        for row in read_json(folder / 'held_metrics.json'):
            fidelity.append(dict(metadata, **row))
        for row in read_json(folder / 'history.json'):
            learning.append(dict(metadata, **row))
        checks.append(dict(metadata, pairs=len(labels), score_max_error=error, prediction_sha256=digest(folder / 'held.npz')))
    source_analyses = []
    seeds = set(config['seeds'])
    for extra in additional_runs:
        extra_config = read_json(extra / 'config.json')
        for key in config.keys() - {'run_id', 'storage_root', 'seeds', 'discovery_run', 'question', 'hypothesis', 'selection', 'decision', 'budget'}:
            assert config[key] == extra_config[key], key
        assert not seeds.intersection(extra_config['seeds'])
        seeds.update(extra_config['seeds'])
        source = extra / 'analysis'
        verification = read_json(source / 'verification.json')
        assert verification['status'] == 'PASS'
        rows.extend(pd.read_csv(source / 'procedures.csv').to_dict('records'))
        fidelity.extend(pd.read_csv(source / 'fidelity_clips.csv').to_dict('records'))
        learning.extend(pd.read_csv(source / 'learning.csv').to_dict('records'))
        checks.extend(verification['checks'])
        source_analyses.append(dict(run=str(extra), procedures_sha256=digest(source / 'procedures.csv')))
    table = pd.DataFrame(rows)
    assert not table.duplicated(['condition', 'method', 'seed', 'video']).any()
    assert table.groupby(['condition', 'method', 'video']).seed.nunique().eq(len(seeds)).all()
    table.to_csv(output / 'procedures.csv', index=False)
    metrics = ['auroc', 'recall', 'protection', 'protected99_capacity', 'matched_capacity']
    table.groupby(['condition', 'method', 'seed'])[metrics].mean().reset_index().to_csv(output / 'seeds.csv', index=False)
    aggregate = table.groupby(['condition', 'method'])[metrics].mean().reset_index()
    aggregate.to_csv(output / 'summary.csv', index=False)
    clips = pd.DataFrame(fidelity)
    clips.to_csv(output / 'fidelity_clips.csv', index=False)
    fidelity_metrics = ['reconstruction_nmse', 'native_anchor_nmse', 'representation_drift', 'local_active']
    clips.groupby(['condition', 'method', 'seed', 'fold', 'video_id'])[fidelity_metrics].mean().reset_index().to_csv(output / 'fidelity_procedures.csv', index=False)
    history = pd.DataFrame(learning)
    history.to_csv(output / 'learning.csv', index=False)
    effects = []
    averaged = table.groupby(['condition', 'method', 'video'])[metrics].mean().reset_index()
    for metric in metrics:
        pivot = averaged.pivot(index='video', columns=['condition', 'method'], values=metric)
        assert not pivot.isna().any().any()
        sparse_change = pivot['adaptive', 'token_sparse'] - pivot['frozen', 'token_sparse']
        dense_change = pivot['adaptive', 'token_dense'] - pivot['frozen', 'token_dense']
        for name, values in [('visual_adaptation_sparse', sparse_change), ('visual_adaptation_dense', dense_change),
                             ('adapted_sparse_minus_dense', pivot['adaptive', 'token_sparse'] - pivot['adaptive', 'token_dense']),
                             ('interaction', sparse_change - dense_change)]:
            effects.append(dict(contrast=name, metric=metric, **bootstrap(values)))
    effect = pd.DataFrame(effects)
    effect.to_csv(output / 'paired_effects.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    palette = {'token_sparse': '#0072B2', 'token_dense': '#D55E00'}
    for ax, metric, title in zip(axes, ['auroc', 'recall', 'protection'],
                               ['Identity AUROC', 'Calibrated same-identity recall', 'Other-identity protection'], strict=True):
        for method, color in palette.items():
            subset = averaged[averaged.method == method].pivot(index='video', columns='condition', values=metric)
            for _, row in subset.iterrows():
                ax.plot([0, 1], row[['frozen', 'adaptive']], color=color, alpha=.15, lw=.7)
            ax.plot([0, 1], subset[['frozen', 'adaptive']].mean(), color=color, marker='o', lw=2,
                    label='TopK64 SAE' if method == 'token_sparse' else 'Dense')
        ax.set(xticks=[0, 1], xticklabels=['Frozen visual', 'Adapted visual'], title=title, ylim=(0, 1.02))
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(output / 'factorial_identity.pdf')
    fig.savefig(output / 'factorial_identity.png', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    for (condition, method), values in history.groupby(['condition', 'method']):
        mean = values.groupby('step')[['identity', 'reconstruction']].mean()
        for ax, metric in zip(axes, ['identity', 'reconstruction'], strict=True):
            ax.plot(mean.index, mean[metric], color=palette[method], linestyle='-' if condition == 'adaptive' else '--',
                    label=condition + ' ' + ('SAE' if method == 'token_sparse' else 'Dense'))
            ax.set(xlabel='Updates', ylabel=metric + ' loss')
            ax.spines[['right', 'top']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / 'training_curves.pdf')
    fig.savefig(output / 'training_curves.png', dpi=220)
    plt.close(fig)
    lines = ['\\begin{tabular}{llrrr}', '\\toprule',
             'Visual encoder & Identity code & AUROC & Recall (\\%) & Protection (\\%) \\\\', '\\midrule']
    for row in aggregate.itertuples():
        lines.append(f"{row.condition.title()} & {'TopK64 SAE' if row.method == 'token_sparse' else 'Dense'} & "
                     f'{row.auroc:.3f} & {100 * row.recall:.2f} & {100 * row.protection:.2f} \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    (output / 'factorial_rows.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', models=len(checks), checks=checks,
        procedures=int(table.video.nunique()), paired_rows=len(table), bootstrap_repetitions=10000,
        seeds=sorted(seeds), source_analyses=source_analyses,
        scope='Saved score reconstruction, chronological identity labels, procedure isolation, paired inputs and procedure bootstrap.'))
    report_path = output / 'report.md' if additional_runs else run / 'report.md'
    report_path.write_text('# Visual adaptation and sparse identity\n\n' +
        f"Three fixed procedure-exclusion folds; {len(seeds)} predeclared initializations; {config['steps']} updates in all four conditions. "
        'Thresholds use other eligible excluded procedures from the same fold. These are development measurements.\n\n' +
        aggregate.to_markdown(index=False) + '\n\nPaired procedure effects and95%percentile bootstrap intervals:\n\n' +
        effect.to_markdown(index=False) + '\n', encoding='utf-8')
    print(aggregate.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--additional-run', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summarize(args.run, args.additional_run, args.output)
