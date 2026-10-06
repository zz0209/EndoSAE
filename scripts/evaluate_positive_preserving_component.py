import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import evaluate_identity_feedback_component as feedback
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


@torch.no_grad()
def prepare(run, smoke, resume):
    config, event, _, original, videos = feedback.setup(run, smoke)
    reference = Path(config['reference_run']) / 'selection'
    root = run / ('smoke_selection' if smoke else 'selection')
    root.mkdir(exist_ok=True)
    feedback.torch_setup()
    device = torch.device(config['device'])
    models = feedback.load_models(original, device)
    folders = dict(feedback.model_specs(original))
    begin, all_rows = time.perf_counter(), []
    tolerance = config['positive_preservation_tolerance']
    for video in videos:
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for key, (model, mean, scale) in models.items():
            target = root / video / key
            target.mkdir(parents=True, exist_ok=True)
            old = reference / video / key
            old_receipt = read_json(old / 'complete.json')
            assert digest(old / 'vectors.npz') == old_receipt['vectors_sha256']
            token_paths = {e['episode_id']: Path(original['source_token_root']) / video / 'sources' / e['episode_id'] / 'observed_tokens.npz'
                           for e in manifest['episodes']}
            identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'), source=digest(__file__),
                reference=digest(old / 'complete.json'), model=digest(folders[key] / 'model.npz'), smoke=smoke,
                tokens={k: digest(p) for k, p in token_paths.items()}, events=digest(event / 'inputs' / video / 'events.json'))
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity and saved['vectors_sha256'] == digest(target / 'vectors.npz')
                all_rows.extend(saved['rows'])
                continue
            with np.load(old / 'vectors.npz') as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            rows = []
            for previous in old_receipt['rows']:
                row = dict(previous)
                identifier = row['episode']
                episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
                info = episode['source_info']
                assert not info['future_frames_used'] and not info['ground_truth_regions_used']
                assert all(s['output_index'] <= episode['click']['output_index'] for s in info['support'])
                with np.load(token_paths[identifier]) as saved:
                    raw, positions = saved['tokens'], saved['positions']
                temporal = np.unique(positions[:, 0])
                assert len(temporal) == len(info['support'])
                np.testing.assert_array_equal([int((positions[:, 0] == t).sum()) for t in temporal], [s['tokens'] for s in info['support']])
                values = ((raw.astype(float) - mean) / scale).astype(np.float32)
                assert model.identity_space == 'code' and model.method.startswith('token_')
                _, _, codes = model(torch.from_numpy(values[None]).to(device))
                positives = torch.stack([codes[0, torch.from_numpy(positions[:, 0] == t).to(device)].mean(0) for t in temporal])
                assert (positives.norm(dim=1) > 0).all()
                positives = F.normalize(positives, dim=1).cpu().numpy().astype(float)
                source = arrays[identifier + '__source']
                source_variants = feedback.encode(raw, model, mean, scale, range(len(source)), device)
                np.testing.assert_array_equal(source_variants[-1], source)
                baseline = np.clip(positives @ source, -1., 1.)
                numerator = positives @ source[:, None] - positives * source[None]
                denominator = np.sqrt((np.sum(positives ** 2, axis=1)[:, None] - positives ** 2) * (source @ source - source ** 2)[None])
                assert (denominator > 0).all()
                effects = np.clip(numerator / denominator, -1., 1.)
                feasible = np.all(effects >= baseline[:, None] - tolerance, axis=0)
                feature, selected_score, edited = -1, row['feedback_score'], source
                if row['feedback'] is not None and feasible.any():
                    negative_effects = arrays[identifier + '__feedback_effects']
                    best = int(np.argmin(np.where(feasible, negative_effects, np.inf)))
                    if negative_effects[best] < row['feedback_score'] - tolerance:
                        feature, selected_score, edited = best, float(negative_effects[best]), source_variants[best]
                error = 0.
                if feature >= 0:
                    for i, t in enumerate(temporal):
                        direct = feedback.encode(raw[positions[:, 0] == t], model, mean, scale, [feature], device)
                        score = float(np.clip(direct[feature] @ edited, -1., 1.))
                        error = max(error, abs(score - effects[i, feature]))
                        assert abs(score - effects[i, feature]) <= 2e-6
                        assert score >= baseline[i] - tolerance - 2e-7
                arrays[identifier + '__edited_source'] = edited
                arrays[identifier + '__positives'] = positives
                arrays[identifier + '__positive_baseline'] = baseline
                arrays[identifier + '__positive_effects'] = effects
                arrays[identifier + '__feasible'] = feasible
                row.update(feature=feature, original_feature=previous['feature'], edited_feedback_score=selected_score,
                    positive_views=len(temporal), feasible_coordinates=int(feasible.sum()), direct_positive_max_error=error)
                rows.append(row)
            np.savez_compressed(target / 'vectors.npz', **arrays)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, rows=rows,
                vectors_sha256=digest(target / 'vectors.npz'), future_labels_used=False))
            all_rows.extend(rows)
            print('POSITIVE_SELECTION', video, key, 'sources', len(rows), flush=True)
        atomic_write_json(root / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos)))
        pause_after_checkpoint(root / 'progress.json')
    if not (root / 'summary.json').exists():
        atomic_write_json(root / 'summary.json', dict(status='COMPLETE', rows=all_rows, seconds=time.perf_counter() - begin,
            models=list(models), created_at=datetime.now(timezone.utc).isoformat(), source_sha256=digest(__file__)))


def summarize(run, smoke):
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    config = read_json(run / 'config.json')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    current = pd.read_csv(root / 'sources.csv')
    current['policy'] = current.policy.replace({'single_coordinate': 'positive_preserving'})
    previous = pd.read_csv(Path(config['reference_run']) / 'evaluation/sources.csv')
    previous = previous[(previous.policy == 'single_coordinate') & previous.seed.isin(current.seed.unique()) & previous.video.isin(current.video.unique())].copy()
    previous['policy'] = 'negative_only'
    sources = pd.concat([current, previous], ignore_index=True)
    selected = read_json(run / ('smoke_selection' if smoke else 'selection') / 'summary.json')['rows']
    availability = {(r['video'], r['episode']): (r['positive_views'] > 1 and r['feedback'] is not None) for r in selected}
    sources['multiple_views'] = [availability[v, e] for v, e in zip(sources.video, sources.episode)]
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    groups = ['method', 'seed', 'video', 'policy', 'horizon']
    procedures = []
    for name, subset in [('complete', sources), ('reference_available', sources[sources.feedback_available]),
                         ('multiple_views', sources[sources.multiple_views])]:
        local = subset.groupby(groups)[metrics].mean().reset_index()
        local['population'] = name
        procedures.append(local)
    procedures = pd.concat(procedures, ignore_index=True)
    seeds = procedures.groupby(['method', 'seed', 'policy', 'horizon', 'population'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy', 'horizon', 'population'])[metrics].mean().reset_index()
    for name, table in [('sources', sources), ('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'feedback'} for r in selected]).to_csv(output / 'selections.csv', index=False)
    fig, axes = feedback.plt.subplots(3, 2, figsize=(12, 11), layout='constrained')
    for row, population in enumerate(['complete', 'reference_available', 'multiple_views']):
        for ax, metric in zip(axes[row], ['repeat_removal', 'other_retention']):
            local = summary[(summary.population == population) & (summary.horizon == 'full')]
            for method, group in local.groupby('method'):
                group = group.set_index('policy').loc[['unchanged', 'negative_only', 'positive_preserving']]
                ax.plot(['Unchanged', 'Negative only', 'Positive preserved'], group[metric] * 100, marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
            ax.set_title(population.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title() + ' (%)')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Existing source-window positive constraints | Examined development')
    fig.savefig(output / 'positive_preservation.png', dpi=170)
    feedback.plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', rows=len(summary),
        aggregation='Defined lesion means, then sources, procedures and seeds; input-defined populations and horizons separate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'score', 'evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run, args.smoke, args.resume)
    elif args.phase == 'score':
        feedback.score(args.run, args.smoke, args.resume)
    elif args.phase == 'evaluate':
        feedback.evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
