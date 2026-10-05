import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from evaluate_prompt_event_components import application, load_models, model_specs
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def replay(manifest, episode, positions, scores, threshold):
    lookup = {int(p): i for i, p in enumerate(positions)}
    chosen = [(i, e) for i, e in enumerate(manifest['events']) if e['episode'] == episode]
    retained = np.zeros((len(scores), len(chosen)), dtype=bool)
    columns = {}
    for j, (_, event) in enumerate(chosen):
        columns.setdefault(event['output'], []).append((j, event['lesion']))
    for output, targets in columns.items():
        frame = manifest['frames'][str(output)]
        keep = np.ones((len(scores), len(frame['positions'])), dtype=bool)
        for j, position in enumerate(frame['positions']):
            if position in lookup:
                keep[:, j] = scores[:, lookup[position]] < threshold
        patterns, inverse = np.unique(keep, axis=0, return_inverse=True)
        labels = np.zeros((len(patterns), len(targets)), dtype=bool)
        for j, pattern in enumerate(patterns):
            boxes = [box for box, selected in zip(frame['detections'], pattern) if selected]
            state = application.shared.memory.acknowledgement.detection.overlap(
                boxes, frame['frame']['original_boxes_xyxy'])
            labels[j] = [lesion in state['detected_lesion_ids'] for _, lesion in targets]
        retained[:, [column for column, _ in targets]] = labels[inverse]
    return np.array([i for i, _ in chosen]), retained


def counts(retained, before, same):
    wrong = ~same & ~before
    correct_repeat = same & ~before
    correct_other = ~same & before
    return dict(corrected=(retained & wrong).sum(1),
                repeat_damage=(retained & correct_repeat).sum(1),
                other_damage=(~retained & correct_other).sum(1),
                repeat_gain=(~retained & same & before).sum(1))


@torch.no_grad()
def evaluate(run, smoke, resume):
    config = read_json(run / 'config.json')
    event_run = Path(config['event_run'])
    event_config = read_json(event_run / 'config.json')
    original = read_json(Path(event_config['application_run']) / 'config.json')
    original.update(training_runs=event_config['training_runs'], methods=event_config['methods'],
                    seeds=[config['smoke_seed']] if smoke else event_config['seeds'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    models = load_models(original, device)
    model_folders = dict(model_specs(original))
    videos = [config['smoke_video']] if smoke else original['development_videos']
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    root.mkdir(exist_ok=True)
    begin = time.perf_counter()
    completed = 0
    for video in videos:
        inputs = event_run / 'inputs' / video
        manifest = read_json(inputs / 'events.json')
        receipt = read_json(inputs / 'complete.json')
        for name, expected in receipt['assets'].items():
            assert digest(inputs / name) == expected
        with np.load(inputs / 'tokens.npz') as saved:
            raw, bounds, positions = saved['tokens'].copy(), saved['offsets'].copy(), saved['positions'].copy()
        lookup = {int(p): i for i, p in enumerate(positions)}
        for key, (model, mean, scale) in models.items():
            target = root / video / key
            target.mkdir(parents=True, exist_ok=True)
            source = event_run / 'evaluation' / video / key
            identity = dict(config=digest(run / 'config.json'), source=digest(__file__),
                            inputs=digest(inputs / 'complete.json'), previous=digest(source / 'complete.json'),
                            model=digest(model_folders[key] / 'model.npz'), smoke=smoke)
            if (target / 'complete.json').exists():
                assert resume and read_json(target / 'complete.json')['identity'] == identity
                completed += 1
                continue
            started = time.perf_counter()
            assert model.identity_space == 'code'
            pooled, unit = [], []
            for i in range(len(positions)):
                values = ((raw[bounds[i]:bounds[i + 1]].astype(float) - mean) / scale).astype(np.float32)
                projected, _, local = model(torch.from_numpy(values[None]).to(device))
                pooled.append(local.mean(1)[0])
                unit.append(projected[0].cpu().numpy().astype(float))
            pooled = torch.stack(pooled)
            unit = np.stack(unit)
            with np.load(source / 'scores.npz') as saved:
                np.testing.assert_array_equal(unit, saved['unit_codes'])
                before_scores = {e['episode_id']: saved[e['episode_id'] + '__before'].copy() for e in manifest['episodes']}
            old_events = pd.read_csv(source / 'events.csv').query("variant == 'before'").sort_values('event_id')
            assert old_events.event_id.tolist() == list(range(len(manifest['events'])))
            threshold = float(old_events.threshold.iloc[0])
            before = old_events.retained.to_numpy(bool)
            same = old_events.same_identity.to_numpy(bool)
            dimension = unit.shape[1]
            retained = np.empty((dimension + 1, len(before)), dtype=bool)
            retained[0] = before
            scalar_error, formula_error, boundary_checks = 0., 0., 0
            for episode in manifest['episodes']:
                indices, unchanged = replay(manifest, episode['episode_id'], positions,
                                            before_scores[episode['episode_id']][None], threshold)
                np.testing.assert_array_equal(unchanged[0], before[indices])
            for start in range(0, dimension, config['feature_batch']):
                features = np.arange(start, min(start + config['feature_batch'], dimension))
                edited = pooled[None].expand(len(features), -1, -1).clone()
                edited[torch.arange(len(features), device=device), :, torch.tensor(features, device=device)] = 0
                assert torch.isfinite(edited).all() and (edited.norm(dim=-1) > 0).all()
                changed = F.normalize(edited, dim=-1).cpu().numpy().astype(float)
                for episode in manifest['episodes']:
                    identifier = episode['episode_id']
                    source_index = lookup[episode['source_position']]
                    scores = np.clip(np.einsum('bnd,bd->bn', changed, changed[:, source_index], optimize=False), -1., 1.)
                    direct = np.clip(np.array([value @ changed[0, source_index] for value in changed[0]]), -1., 1.)
                    scalar_error = max(scalar_error, float(np.max(np.abs(direct - scores[0]))))
                    np.testing.assert_allclose(direct, scores[0], atol=2e-12, rtol=0)
                    near = np.argwhere(np.abs(scores - threshold) < 1e-10)
                    for row, column in near:
                        scores[row, column] = np.clip(changed[row, column] @ changed[row, source_index], -1., 1.)
                    boundary_checks += len(near)
                    untouched = (unit[:, features].T == 0) & (unit[source_index, features, None] == 0)
                    scores[untouched] = np.broadcast_to(before_scores[identifier], scores.shape)[untouched]
                    raw_dot = np.array([value @ unit[source_index] for value in unit])
                    numerator = raw_dot[None] - unit[:, features].T * unit[source_index, features, None]
                    denominator = np.sqrt((np.square(unit).sum(1)[None] - np.square(unit[:, features].T)) *
                        (np.square(unit[source_index]).sum() - np.square(unit[source_index, features, None])))
                    closed = np.clip(numerator / denominator, -1., 1.)
                    formula_error = max(formula_error, float(np.max(np.abs(closed - scores))))
                    np.testing.assert_allclose(closed, scores, atol=2e-6, rtol=0)
                    indices, local = replay(manifest, identifier, positions, scores, threshold)
                    retained[np.ix_(features + 1, indices)] = local
                print('CAPACITY_FEATURES', video, key, int(features[-1]) + 1, '/', dimension, flush=True)
            metrics = counts(retained, before, same)
            table = pd.DataFrame(dict(feature=np.arange(-1, dimension), **metrics))
            table.to_csv(target / 'features.csv', index=False)
            np.savez_compressed(target / 'effects.npz', retained=retained, before=before, same=same,
                                features=np.arange(-1, dimension), unit_codes=unit, positions=positions)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity,
                events=len(before), dimensions=dimension, unchanged_replay_exact=True, original_codes_exact=True,
                scalar_max_error=scalar_error, formula_max_error=formula_error, boundary_checks=boundary_checks,
                seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            completed += 1
            atomic_write_json(root / 'progress.json', dict(completed=completed, total=len(videos) * len(models),
                              video=video, model=key))
            pause_after_checkpoint(target / 'complete.json')
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', models=completed,
        seconds=time.perf_counter() - begin, torch=str(torch.__version__), numpy=np.__version__))


def best_index(retained, before, same):
    metrics = counts(retained, before, same)
    safe = (metrics['repeat_damage'] + metrics['other_damage']) == 0
    candidates = np.flatnonzero(safe)
    return min(candidates, key=lambda i: (-metrics['corrected'][i], -metrics['repeat_gain'][i], i))


def summarize(run, smoke):
    config = read_json(run / 'config.json')
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    groups, details, feature_tables = {}, [], []
    for folder in sorted(root.glob('*/*/complete.json')):
        video, key = folder.parent.parent.name, folder.parent.name
        manifest = read_json(Path(config['event_run']) / 'inputs' / video / 'events.json')
        with np.load(folder.parent / 'effects.npz') as saved:
            retained, before, same = saved['retained'].copy(), saved['before'].copy(), saved['same'].copy()
        groups.setdefault(key, []).append((video, manifest, retained, before, same))
        frame = pd.read_csv(folder.parent / 'features.csv')
        frame['video'], frame['model'] = video, key
        feature_tables.append(frame)
    for key, items in groups.items():
        all_retained = np.concatenate([item[2] for item in items], axis=1)
        all_before = np.concatenate([item[3] for item in items])
        all_same = np.concatenate([item[4] for item in items])
        global_index = best_index(all_retained, all_before, all_same)
        method, seed = key.rsplit('_seed', 1)
        for video, manifest, retained, before, same in items:
            for episode in manifest['episodes']:
                indices = np.array([i for i, event in enumerate(manifest['events']) if event['episode'] == episode['episode_id']])
                local, old, labels = retained[:, indices], before[indices], same[indices]
                source_index = best_index(local, old, labels)
                measures = counts(local, old, labels)
                safe = measures['repeat_damage'] + measures['other_damage'] == 0
                wrong = ~labels & ~old
                rescuable = int(local[safe][:, wrong].any(0).sum())
                for policy, index in [('unchanged', 0), ('global_oracle', global_index), ('source_oracle', source_index)]:
                    after = local[index]
                    first = np.array([manifest['events'][i]['first_prompt'] for i in indices]) & ~labels
                    details.append(dict(method=method, seed=int(seed), video=video, episode=episode['episode_id'],
                        policy=policy, feature=int(index - 1), events=len(indices), original_wrong=int(wrong.sum()),
                        original_correct_repeat=int((labels & ~old).sum()),
                        corrected=int(measures['corrected'][index]), repeat_damage=int(measures['repeat_damage'][index]),
                        other_damage=int(measures['other_damage'][index]), repeat_gain=int(measures['repeat_gain'][index]),
                        first_damage=int((first & old & ~after).sum()), individually_rescuable=rescuable,
                        safe_features=int(safe.sum() - 1)))
    sources = pd.DataFrame(details)
    sources.to_csv(output / 'sources.csv', index=False)
    metrics = ['original_wrong', 'original_correct_repeat', 'corrected', 'repeat_damage', 'other_damage',
               'repeat_gain', 'first_damage', 'individually_rescuable']
    procedures = sources.groupby(['method', 'seed', 'video', 'policy'])[metrics].sum().reset_index()
    procedures.to_csv(output / 'procedures.csv', index=False)
    seeds = procedures.groupby(['method', 'seed', 'policy'])[metrics].sum().reset_index()
    seeds.to_csv(output / 'seeds.csv', index=False)
    summary = seeds.groupby(['method', 'policy'])[metrics].sum().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    features = pd.concat(feature_tables, ignore_index=True)
    features.to_csv(output / 'all_features.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, (method, group) in zip(axes, summary.groupby('method', sort=True)):
        group = group.set_index('policy').loc[['unchanged', 'global_oracle', 'source_oracle']]
        ax.bar(range(3), group.corrected, color='#126A89' if method.endswith('sparse') else '#CB6549')
        ax.set_xticks(range(3), ['Unchanged', 'Global oracle', 'Source oracle'], rotation=15)
        ax.set_title('SAE' if method.endswith('sparse') else 'Dense')
        ax.set_ylabel('Corrected wrong-removal event instances')
        ax.set_ylim(0, max(1, int(group.original_wrong.max())) * 1.05)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Single-coordinate capacity | Label-informed selection | Zero damage constraint\nCounts include repeated events across seeds; not independent patients')
    fig.savefig(output / 'capacity.png', dpi=170)
    plt.close(fig)
    assert (sources[['repeat_damage', 'other_damage', 'first_damage']].to_numpy() == 0).all()
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', source_rows=len(sources),
                      procedure_rows=len(procedures), seed_rows=len(seeds), selection='label-informed diagnostic'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
