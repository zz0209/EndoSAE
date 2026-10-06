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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch

import evaluate_pair_component_capacity as pair
import evaluate_source_component_transfer as transfer
import encode_causal_detection_identity as native
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def settings(run, smoke):
    config = read_json(run / 'config.json')
    capacity = Path(config['capacity_run'])
    event = Path(read_json(capacity / 'config.json')['event_run'])
    definition = read_json(event / 'config.json')
    original = read_json(Path(definition['application_run']) / 'config.json')
    original.update({key: definition[key] for key in ['training_runs', 'methods', 'seeds']})
    if smoke:
        original['development_videos'] = [config['smoke_video']]
        original['seeds'] = [config['smoke_seed']]
    root = run / 'smoke' if smoke else run
    root.mkdir(exist_ok=True)
    return config, capacity, event, original, root


def scores_for(unit, source, coordinates, original):
    norm = np.square(unit).sum(1)
    mass = np.square(unit[:, coordinates]).sum(1)
    removed = (unit[:, coordinates] * unit[source, coordinates]).sum(1)
    numerator = unit @ unit[source] - removed
    assert np.all(norm - mass > 0)
    original_denominator = np.sqrt(norm * norm[source])
    changed_denominator = np.sqrt((norm - mass) * (norm[source] - mass[source]))
    scores = dict(unchanged=original.copy(),
        numerator_only=numerator / original_denominator,
        source_only=numerator / np.sqrt(norm * (norm[source] - mass[source])),
        query_only=numerator / np.sqrt((norm - mass) * norm[source]),
        bilateral=numerator / changed_denominator,
        normalization_only=(unit @ unit[source]) / changed_denominator)
    changed = unit.copy()
    changed[:, coordinates] = 0
    changed /= np.linalg.norm(changed, axis=1, keepdims=True)
    normalized = unit / np.sqrt(norm[:, None])
    direct = dict(source_only=normalized @ changed[source], query_only=changed @ normalized[source],
                  bilateral=changed @ changed[source])
    error = 0.
    for action, values in scores.items():
        if action in direct:
            error = max(error, float(np.max(np.abs(values - direct[action]))))
            np.testing.assert_allclose(values, direct[action], atol=2e-12, rtol=0)
        values[:] = np.clip(values, -1, 1)
        values[(mass == 0) & (mass[source] == 0)] = original[(mass == 0) & (mass[source] == 0)]
        assert np.isfinite(values).all()
    np.testing.assert_allclose(scores['bilateral'], pair.direct_scores(unit, source, coordinates, original), atol=2e-12, rtol=0)
    return scores, removed, mass, error


def evaluate(run, smoke):
    config, capacity, event, original, root = settings(run, smoke)
    choices = read_json(Path(config['bank_run']) / 'selection/summary.json')['choices']
    bank = {(r['video'], r['model'], r['episode']): r for r in choices if r['policy'] == 'bank_pair'}
    begin, completed, errors = time.perf_counter(), 0, []
    total = len(original['seeds']) * len(original['methods']) * sum(
        len(read_json(event / 'inputs' / video / 'events.json')['episodes']) for video in original['development_videos'])
    for video in original['development_videos']:
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for key, _ in transfer.model_specs(original):
            source_path = capacity / 'evaluation' / video / key / 'effects.npz'
            with np.load(source_path) as saved:
                unit, positions, before, same = [saved[name] for name in ['unit_codes', 'positions', 'before', 'same']]
            lookup = {int(p): i for i, p in enumerate(positions)}
            for episode in manifest['episodes']:
                identifier = episode['episode_id']
                target = root / 'evaluation' / video / key / identifier
                target.mkdir(parents=True, exist_ok=True)
                selected_path = Path(config['pair_run']) / 'evaluation' / video / key / identifier / 'selected.json'
                identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'),
                    script=digest(__file__), source=digest(source_path), selected=digest(selected_path),
                    bank=digest(Path(config['bank_run']) / 'selection/summary.json'))
                if (target / 'complete.json').exists():
                    receipt = read_json(target / 'complete.json')
                    assert receipt['identity'] == identity
                    assert digest(target / 'events.csv') == receipt['events_sha256']
                    completed += 1
                    continue
                choice = bank[video, key, identifier]
                label = next(r for r in read_json(selected_path) if r['mode'] == 'pair_exact')
                with np.load(event / 'evaluation' / video / key / 'scores.npz') as saved:
                    old_scores = saved[identifier + '__before']
                source = lookup[episode['source_position']]
                event_ids, indices, tables = pair.event_lookup(manifest, identifier, positions)
                np.testing.assert_array_equal(pair.lookup_replay(old_scores[None], choice['threshold'], indices, tables)[0], before[event_ids])
                active = np.array([manifest['events'][i]['frame'] > choice['activation_frame'] for i in event_ids])
                rows, arrays = [], dict(event_ids=event_ids, positions=positions)
                for policy, coordinates in [('bank_pair', choice['coordinates']), ('label_pair', label['features'])]:
                    scores, removed, mass, error = scores_for(unit, source, coordinates, old_scores)
                    errors.append(error)
                    values = np.array([scores[action] for action in config['actions']])
                    retained = pair.lookup_replay(values, choice['threshold'], indices, tables)
                    if policy == 'label_pair':
                        old = Path(config['pair_run']) / 'evaluation' / video / key / identifier / 'selected_predictions.npz'
                        with np.load(old) as reference:
                            np.testing.assert_array_equal(retained[config['actions'].index('bilateral')], reference['pair_exact_retained'])
                    retained[:, ~active] = before[event_ids[~active]][None]
                    if policy == 'bank_pair':
                        old = Path(config['bank_run']) / 'analysis' / video / key / identifier / 'bank_pair.npz'
                        with np.load(old) as reference:
                            np.testing.assert_array_equal(reference['ids'], event_ids)
                            np.testing.assert_array_equal(retained[config['actions'].index('bilateral')], reference['retained'])
                    for action_index, action in enumerate(config['actions']):
                        arrays[policy + '__' + action] = scores[action]
                        for column, event_id in enumerate(event_ids):
                            row = manifest['events'][int(event_id)]
                            frame = manifest['frames'][str(row['output'])]
                            match = next(m for m in frame['matches'] if m['lesion_id'] == row['lesion'])
                            position = int(frame['positions'][match['prediction_index']])
                            local = lookup.get(position)
                            rows.append(dict(video=video, model=key, method=key.rsplit('_seed', 1)[0],
                                seed=int(key.rsplit('_seed', 1)[1]), episode=identifier, event_id=int(event_id),
                                policy=policy, action=action, coordinates=','.join(map(str, coordinates)),
                                after_activation=bool(active[column]), before=bool(before[event_id]),
                                retained=bool(retained[action_index, column]), threshold=choice['threshold'],
                                matched_position=position, shared_product=float(removed[local]) if local is not None else np.nan,
                                source_mass=float(mass[source]), query_mass=float(mass[local]) if local is not None else np.nan,
                                original_score=float(old_scores[local]) if local is not None else np.nan,
                                changed_score=float(scores[action][local]) if local is not None else np.nan,
                                **{k: row[k] for k in ['frame', 'lesion', 'same_identity', 'first_prompt', 'stratum', 'output']}))
                pd.DataFrame(rows).to_csv(target / 'events.csv', index=False)
                np.savez_compressed(target / 'scores.npz', **arrays)
                atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity,
                    events_sha256=digest(target / 'events.csv'), score_sha256=digest(target / 'scores.npz'),
                    direct_error=max(errors), original_actions_exact=True))
                completed += 1
                atomic_write_json(root / 'progress.json', dict(completed=completed, total=total, video=video,
                    model=key, episode=identifier, seconds=time.perf_counter() - begin))
                print('ACTION_MECHANISM', completed, '/', total, video, key, identifier, flush=True)
                pause_after_checkpoint(target / 'complete.json')
    atomic_write_json(root / 'evaluation/summary.json', dict(status='COMPLETE', jobs=completed, total=total,
        seconds=time.perf_counter() - begin, numpy=np.__version__, pandas=pd.__version__))


def summarize(run, smoke):
    config, _, _, _, root = settings(run, smoke)
    receipts = [read_json(p) for p in sorted((root / 'evaluation').glob('*/*/*/complete.json'))]
    frame = pd.concat([pd.read_csv(p) for p in sorted((root / 'evaluation').glob('*/*/*/events.csv'))], ignore_index=True)
    frame['original_wrong'] = ~frame.same_identity & ~frame.before
    frame['corrected'] = frame.original_wrong & frame.retained
    frame['repeat_damage'] = frame.same_identity & ~frame.before & frame.retained
    frame['other_damage'] = ~frame.same_identity & frame.before & ~frame.retained
    frame['repeat_gain'] = frame.same_identity & frame.before & ~frame.retained
    frame['first_damage'] = frame.first_prompt & ~frame.same_identity & frame.before & ~frame.retained
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    frame.to_csv(output / 'events.csv', index=False)
    metrics = ['original_wrong', 'corrected', 'repeat_damage', 'other_damage', 'repeat_gain', 'first_damage']
    selected = frame[frame.after_activation]
    for name, keys in [('sources', ['method', 'seed', 'video', 'episode', 'policy', 'action']),
                       ('procedures', ['method', 'seed', 'video', 'policy', 'action']),
                       ('seeds', ['method', 'seed', 'policy', 'action']),
                       ('summary', ['method', 'policy', 'action'])]:
        selected.groupby(keys)[metrics].sum().reset_index().to_csv(output / (name + '.csv'), index=False)
    image_cases = []
    eligible = selected[(selected.method == 'p27v4_token_sparse') & (selected.policy == 'bank_pair') & (selected.action == 'bilateral')].copy()
    eligible['persistent_error'] = eligible.original_wrong & ~eligible.retained
    eligible['damaged_repeat'] = eligible.repeat_damage
    for video, local in eligible.groupby('video'):
        for role in config['image_roles']:
            candidates = local[local[role]].sort_values(['seed', 'frame', 'episode', 'event_id'])
            if len(candidates):
                row = candidates.iloc[0]
                image_cases.append(dict(video=video, role=role, model=row.model, episode=row.episode,
                    event_id=int(row.event_id), seed=int(row.seed),
                    coordinates=[] if pd.isna(row.coordinates) else [int(v) for v in str(row.coordinates).split(',')],
                    rule='Within each procedure/outcome: smallest seed, earliest frame, episode, event index.'))
    atomic_write_json(output / 'image_cases.json', image_cases)
    summary = pd.read_csv(output / 'summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for row_index, method in enumerate(sorted(frame.method.unique())):
        for ax, metric in zip(axes[row_index], ['corrected', 'repeat_damage', 'other_damage']):
            for policy in config['policies']:
                local = summary[(summary.method == method) & (summary.policy == policy)].set_index('action').loc[config['actions']]
                ax.plot(range(len(local)), local[metric], marker='o', label=policy)
            ax.set_xticks(range(6), ['Original', 'Product', 'Source', 'Query', 'Both', 'Norm'], rotation=25)
            ax.set_title(('SAE' if method.endswith('sparse') else 'Dense') + ' | ' + metric.replace('_', ' '))
            ax.set_ylabel('Event counts across three seeds' if not smoke else 'Event counts, one smoke seed')
            ax.legend()
    fig.suptitle('Fixed coordinates and threshold | After initial visibility | Examined development')
    fig.savefig(output / 'action_effects.png', dpi=160)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', source_models=len(receipts),
        maximum_direct_error=max(r['direct_error'] for r in receipts), original_actions_exact=True,
        cases=len(image_cases), event_rows=len(frame), independent_data_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['evaluate', 'summarize'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (evaluate if args.phase == 'evaluate' else summarize)(args.run, args.smoke)
