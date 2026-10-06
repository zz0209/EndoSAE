import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from evaluate_single_component_capacity import best_index, counts, application, load_models, model_specs
from encode_token_causal_identity import video_inputs, ROOT
from evaluate_prompt_event_components import event_rows
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def inputs(run):
    config = read_json(run / 'config.json')
    capacity = Path(config['capacity_run'])
    event = Path(read_json(capacity / 'config.json')['event_run'])
    event_config = read_json(event / 'config.json')
    original = read_json(Path(event_config['application_run']) / 'config.json')
    original.update(training_runs=event_config['training_runs'], methods=event_config['methods'], seeds=event_config['seeds'])
    return config, capacity, event, event_config, original


def select(run, resume):
    config, capacity, event, _, original = inputs(run)
    target = run / 'selection.json'
    identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'), source=digest(__file__),
                    capacity=digest(capacity / 'analysis/summary.json'))
    if target.exists():
        assert resume and read_json(target)['identity'] == identity
        return
    choices = []
    manifests = {v: read_json(event / 'inputs' / v / 'events.json') for v in original['development_videos']}
    for key, _ in model_specs(original):
        for video in original['development_videos']:
            bank, combined, old, labels = [], [], [], []
            for reference in sorted(v for v in manifests if v != video):
                manifest = manifests[reference]
                with np.load(capacity / 'evaluation' / reference / key / 'effects.npz') as saved:
                    retained, before, same = saved['retained'], saved['before'], saved['same']
                    combined.append(retained)
                    old.append(before)
                    labels.append(same)
                    for episode in manifest['episodes']:
                        indices = np.array([i for i, e in enumerate(manifest['events']) if e['episode'] == episode['episode_id']])
                        index = best_index(retained[:, indices], before[indices], same[indices])
                        position = np.flatnonzero(saved['positions'] == episode['source_position']).item()
                        bank.append(dict(video=reference, episode=episode['episode_id'], feature=int(index - 1),
                                         code=saved['unit_codes'][position]))
            global_feature = int(best_index(np.concatenate(combined, axis=1), np.concatenate(old), np.concatenate(labels)) - 1)
            assert all(row['video'] != video for row in bank)
            with np.load(capacity / 'evaluation' / video / key / 'effects.npz') as saved:
                for episode in manifests[video]['episodes']:
                    identifier = episode['episode_id']
                    position = np.flatnonzero(saved['positions'] == episode['source_position']).item()
                    code = saved['unit_codes'][position]
                    similarities = np.array([code @ row['code'] for row in bank])
                    nearest = int(np.argmax(similarities))
                    seed = int.from_bytes(hashlib.sha256(f"{config['random_seed']}:{key}:{video}:{identifier}".encode()).digest()[:8], 'little')
                    random = int(np.random.default_rng(seed).integers(len(bank)))
                    routes = {'unchanged': (-1, None), 'global_lopo': (global_feature, None),
                              'nearest_source': (bank[nearest]['feature'], nearest),
                              'random_source': (bank[random]['feature'], random)}
                    for policy, (feature, index) in routes.items():
                        choices.append(dict(model=key, video=video, episode=identifier, policy=policy, feature=feature,
                            reference_video=bank[index]['video'] if index is not None else None,
                            reference_episode=bank[index]['episode'] if index is not None else None,
                            similarity=float(similarities[index]) if index is not None else None,
                            bank_procedures=sorted({row['video'] for row in bank}), bank_sources=len(bank)))
            print('SELECT_COMPONENT', key, video, 'bank_sources', len(bank), flush=True)
    atomic_write_json(target, dict(status='COMPLETE', identity=identity, choices=choices,
        created_at=datetime.now(timezone.utc).isoformat(), target_outcomes_used_for_selection=False))


@torch.no_grad()
def encode(raw, model, mean, scale, features, device):
    values = ((raw.astype(float) - mean) / scale).astype(np.float32)
    projected, _, codes = model(torch.from_numpy(values[None]).to(device))
    assert model.identity_space == 'code'
    pooled = codes.mean(1)[0]
    vectors = {-1: projected[0].cpu().numpy().astype(float)}
    for feature in features:
        if isinstance(feature, tuple):
            changed = vectors[-1].copy()
            if feature:
                changed[list(feature)] = 0
                assert np.isfinite(changed).all() and np.linalg.norm(changed) > 0
                changed /= np.linalg.norm(changed)
            vectors[feature] = changed
            continue
        if feature < 0:
            continue
        changed = pooled.clone()
        changed[feature] = 0
        assert torch.isfinite(changed).all() and changed.norm() > 0
        vectors[feature] = F.normalize(changed, dim=0).cpu().numpy().astype(float)
    return vectors


def action_key(row):
    return tuple(row['coordinates']) if 'coordinates' in row else row['feature']


def score(run, smoke, resume, selection_name='selection.json'):
    config, capacity, event, _, original = inputs(run)
    selection = read_json(run / selection_name)
    assert selection['status'] == 'COMPLETE'
    assert not selection.get('future_outcomes_used_for_selection', selection['target_outcomes_used_for_selection'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    if smoke:
        original['seeds'] = [config['smoke_seed']]
    models = load_models(original, device)
    videos = [config['smoke_video']] if smoke else original['development_videos']
    root = run / ('smoke_scores' if smoke else 'scores')
    root.mkdir(exist_ok=True)
    begin, completed = time.perf_counter(), 0
    total = sum(read_json(Path(original['raw_root']) / v / 'complete.json')['detections'] for v in videos)
    for video in videos:
        folder = root / video
        folder.mkdir(exist_ok=True)
        _, _, _, tracks, available, _, by_output, _ = video_inputs(original, video)
        detection_frames = np.repeat(tracks['frame_indices'], np.diff(tracks['offsets']))
        raw_root = Path(original['raw_root']) / video
        receipt = read_json(raw_root / 'complete.json')
        encoded = np.load(raw_root / 'encoded.npy')
        np.testing.assert_array_equal(encoded, available)
        identity = dict(selection=digest(run / selection_name), source=digest(__file__), smoke=smoke,
                        raw=digest(raw_root / 'complete.json'), models={key: digest(path / 'model.npz') for key, path in model_specs(original)})
        if (folder / 'identity.json').exists():
            assert resume and read_json(folder / 'identity.json') == identity
        else:
            atomic_write_json(folder / 'identity.json', identity)
        if (folder / 'complete.json').exists():
            assert resume
            completed += int(encoded.sum())
            continue
        routes = [row for row in selection['choices'] if row['video'] == video and row['model'] in models]
        features = {key: sorted({action_key(row) for row in routes if row['model'] == key}, key=repr) for key in models}
        sources, originals, arrays = {}, {}, {}
        progress_path = folder / 'progress.json'
        previous = read_json(progress_path) if progress_path.exists() else dict(shards=0, processed=0, seconds=0., max_score_error=0.)
        assert previous['shards'] == 0 or resume
        for episode in [e for group in by_output.values() for e in group]:
            identifier = episode['episode_id']
            source = Path(original['source_token_root']) / video / 'sources' / identifier
            info = read_json(source / 'source.json')
            assert not info['future_frames_used'] and not info['ground_truth_regions_used']
            with np.load(source / 'observed_tokens.npz') as saved:
                raw = saved['tokens'].copy()
            for key, (model, mean, scale) in models.items():
                sources[identifier, key] = encode(raw, model, mean, scale, features[key], device)
                method, seed = key.rsplit('_seed', 1)
                originals[identifier, key] = np.load(Path(original['embedding_root']) / video / 'sources' / identifier / f'{method}_pooled_cosine_seed{seed}.npy', mmap_mode='r')
                with np.load(capacity / 'evaluation' / video / key / 'effects.npz') as saved:
                    position = np.flatnonzero(saved['positions'] == info['position']).item()
                    np.testing.assert_array_equal(sources[identifier, key][-1], saved['unit_codes'][position])
        for row in routes:
            path = folder / row['model'] / row['episode'] / (row['policy'] + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float64, shape=available.shape)
            if previous['shards'] == 0:
                array[:] = np.nan
            arrays[row['episode'], row['model'], row['policy']] = array
        local_begin = time.perf_counter()
        error, processed = previous['max_score_error'], previous['processed']
        points = {seed: read_json(Path(inputs(run)[3]['application_run']) / 'evaluation' / f'seed{seed}' / 'operating_points.json')['methods'] for seed in original['seeds']}
        for shard_index in range(previous['shards'], len(receipt['shards'])):
            shard = receipt['shards'][shard_index]
            path = raw_root / shard['file']
            assert digest(path) == shard['sha256']
            with np.load(path) as saved:
                values, offsets, positions = saved['tokens'], saved['offsets'], saved['detection_positions']
            assert np.all(encoded[positions])
            for i, position in enumerate(positions):
                for key, (model, mean, scale) in models.items():
                    vectors = encode(values[offsets[i]:offsets[i + 1]], model, mean, scale, features[key], device)
                    method, seed = key.rsplit('_seed', 1)
                    threshold = points[int(seed)][method + '_pooled_cosine']['threshold']
                    for row in (r for r in routes if r['model'] == key):
                        identifier, feature = row['episode'], action_key(row)
                        source = sources[identifier, key]
                        before = float(originals[identifier, key][position])
                        unchanged = float(np.clip(vectors[-1] @ source[-1], -1., 1.))
                        error = max(error, abs(before - unchanged))
                        assert abs(before - unchanged) <= 2e-6 and (before < threshold) == (unchanged < threshold)
                        coordinates = list(feature) if isinstance(feature, tuple) else ([] if feature < 0 else [feature])
                        unaffected = not coordinates or (np.all(vectors[-1][coordinates] == 0) and np.all(source[-1][coordinates] == 0))
                        score_value = before if unaffected else float(np.clip(vectors[feature] @ source[feature], -1., 1.))
                        if 'activation_frame' in row and detection_frames[position] <= row['activation_frame']:
                            score_value = before
                        arrays[identifier, key, row['policy']][position] = score_value
                processed += 1
            for array in arrays.values():
                array.flush()
            atomic_write_json(progress_path, dict(shards=shard_index + 1, processed=processed,
                seconds=previous['seconds'] + time.perf_counter() - local_begin, max_score_error=error))
            atomic_write_json(root / 'progress.json', dict(completed=completed + processed, total=total, video=video))
            print('TRANSFER_SCORES', video, processed, '/', int(encoded.sum()), 'total', completed + processed, '/', total, flush=True)
            pause_after_checkpoint(progress_path)
        assert processed == int(encoded.sum())
        for array in arrays.values():
            assert np.isfinite(array[encoded]).all() and np.isnan(array[~encoded]).all()
        atomic_write_json(folder / 'complete.json', dict(status='COMPLETE', **read_json(progress_path),
                          original_decisions_exact=True, peak_cuda_bytes=torch.cuda.max_memory_allocated(), identity=identity))
        completed += processed
    if not (root / 'summary.json').exists():
        atomic_write_json(root / 'summary.json', dict(status='COMPLETE', detections=completed, models=list(models),
            seconds=time.perf_counter() - begin, torch=str(torch.__version__), numpy=np.__version__))


def evaluate(run, smoke, resume):
    config, capacity, event, event_config, original = inputs(run)
    root = run / ('smoke_scores' if smoke else 'scores')
    output = run / ('smoke_evaluation' if smoke else 'evaluation')
    output.mkdir(exist_ok=True)
    scores_receipt = read_json(root / 'summary.json')
    assert scores_receipt['status'] == 'COMPLETE'
    videos = [config['smoke_video']] if smoke else original['development_videos']
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    choices = read_json(run / 'selection.json')['choices']
    all_rows, event_results = [], []
    begin = time.perf_counter()
    for video in videos:
        receipt, frames, records, _, offsets, _, available, first = application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for key in scores_receipt['models']:
            method, seed = key.rsplit('_seed', 1)
            threshold = read_json(Path(event_config['application_run']) / 'evaluation' / f'seed{seed}' / 'operating_points.json')['methods'][method + '_pooled_cosine']['threshold']
            for episode in manifest['episodes']:
                identifier = episode['episode_id']
                target = output / video / key / identifier
                target.mkdir(parents=True, exist_ok=True)
                if (target / 'complete.json').exists():
                    assert resume
                    all_rows.extend(read_json(target / 'complete.json')['rows'])
                    event_results.extend(pd.read_csv(target / 'events.csv').to_dict('records'))
                    continue
                with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                    data = {name: saved[name] for name in saved.files}
                before = np.load(root / video / key / identifier / 'unchanged.npy')
                source_rows, source_events, keeps = [], [], {}
                for policy in config['policies']:
                    scores = np.load(root / video / key / identifier / (policy + '.npy'))
                    curve, checks, keep = application.fixed_curve(scores, threshold, offsets, records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                    for group, columns in episode['groups'].items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    result = application.result_row(curve, episode)
                    others = [lesion for lesion in result['lesions'] if 'all_other' in lesion['groups']]
                    denominator = sum(lesion['baseline_seconds'] for lesion in others)
                    retained = sum(lesion['retained_baseline_seconds'] for lesion in others)
                    firsts = [lesion for lesion in others if lesion['baseline_first_prompt_time'] is not None]
                    first_retained = sum(lesion['first_prompt_time'] == lesion['baseline_first_prompt_time'] for lesion in firsts)
                    retentions = [lesion['retention'] for lesion in others if lesion['retention'] is not None]
                    source_rows.append(dict(method=method, seed=int(seed), video=video, episode=identifier, policy=policy,
                        repeat_removal=result['source_removal_fraction'], other_retention=float(np.mean(retentions)) if retentions else None,
                        other_baseline_seconds=denominator, other_retained_seconds=retained,
                        first_count=len(firsts), first_retained=first_retained, first_retention=first_retained / len(firsts) if firsts else None))
                    np.savez_compressed(target / (policy + '_curve.npz'), **curve)
                    keeps[policy] = keep
                    events = event_rows(dict(manifest, events=[e for e in manifest['events'] if e['episode'] == identifier]), {identifier: scores}, {identifier: before}, threshold, key, policy)
                    choice = next(r for r in choices if r['video'] == video and r['model'] == key and r['episode'] == identifier and r['policy'] == policy)
                    indices = np.array([i for i, e in enumerate(manifest['events']) if e['episode'] == identifier])
                    with np.load(capacity / 'evaluation' / video / key / 'effects.npz') as saved:
                        np.testing.assert_array_equal([e['retained'] for e in events], saved['retained'][choice['feature'] + 1, indices])
                    source_events.extend(events)
                np.savez_compressed(target / 'fixed_keep.npz', **keeps)
                pd.DataFrame(source_events).to_csv(target / 'events.csv', index=False)
                atomic_write_json(target / 'complete.json', dict(status='COMPLETE', rows=source_rows,
                    event_effects_exact=True, selection_sha256=digest(run / 'selection.json'), source_sha256=digest(__file__), checks=checks))
                all_rows.extend(source_rows)
                event_results.extend(source_events)
                print('TRANSFER_REPLAY', video, key, identifier, flush=True)
        atomic_write_json(output / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos), video=video))
        pause_after_checkpoint(output / 'progress.json')
    pd.DataFrame(all_rows).to_csv(output / 'sources.csv', index=False)
    pd.DataFrame(event_results).to_csv(output / 'events.csv', index=False)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', seconds=time.perf_counter() - begin,
                      population='examined development', source_rows=len(all_rows), all_event_effects_exact=True))


def summarize(run, smoke):
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    sources = pd.read_csv(root / 'sources.csv')
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    procedures = sources.groupby(['method', 'seed', 'video', 'policy'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed', 'policy'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy'])[metrics].mean().reset_index()
    for name, table in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    events = pd.read_csv(root / 'events.csv')
    rows = []
    for keys, group in events.groupby(['method', 'seed', 'video', 'variant']):
        measures = counts(group.retained.to_numpy(bool)[None], group.before_retained.to_numpy(bool), group.same_identity.to_numpy(bool))
        rows.append(dict(zip(['method', 'seed', 'video', 'policy'], keys), **{k: int(v[0]) for k, v in measures.items()}))
    pd.DataFrame(rows).to_csv(output / 'event_effects.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, metric in zip(axes, ['repeat_removal', 'other_retention']):
        for method, group in summary.groupby('method'):
            ax.plot(group.policy, group[metric] * 100, marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
        ax.set_ylabel(metric.replace('_', ' ').title() + ' (%)')
        ax.tick_params(axis='x', rotation=18)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Source-conditioned coordinate transfer | Examined development\nProcedure-excluded component selection; original development thresholds')
    fig.savefig(output / 'transfer.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', rows=len(summary),
        aggregation='Defined lesion means within source, source means within procedure, then procedures, then seeds'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['select', 'score', 'evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'select':
        select(args.run, args.resume)
    elif args.phase == 'score':
        score(args.run, args.smoke, args.resume)
    elif args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
