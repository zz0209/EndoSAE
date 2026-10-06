import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
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

from evaluate_source_component_transfer import inputs, encode, application, load_models, model_specs, video_inputs
from evaluate_prompt_event_components import event_rows
from evaluate_single_component_capacity import counts
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def setup(run, smoke):
    config, _, event, event_config, original = inputs(run)
    if smoke:
        original['seeds'] = [config['smoke_seed']]
    videos = [config['smoke_video']] if smoke else original['development_videos']
    return config, event, event_config, original, videos


def torch_setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def feedback_event(manifest, episode, positions, mode='first_later_other'):
    if mode == 'previous_confirmation':
        candidates = [e for e in manifest['episodes'] if e['click']['output_index'] < episode['click']['output_index']
                      and e['source_lesion_id'] != episode['source_lesion_id'] and e['source_position'] in positions]
        if not candidates:
            return None
        previous = max(candidates, key=lambda e: (e['click']['output_index'], e['episode_id']))
        assert previous['click']['time'] < episode['click']['time']
        assert not previous['source_info']['future_frames_used'] and not previous['source_info']['ground_truth_regions_used']
        return dict(frame=episode['click']['input_frame'], output=episode['click']['output_index'],
                    position=previous['source_position'], lesion=previous['source_lesion_id'],
                    observed_frame=previous['click']['input_frame'], observed_output=previous['click']['output_index'],
                    reference_episode=previous['episode_id'])
    assert mode == 'first_later_other'
    candidates = sorted((e for e in manifest['events'] if e['episode'] == episode['episode_id'] and not e['same_identity']),
                        key=lambda e: (e['frame'], e['lesion'], e['output']))
    for event in candidates:
        frame = manifest['frames'][str(event['output'])]
        match = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
        position = frame['positions'][match['prediction_index']]
        if position in positions:
            assert event['output'] > episode['click']['output_index']
            return dict(frame=event['frame'], output=event['output'], position=position, lesion=event['lesion'])
    return None


@torch.no_grad()
def prepare(run, smoke, resume):
    config, event, event_config, original, videos = setup(run, smoke)
    root = run / ('smoke_selection' if smoke else 'selection')
    root.mkdir(exist_ok=True)
    torch_setup()
    device = torch.device(config['device'])
    models = load_models(original, device)
    folders = dict(model_specs(original))
    begin, all_rows = time.perf_counter(), []
    for video in videos:
        folder = event / 'inputs' / video
        receipt = read_json(folder / 'complete.json')
        for name, expected in receipt['assets'].items():
            assert digest(folder / name) == expected
        manifest = read_json(folder / 'events.json')
        with np.load(folder / 'tokens.npz') as saved:
            raw, bounds, positions = saved['tokens'], saved['offsets'], saved['positions']
        lookup = {int(p): i for i, p in enumerate(positions)}
        for key, (model, mean, scale) in models.items():
            target = root / video / key
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'), source=digest(__file__),
                            inputs=digest(folder / 'complete.json'), model=digest(folders[key] / 'model.npz'), smoke=smoke)
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity
                all_rows.extend(saved['rows'])
                continue
            method, seed = key.rsplit('_seed', 1)
            threshold = read_json(Path(event_config['application_run']) / 'evaluation' / f'seed{seed}' / 'operating_points.json')['methods'][method + '_pooled_cosine']['threshold']
            assert -1 < threshold < 1
            rows, arrays = [], {}
            for episode in manifest['episodes']:
                identifier = episode['episode_id']
                feedback = feedback_event(manifest, episode, lookup, config.get('feedback_mode', 'first_later_other'))
                source_index = lookup[episode['source_position']]
                source_raw = raw[bounds[source_index]:bounds[source_index + 1]]
                source = encode(source_raw, model, mean, scale, [], device)[-1]
                info = episode['source_info']
                assert not info['future_frames_used'] and not info['ground_truth_regions_used']
                feature, old_score, selected_score, raised = -1, None, None, threshold
                negative, edited = np.zeros_like(source), source.copy()
                if feedback is not None:
                    negative_index = lookup[feedback['position']]
                    negative_raw = raw[bounds[negative_index]:bounds[negative_index + 1]]
                    negative = encode(negative_raw, model, mean, scale, [], device)[-1]
                    old_score = float(np.clip(source @ negative, -1., 1.))
                    raised = max(threshold, float(np.nextafter(old_score, np.inf)))
                    selected_score = old_score
                    source_variants = encode(source_raw, model, mean, scale, range(len(source)), device)
                    negative_variants = encode(negative_raw, model, mean, scale, range(len(source)), device)
                    effects = np.array([np.clip(source_variants[j] @ negative_variants[j], -1., 1.) for j in range(len(source))])
                    best = int(np.argmin(effects))
                    if effects[best] < old_score:
                        feature, selected_score, edited = best, float(effects[best]), source_variants[best]
                    arrays[identifier + '__feedback_effects'] = effects
                arrays[identifier + '__source'] = source
                arrays[identifier + '__negative'] = negative
                arrays[identifier + '__edited_source'] = edited
                rows.append(dict(model=key, video=video, episode=identifier, feature=feature, feedback=feedback,
                    threshold=threshold, raised_threshold=raised, feedback_score=old_score, edited_feedback_score=selected_score))
            np.savez_compressed(target / 'vectors.npz', **arrays)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, rows=rows,
                vectors_sha256=digest(target / 'vectors.npz'), future_labels_used=False))
            all_rows.extend(rows)
            print('FEEDBACK_SELECTION', video, key, 'sources', len(rows), flush=True)
        atomic_write_json(root / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos)))
        pause_after_checkpoint(root / 'progress.json')
    if not (root / 'summary.json').exists():
        atomic_write_json(root / 'summary.json', dict(status='COMPLETE', rows=all_rows, seconds=time.perf_counter() - begin,
            models=list(models), created_at=datetime.now(timezone.utc).isoformat(), source_sha256=digest(__file__)))


def apply_feedback(before, vector, changed, source, negative, edited_source, choice, after):
    values = dict(unchanged=before, single_coordinate=before, negative_prototype=before, feedback_threshold=before)
    if not after or choice['feedback'] is None:
        return values
    feature = choice['feature']
    if feature >= 0 and (vector[feature] != 0 or source[feature] != 0):
        values['single_coordinate'] = float(np.clip(changed @ edited_source, -1., 1.))
    negative_score = float(np.clip(vector @ negative, -1., 1.))
    if before <= negative_score:
        values['negative_prototype'] = -1.
    if before < choice['raised_threshold']:
        values['feedback_threshold'] = -1.
    return values


@torch.no_grad()
def score(run, smoke, resume):
    config, _, _, original, videos = setup(run, smoke)
    selection = run / ('smoke_selection' if smoke else 'selection')
    selected = read_json(selection / 'summary.json')
    assert selected['status'] == 'COMPLETE'
    torch_setup()
    device = torch.device(config['device'])
    models = load_models(original, device)
    root = run / ('smoke_scores' if smoke else 'scores')
    root.mkdir(exist_ok=True)
    begin, completed = time.perf_counter(), 0
    total = sum(read_json(Path(original['raw_root']) / v / 'complete.json')['detections'] for v in videos)
    for video in videos:
        folder = root / video
        folder.mkdir(exist_ok=True)
        _, _, _, _, available, _, _, _ = video_inputs(original, video)
        raw_root = Path(original['raw_root']) / video
        receipt = read_json(raw_root / 'complete.json')
        encoded = np.load(raw_root / 'encoded.npy')
        np.testing.assert_array_equal(encoded, available)
        identity = dict(selection=digest(selection / 'summary.json'), source=digest(__file__), smoke=smoke,
                        raw=digest(raw_root / 'complete.json'))
        if (folder / 'identity.json').exists():
            assert resume and read_json(folder / 'identity.json') == identity
        else:
            atomic_write_json(folder / 'identity.json', identity)
        if (folder / 'complete.json').exists():
            assert resume
            completed += int(encoded.sum())
            continue
        rows = [row for row in selected['rows'] if row['video'] == video]
        features = {key: sorted({r['feature'] for r in rows if r['model'] == key}) for key in models}
        sources, originals, arrays, future = {}, {}, {}, {}
        base = Path(original['development_base'])
        offsets = np.load(base / video / 'indices.npz')['offsets']
        progress_path = folder / 'progress.json'
        previous = read_json(progress_path) if progress_path.exists() else dict(shards=0, processed=0, seconds=0., max_score_error=0.)
        assert previous['shards'] == 0 or resume
        for key in models:
            local = selection / video / key
            assert digest(local / 'vectors.npz') == read_json(local / 'complete.json')['vectors_sha256']
            with np.load(local / 'vectors.npz') as saved:
                sources[key] = {name: saved[name].copy() for name in saved.files}
        for row in rows:
            identifier, key = row['episode'], row['model']
            method, seed = key.rsplit('_seed', 1)
            originals[identifier, key] = np.load(Path(original['embedding_root']) / video / 'sources' / identifier / f'{method}_pooled_cosine_seed{seed}.npy', mmap_mode='r')
            with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                future[identifier] = np.repeat(saved['frame'] > row['feedback']['frame'], np.diff(offsets)) if row['feedback'] else np.zeros(len(available), bool)
            for policy in config['policies']:
                path = folder / key / identifier / (policy + '.npy')
                path.parent.mkdir(parents=True, exist_ok=True)
                array = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float64, shape=available.shape)
                if previous['shards'] == 0:
                    array[:] = np.nan
                arrays[identifier, key, policy] = array
        local_begin = time.perf_counter()
        error, processed = previous['max_score_error'], previous['processed']
        for shard_index in range(previous['shards'], len(receipt['shards'])):
            shard = receipt['shards'][shard_index]
            path = raw_root / shard['file']
            assert digest(path) == shard['sha256']
            with np.load(path) as saved:
                values, bounds, positions = saved['tokens'], saved['offsets'], saved['detection_positions']
            for i, position in enumerate(positions):
                for key, (model, mean, scale) in models.items():
                    vectors = encode(values[bounds[i]:bounds[i + 1]], model, mean, scale, features[key], device)
                    for row in (r for r in rows if r['model'] == key):
                        identifier, feature = row['episode'], row['feature']
                        source = sources[key][identifier + '__source']
                        before = float(originals[identifier, key][position])
                        original_score = float(np.clip(vectors[-1] @ source, -1., 1.))
                        error = max(error, abs(before - original_score))
                        assert abs(before - original_score) <= 2e-6 and (before < row['threshold']) == (original_score < row['threshold'])
                        result = apply_feedback(before, vectors[-1], vectors[feature], source,
                            sources[key][identifier + '__negative'], sources[key][identifier + '__edited_source'], row, future[identifier][position])
                        for policy, value in result.items():
                            arrays[identifier, key, policy][position] = value
                processed += 1
            for array in arrays.values():
                array.flush()
            atomic_write_json(progress_path, dict(shards=shard_index + 1, processed=processed,
                seconds=previous['seconds'] + time.perf_counter() - local_begin, max_score_error=error))
            atomic_write_json(root / 'progress.json', dict(completed=completed + processed, total=total, video=video))
            print('FEEDBACK_SCORES', video, processed, '/', int(encoded.sum()), 'total', completed + processed, '/', total, flush=True)
            pause_after_checkpoint(progress_path)
        assert processed == int(encoded.sum())
        for (identifier, key, policy), array in arrays.items():
            assert np.isfinite(array[encoded]).all() and np.isnan(array[~encoded]).all()
            np.testing.assert_array_equal(array[encoded & ~future[identifier]], originals[identifier, key][encoded & ~future[identifier]])
        atomic_write_json(folder / 'complete.json', dict(status='COMPLETE', **read_json(progress_path),
            original_decisions_exact=True, pre_feedback_unchanged=True, peak_cuda_bytes=torch.cuda.max_memory_allocated(), identity=identity))
        completed += processed
    if not (root / 'summary.json').exists():
        atomic_write_json(root / 'summary.json', dict(status='COMPLETE', detections=completed, models=list(models),
            seconds=time.perf_counter() - begin, torch=str(torch.__version__), numpy=np.__version__))


def evaluate(run, smoke, resume):
    config, event, _, original, videos = setup(run, smoke)
    selection = run / ('smoke_selection' if smoke else 'selection')
    choices = read_json(selection / 'summary.json')['rows']
    root = run / ('smoke_scores' if smoke else 'scores')
    output = run / ('smoke_evaluation' if smoke else 'evaluation')
    output.mkdir(exist_ok=True)
    receipt = read_json(root / 'summary.json')
    assert receipt['status'] == 'COMPLETE'
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    all_rows, all_events = [], []
    begin = time.perf_counter()
    for video in videos:
        video_receipt, frames, records, _, offsets, _, _, first = application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for choice in (r for r in choices if r['video'] == video):
            key, identifier, threshold = choice['model'], choice['episode'], choice['threshold']
            method, seed = key.rsplit('_seed', 1)
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
            target = output / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'complete.json').exists():
                assert resume
                previous = read_json(target / 'complete.json')
                assert previous['source_sha256'] == digest(__file__)
                assert previous['selection_sha256'] == digest(selection / 'summary.json')
                all_rows.extend(previous['rows'])
                all_events.extend(pd.read_csv(target / 'events.csv').to_dict('records'))
                continue
            with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                data = {name: saved[name] for name in saved.files}
            before = np.load(root / video / key / identifier / 'unchanged.npy')
            source_rows, source_events, keeps = [], [], {}
            for policy in config['policies']:
                scores = np.load(root / video / key / identifier / (policy + '.npy'))
                for horizon in ['full', 'after_feedback']:
                    if horizon == 'after_feedback' and choice['feedback'] is None:
                        continue
                    local = data if horizon == 'full' else dict(data, known=data['known'] & (data['frame'] > choice['feedback']['frame']))
                    curve, checks, keep = application.fixed_curve(scores, threshold, offsets, records, frames, local, episode['source_lesion_id'], first, video_receipt['fps'])
                    for group, columns in episode['groups'].items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    result = application.result_row(curve, episode)
                    others = [l for l in result['lesions'] if 'all_other' in l['groups']]
                    retentions = [l['retention'] for l in others if l['retention'] is not None]
                    firsts = [l for l in others if l['baseline_first_prompt_time'] is not None]
                    source_rows.append(dict(method=method, seed=int(seed), video=video, episode=identifier, policy=policy, horizon=horizon,
                        repeat_removal=result['source_removal_fraction'], other_retention=float(np.mean(retentions)) if retentions else None,
                        source_baseline_seconds=result['source_baseline_seconds'], source_removed_seconds=result['source_removed_seconds'],
                        other_baseline_seconds=sum(l['baseline_seconds'] for l in others),
                        other_retained_seconds=sum(l['retained_baseline_seconds'] for l in others),
                        first_retention=sum(l['first_prompt_time'] == l['baseline_first_prompt_time'] for l in firsts) / len(firsts) if firsts else None,
                        feedback_available=choice['feedback'] is not None))
                    np.savez_compressed(target / (policy + '_' + horizon + '_curve.npz'), **curve)
                    keeps[policy] = keep
                local_events = [e for e in manifest['events'] if e['episode'] == identifier]
                events = event_rows(dict(manifest, events=local_events), {identifier: scores}, {identifier: before}, threshold, key, policy)
                for row in events:
                    row['after_feedback'] = choice['feedback'] is not None and row['frame'] > choice['feedback']['frame']
                source_events.extend(events)
            np.savez_compressed(target / 'fixed_keep.npz', **keeps)
            pd.DataFrame(source_events).to_csv(target / 'events.csv', index=False)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', rows=source_rows,
                selection_sha256=digest(selection / 'summary.json'), source_sha256=digest(__file__), checks=checks))
            all_rows.extend(source_rows)
            all_events.extend(source_events)
            print('FEEDBACK_REPLAY', video, key, identifier, flush=True)
        atomic_write_json(output / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos), video=video))
        pause_after_checkpoint(output / 'progress.json')
    if (output / 'summary.json').exists():
        assert resume
        return
    pd.DataFrame(all_rows).to_csv(output / 'sources.csv', index=False)
    pd.DataFrame(all_events).to_csv(output / 'events.csv', index=False)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', seconds=time.perf_counter() - begin, source_rows=len(all_rows)))


def summarize(run, smoke):
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    sources = pd.read_csv(root / 'sources.csv')
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    procedures = sources.groupby(['method', 'seed', 'video', 'policy', 'horizon'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed', 'policy', 'horizon'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy', 'horizon'])[metrics].mean().reset_index()
    for name, table in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    events = pd.read_csv(root / 'events.csv')
    rows = []
    for keys, group in events.groupby(['method', 'seed', 'video', 'variant', 'after_feedback']):
        measures = counts(group.retained.to_numpy(bool)[None], group.before_retained.to_numpy(bool), group.same_identity.to_numpy(bool))
        rows.append(dict(zip(['method', 'seed', 'video', 'policy', 'after_feedback'], keys),
            **{k: int(v[0]) for k, v in measures.items()},
            original_wrong=int((~group.same_identity & ~group.before_retained).sum()), events=len(group)))
    pd.DataFrame(rows).to_csv(output / 'event_effects.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row, horizon in enumerate(['full', 'after_feedback']):
        for ax, metric in zip(axes[row], ['repeat_removal', 'other_retention']):
            for method, group in summary[summary.horizon == horizon].groupby('method'):
                ax.plot(group.policy, group[metric] * 100, marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
            if not summary.loc[summary.horizon == horizon, metric].notna().any():
                ax.text(.5, .5, 'Undefined: no qualifying prompts', ha='center', va='center', transform=ax.transAxes)
                ax.set_yticks([])
            policies = sorted(summary.policy.unique())
            ax.set_xticks(range(len(policies)), policies)
            ax.set_title(horizon.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title() + ' (%)')
            ax.tick_params(axis='x', rotation=20)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('One simulated identity contrast | Examined development\nActivation frame excluded from subsequent outcomes')
    fig.savefig(output / 'feedback.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', rows=len(summary),
        aggregation='Defined lesion means, then sources, procedures and seeds; horizons reported separately'))


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
        score(args.run, args.smoke, args.resume)
    elif args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
