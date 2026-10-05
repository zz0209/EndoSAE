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

from score_local_causal_identity import ROOT, load_models, model_specs, vectors, video_inputs
from analyze_temporal_identity_components import remove_components
import evaluate_acknowledgement_sae as application
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def select_events(data, episode):
    rows = []
    eligible = data['known'] & data['post_click']
    for column, lesion in enumerate(data['lesion_ids']):
        for stratum in ['current_visibility', 'unknown_continuity', 'later_reappearance']:
            indices = np.flatnonzero(eligible & data['baseline'][:, column] &
                                     (data['acknowledged_stratum'] == stratum))
            if not len(indices):
                continue
            boundaries = np.flatnonzero((np.diff(indices) != 1) | (np.diff(data['frame'][indices]) != 1)) + 1
            for bout, block in enumerate(np.split(indices, boundaries)):
                chosen = block[np.unique([0, len(block) // 2, len(block) - 1])]
                for output in chosen:
                    rows.append(dict(episode=episode['episode_id'], output=int(output),
                        frame=int(data['frame'][output]), lesion=str(lesion), stratum=stratum, bout=bout,
                        same_identity=bool(lesion == episode['source_lesion_id']),
                        first_prompt=bool(data['first_frame'][output, column]),
                        represented_prompt_frames=len(block)))
    return rows


def prepare(run, smoke, resume):
    config = read_json(run / 'config.json')
    original = read_json(Path(config['application_run']) / 'config.json')
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    root = run / ('smoke_inputs' if smoke else 'inputs')
    root.mkdir(exist_ok=True)
    videos = original['development_videos'][:1] if smoke else original['development_videos']
    begin = time.perf_counter()
    receipts = []
    for video in videos:
        folder = root / video
        folder.mkdir(exist_ok=True)
        identity = dict(config=digest(run / 'config.json'), source=digest(__file__), smoke=smoke,
                        raw=digest(Path(original['raw_root']) / video / 'complete.json'))
        if (folder / 'complete.json').exists():
            receipt = read_json(folder / 'complete.json')
            assert resume and receipt['identity'] == identity
            receipts.append(receipt)
            continue
        receipt, frames, records, _, offsets, _, available, _ = application.load_video(base, settings, video)
        definition, _, _, _, _, targets, _, _ = video_inputs(original, video)
        episodes = [e for e in read_json(base / video / 'summary.json')['episodes'] if e['click']['available']]
        events = []
        for episode in episodes:
            with np.load(base / video / 'sources' / episode['episode_id'] / 'frame_data.npz') as saved:
                data = {key: saved[key].copy() for key in saved.files}
            local = select_events(data, episode)
            firsts = {(int(i), str(data['lesion_ids'][j])) for i, j in zip(*np.where(data['first_frame']))}
            assert firsts <= {(e['output'], e['lesion']) for e in local}
            events.extend(local)
        outputs = sorted({e['output'] for e in events})
        requested = {int(p) for i in outputs for p in range(offsets[i], offsets[i + 1]) if available[p]}
        requested.update(int(offsets[e['click']['output_index']]) + e['click']['detection_index'] for e in episodes)
        raw_root = Path(original['raw_root']) / video
        raw_receipt = read_json(raw_root / 'complete.json')
        selected = {}
        for number, shard in enumerate(raw_receipt['shards']):
            shard_outputs = targets[shard['first']:shard['stop']]
            if not any(offsets[i] <= p < offsets[i + 1] for i in shard_outputs for p in requested):
                continue
            path = raw_root / shard['file']
            assert digest(path) == shard['sha256']
            with np.load(path) as saved:
                positions = saved['detection_positions']
                wanted = np.flatnonzero(np.isin(positions, list(requested)))
                values, bounds = saved['tokens'], saved['offsets']
                for i in wanted:
                    selected[int(positions[i])] = values[bounds[i]:bounds[i + 1]].copy()
            print('EVENT_INPUT_SHARD', video, number + 1, '/', len(raw_receipt['shards']),
                  'detections', len(selected), '/', len(requested), flush=True)
        assert set(selected) == requested
        positions = np.array(sorted(selected), dtype=int)
        bounds = np.concatenate([[0], np.cumsum([len(selected[p]) for p in positions])])
        np.savez_compressed(folder / 'tokens.npz', positions=positions, offsets=bounds,
                            tokens=np.concatenate([selected[p] for p in positions]))
        frame_rows = {}
        for output in outputs:
            frame = frames[records[output]['frame_index']]
            boxes = records[output]['detections']
            matches = application.shared.memory.acknowledgement.detection.overlap(boxes, frame['original_boxes_xyxy'])
            frame_rows[str(output)] = dict(frame=frame, detections=boxes,
                positions=list(range(int(offsets[output]), int(offsets[output + 1]))), matches=matches['matches'])
        for episode in episodes:
            episode['source_position'] = int(offsets[episode['click']['output_index']]) + episode['click']['detection_index']
            source = Path(original['source_token_root']) / video / 'sources' / episode['episode_id']
            info = read_json(source / 'source.json')
            assert not info['future_frames_used'] and not info['ground_truth_regions_used']
            with np.load(source / 'observed_tokens.npz') as saved:
                np.testing.assert_array_equal(selected[episode['source_position']], saved['tokens'])
            episode['source_info'] = info
        atomic_write_json(folder / 'events.json', dict(video=video, events=events, frames=frame_rows,
            episodes=episodes, definition=definition, fps=receipt['fps']))
        result = dict(status='COMPLETE', identity=identity, video=video, events=len(events),
            frames=len(outputs), detections=len(positions), sources=len(episodes), tokens=int(bounds[-1]),
            all_first_prompts_included=True, all_source_tokens_exact=True,
            assets={name: digest(folder / name) for name in ['tokens.npz', 'events.json']})
        atomic_write_json(folder / 'complete.json', result)
        receipts.append(result)
        atomic_write_json(root / 'progress.json', dict(completed=len(receipts), total=len(videos), video=video))
        pause_after_checkpoint(folder / 'complete.json')
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', videos=receipts,
                      seconds=time.perf_counter() - begin))


def event_rows(manifest, scores, before_scores, threshold, key, variant, terms=None):
    detection = application.shared.memory.acknowledgement.detection
    rows = []
    for index, event in enumerate(manifest['events']):
        frame = manifest['frames'][str(event['output'])]
        positions = np.array(frame['positions'])
        local = scores[event['episode']][positions]
        old = before_scores[event['episode']][positions]
        keep = ~np.isfinite(local) | (local < threshold)
        old_keep = ~np.isfinite(old) | (old < threshold)
        boxes = frame['detections']
        annotation = frame['frame']['original_boxes_xyxy']
        state = detection.overlap([b for b, k in zip(boxes, keep) if k], annotation)
        before = detection.overlap([b for b, k in zip(boxes, old_keep) if k], annotation)
        assert event['lesion'] in [m['lesion_id'] for m in frame['matches']]
        matched = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
        position = int(positions[matched['prediction_index']])
        retained = event['lesion'] in state['detected_lesion_ids']
        before_retained = event['lesion'] in before['detected_lesion_ids']
        method, seed = key.rsplit('_seed', 1)
        rows.append(dict(event_id=index, video=manifest['video'], method=method, seed=int(seed),
            variant=variant, threshold=threshold, **event, retained=retained,
            before_retained=before_retained, matched_position=position,
            matched_score=float(scores[event['episode']][position]),
            matched_before_score=float(before_scores[event['episode']][position]),
            component_contribution=float(terms[event['episode']][position]) if terms is not None else None,
            detections=len(positions)))
    return rows


@torch.no_grad()
def evaluate(run, smoke, resume):
    config = read_json(run / 'config.json')
    original = read_json(Path(config['application_run']) / 'config.json')
    original.update(training_runs=config['training_runs'], methods=config['methods'], seeds=config['seeds'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    inputs = run / ('smoke_inputs' if smoke else 'inputs')
    output = run / ('smoke_evaluation' if smoke else 'evaluation')
    output.mkdir(exist_ok=True)
    videos = read_json(inputs / 'summary.json')['videos']
    models = load_models(original, device)
    folders = dict(model_specs(original))
    begin = time.perf_counter()
    completed = 0
    for receipt in videos:
        folder = inputs / receipt['video']
        for name, expected in receipt['assets'].items():
            assert digest(folder / name) == expected
        manifest = read_json(folder / 'events.json')
        with np.load(folder / 'tokens.npz') as saved:
            raw, offsets, positions = saved['tokens'].copy(), saved['offsets'].copy(), saved['positions'].copy()
        lookup = {int(p): i for i, p in enumerate(positions)}
        for key, (model, mean, scale) in models.items():
            method, seed_text = key.rsplit('_seed', 1)
            seed = int(seed_text)
            component = Path(config['component_root']) / method.removeprefix('p27v4_') / f'seed{seed}' / 'foldfull'
            component_receipt = read_json(component / 'complete.json')
            assert component_receipt['identity']['model_sha256'] == digest(folders[key] / 'model.npz')
            target = output / manifest['video'] / key
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(inputs=digest(folder / 'complete.json'), config=digest(run / 'config.json'),
                model=digest(folders[key] / 'model.npz'), source=digest(__file__),
                components=digest(component / 'effects.npz'), smoke=smoke)
            if (target / 'complete.json').exists():
                assert resume and read_json(target / 'complete.json')['identity'] == identity
                completed += 1
                continue
            assert model.identity_space == 'code'
            representations, pooled = [], []
            for i in range(len(positions)):
                values = ((raw[offsets[i]:offsets[i + 1]].astype(float) - mean) / scale).astype(np.float32)
                projected, _, local = model(torch.from_numpy(values[None]).to(device))
                representations.append(projected[0].cpu().numpy().astype(float))
                pooled.append(local.mean(dim=1)[0])
            unit = np.stack(representations)
            with np.load(component / 'effects.npz') as saved:
                features = {name: saved[name + '_features'].copy() for name in ['selected', 'random0', 'random1']}
            modified = {}
            for variant, selected in features.items():
                changed = []
                for value in pooled:
                    intervention = value.clone()
                    intervention[selected] = 0
                    if not torch.isfinite(intervention).all() or intervention.norm() <= 0:
                        raise ValueError('Invalid intervened pooled representation')
                    changed.append(F.normalize(intervention[None], dim=-1)[0].cpu().numpy().astype(float))
                modified[variant] = np.stack(changed)
            points = read_json(Path(config['application_run']) / 'evaluation' / f'seed{seed}' / 'operating_points.json')['methods']
            threshold = points[method + '_pooled_cosine']['threshold']
            source_arrays, variant_arrays, terms = {}, {name: {} for name in features}, {name: {} for name in features}
            max_error = 0.
            for episode in manifest['episodes']:
                identifier = episode['episode_id']
                saved_path = Path(original['embedding_root']) / manifest['video'] / 'sources' / identifier / f'{method}_pooled_cosine_seed{seed}.npy'
                saved_scores = np.load(saved_path)
                source = np.full(len(positions), lookup[episode['source_position']], dtype=int)
                query = np.arange(len(positions))
                before = np.clip(np.array([value @ unit[source[0]] for value in unit]), -1., 1.)
                np.testing.assert_allclose(before, saved_scores[positions], atol=2e-6, rtol=0)
                np.testing.assert_array_equal(before < threshold, saved_scores[positions] < threshold)
                max_error = max(max_error, float(np.max(np.abs(before - saved_scores[positions]))))
                source_arrays[identifier] = saved_scores.copy()
                for variant, selected in features.items():
                    closed_form = remove_components(unit, source, query, selected)
                    explicit = unit.copy()
                    explicit[:, selected] = 0
                    explicit /= np.linalg.norm(explicit, axis=1, keepdims=True)
                    np.testing.assert_allclose(closed_form, explicit @ explicit[source[0]], atol=2e-12, rtol=0)
                    after = np.clip(np.array([value @ modified[variant][source[0]] for value in modified[variant]]), -1., 1.)
                    np.testing.assert_allclose(after, closed_form, atol=2e-6, rtol=0)
                    unchanged = (unit[:, selected] == 0).all(1) & (unit[source[0], selected] == 0).all()
                    np.testing.assert_array_equal(after[unchanged], before[unchanged])
                    values = np.full_like(saved_scores, np.nan)
                    values[positions] = np.clip(after, -1., 1.)
                    variant_arrays[variant][identifier] = values
                    contribution = np.full_like(saved_scores, np.nan)
                    contribution[positions] = (unit[source][:, selected] * unit[:, selected]).sum(1)
                    terms[variant][identifier] = contribution
            rows = event_rows(manifest, source_arrays, source_arrays, threshold, key, 'before')
            for variant in features:
                rows.extend(event_rows(manifest, variant_arrays[variant], source_arrays, threshold, key, variant, terms[variant]))
            pd.DataFrame(rows).to_csv(target / 'events.csv', index=False)
            arrays = {f'{identifier}__before': values[positions] for identifier, values in source_arrays.items()}
            arrays.update({f'{identifier}__{variant}': values[positions] for variant, sources in variant_arrays.items()
                           for identifier, values in sources.items()})
            np.savez_compressed(target / 'scores.npz', positions=positions, unit_codes=unit, **arrays,
                                **{name + '_features': values for name, values in features.items()})
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity,
                original_score_max_error=max_error, original_decisions_exact=True,
                explicit_component_removal_checked=True, rows=len(rows),
                peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            completed += 1
            atomic_write_json(output / 'progress.json', dict(completed=completed, total=len(videos) * len(models),
                              video=manifest['video'], model=key))
            print('EVENT_MODEL', completed, '/', len(videos) * len(models), manifest['video'], key, flush=True)
            pause_after_checkpoint(target / 'complete.json')
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', models=completed,
        seconds=time.perf_counter() - begin, torch=str(torch.__version__), numpy=np.__version__))


def summarize(run, smoke):
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    events = pd.concat([pd.read_csv(p) for p in sorted(root.glob('*/*/events.csv'))], ignore_index=True)
    events.to_csv(output / 'events.csv', index=False)
    rows = []
    for keys, group in events.groupby(['video', 'method', 'seed', 'variant'], sort=True):
        same, other = group[group.same_identity], group[~group.same_identity]
        first = other[other.first_prompt]
        wrong = other[~other.before_retained]
        correct = same[~same.before_retained]
        rows.append(dict(zip(['video', 'method', 'seed', 'variant'], keys),
            repeat_removal=(~same.retained).mean(), other_retention=other.retained.mean(),
            first_retention=first.retained.mean(), false_removal_correction=wrong.retained.mean(),
            correct_repeat_damage=correct.retained.mean(), repeat_events=len(same), other_events=len(other),
            first_events=len(first), original_false_removals=len(wrong), original_correct_removals=len(correct)))
    procedures = pd.DataFrame(rows)
    procedures.to_csv(output / 'procedures.csv', index=False)
    metrics = ['repeat_removal', 'other_retention', 'first_retention', 'false_removal_correction', 'correct_repeat_damage']
    seeds = procedures.groupby(['method', 'variant', 'seed'])[metrics].mean().reset_index()
    seeds.to_csv(output / 'seeds.csv', index=False)
    summary = seeds.groupby(['method', 'variant'])[metrics].mean().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    contribution = events[events.variant != 'before'].groupby(
        ['video', 'episode', 'method', 'seed', 'variant', 'same_identity']).component_contribution.mean().unstack()
    contribution.columns = ['other_contribution' if not c else 'same_contribution' for c in contribution.columns]
    contribution['other_minus_same'] = contribution.other_contribution - contribution.same_contribution
    contribution.to_csv(output / 'source_contributions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
    colors = {'p27v4_token_sparse': '#126A89', 'p27v4_token_dense': '#CB6549'}
    variants = ['before', 'selected', 'random0', 'random1']
    for ax, metric, title in zip(axes, metrics[:3], ['Repeat prompts removed', 'Other prompts retained', 'First other prompts retained']):
        for method, group in summary.groupby('method'):
            values = group.set_index('variant').loc[variants, metric].to_numpy() * 100
            ax.plot(range(4), values, marker='o', color=colors[method], label='SAE' if method.endswith('sparse') else 'Dense')
        ax.set_xticks(range(4), ['Before', 'Selected', 'Random 0', 'Random 1'], rotation=20)
        ax.set_title(title)
        ax.set_ylabel('Procedure mean (%)')
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Fixed development events | Frozen models and original thresholds')
    fig.savefig(output / 'event_interventions.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', events=len(events),
        procedure_rows=len(procedures), seed_rows=len(seeds), source_contrasts=len(contribution),
        scope='Candidate-independent sampled development events; not time-weighted full-video effectiveness'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run, args.smoke, args.resume)
    elif args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
