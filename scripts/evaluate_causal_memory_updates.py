import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_acknowledgement_sae as application
import encode_causal_detection_identity as reference_api
from evaluate_time_identity_policy import retain_policy_points
from train_acknowledgement_sae import chronological_pairs
from train_region_identity_sae import load_inputs
from src.causal_identity_memory import METHODS, memory_scores, replay_memory
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest
from src.temporal_shared_sae import load_predictor


@torch.no_grad()
def shared_codes(predictor, raw):
    values = predictor._input(raw)
    result = []
    for start in range(0, len(values), 256):
        code, _ = predictor.model.encode_parts(torch.from_numpy(values[start:start + 256]))
        if not torch.isfinite(code).all() or torch.any(code.norm(dim=-1) <= 0):
            raise ValueError('Invalid shared component code')
        result.append(F.normalize(code, dim=-1).numpy())
    return np.concatenate(result)


def prepare(run, seed):
    config = read_json(run / 'config.json')
    parent = Path(config['component_run'])
    parent_config = read_json(parent / 'config.json')
    views, _, records, _ = load_inputs(parent_config)
    pairs = chronological_pairs(records, parent_config['validation_video_ids'])
    source = np.asarray([p['source_index'] for p in pairs])
    query = np.asarray([p['query_index'] for p in pairs])
    labels = np.asarray([p['same_identity'] for p in pairs])
    reference = reference_api.Representations(config['reference_fit'])
    paths = [Path(config['reference_fit']) / name for name in ('model.npz', 'normalization.npz')]
    models = {}
    codes = {'reference': reference(views[:, 0])['supcon_l2']}
    for method, original in (('sparse_guard', 'sparse_shared'), ('dense_guard', 'dense_shared')):
        directory = parent / 'fit' / original / f'seed{seed}'
        models[method] = load_predictor(directory)
        paths.extend(directory / name for name in ('model.npz', 'normalization.npz', 'model_config.json'))
        codes[method] = shared_codes(models[method], views[:, 0])
    weights = np.zeros(len(pairs))
    for video in parent_config['validation_video_ids']:
        negative = np.asarray([p['video_id'] == video for p in pairs]) & ~labels
        if negative.any():
            weights[negative] = 1 / negative.sum()
    negative = weights > 0
    thresholds, results = {}, {}
    for method, values in codes.items():
        scores = (values[source].astype(np.float64) * values[query].astype(np.float64)).sum(-1)
        threshold = float(np.quantile(scores[negative], config['calibration_quantile'],
                                     method='inverted_cdf', weights=weights[negative]))
        thresholds[method] = threshold
        results[method] = dict(threshold=threshold, same_recall=float(np.mean(scores[labels] > threshold)),
                               weighted_other_acceptance=float(np.average(scores[negative] > threshold, weights=weights[negative])))
    report = dict(thresholds=thresholds, methods=results, seed=seed, calibration_pairs=len(pairs),
                  negative_pairs=int(negative.sum()), calibration_videos=parent_config['validation_video_ids'],
                  population='Previously exposed validation GT-ROI chronological pairs',
                  model_files={str(p): digest(p) for p in paths})
    destination = run / 'calibration' / f'seed{seed}.json'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and read_json(destination) != report:
        raise ValueError('Saved calibration changed')
    atomic_write_json(destination, report)
    return config, models, thresholds, report


def verify_causality(codes, available, offsets, times, tracks, click, reference, guards, thresholds, config,
                     scores, events):
    zero, _ = replay_memory(codes, available, offsets, reference, [], config['memory_capacity'])
    assert np.array_equal(zero, reference, equal_nan=True)
    event_outputs = [e['output_index'] for values in events.values() for e in values]
    cut = min(len(times) - 1, max(click['output_index'] + 2, event_outputs[0] + 2 if event_outputs else len(times) // 2))
    count = int(offsets[cut])
    prefix, prefix_events, _ = memory_scores(codes[:count], available[:count], offsets[:cut + 1], times[:cut], tracks[:count],
                                 click, reference[:count], {k: v[:count] for k, v in guards.items()}, thresholds, config)
    maximum_error = 0.
    for method in METHODS:
        difference = np.abs(prefix[method] - scores[method][:count])
        maximum_error = max(maximum_error, float(np.nanmax(difference)))
        assert np.allclose(prefix[method], scores[method][:count], atol=1e-14, rtol=0, equal_nan=True), method
        assert prefix_events[method] == [event for event in events[method] if event['output_index'] < cut]
        for position in np.flatnonzero(available)[::max(1, int(available.sum()) // 17)]:
            output = int(np.searchsorted(offsets, position, side='right') - 1)
            prior = [e['detection_position'] for e in events[method] if e['output_index'] < output]
            prior = prior[-(config['memory_capacity'] - 1):]
            expected = max([float(reference[position])] +
                           [float(np.clip(np.dot(codes[position].astype(float), codes[i].astype(float)), -1, 1)) for i in prior])
            assert abs(expected - scores[method][position]) < 1e-12, (method, position)
    return dict(no_write_exact=True, prefix_max_absolute_error=maximum_error, prefix_tolerance=1e-14,
                prefix_events_exact=True, prefix_frames=cut, independent_scalar_replay=True)


def label_writes(events, records, frames, offsets, source):
    matches = {}
    for output in sorted({e['output_index'] for rows in events.values() for e in rows}):
        record = records[output]
        frame = frames.get(record['frame_index'])
        matches[output] = None if frame is None else {
            row['prediction_index']: row['lesion_id'] for row in
            application.shared.memory.acknowledgement.detection.overlap(record['detections'], frame['original_boxes_xyxy'])['matches']}
    result = {}
    for method, rows in events.items():
        result[method] = []
        for event in rows:
            matching = matches[event['output_index']]
            local = event['detection_position'] - offsets[event['output_index']]
            lesion = matching.get(local) if matching is not None else None
            status = 'unknown' if matching is None else 'unmatched' if lesion is None else 'source' if lesion == source else 'other'
            result[method].append(dict(event, annotation_status=status, matched_lesion_id=lesion))
    return result


def phase(run, output, config, models, thresholds, settings, population, points, smoke, resume, stop_after):
    base = Path(settings[population + '_base'])
    population_config = read_json(base / 'config.json')
    videos = settings[population + '_videos'][:1] if smoke else settings[population + '_videos']
    destination = output / population
    destination.mkdir(exist_ok=True)
    conditions = [f'{method}_mute{seconds}' for method in METHODS for seconds in config['mute_seconds']]
    completed_sources = 0
    for number, video in enumerate(videos):
        saved_video = destination / video / 'summary.json'
        if saved_video.exists():
            if not resume:
                raise FileExistsError(saved_video)
            continue
        receipt, frames, records, directory, offsets, raw, available, first = application.load_video(base, population_config, video)
        codes = np.load(directory / 'supcon_l2.npy')
        gate_codes = {method: shared_codes(model, np.asarray(raw[available])) for method, model in models.items()}
        original = read_json(base / video / 'summary.json')
        episodes = []
        for episode in original['episodes'][:1] if smoke else original['episodes']:
            target = destination / video / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'summary.json').exists():
                if not resume:
                    raise FileExistsError(target)
                episodes.append(read_json(target / 'summary.json'))
                continue
            if not episode['click']['available']:
                result = dict(episode, conditions={}, display_action='retain_baseline')
                atomic_write_json(target / 'summary.json', result)
                episodes.append(result)
                continue
            source_directory = directory / 'sources' / episode['episode_id']
            with np.load(source_directory / 'track_memory.npz', allow_pickle=False) as stored:
                source_raw, source_available = stored['raw_mean'].copy(), bool(stored['available'])
            stored_source = base / video / 'sources' / episode['episode_id']
            with np.load(stored_source / 'frame_data.npz', allow_pickle=False) as stored:
                data = {key: stored[key].copy() for key in stored.files}
            with np.load(stored_source / 'cosines.npz', allow_pickle=False) as stored:
                reference = stored['track__supcon_l2'].copy()
            track_path = Path(population_config['episodes']) / video / 'sources' / episode['source_lesion_id'] / 'tracks.npz'
            with np.load(track_path, allow_pickle=False) as stored:
                assert np.array_equal(offsets, stored['offsets'])
                assert np.array_equal(data['frame'], stored['frame_indices'])
                tracks = stored['track_ids'].copy()
            guards = {}
            for method, model in models.items():
                guards[method] = np.full(len(available), np.nan)
                if source_available:
                    source_code = shared_codes(model, source_raw.reshape(1, -1))[0]
                    guards[method][available] = gate_codes[method].astype(float) @ source_code.astype(float)
            if source_available:
                scores, events, winners = memory_scores(codes, available, offsets, data['start'], tracks, episode['click'],
                                                        reference, guards, thresholds, config)
                verification = verify_causality(codes, available, offsets, data['start'], tracks, episode['click'], reference,
                                                 guards, thresholds, config, scores, events) if smoke else {}
            else:
                scores = {method: reference.copy() for method in METHODS}
                events, winners, verification = {method: [] for method in METHODS}, {}, {}
            assert np.array_equal(scores['static'], reference, equal_nan=True)
            labeled = label_writes(events, records, frames, offsets, episode['source_lesion_id'])
            atomic_write_json(target / 'writes.json', labeled)
            np.savez_compressed(target / 'memory_scores.npz', **scores)
            np.savez_compressed(target / 'memory_winners.npz', **winners)
            summaries, fixed, keeps = {}, {}, {}
            elapsed = np.repeat(data['start'] - episode['click']['time'], np.diff(offsets))
            for method in METHODS:
                for mute in config['mute_seconds']:
                    name = f'{method}_mute{mute}'
                    values = scores[method].copy()
                    if mute:
                        values[(elapsed >= -1e-9) & (elapsed < mute)] = 1.5
                    threshold = points[name]['threshold'] if points else None
                    curve, checks, keep = application.condition_curve(values, population, threshold, offsets, records,
                                                                      frames, data, episode['source_lesion_id'], first, receipt['fps'])
                    if population == 'development' and mute:
                        curve = retain_policy_points(curve, 1.5)
                    if smoke and population == 'development' and name == 'static_mute0':
                        with np.load(stored_source / 'track__supcon_l2_curve.npz') as old:
                            for key in ('threshold', 'acknowledged_removed_seconds', 'retained_baseline_seconds'):
                                assert np.array_equal(curve[key], old[key], equal_nan=True)
                    for group, columns in episode['groups'].items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    np.savez_compressed(target / (name + '__score_curve.npz'), **curve)
                    summaries[name + '__score'] = dict(points=len(curve['threshold']), threshold=threshold,
                                                       direct_mask_verification=checks)
                    if keep is not None:
                        fixed[name], keeps[name] = application.result_row(curve, episode), keep
            if keeps:
                np.savez_compressed(target / 'fixed_keep.npz', **keeps)
            result = dict(episode, conditions=summaries, fixed_results=fixed, causal_checks=verification,
                          input_identity={str(p): digest(p) for p in (track_path, directory / 'identity.json',
                              source_directory / 'track_memory.npz', stored_source / 'frame_data.npz')})
            atomic_write_json(target / 'summary.json', result)
            episodes.append(result)
            completed_sources += 1
            print('MEMORY_REPLAY', population, episode['episode_id'],
                  {name: len(rows) for name, rows in events.items()}, flush=True)
            if stop_after and completed_sources >= stop_after:
                raise SystemExit(75)
        atomic_write_json(saved_video, dict(video=video, episodes=episodes, status='COMPLETE', smoke=smoke))
        atomic_write_json(output / 'progress.json', dict(completed=number + 1 + (len(videos) if population == 'extension' else 0),
                                                       total=2 * len(videos), phase=population, video=video))
    application.shared.aggregate(destination, dict(source_variants=conditions, representations=['score']), videos)
    atomic_write_json(destination / 'summary.json', dict(status='COMPLETE', videos=videos, conditions=conditions))
    return conditions


def evaluate(run, seed, smoke, resume, override, stop_after):
    torch.set_num_threads(1)
    started = time.perf_counter()
    config, models, thresholds, calibration = prepare(run, seed)
    if seed not in config['seeds']:
        raise ValueError('Unspecified seed')
    output = override or run / ('smoke' if smoke else 'evaluation') / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=resume)
    files = [run / 'config.json', run / 'protocol.json', Path(__file__), ROOT / 'src/causal_identity_memory.py',
             Path(application.__file__), Path(application.shared.__file__), Path(application.shared.memory.__file__)]
    identity = dict(files={str(p): digest(p) for p in files}, calibration=calibration, seed=seed, smoke=smoke,
                    runtime=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__), threads=1))
    if (output / 'identity.json').exists() and read_json(output / 'identity.json') != identity:
        raise ValueError('Run source or calibration changed')
    atomic_write_json(output / 'identity.json', identity)
    settings = read_json(config['settings_file'])
    names = phase(run, output, config, models, thresholds, settings, 'development', None, smoke, resume, stop_after)
    points = application.calibrate(output, names, config['retention_floor'])
    phase(run, output, config, models, thresholds, settings, 'extension', points, smoke, resume, stop_after)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', seed=seed, smoke=smoke,
                                                  seconds=time.perf_counter() - started, methods=names,
                                                  population='Previously examined development and extension'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after-sources', type=int)
    args = parser.parse_args()
    evaluate(args.run, args.seed, args.smoke, args.resume, args.output, args.stop_after_sources)
