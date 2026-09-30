import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import evaluate_shared_acknowledgement_memory as shared
from evaluate_fixed_acknowledgement_policy import one_point, result_row
from src import acknowledgement_sae as model_api
from src.acknowledgement_sae import load_predictor
from src.evaluation.realcolon_task import digest, write_json


def progress(output, config, smoke, phase, number, video):
    counts = [1, 1, 1, 1] if smoke else [len(config['development_videos'])] * 2 + [len(config['extension_videos'])] * 2
    phases = ['development', 'development_interventions', 'extension', 'extension_interventions']
    if not config.get('include_interventions', True):
        counts = [counts[0], counts[2]]
        phases = ['development', 'extension']
    index = phases.index(phase)
    write_json(output / 'progress.json', dict(completed=sum(counts[:index]) + number,
               total=sum(counts), phase=phase, video=video))


def fixed_curve(scores, threshold, offsets, records, frames, data, source, first, fps):
    keep = ~np.isfinite(scores) | (scores < threshold)
    decision = np.where(keep, np.nan, 1.)
    curve, checks = shared.episode_curve(decision, offsets, records, frames, data, source, first, fps)
    curve = one_point(curve)
    return curve, checks, keep


def load_video(base, settings, video):
    detection = shared.memory.acknowledgement.detection
    clips, receipt, frames, _, _ = detection.full_frame_metadata(Path(settings['metadata']), video)
    records, _ = detection.load_predictions(Path(settings['predictions']) / video / 'detections.jsonl',
                                            clips, receipt, camera_rate=True)
    directory = Path(settings['descriptors']) / video
    assert shared.read(directory / 'complete.json')['status'] == 'COMPLETE'
    with np.load(directory / 'indices.npz') as saved:
        offsets = saved['offsets'].copy()
        assert np.array_equal(saved['frame_indices'], [record['frame_index'] for record in records])
        assert np.array_equal(np.diff(offsets), [len(record['detections']) for record in records])
    with np.load(base / video / 'indices.npz') as saved:
        assert np.array_equal(saved['original_detection_positions'], np.arange(offsets[-1]))
        assert np.array_equal(saved['offsets'], offsets)
    raw = np.load(directory / 'raw_mean.npy', mmap_mode='r')
    available = np.load(directory / 'available.npy')
    assert len(raw) == len(available) == offsets[-1]
    assert np.all(np.load(directory / 'status_code.npy') != 0)
    assert np.isfinite(raw[available]).all()
    annotation = next(row for row in shared.read(settings['annotations'])['videos'] if row['video_id'] == video)
    first = {row['lesion_id']: row['first_frame'] for row in annotation['lesions']}
    return receipt, frames, records, directory, offsets, raw, available, first


def source_scores(predictor, source_raw, codes, available):
    memory = np.asarray(predictor.memory(source_raw)).reshape(-1)
    scores = np.full(len(available), np.nan)
    scores[available] = predictor.score_encoded(memory, codes)
    assert np.isfinite(scores[available]).all() and np.max(np.abs(scores[available]), initial=0) <= 1.00001
    return np.clip(scores, -1., 1.), memory


def matched_interventions(memory, seed, episode_id, method):
    active = np.flatnonzero(np.abs(memory) > 0.)
    if len(active) < 2:
        return dict(status='UNDEFINED', active_components=len(active)), {}
    selected = int(active[np.argmax(np.abs(memory[active]))])
    stream = int.from_bytes(hashlib.sha256(f'{seed}:{episode_id}:{method}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(stream)
    control = int(rng.choice(active[active != selected]))
    dose = float(min(abs(memory[selected]), abs(memory[control])))
    vectors = {}
    for name, index in [('selected', selected), ('random_other', control)]:
        value = memory.copy()
        value[index] -= np.sign(value[index]) * dose
        vectors[name] = value
    actual_norms = {name: float(np.linalg.norm(memory - value)) for name, value in vectors.items()}
    assert np.isclose(actual_norms['selected'], actual_norms['random_other'], atol=1e-7, rtol=1e-6)
    return dict(status='DEFINED', active_components=len(active), selected_index=selected,
                control_index=control, seed=seed, derived_seed=stream,
                selected_natural_removed_norm=float(abs(memory[selected])),
                control_natural_removed_norm=float(abs(memory[control])), applied_norm=dose,
                actual_applied_norms=actual_norms,
                renormalized=False), vectors


def condition_curve(scores, phase, threshold, offsets, records, frames, data, source, first, fps):
    if phase == 'development':
        curve, checks = shared.episode_curve(scores, offsets, records, frames, data, source, first, fps)
        return curve, checks, None
    return fixed_curve(scores, threshold, offsets, records, frames, data, source, first, fps)


def evaluate_phase(config, output, phase, predictors, thresholds, smoke, resume):
    base = Path(config[phase + '_base'])
    settings = shared.read(base / 'config.json')
    videos = config[phase + '_videos'][:1] if smoke else config[phase + '_videos']
    conditions = list(predictors) + ['reference_supcon']
    destination = output / phase
    destination.mkdir(exist_ok=True)
    aggregate_config = dict(source_variants=conditions, representations=['score'])
    for number, video in enumerate(videos):
        saved = destination / video / 'summary.json'
        if saved.exists():
            assert resume
            continue
        receipt, frames, records, directory, offsets, raw, available, first = load_video(base, settings, video)
        codes = {name: predictor.encode(np.asarray(raw[available]), batch_size=512)
                 for name, predictor in predictors.items()}
        original = shared.read(base / video / 'summary.json')
        assert original['status'] == 'COMPLETE'
        episodes = []
        source_list = original['episodes'][:1] if smoke else original['episodes']
        for episode in source_list:
            target = destination / video / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'summary.json').exists():
                assert resume
                episodes.append(shared.read(target / 'summary.json'))
                continue
            if not episode['click']['available']:
                result = dict(episode, conditions={}, display_action='retain_baseline',
                              acknowledgement_status='UNAVAILABLE')
                write_json(target / 'summary.json', result)
                episodes.append(result)
                continue
            source_directory = directory / 'sources' / episode['episode_id']
            source_info = shared.read(source_directory / 'source.json')
            for key in ['output_index', 'detection_index', 'input_frame', 'time']:
                assert source_info['click'][key] == episode['click'][key]
            with np.load(source_directory / 'track_memory.npz') as saved_source:
                source_available = bool(saved_source['available'])
                source_raw = saved_source['raw_mean'].copy()
            if source_available:
                position = offsets[episode['click']['output_index']] + episode['click']['detection_index']
                assert available[position] and np.allclose(source_raw, raw[position], atol=1e-10, rtol=0)
            stored_source = base / video / 'sources' / episode['episode_id']
            with np.load(stored_source / 'frame_data.npz') as saved_data:
                data = {key: saved_data[key].copy() for key in saved_data.files}
            assert np.array_equal(data['frame'], [record['frame_index'] for record in records])
            with np.load(stored_source / 'cosines.npz') as saved_scores:
                scores_by_method = dict(reference_supcon=saved_scores['track__supcon_l2'].copy())
            memories = {}
            interface_checks = {}
            for name, predictor in predictors.items():
                scores = np.full(len(raw), np.nan)
                if source_available:
                    scores, memories[name] = source_scores(predictor, source_raw, codes[name], available)
                    if hasattr(predictor, 'score_diagnostics'):
                        np.savez_compressed(target / (name + '__diagnostics.npz'),
                            detection_positions=np.flatnonzero(available),
                            **predictor.score_diagnostics(memories[name], codes[name]))
                    if smoke:
                        positions = np.flatnonzero(available)[:16]
                        direct = predictor.score(source_raw, np.asarray(raw[positions]), batch_size=16)
                        error = float(np.max(np.abs(scores[positions] - direct)))
                        assert error < 1e-5, (name, error)
                        interface_checks[name] = dict(real_queries=len(positions), direct_score_max_error=error)
                scores_by_method[name] = scores
            summaries, fixed_results, keeps = {}, {}, {}
            for name in conditions:
                condition = name + '__score'
                threshold = thresholds[name]['threshold'] if thresholds else None
                curve, checks, keep = condition_curve(scores_by_method[name], phase, threshold,
                    offsets, records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                if smoke and phase == 'development' and name == 'reference_supcon':
                    with np.load(stored_source / 'track__supcon_l2_curve.npz') as reference:
                        for key in ['threshold', 'acknowledged_removed_seconds', 'retained_baseline_seconds',
                                    'first_postclick_correct_prompt_time']:
                            assert np.array_equal(curve[key], reference[key], equal_nan=True), key
                    interface_checks['saved_reference'] = dict(original_curve_exact=True)
                np.savez_compressed(target / (condition + '_curve.npz'), **curve)
                summaries[condition] = dict(points=len(curve['threshold']), source_available=source_available,
                    direct_mask_verification=checks, threshold=threshold)
                if keep is not None:
                    fixed_results[name] = result_row(curve, episode)
                    keeps[name] = keep
            np.savez_compressed(target / 'scores.npz', **scores_by_method)
            if keeps:
                np.savez_compressed(target / 'fixed_keep.npz', **keeps)
            result = dict(episode, conditions=summaries, fixed_results=fixed_results, interface_checks=interface_checks,
                evaluation_population='development' if phase == 'development' else 'previously_examined_test_extension',
                input_identity={str(path): digest(path) for path in [directory / 'identity.json',
                    source_directory / 'track_memory.npz', stored_source / 'frame_data.npz']})
            write_json(target / 'summary.json', result)
            episodes.append(result)
            print('SAE_EPISODE', phase, video, episode['episode_id'], flush=True)
        write_json(destination / video / 'summary.json', dict(video=video, episodes=episodes,
                   status='COMPLETE', smoke=smoke))
        progress(output, config, smoke, phase, number + 1, video)
    shared.aggregate(destination, aggregate_config, videos)
    write_json(destination / 'summary.json', dict(status='SMOKE_COMPLETE' if smoke else 'COMPLETE', videos=videos,
        conditions=conditions, grouping='Lesion then episode then procedure, equal procedure weights.',
        population='Development validation' if phase == 'development' else 'Previously examined test extension'))
    return videos


def calibrate(output, conditions, floor):
    result = {}
    for method in conditions:
        with np.load(output / 'development' / (method + '__score_procedure_curve.npz')) as curve:
            removal = curve['source_suppression__procedure_mean']
            retention = curve['all_other__baseline_qualified_retention__procedure_mean']
            first = curve['all_other__first_baseline_frame_retention__procedure_values']
            admissible = (retention >= floor - 1e-9) & np.all(~np.isfinite(first) | (first >= 1. - 1e-9), axis=0)
            eligible = np.flatnonzero(admissible & np.isfinite(removal))
            assert len(eligible), method
            maximum = removal[eligible].max()
            index = int(eligible[np.flatnonzero(removal[eligible] >= maximum - 1e-12)[-1]])
            result[method] = dict(threshold=float(curve['threshold'][index]),
                removal=float(removal[index]), retention=float(retention[index]),
                per_procedure=[dict(video=video,
                    removal=float(curve['source_suppression__procedure_values'][number, index]),
                    retention=float(curve['all_other__baseline_qualified_retention__procedure_values'][number, index]))
                    for number, video in enumerate(curve['videos'].tolist())])
    write_json(output / 'operating_points.json', dict(selection_population='development_only',
        protected_group='all_other', retention_floor=floor, preserve_every_first_prompt=True, methods=result))
    return result


def interventions(config, output, phase, predictors, points, smoke):
    base = Path(config[phase + '_base'])
    settings = shared.read(base / 'config.json')
    videos = config[phase + '_videos'][:1] if smoke else config[phase + '_videos']
    for number, video in enumerate(videos):
        result_path = output / phase / video / 'interventions.json'
        if result_path.exists():
            continue
        receipt, frames, records, directory, offsets, raw, available, first = load_video(base, settings, video)
        codes = {name: predictor.encode(np.asarray(raw[available]), batch_size=512)
                 for name, predictor in predictors.items()}
        original = shared.read(base / video / 'summary.json')
        results = []
        for episode in original['episodes'][:1] if smoke else original['episodes']:
            if not episode['click']['available']:
                results.append(dict(episode_id=episode['episode_id'], status='CLICK_UNAVAILABLE'))
                continue
            with np.load(directory / 'sources' / episode['episode_id'] / 'track_memory.npz') as source:
                if not bool(source['available']):
                    results.append(dict(episode_id=episode['episode_id'], status='SOURCE_DESCRIPTOR_UNAVAILABLE'))
                    continue
                source_raw = source['raw_mean'].copy()
            with np.load(base / video / 'sources' / episode['episode_id'] / 'frame_data.npz') as saved:
                data = {key: saved[key].copy() for key in saved.files}
            for method, predictor in predictors.items():
                memory = np.asarray(predictor.memory(source_raw)).reshape(-1)
                description, changed = matched_interventions(memory, config['intervention_seed'], episode['episode_id'], method)
                row = dict(video=video, episode_id=episode['episode_id'], method=method,
                           threshold=points[method]['threshold'], **description)
                if changed:
                    variants = dict(full=memory, **changed)
                    outcomes, scores_saved = {}, {}
                    for name, vector in variants.items():
                        scores = np.full(len(raw), np.nan)
                        scores[available] = predictor.score_encoded(vector, codes[method])
                        curve, _, _ = fixed_curve(scores, row['threshold'], offsets, records, frames, data,
                            episode['source_lesion_id'], first, receipt['fps'])
                        outcomes[name] = result_row(curve, episode)
                        scores_saved[name] = scores
                    row['outcomes'] = outcomes
                    target = output / phase / video / 'sources' / episode['episode_id']
                    np.savez_compressed(target / (method + '_component_intervention.npz'), **scores_saved,
                        source_memory=memory, selected_memory=changed['selected'], random_memory=changed['random_other'])
                results.append(row)
        write_json(result_path, dict(video=video, population=phase, results=results,
            interpretation='Fixed-threshold prompt changes after equal-norm single-component attenuation; no renormalization.'))
        progress(output, config, smoke, phase + '_interventions', number + 1, video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = shared.read(args.config)
    started = time.perf_counter()
    identity = dict(config_sha256=digest(args.config), source_sha256=digest(__file__),
        protocol_sha256=digest(config['protocol']), smoke=args.smoke,
        model_source_sha256=digest(model_api.__file__),
        metric_source_sha256=digest(shared.memory.__file__), shared_source_sha256=digest(shared.__file__),
        model_assets={name: {file: digest(Path(directory) / file) for file in
            ['model.npz', 'normalization.npz', 'model_config.json']} for name, directory in config['methods'].items()})
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output / 'identity.json').exists():
        assert args.resume and shared.read(args.output / 'identity.json') == identity
    else:
        write_json(args.output / 'identity.json', identity)
        write_json(args.output / 'config.json', config)
        shutil.copyfile(__file__, args.output / Path(__file__).name)
    predictors = {name: load_predictor(directory, device='cpu') for name, directory in config['methods'].items()}
    conditions = list(predictors) + ['reference_supcon']
    evaluate_phase(config, args.output, 'development', predictors, None, args.smoke, args.resume)
    points_path = args.output / 'operating_points.json'
    points = shared.read(points_path)['methods'] if points_path.exists() else calibrate(args.output, conditions, config['retention_floor'])
    interventions(config, args.output, 'development', predictors, points, args.smoke)
    if not args.development_only:
        evaluate_phase(config, args.output, 'extension', predictors, points, args.smoke, args.resume)
        interventions(config, args.output, 'extension', predictors, points, args.smoke)
    write_json(args.output / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if args.smoke else 'COMPLETE',
        development_videos=config['development_videos'][:1] if args.smoke else config['development_videos'],
        extension_videos=[] if args.development_only else config['extension_videos'][:1] if args.smoke else config['extension_videos'],
        operating_points=points, seconds=time.perf_counter() - started,
        extension_population='Previously examined test procedures; no extension fitting or threshold selection.'))
    total = 4 if args.smoke else 2 * (len(config['development_videos']) + len(config['extension_videos']))
    write_json(args.output / 'progress.json', dict(completed=total // 2 if args.development_only else total,
               total=total, phase='DEVELOPMENT_COMPLETE' if args.development_only else 'COMPLETE', video=None))
    print('SAE_EVALUATION_COMPLETE', round(time.perf_counter() - started, 3), 'seconds', flush=True)


if __name__ == '__main__':
    main()
