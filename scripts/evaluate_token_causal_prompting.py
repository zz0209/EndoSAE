import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_acknowledgement_sae as application
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def methods(config):
    return [f'{space}_{method}' for space in config['training_runs'] for method in config['methods']]


def evaluate_phase(run, config, output, root, seed, phase, points, smoke, resume):
    base = Path(config[phase + '_base'])
    settings = read_json(base / 'config.json')
    videos = config[phase + '_videos'][:1] if smoke else config[phase + '_videos']
    names = methods(config) + ['reference_supcon']
    destination = output / phase
    destination.mkdir(exist_ok=True)
    for number, video in enumerate(videos):
        if (destination / video / 'summary.json').exists():
            assert resume
            application.progress(output, config, smoke, phase, number + 1, video)
            continue
        receipt, frames, records, directory, offsets, raw, available, first = application.load_video(base, settings, video)
        encoded = np.load(root / video / 'encoded.npy', allow_pickle=False)
        assert not np.any(encoded & ~available)
        if not smoke:
            np.testing.assert_array_equal(encoded, available)
        codes = {name: np.load(root / video / f'{name}_seed{seed}.npy', mmap_mode='r') for name in methods(config)}
        for value in codes.values():
            assert len(value) == len(raw) and np.isfinite(value[encoded]).all()
            np.testing.assert_allclose(np.linalg.norm(value[encoded], axis=1), 1., atol=1e-5)
        episodes = []
        for episode in read_json(base / video / 'summary.json')['episodes']:
            target = destination / video / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'summary.json').exists():
                assert resume
                episodes.append(read_json(target / 'summary.json'))
                continue
            if not episode['click']['available']:
                result = dict(episode, conditions={}, display_action='retain_baseline', acknowledgement_status='UNAVAILABLE')
                atomic_write_json(target / 'summary.json', result)
                episodes.append(result)
                continue
            source_directory = directory / 'sources' / episode['episode_id']
            info = read_json(source_directory / 'source.json')
            for key in ['output_index', 'detection_index', 'input_frame', 'time']:
                assert info['click'][key] == episode['click'][key]
            with np.load(source_directory / 'track_memory.npz') as original:
                source_available = bool(original['available'])
                position = int(offsets[episode['click']['output_index']] + episode['click']['detection_index'])
                if source_available:
                    assert encoded[position]
                    np.testing.assert_allclose(original['raw_mean'], raw[position], atol=1e-10, rtol=0)
            stored = base / video / 'sources' / episode['episode_id']
            with np.load(stored / 'frame_data.npz') as data_file:
                data = {key: data_file[key].copy() for key in data_file.files}
            np.testing.assert_array_equal(data['frame'], [r['frame_index'] for r in records])
            with np.load(stored / 'cosines.npz') as original:
                scores = dict(reference_supcon=original['track__supcon_l2'].copy())
            checks = {}
            for name, value in codes.items():
                score = np.full(len(raw), np.nan)
                if source_available:
                    query = np.asarray(value[encoded], dtype=np.float64)
                    source = np.asarray(value[position], dtype=np.float64)
                    score[encoded] = np.clip(query @ source, -1., 1.)
                    assert np.isclose(score[position], 1., atol=1e-5)
                    checks[name] = dict(queries=int(encoded.sum()), source_position=position, self_score=float(score[position]))
                scores[name] = score
            summaries, fixed, keeps = {}, {}, {}
            for name in names:
                threshold = points[name]['threshold'] if points else None
                curve, verification, keep = application.condition_curve(scores[name], phase, threshold,
                    offsets, records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                if phase == 'development' and name == 'reference_supcon':
                    with np.load(stored / 'track__supcon_l2_curve.npz') as reference:
                        for key in ['threshold', 'acknowledged_removed_seconds', 'retained_baseline_seconds',
                                    'first_postclick_correct_prompt_time']:
                            np.testing.assert_array_equal(curve[key], reference[key])
                    checks['saved_reference'] = dict(original_curve_exact=True)
                condition = name + '__score'
                np.savez_compressed(target / (condition + '_curve.npz'), **curve)
                summaries[condition] = dict(points=len(curve['threshold']), source_available=source_available,
                                           direct_mask_verification=verification, threshold=threshold)
                if keep is not None:
                    fixed[name] = application.result_row(curve, episode)
                    keeps[name] = keep
            np.savez_compressed(target / 'scores.npz', **scores)
            if keeps:
                np.savez_compressed(target / 'fixed_keep.npz', **keeps)
            result = dict(episode, conditions=summaries, fixed_results=fixed, interface_checks=checks,
                smoke=smoke, population=phase, encoding_identity=digest(root / video / 'identity.json'))
            atomic_write_json(target / 'summary.json', result)
            episodes.append(result)
            print('TOKEN_APPLICATION', seed, phase, video, episode['episode_id'], flush=True)
        atomic_write_json(destination / video / 'summary.json', dict(video=video, episodes=episodes, status='COMPLETE', smoke=smoke))
        application.progress(output, config, smoke, phase, number + 1, video)
    application.shared.aggregate(destination, dict(source_variants=names, representations=['score']), videos)
    atomic_write_json(destination / 'summary.json', dict(status='COMPLETE', videos=videos, smoke=smoke))


def evaluate(run, seed, smoke, resume):
    config = read_json(run / 'config.json')
    assert seed in config['seeds']
    config['include_interventions'] = False
    root = run / 'smoke_encoding' if smoke else Path(config['embedding_root'])
    output = run / ('smoke_evaluation' if smoke else 'evaluation') / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=resume)
    videos = [config['development_videos'][0], config['extension_videos'][0]] if smoke else config['development_videos'] + config['extension_videos']
    paths = [run / 'config.json', run / 'protocol.json', Path(__file__), Path(application.__file__),
             Path(application.shared.__file__), Path(application.shared.memory.__file__)]
    paths += [root / video / name for video in videos for name in ('identity.json', 'complete.json')]
    for video in videos:
        assert read_json(root / video / 'complete.json')['status'] == 'COMPLETE'
    identity = dict(seed=seed, smoke=smoke, files={str(p): digest(p) for p in paths})
    if (output / 'identity.json').exists():
        assert resume and read_json(output / 'identity.json') == identity
    if (output / 'summary.json').exists():
        assert resume and read_json(output / 'summary.json')['status'] == 'COMPLETE'
        return
    atomic_write_json(output / 'identity.json', identity)
    started = time.perf_counter()
    evaluate_phase(run, config, output, root, seed, 'development', None, smoke, resume)
    points_path = output / 'operating_points.json'
    points = read_json(points_path)['methods'] if points_path.exists() else application.calibrate(
        output, methods(config) + ['reference_supcon'], config['application_retention_floor'])
    evaluate_phase(run, config, output, root, seed, 'extension', points, smoke, resume)
    assert identity['files'] == {str(p): digest(p) for p in paths}
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', smoke=smoke, seed=seed,
        operating_points=points, seconds=time.perf_counter() - started,
        scope='Technical partial-query smoke' if smoke else 'Previously examined development and extension; exploratory application evidence'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluate(args.run, args.seed, args.smoke, args.resume)
