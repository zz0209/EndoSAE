import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import evaluate_shared_acknowledgement_memory as shared
from src.evaluation.realcolon_task import digest, write_json


def retain_policy_points(curve, upper):
    selected = curve['threshold'] < upper
    points = len(selected)
    scalar_curves = {'threshold', 'acknowledged_removed_seconds', 'acknowledged_suppression_fraction'}
    return {key: value[selected] if key in scalar_curves or
            (value.ndim == 2 and value.shape[0] == points) else value
            for key, value in curve.items()}


def run(config, output, smoke, resume):
    output.mkdir(parents=True, exist_ok=resume)
    write_json(output / 'config.json', config)
    shutil.copyfile(__file__, output / Path(__file__).name)
    base = Path(config['base'])
    native = Path(config['ntssl'])
    settings = shared.read(base / 'config.json')
    detection = shared.memory.acknowledgement.detection
    annotation = shared.read(settings['annotations'])
    completed = []
    started = time.perf_counter()
    for video in config['videos'][:1] if smoke else config['videos']:
        saved = output / video / 'summary.json'
        if saved.exists():
            assert resume
            completed.append(video)
            continue
        clips, receipt, frames, _, _ = detection.full_frame_metadata(Path(settings['metadata']), video)
        records, _ = detection.load_predictions(Path(settings['predictions']) / video / 'detections.jsonl',
                                                clips, receipt, camera_rate=True)
        with np.load(base / video / 'indices.npz') as index:
            offsets = index['offsets'].copy()
            assert np.array_equal(index['frame_indices'], [r['frame_index'] for r in records])
        original = shared.read(base / video / 'summary.json')
        first = {r['lesion_id']: r['first_frame'] for r in
                 next(r for r in annotation['videos'] if r['video_id'] == video)['lesions']}
        episodes = []
        for episode in original['episodes'][:1] if smoke else original['episodes']:
            target = output / video / 'sources' / episode['episode_id']
            receipt_path = target / 'summary.json'
            if receipt_path.exists():
                assert resume
                episodes.append(shared.read(receipt_path))
                continue
            assert episode['click']['available']
            target.mkdir(parents=True, exist_ok=resume)
            source = base / video / 'sources' / episode['episode_id']
            with np.load(source / 'frame_data.npz') as stored:
                data = {key: stored[key].copy() for key in stored.files}
            elapsed = data['start'] - episode['click']['time']
            with np.load(source / 'cosines.npz') as stored:
                memory_scores = {name: stored['track__' + name + '_l2'].copy()
                                 for name in ['raw', 'standardized', 'supcon']}
            native_source = native / video / 'sources' / episode['episode_id']
            with np.load(native_source / 'cosines.npz') as stored:
                memory_scores['ntssl'] = stored['track__ntssl_l2'].copy()
            assert all(len(scores) == offsets[-1] for scores in memory_scores.values())
            repeated_elapsed = np.repeat(elapsed, np.diff(offsets))
            summaries = {}
            for variant in config['source_variants']:
                if variant == 'timer':
                    post_elapsed = np.maximum(repeated_elapsed, 0.)
                    scores = -post_elapsed / (1. + post_elapsed)
                else:
                    duration = 5. if variant.startswith('mute5_') else 30.
                    scores = memory_scores[variant.rsplit('_', 1)[-1]].copy()
                    scores[repeated_elapsed < duration] = 1.5
                curve, checks = shared.episode_curve(scores, offsets, records, frames, data,
                                                     episode['source_lesion_id'], first, receipt['fps'])
                if variant != 'timer':
                    curve = retain_policy_points(curve, 1.5)
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                condition = variant + '__policy'
                np.savez_compressed(target / (condition + '_curve.npz'), **curve)
                summaries[condition] = dict(points=len(curve['threshold']), direct_mask_verification=checks)
                print('POLICY_CURVE', video, episode['episode_id'], variant, len(curve['threshold']), flush=True)
            result = dict(episode, conditions=summaries,
                          policy_input_identity={str(p): digest(p) for p in
                              [source / 'cosines.npz', source / 'frame_data.npz', native_source / 'cosines.npz']})
            write_json(receipt_path, result)
            episodes.append(result)
        write_json(saved, dict(video=video, episodes=episodes, smoke=smoke, status='COMPLETE'))
        completed.append(video)
        shared.aggregate(output, config, completed)
        write_json(output / 'summary.json', dict(status='SMOKE_COMPLETE' if smoke else
                   ('COMPLETE' if completed == config['videos'] else 'PARTIAL'), videos=completed,
                   planned_videos=config['videos'], seconds=time.perf_counter()-started,
                   python=sys.version, numpy=np.__version__))
        print('POLICY_PROCEDURES', len(completed), '/', len(config['videos']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(shared.read(args.config), args.output, args.smoke, args.resume)
