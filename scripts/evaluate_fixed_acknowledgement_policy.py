import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import evaluate_shared_acknowledgement_memory as shared
from src.evaluation.realcolon_task import digest, write_json


def number(value):
    return float(value) if np.isfinite(value) else None


def fixed_keep(point, elapsed, post, scores):
    keep = np.ones(len(elapsed), dtype=bool)
    if 'score_threshold' in point:
        positive = np.maximum(elapsed, 0.)
        keep[post] = (-positive / (1. + positive))[post] < point['score_threshold']
    else:
        if 'cosine_threshold' in point:
            keep[post] = (~np.isfinite(scores) | (scores < point['cosine_threshold']))[post]
        keep[post & (elapsed < point['mute_seconds'])] = False
    return keep


def one_point(curve):
    count = len(curve['threshold'])
    scalars = {'threshold', 'acknowledged_removed_seconds', 'acknowledged_suppression_fraction'}
    return {key: value[:1] if key in scalars or (value.ndim == 2 and value.shape[0] == count) else value
            for key, value in curve.items()}


def result_row(curve, episode):
    source = list(curve['lesion_ids']).index(episode['source_lesion_id'])
    lesions = []
    for column, lesion in enumerate(curve['lesion_ids']):
        lesions.append(dict(lesion_id=str(lesion), groups=[name for name, columns in episode['groups'].items() if column in columns],
            baseline_seconds=number(curve['baseline_seconds'][column]),
            retained_baseline_seconds=number(curve['retained_baseline_seconds'][0, column]),
            retention=number(curve['baseline_qualified_retention'][0, column]),
            visible_seconds=number(curve['visible_seconds'][column]),
            actual_correct_seconds=number(curve['actual_correct_seconds'][0, column]),
            first_frame_retention=number(curve['first_baseline_frame_retention'][0, column]),
            first_bout_retention=number(curve['first_baseline_bout_retention'][0, column]),
            baseline_first_prompt_time=number(curve['baseline_first_postclick_correct_prompt_time'][column]),
            first_prompt_time=number(curve['first_postclick_correct_prompt_time'][0, column]),
            added_first_prompt_delay_seconds=number(curve['added_postclick_delay_seconds'][0, column]),
            completely_hidden=bool(curve['completely_hidden_baseline_detected_lesion'][0, column])))
    return dict(source_baseline_seconds=number(curve['baseline_seconds'][source]),
        source_removed_seconds=number(curve['acknowledged_removed_seconds'][0]),
        source_removal_fraction=number(curve['acknowledged_suppression_fraction'][0]), lesions=lesions)


def run(config_path, output, videos, smoke, resume):
    config = shared.read(config_path)
    protocol_path = Path(config['protocol'])
    protocol = shared.read(protocol_path)
    points = protocol['operating_points']
    identity = dict(config_sha256=digest(config_path), protocol_sha256=digest(protocol_path),
        source_sha256=digest(__file__), shared_source_sha256=digest(shared.__file__),
        metric_source_sha256=digest(shared.memory.__file__), smoke=smoke)
    output.mkdir(parents=True, exist_ok=resume)
    if (output / 'identity.json').exists():
        assert resume and shared.read(output / 'identity.json') == identity
    else:
        write_json(output / 'identity.json', identity)
        write_json(output / 'config.json', config)
        write_json(output / 'confirmation_protocol.json', protocol)
        for path in [Path(__file__), Path(shared.__file__), Path(shared.memory.__file__)]:
            shutil.copyfile(path, output / path.name)
    base, native = Path(config['base']), Path(config['ntssl'])
    settings = shared.read(base / 'config.json')
    annotations = shared.read(settings['annotations'])
    detection = shared.memory.acknowledgement.detection
    started = time.perf_counter()
    for video in videos[:1] if smoke else videos:
        assert video in config['videos']
        destination = output / video
        if (destination / 'summary.json').exists():
            assert resume
            continue
        original = shared.read(base / video / 'summary.json')
        native_summary = shared.read(native / video / 'summary.json')
        assert original['status'] == native_summary['status'] == 'COMPLETE'
        assert not original['smoke'] and not native_summary['smoke']
        clips, receipt, frames, _, _ = detection.full_frame_metadata(Path(settings['metadata']), video)
        records, _ = detection.load_predictions(Path(settings['predictions']) / video / 'detections.jsonl', clips, receipt, camera_rate=True)
        with np.load(base / video / 'indices.npz') as index:
            offsets = index['offsets'].copy()
            assert np.array_equal(index['frame_indices'], [row['frame_index'] for row in records])
        first = {row['lesion_id']: row['first_frame'] for row in next(row for row in annotations['videos'] if row['video_id'] == video)['lesions']}
        results = []
        for episode in original['episodes'][:1] if smoke else original['episodes']:
            target = destination / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=resume)
            if not episode['click']['available']:
                result = dict(episode, fixed_results={name: None for name in points},
                    fixed_policy_status='ACKNOWLEDGEMENT_UNAVAILABLE', display_action='retain_baseline')
                write_json(target / 'summary.json', result)
                results.append(result)
                continue
            source = base / video / 'sources' / episode['episode_id']
            native_source = native / video / 'sources' / episode['episode_id']
            with np.load(source / 'frame_data.npz') as saved:
                data = {key: saved[key].copy() for key in saved.files}
            with np.load(source / 'cosines.npz') as saved:
                scores = {name: saved['track__' + name].copy() for name in ['raw_l2', 'standardized_l2', 'supcon_l2']}
            with np.load(native_source / 'cosines.npz') as saved:
                scores['ntssl_l2'] = saved['track__ntssl_l2'].copy()
            assert all(len(value) == offsets[-1] for value in scores.values())
            elapsed = np.repeat(data['start'] - episode['click']['time'], np.diff(offsets))
            post = np.repeat(data['post_click'], np.diff(offsets))
            summaries, measures, decisions = {}, {}, {}
            for name, point in points.items():
                keep = fixed_keep(point, elapsed, post, scores.get(point.get('representation')))
                decisions[name] = keep
                metric_scores = np.where(keep, np.nan, 1.)
                curve, checks = shared.episode_curve(metric_scores, offsets, records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                assert np.array_equal(~np.isfinite(metric_scores) | (metric_scores < curve['threshold'][0]), keep)
                curve = one_point(curve)
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                condition = name + '__fixed'
                np.savez_compressed(target / (condition + '_curve.npz'), **curve)
                summaries[condition] = dict(operating_point=point, direct_mask_verification=checks,
                    metric_state='Exact frozen keep decisions; first state retains every NaN-encoded keep and removes each finite-encoded hide.')
                measures[name] = result_row(curve, episode)
            np.savez_compressed(target / 'fixed_keep.npz', **decisions)
            result = dict(episode, conditions=summaries, fixed_results=measures, fixed_policy_status='EVALUATED',
                fixed_input_identity={str(path): digest(path) for path in [source / 'frame_data.npz', source / 'cosines.npz', native_source / 'cosines.npz']})
            write_json(target / 'summary.json', result)
            results.append(result)
            print('FIXED_POLICY', video, episode['episode_id'], len(points), 'frozen methods', flush=True)
        write_json(destination / 'summary.json', dict(video=video, episodes=results, status='COMPLETE', smoke=smoke))
    completed = [video for video in config['videos'] if (output / video / 'summary.json').exists()]
    shared.aggregate(output, dict(config, source_variants=list(points), representations=['fixed']), completed)
    conditions = {}
    for name in points:
        with np.load(output / (name + '__fixed_procedure_curve.npz')) as saved:
            conditions[name] = {key: (np.where(np.isfinite(value), value, None).tolist()
                if value.dtype.kind in 'fc' else value.tolist()) for key, value in saved.items() if key != 'threshold'}
    write_json(output / 'summary.json', dict(status='SMOKE_COMPLETE' if smoke else 'COMPLETE' if completed == config['videos'] else 'PARTIAL',
        videos=completed, planned_videos=config['videos'], methods=list(points), seconds=time.perf_counter()-started,
        episodes=sum(len(shared.read(output / video / 'summary.json')['episodes']) for video in completed)))
    write_json(output / 'fixed_results.json', conditions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = shared.read(args.config)
    run(args.config, args.output, args.videos or config['videos'], args.smoke, args.resume)
