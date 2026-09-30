import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import evaluate_acknowledgement_memory as memory
import evaluate_multisource_acknowledgement as multisource
from src.evaluation.realcolon_task import digest, write_json


read = multisource.read
GROUPS = ['all_other', 'new_after_click', 'previously_seen']
METRICS = ['baseline_qualified_retention', 'actual_visible_coverage', 'first_baseline_frame_retention',
           'first_baseline_bout_retention', 'later_baseline_prompt_retention']


def mean_defined(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count, out=np.full(count.shape, np.nan), where=count > 0), count


def episode_curve(scores, offsets, records, frames, data, source, first, fps):
    post_data = dict(data, known=data['known'] & data['post_click'])
    curve, verification = memory.evaluate_curve(scores, offsets, records, frames, post_data)
    baseline_first = np.asarray([data['start'][np.flatnonzero(data['baseline'][:, column] & data['known'] & data['post_click'])[0]]
        if np.any(data['baseline'][:, column] & data['known'] & data['post_click']) else np.nan
        for column in range(len(data['lesion_ids']))])
    global_first = np.asarray([data['start'][np.flatnonzero(data['baseline'][:, column] & data['known'])[0]]
        if np.any(data['baseline'][:, column] & data['known']) else np.nan for column in range(len(data['lesion_ids']))])
    origins = np.asarray([first[str(lesion)] / fps for lesion in data['lesion_ids']])
    curve['first_postclick_correct_prompt_time'] = curve['first_correct_prompt_time'].copy()
    curve['baseline_first_postclick_correct_prompt_time'] = baseline_first
    curve['baseline_first_correct_prompt_time'] = global_first
    curve['baseline_detector_delay_seconds'] = global_first - origins
    curve['first_prompt_delay_from_annotation_seconds'] = curve['first_correct_prompt_time'] - origins[None]
    curve['added_postclick_delay_seconds'] = curve['first_correct_prompt_time'] - baseline_first[None]
    curve['completely_hidden_baseline_detected_lesion'] = np.isfinite(baseline_first)[None] & ~np.isfinite(curve['first_correct_prompt_time'])
    column = list(curve['lesion_ids']).index(source)
    curve['acknowledged_removed_seconds'] = curve['baseline_seconds'][column] - curve['retained_baseline_seconds'][:, column]
    curve['acknowledged_suppression_fraction'] = 1. - curve['baseline_qualified_retention'][:, column]
    return curve, verification


def run_video(video, config, output, smoke):
    started = time.perf_counter()
    detection = memory.acknowledgement.detection
    metadata = Path(config['metadata'])
    prediction = Path(config['predictions']) / video / 'detections.jsonl'
    directory = Path(config['descriptors']) / video
    completion = read(directory / 'complete.json')
    assert completion['status'] == ('PASS_REAL_SMOKE' if smoke else 'COMPLETE')
    episodes_path = Path(config['episodes']) / video / 'episodes.json'
    episodes = read(episodes_path)
    if smoke:
        episodes = [episode for episode in episodes if episode['episode_id'] in config['smoke_source_directories']]
        assert episodes
    clips, receipt, frames, segments, _ = detection.full_frame_metadata(metadata, video)
    full_records, completed = detection.load_predictions(prediction, clips, receipt, camera_rate=True)
    full_intervals, _ = detection.displayed_intervals(receipt, frames, segments, full_records, completed)
    saved_indices = np.load(directory / 'indices.npz')
    assert np.array_equal(saved_indices['frame_indices'], [row['frame_index'] for row in full_records])
    assert np.array_equal(np.diff(saved_indices['offsets']), [len(row['detections']) for row in full_records])
    selected = completion['processed_output_indices'] if smoke else list(range(len(full_records)))
    records = [full_records[index] for index in selected]
    selected_frames = {row['frame_index'] for row in records}
    intervals = [row for row in full_intervals if row['frame'] in selected_frames]
    positions = np.concatenate([np.arange(saved_indices['offsets'][index], saved_indices['offsets'][index+1]) for index in selected]).astype(int)
    offsets = np.cumsum([0] + [len(row['detections']) for row in records])
    available = np.load(directory / 'available.npy')[positions]
    assert np.all(np.load(directory / 'status_code.npy')[positions] != 0)
    vectors = {name: np.asarray(np.load(directory / (name + '.npy'), mmap_mode='r')[positions]) for name in config['representations']}
    for values in vectors.values():
        assert len(values) == len(available) and np.isfinite(values[available]).all() and np.isnan(values[~available]).all()
    annotations = next(row for row in read(config['annotations'])['videos'] if row['video_id'] == video)
    first = {row['lesion_id']: row['first_frame'] for row in annotations['lesions']}
    destination = output / video
    destination.mkdir(parents=True, exist_ok=True)
    identity_paths = [directory / 'complete.json', directory / 'identity.json', episodes_path,
        prediction.parent / 'identity.json', metadata / (video + '.json'), metadata / (video + '.jsonl')]
    np.savez_compressed(destination / 'indices.npz', offsets=offsets,
        original_detection_positions=positions, original_output_indices=selected,
        frame_indices=[row['frame_index'] for row in records], base_track_ids=saved_indices['track_ids'][positions])
    results = []
    for episode in episodes:
        episode_id = episode['episode_id']
        target = destination / 'sources' / episode_id
        target.mkdir(parents=True, exist_ok=True)
        if (target / 'summary.json').exists():
            saved = read(target / 'summary.json')
            assert all(digest(path) == value for path, value in saved['input_identity'].items())
            results.append(saved)
            continue
        source = episode['source_lesion_id']
        click = episode['click']
        source_dir = (directory / 'sources' / config['smoke_source_directories'][episode_id] if smoke
                      else directory / 'sources' / episode_id)
        source_info = read(source_dir / 'source.json')
        if click['available']:
            for key in ['output_index', 'detection_index', 'input_frame', 'time']:
                assert source_info['click'][key] == click[key]
        paths = identity_paths + [source_dir / 'source.json', Path(episode['summary'])]
        control_result = read(episode['summary'])
        controls = multisource.episode_metrics(control_result, episode)
        summaries, cosines = {}, {}
        bouts, ids, groups = {}, [], {}
        timing = dict(observed_postclick_seconds=None, unknown_postclick_seconds=None)
        if click['available']:
            data, bouts = memory.frame_data(receipt, frames, intervals, records, source, click)
            ids = data['lesion_ids'].tolist()
            origins = np.asarray([first[lesion] / receipt['fps'] for lesion in ids])
            other = np.asarray(ids) != source
            visible_post = np.any(data['visible'] & data['post_click'][:, None] & data['known'][:, None], axis=0)
            groups = dict(all_other=np.flatnonzero(other & visible_post).tolist(),
                new_after_click=np.flatnonzero(other & visible_post & (origins > click['time'])).tolist(),
                previously_seen=np.flatnonzero(other & visible_post & (origins <= click['time'])).tolist())
            dt = data['end'] - data['start']
            timing = dict(observed_postclick_seconds=float(dt[data['post_click'] & data['known']].sum()),
                unknown_postclick_seconds=float(dt[data['post_click'] & ~data['known']].sum()))
            np.savez_compressed(target / 'frame_data.npz', **data)
            for variant in config['source_variants']:
                path = source_dir / (variant + '_memory.npz')
                paths.append(path)
                fixed = np.load(path)
                for name in config['representations']:
                    condition = variant + '__' + name
                    scores = np.full(len(positions), np.nan)
                    if bool(fixed['available']):
                        reference = fixed[name].reshape(-1).astype(np.float64)
                        assert np.isclose(np.linalg.norm(reference), 1, atol=1e-5)
                        scores[available] = np.clip(vectors[name][available].astype(np.float64) @ reference, -1, 1)
                    curve, verification = episode_curve(scores, offsets, records, frames, data, source, first, receipt['fps'])
                    for group, columns in groups.items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    np.savez_compressed(target / (condition + '_curve.npz'), **curve)
                    cosines[condition] = scores
                    summaries[condition] = dict(curve=condition + '_curve.npz', points=len(curve['threshold']),
                        memory_available=bool(fixed['available']), source_support_tokens=fixed['support_tokens'].tolist(),
                        direct_mask_verification=verification)
                    print('SOURCE_CURVE', video, episode_id, condition, len(curve['threshold']), flush=True)
            np.savez_compressed(target / 'cosines.npz', **cosines)
        summary = dict(video=video, episode_id=episode_id, source_lesion_id=source,
            earliest_source=episode['earliest_source'], click=click, lesion_ids=ids, groups=groups,
            baseline_prompt_bouts=bouts, timing=timing, conditions=summaries,
            controls=controls, control_summary=episode['summary'],
            status='REAL_INTERFACE_SMOKE' if smoke else 'COMPLETE',
            undefined_click_scope='Post-click quantities undefined when acknowledgement is unavailable; unchanged detector display is retained in the saved controls.',
            input_identity={str(path): digest(path) for path in paths})
        write_json(target / 'summary.json', summary)
        results.append(summary)
    write_json(destination / 'summary.json', dict(video=video, episodes=results, smoke=smoke,
        processed_camera_outputs=len(records), original_camera_outputs=len(full_records),
        seconds=time.perf_counter()-started, status='REAL_INTERFACE_SMOKE' if smoke else 'COMPLETE'))
    return results


def aggregate(output, config, completed_videos):
    procedures = [read(output / video / 'summary.json') for video in completed_videos]
    for variant in config['source_variants']:
        for name in config['representations']:
            condition = variant + '__' + name
            loaded = {}
            for procedure in procedures:
                for episode in procedure['episodes']:
                    if episode['click']['available']:
                        path = output / procedure['video'] / 'sources' / episode['episode_id'] / (condition + '_curve.npz')
                        loaded[episode['episode_id']] = np.load(path)
            grid = np.unique(np.concatenate([curve['threshold'] for curve in loaded.values()])) if loaded else np.asarray([-2., 2.])
            values = {group + '__' + metric: [] for group in GROUPS for metric in METRICS}
            values['source_suppression'] = []
            defined = {key: [] for key in values}
            for procedure in procedures:
                local = {key: [] for key in values}
                for episode in procedure['episodes']:
                    if not episode['click']['available']:
                        for key in local:
                            local[key].append(np.full(len(grid), np.nan))
                        continue
                    curve = loaded[episode['episode_id']]
                    indices = np.searchsorted(curve['threshold'], grid, side='right') - 1
                    local['source_suppression'].append(curve['acknowledged_suppression_fraction'][indices])
                    for group in GROUPS:
                        columns = curve[group + '_lesion_columns']
                        for metric in METRICS:
                            array = curve[metric][indices][:, columns]
                            mean, _ = mean_defined(array, axis=1)
                            local[group + '__' + metric].append(mean)
                for key in values:
                    mean, count = mean_defined(np.stack(local[key]), axis=0)
                    values[key].append(mean)
                    defined[key].append(count)
            saved = dict(threshold=grid, videos=np.asarray(completed_videos))
            for key, rows in values.items():
                per_procedure = np.stack(rows)
                mean, count = mean_defined(per_procedure, axis=0)
                saved[key + '__procedure_values'] = per_procedure
                saved[key + '__procedure_mean'] = mean
                saved[key + '__defined_procedures'] = count
                saved[key + '__defined_episodes_per_procedure'] = np.stack(defined[key])
            np.savez_compressed(output / (condition + '_procedure_curve.npz'), **saved)
    controls = [dict(video=procedure['video'],
        all_sources=multisource.aggregate_episodes([episode['controls'] for episode in procedure['episodes']]),
        earliest_source=multisource.aggregate_episodes([episode['controls'] for episode in procedure['episodes'] if episode['earliest_source']]))
        for procedure in procedures]
    write_json(output / 'control_points.json', dict(all_sources=multisource.aggregate_procedures(controls, 'all_sources'),
        earliest_source=multisource.aggregate_procedures(controls, 'earliest_source'), procedures=controls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    assert set(args.videos) <= set(config['videos'])
    args.output.mkdir(parents=True, exist_ok=args.resume)
    paths = [args.config, Path(__file__), Path(memory.__file__), Path(multisource.__file__),
             Path(memory.acknowledgement.__file__), Path(memory.acknowledgement.detection.__file__)]
    identity = dict(files={str(path): digest(path) for path in paths}, smoke=args.smoke)
    if (args.output / 'identity.json').exists():
        assert args.resume and read(args.output / 'identity.json') == identity
    else:
        write_json(args.output / 'identity.json', identity)
        write_json(args.output / 'config.json', config)
        for path in paths[1:]:
            shutil.copyfile(path, args.output / path.name)
    started = time.perf_counter()
    for video in args.videos:
        run_video(video, config, args.output, args.smoke)
    completed = [video for video in config['videos'] if (args.output / video / 'summary.json').exists()]
    if not args.smoke:
        aggregate(args.output, config, completed)
    summaries = [read(args.output / video / 'summary.json') for video in completed]
    write_json(args.output / 'summary.json', dict(status='REAL_INTERFACE_SMOKE' if args.smoke else
        'COMPLETE' if len(completed) == len(config['videos']) else 'PARTIAL',
        videos=completed, planned_videos=config['videos'], episodes=sum(len(row['episodes']) for row in summaries),
        seconds_this_invocation=time.perf_counter()-started,
        aggregation='At each shared cosine threshold: defined lesion means within episode, episode means within procedure, then equal procedure means. Undefined quantities retain explicit counts.'))


if __name__ == '__main__':
    main()
