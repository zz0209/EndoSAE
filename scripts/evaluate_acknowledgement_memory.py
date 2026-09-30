import argparse
import heapq
import json
import shutil
from pathlib import Path

import numpy as np

import evaluate_acknowledged_prompting as acknowledgement
from src.evaluation.realcolon_task import digest, write_json


REPRESENTATIONS = ['raw_l2', 'standardized_l2', 'supcon_l2']


def frame_data(receipt, frames, intervals, records, source, click):
    ids = sorted({box['lesion_id'] for frame in frames.values() for box in frame['original_boxes_xyxy']})
    lookup = {row['frame']: row for row in intervals}
    assert len(lookup) == len(intervals)
    camera = np.asarray([row['frame_index'] for row in records])
    known = np.asarray([int(frame) in frames for frame in camera])
    baseline = np.zeros((len(camera), len(ids)), dtype=bool)
    visible = np.zeros_like(baseline)
    for row, frame in enumerate(camera):
        if known[row]:
            baseline[row] = [lesion in lookup[int(frame)]['detected_lesion_ids'] for lesion in ids]
            visible[row] = [lesion in lookup[int(frame)]['visible_lesion_ids'] for lesion in ids]
    start = camera / receipt['fps']
    post = start >= click['time'] - 1e-9
    data = dict(frame=camera, start=start, end=(camera + 1) / receipt['fps'], known=known,
                baseline=baseline, visible=visible, post_click=post, lesion_ids=np.asarray(ids))
    strata = np.full(len(camera), 'before_click', dtype='<U24')
    absent, crossed_unknown = False, False
    source_column = ids.index(source)
    for output in np.flatnonzero(post):
        if not known[output]:
            crossed_unknown = True
        elif not visible[output, source_column]:
            absent = True
        strata[output] = ('later_reappearance' if absent else
                         'unknown_continuity' if crossed_unknown else 'current_visibility')
    data['acknowledged_stratum'] = strata
    first_frame, first_bout, remaining = [np.zeros_like(baseline) for _ in range(3)]
    bouts = {}
    for column, lesion in enumerate(ids):
        selected = baseline[:, column] & known & post
        blocks = acknowledgement.bouts(data, selected, post)
        bouts[lesion] = blocks
        positions = np.flatnonzero(selected)
        if len(positions):
            first_frame[positions[0], column] = True
            end = positions[0]
            while end < len(selected) and selected[end]:
                first_bout[end, column] = True
                end += 1
        remaining[:, column] = selected & ~first_bout[:, column]
    data.update(first_frame=first_frame, first_bout=first_bout, remaining=remaining)
    return data, bouts


def detected_vector(boxes, frame, ids):
    matched = acknowledgement.detection.overlap(boxes, frame['original_boxes_xyxy'])
    return np.asarray([lesion in matched['detected_lesion_ids'] for lesion in ids], dtype=bool)


def evaluate_curve(scores, offsets, records, frames, data):
    ids = list(data['lesion_ids'])
    eligible = data['known'] & data['post_click']
    dt = data['end'] - data['start']
    current = data['baseline'].copy()
    events = []
    thresholds = set()
    for output in np.flatnonzero(data['post_click']):
        start, end = offsets[output:output + 2]
        local = scores[start:end]
        values = np.unique(local[np.isfinite(local)])
        thresholds.update(float(value) for value in values)
        if not data['known'][output]:
            continue
        frame = frames[int(data['frame'][output])]
        boxes = records[output]['detections']
        keep = ~np.isfinite(local)
        current[output] = detected_vector([box for box, selected in zip(boxes, keep) if selected], frame, ids)
        for value in values:
            keep |= local == value
            state = detected_vector([box for box, selected in zip(boxes, keep) if selected], frame, ids)
            events.append((float(value), int(output), state))
    values = np.asarray(sorted(thresholds), dtype=np.float64)
    grid = np.concatenate([[-2.], np.nextafter(values, np.inf), [2.]])
    events.sort(key=lambda event: (event[0], event[1]))
    source_strata = [data['baseline'] & (data['acknowledged_stratum'] == name)[:, None]
                     for name in ['current_visibility', 'later_reappearance', 'unknown_continuity']]
    weights = np.stack([data['baseline'], np.ones_like(current), data['first_frame'],
                        data['first_bout'], data['remaining']] + source_strata, axis=0) * (dt * eligible)[None, :, None]
    totals = (weights * current[None]).sum(axis=1)
    output_totals = np.zeros((len(grid), len(weights), len(ids)), dtype=np.float64)
    first_times = np.full((len(grid), len(ids)), np.nan)
    heaps = [np.flatnonzero(current[:, column] & data['known']).tolist() for column in range(len(ids))]
    cursor = 0
    for point, threshold in enumerate(grid):
        while cursor < len(events) and events[cursor][0] < threshold:
            _, output, state = events[cursor]
            old = current[output].copy()
            totals += weights[:, output] * (state.astype(int) - old.astype(int))[None]
            current[output] = state
            for column in np.flatnonzero(state & ~old):
                heapq.heappush(heaps[column], output)
            cursor += 1
        output_totals[point] = totals
        for column, heap in enumerate(heaps):
            while heap and not current[heap[0], column]:
                heapq.heappop(heap)
            if heap:
                first_times[point, column] = data['start'][heap[0]]
    base_seconds = (data['baseline'] * (dt * eligible)[:, None]).sum(axis=0)
    visible_seconds = (data['visible'] * (dt * eligible)[:, None]).sum(axis=0)
    first_frame_seconds = (data['first_frame'] * dt[:, None]).sum(axis=0)
    first_bout_seconds = (data['first_bout'] * dt[:, None]).sum(axis=0)
    remaining_seconds = (data['remaining'] * dt[:, None]).sum(axis=0)
    curve = dict(threshold=grid, retained_baseline_seconds=output_totals[:, 0],
        actual_correct_seconds=output_totals[:, 1], first_baseline_frame_retained_seconds=output_totals[:, 2],
        first_baseline_bout_retained_seconds=output_totals[:, 3], later_baseline_prompt_retained_seconds=output_totals[:, 4],
        first_correct_prompt_time=first_times, baseline_seconds=base_seconds, visible_seconds=visible_seconds,
        first_baseline_frame_seconds=first_frame_seconds, first_baseline_bout_seconds=first_bout_seconds,
        later_baseline_prompt_seconds=remaining_seconds, lesion_ids=data['lesion_ids'])
    for numerator, denominator, name in [
            ('retained_baseline_seconds', 'baseline_seconds', 'baseline_qualified_retention'),
            ('actual_correct_seconds', 'visible_seconds', 'actual_visible_coverage'),
            ('first_baseline_frame_retained_seconds', 'first_baseline_frame_seconds', 'first_baseline_frame_retention'),
            ('first_baseline_bout_retained_seconds', 'first_baseline_bout_seconds', 'first_baseline_bout_retention'),
            ('later_baseline_prompt_retained_seconds', 'later_baseline_prompt_seconds', 'later_baseline_prompt_retention')]:
        curve[name] = np.divide(curve[numerator], curve[denominator][None],
            out=np.full_like(curve[numerator], np.nan), where=curve[denominator][None] > 0)
    for index, name in enumerate(['current_visibility', 'later_reappearance', 'unknown_continuity']):
        curve[name + '_baseline_seconds'] = weights[index + 5].sum(axis=0)
        curve[name + '_retained_seconds'] = output_totals[:, index + 5]
    assert np.all(curve['retained_baseline_seconds'] <= base_seconds[None] + 1e-7)
    assert np.allclose(curve['retained_baseline_seconds'][-1], base_seconds, atol=1e-7)
    verification = []
    for point in np.unique(np.linspace(0, len(grid) - 1, min(7, len(grid))).astype(int)):
        direct = data['baseline'].copy()
        for output in np.flatnonzero(eligible):
            start, end = offsets[output:output + 2]
            keep = ~np.isfinite(scores[start:end]) | (scores[start:end] < grid[point])
            boxes = [box for box, selected in zip(records[output]['detections'], keep) if selected]
            direct[output] = detected_vector(boxes, frames[int(data['frame'][output])], ids)
        actual = (weights * direct[None]).sum(axis=1)
        error = float(np.max(np.abs(actual - output_totals[point])))
        assert error < 1e-7
        direct_first = np.asarray([data['start'][np.flatnonzero(direct[:, column] & data['known'])[0]]
            if np.any(direct[:, column] & data['known']) else np.nan for column in range(len(ids))])
        assert np.allclose(direct_first, first_times[point], atol=1e-9, equal_nan=True)
        verification.append(dict(point=int(point), threshold=float(grid[point]), maximum_seconds_error=error,
            first_prompt_times_equal=True, evaluated_frames=int(eligible.sum()),
            multi_detection_frames=sum(len(records[index]['detections']) > 1 for index in np.flatnonzero(eligible))))
    return curve, verification


def evaluate_video(video, config, output):
    detection = acknowledgement.detection
    metadata = Path(config['metadata'])
    prediction = Path(config['predictions']) / video / 'detections.jsonl'
    directory = Path(config['descriptors']) / video
    completion = detection.temporal.read(directory / 'complete.json')
    assert completion['status'] == 'COMPLETE'
    clips, receipt, frames, segments, _ = detection.full_frame_metadata(metadata, video)
    records, completed = detection.load_predictions(prediction, clips, receipt, camera_rate=True)
    intervals, matches = detection.displayed_intervals(receipt, frames, segments, records, completed)
    annotations = next(row for row in detection.temporal.read(Path(config['annotations']))['videos'] if row['video_id'] == video)
    first = {row['lesion_id']: row['first_frame'] for row in annotations['lesions']}
    source = min(first, key=lambda lesion: (first[lesion], lesion))
    click = acknowledgement.choose_click(intervals, matches, records, source, config['reaction_seconds'])
    assert click['available']
    indices = np.load(directory / 'indices.npz')
    offsets = indices['offsets']
    assert np.array_equal(indices['frame_indices'], [record['frame_index'] for record in records])
    assert np.array_equal(np.diff(offsets), [len(record['detections']) for record in records])
    available = np.load(directory / 'available.npy')
    assert np.all(np.load(directory / 'status_code.npy') != 0)
    memory = np.load(directory / 'source_memory.npz')
    memory_info = detection.temporal.read(directory / 'source_memory.json')
    for key in ['output_index', 'detection_index', 'input_frame']:
        assert memory_info['click'][key] == click[key]
    data, bouts = frame_data(receipt, frames, intervals, records, source, click)
    destination = output / video
    destination.mkdir(exist_ok=False)
    score_output, summaries = {}, {}
    for name in REPRESENTATIONS:
        vectors = np.load(directory / (name + '.npy'), mmap_mode='r')
        assert len(vectors) == len(available) == offsets[-1]
        assert np.all(np.isfinite(vectors[available])) and np.all(np.isnan(vectors[~available]))
        scores = np.full(len(available), np.nan)
        memory_available = bool(np.asarray(memory['available']).item())
        if memory_available:
            reference = np.asarray(memory[name]).reshape(-1)
            assert np.isclose(np.linalg.norm(reference), 1., atol=1e-5)
            scores[available] = np.asarray(vectors[available], dtype=np.float64) @ reference.astype(np.float64)
            scores[available] = np.clip(scores[available], -1., 1.)
        score_output[name] = scores
        curve, verification = evaluate_curve(scores, offsets, records, frames, data)
        origins = np.asarray([first[str(lesion)] / receipt['fps'] for lesion in data['lesion_ids']])
        curve['first_prompt_delay_seconds'] = curve['first_correct_prompt_time'] - origins[None]
        baseline_first = np.asarray([data['start'][np.flatnonzero(data['baseline'][:, column])[0]]
            if np.any(data['baseline'][:, column]) else np.nan for column in range(len(data['lesion_ids']))])
        curve['baseline_first_correct_prompt_time'] = baseline_first
        curve['baseline_detector_delay_seconds'] = baseline_first - origins
        curve['added_delay_from_baseline_seconds'] = curve['first_correct_prompt_time'] - baseline_first[None]
        curve['completely_hidden_baseline_detected_lesion'] = np.isfinite(baseline_first)[None] & ~np.isfinite(curve['first_correct_prompt_time'])
        source_column = list(data['lesion_ids']).index(source)
        curve['acknowledged_removed_seconds'] = curve['baseline_seconds'][source_column] - curve['retained_baseline_seconds'][:, source_column]
        curve['acknowledged_suppression_fraction'] = 1. - curve['baseline_qualified_retention'][:, source_column]
        np.savez_compressed(destination / (name + '_curve.npz'), **curve)
        summaries[name] = dict(threshold_count=len(curve['threshold']), source_memory_available=memory_available,
            available_detections=int(np.isfinite(scores).sum()), total_detections=len(scores),
            direct_mask_verification=verification, curve=name + '_curve.npz')
    np.savez_compressed(destination / 'cosines.npz', offsets=offsets, frame_indices=indices['frame_indices'], **score_output)
    np.savez_compressed(destination / 'frame_data.npz', **data)
    matching_by_frame = {row['frame_index']: row for row in matches}
    with (destination / 'detection_events.jsonl').open('w', encoding='utf-8') as stream:
        for output_index, record in enumerate(records):
            frame = record['frame_index']
            matching = matching_by_frame.get(frame)
            assignments = {row['prediction_index']: row for row in matching['matches']} if matching else {}
            for index, box in enumerate(record['detections']):
                position = offsets[output_index] + index
                event = dict(frame_index=frame, output_index=output_index, detection_index=index,
                    time=record['available_time_seconds'], detection=box, baseline_displayed=True,
                    post_click=bool(data['post_click'][output_index]), annotation_known=frame in frames,
                    gt_match=assignments.get(index), track_id=int(indices['track_ids'][position]),
                    descriptor_available=bool(available[position]),
                    cosine={name: float(scores[position]) if np.isfinite(scores[position]) else None
                            for name, scores in score_output.items()})
                stream.write(json.dumps(event, allow_nan=False) + '\n')
    result = dict(video=video, split=receipt['split'], camera_frames=len(records), source_lesion=source, click=click,
        lesion_ids=data['lesion_ids'].tolist(), baseline_prompt_bouts=bouts, representations=summaries,
        missing_candidate_convention='Frames with no detections have equal consecutive offsets and no detection_events row; no cosine is imputed.',
        descriptor_completion=completion, input_identity={str(path): digest(path) for path in
            [directory / 'complete.json', directory / 'source_memory.npz', directory / 'source_memory.json',
             prediction.parent / 'identity.json', metadata / (video + '.json'), metadata / (video + '.jsonl')]})
    write_json(destination / 'summary.json', result)
    print('MEMORY_CURVES_COMPLETED', video, {name: row['threshold_count'] for name, row in summaries.items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = acknowledgement.detection.temporal.read(args.config)
    assert set(args.videos) <= set(config['videos'])
    args.output.mkdir(parents=True, exist_ok=args.resume)
    files = [Path(__file__), Path(acknowledgement.__file__), Path(acknowledgement.detection.__file__), args.config]
    identity = {str(path): digest(path) for path in files}
    if (args.output / 'identity.json').exists():
        assert args.resume and acknowledgement.detection.temporal.read(args.output / 'identity.json') == identity
    else:
        write_json(args.output / 'identity.json', identity)
        write_json(args.output / 'config.json', config)
        for path in files[:-1]:
            shutil.copyfile(path, args.output / path.name)
    for video in args.videos:
        saved = args.output / video / 'summary.json'
        if saved.exists():
            assert args.resume
            result = acknowledgement.detection.temporal.read(saved)
            assert all(digest(path) == value for path, value in result['input_identity'].items())
        else:
            evaluate_video(video, config, args.output)
    summaries = [acknowledgement.detection.temporal.read(args.output / video / 'summary.json')
                 for video in config['videos'] if (args.output / video / 'summary.json').exists()]
    if len(summaries) == len(config['videos']):
        for name in REPRESENTATIONS:
            curves = [np.load(args.output / row['video'] / (name + '_curve.npz')) for row in summaries]
            grid = np.unique(np.concatenate([curve['threshold'] for curve in curves]))
            aligned = {}
            for metric in ['baseline_qualified_retention', 'first_baseline_frame_retention',
                           'first_baseline_bout_retention', 'later_baseline_prompt_retention']:
                values = []
                for summary, curve in zip(summaries, curves):
                    positions = np.searchsorted(curve['threshold'], grid, side='right') - 1
                    protected = (curve['lesion_ids'] != summary['source_lesion']) & (curve['baseline_seconds'] > 0)
                    values.append(curve[metric][positions][:, protected].mean(axis=1)
                                  if np.any(protected) else np.full(len(grid), np.nan))
                per_video = np.stack(values)
                count = np.isfinite(per_video).sum(axis=0)
                aligned[metric + '_per_video'] = per_video
                aligned[metric + '_video_mean'] = np.divide(np.nansum(per_video, axis=0), count,
                    out=np.full(len(grid), np.nan), where=count > 0)
            suppression = np.stack([curve['acknowledged_suppression_fraction'][
                np.searchsorted(curve['threshold'], grid, side='right') - 1] for curve in curves])
            aligned['acknowledged_suppression_per_video'] = suppression
            aligned['acknowledged_suppression_video_mean'] = suppression.mean(axis=0)
            np.savez_compressed(args.output / (name + '_video_mean_curve.npz'), threshold=grid,
                videos=np.asarray([row['video'] for row in summaries]), **aligned)
    write_json(args.output / 'summary.json', dict(status='COMPLETE' if len(summaries) == len(config['videos']) else 'PARTIAL',
        independent_procedures=len(summaries), videos=summaries))


if __name__ == '__main__':
    main()
