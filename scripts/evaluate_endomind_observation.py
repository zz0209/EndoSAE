import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import evaluate_continuous_rejection as temporal
from src.evaluation.realcolon_task import digest, write_json


def full_frame_metadata(directory, video, scope=None):
    clips = temporal.rows(directory / (video + '.jsonl'))
    receipt = temporal.read(directory / (video + '.json'))
    all_frames = [frame for clip in clips for frame in clip['frames']] + receipt['terminal_frame_records']
    assert [frame['frame_index'] for frame in all_frames] == list(range(receipt['start_frame'], receipt['end_frame_exclusive']))
    if scope is not None:
        start, end = scope
        assert receipt['start_frame'] <= start < end <= receipt['end_frame_exclusive']
        all_frames = [frame for frame in all_frames if start <= frame['frame_index'] < end]
        receipt = dict(receipt, start_frame=start, end_frame_exclusive=end, requested_frames=end - start)
    frames, segments, previous, segment = {}, {}, None, -1
    for frame in all_frames:
        known = frame['xml_member'] is not None and frame['annotation_anomalies'] == []
        if not known:
            continue
        boxes = frame['original_boxes_xyxy']
        assert boxes is not None and frame['width'] > 0 and frame['height'] > 0
        for box in boxes:
            left, top, right, bottom = box['box']
            assert 0 <= left < right <= frame['width'] and 0 <= top < bottom <= frame['height']
        index = frame['frame_index']
        if previous is None or index != previous + 1:
            segment += 1
        frames[index] = dict(frame, class_full_frame=int(bool(boxes)))
        segments[index], previous = segment, index
    ends = {value: (max(index for index in segments if segments[index] == value) + 1) / receipt['fps']
            for value in set(segments.values())}
    return clips, receipt, frames, segments, ends


def overlap(detections, boxes):
    prediction = np.asarray([row['xyxy'] for row in detections], dtype=float).reshape(-1, 4)
    target = np.asarray([row['box'] for row in boxes], dtype=float).reshape(-1, 4)
    size = np.maximum(0., np.minimum(prediction[:, None, 2:], target[None, :, 2:]) -
                      np.maximum(prediction[:, None, :2], target[None, :, :2]))
    intersection = size.prod(-1)
    pred_area = (prediction[:, 2:] - prediction[:, :2]).prod(-1)
    target_area = (target[:, 2:] - target[:, :2]).prod(-1)
    iou = intersection / (pred_area[:, None] + target_area[None, :] - intersection)
    matched, used = [], set()
    for prediction_index in sorted(range(len(detections)), key=lambda index: (-detections[index]['score'], index)):
        candidates = [index for index in range(len(boxes)) if index not in used and iou[prediction_index, index] >= .5]
        if candidates:
            target_index = max(candidates, key=lambda index: (iou[prediction_index, index], -index))
            used.add(target_index)
            matched.append(dict(prediction_index=prediction_index, target_index=target_index,
                                lesion_id=boxes[target_index]['lesion_id'], iou=float(iou[prediction_index, target_index])))
    return dict(iou=iou.tolist(), intersection_area=intersection.tolist(), matches=matched,
                detected_lesion_ids=sorted({row['lesion_id'] for row in matched}),
                overlap_lesion_ids=sorted({box['lesion_id'] for index, box in enumerate(boxes)
                                          if np.any(intersection[:, index] > 0)}))


def load_predictions(path, clips, receipt, camera_rate=False, smoke=False):
    completion = temporal.read(path.parent / 'complete.json')
    assert completion['status'] == ('PASS' if smoke else 'complete') and completion['video'] == receipt['video_id']
    assert completion['confidence'] == .2
    if camera_rate:
        assert completion['input_schedule'] == 'every_camera_frame' and completion['is_smoke'] == smoke
        assert completion['start_frame'] == receipt['start_frame'] and completion['end_frame_exclusive'] == receipt['end_frame_exclusive']
        expected = list(range(receipt['start_frame'], receipt['end_frame_exclusive']))
    else:
        assert not smoke
        expected = [clip['frames'][-1]['frame_index'] for clip in clips]
    assert completion['inputs'] == len(expected)
    records = temporal.rows(path)
    assert [row['frame_index'] for row in records] == expected
    metadata_frames = {frame['frame_index']: frame for clip in clips for frame in clip['frames']}
    metadata_frames.update({frame['frame_index']: frame for frame in receipt['terminal_frame_records']})
    for record in records:
        assert record['video_id'] == receipt['video_id']
        acquisition = record['frame_index'] / receipt['fps']
        assert abs(record['acquired_time_seconds'] - acquisition) < 1e-8
        assert abs(record['available_time_seconds'] - acquisition) < 1e-8
        assert np.isfinite(record['processing_seconds']) and record['processing_seconds'] >= 0
        frame = metadata_frames[record['frame_index']]
        if frame['width'] is not None:
            assert record['width'] == frame['width'] and record['height'] == frame['height']
        for detection in record['detections']:
            left, top, right, bottom = detection['xyxy']
            assert 0 <= left < right <= record['width'] and 0 <= top < bottom <= record['height']
            assert detection['class_id'] == 0 and .2 <= detection['score'] <= 1
    completed = np.array([row['available_time_seconds'] for row in records], dtype=float)
    assert np.all(np.diff(completed) > 0)
    return records, completed


def displayed_intervals(receipt, frames, segments, records, completed):
    intervals, matches = [], []
    fps = receipt['fps']
    for index, frame in frames.items():
        begin, end = index / fps, (index + 1) / fps
        boundaries = [begin] + completed[(completed > begin + 1e-10) & (completed < end - 1e-10)].tolist() + [end]
        for start, finish in zip(boundaries[:-1], boundaries[1:]):
            output = int(np.searchsorted(completed, start + 1e-10, side='right') - 1)
            detections = records[output]['detections'] if output >= 0 else []
            if output >= 0:
                assert records[output]['width'] == frame['width'] and records[output]['height'] == frame['height']
            matching = overlap(detections, frame['original_boxes_xyxy'])
            visible = sorted({box['lesion_id'] for box in frame['original_boxes_xyxy']})
            intervals.append(dict(start=start, end=finish, frame=index, output=output, segment=segments[index],
                negative=not visible, alarm=bool(detections), target_detected=bool(matching['detected_lesion_ids']),
                overlap_target_detected=bool(matching['overlap_lesion_ids']), visible_lesion_ids=visible,
                detected_lesion_ids=matching['detected_lesion_ids'], overlap_lesion_ids=matching['overlap_lesion_ids'],
                unmatched_detection_count=len(detections) - len(matching['matches'])))
            matches.append(dict(frame_index=index, output=output, start=start, end=finish,
                                original_boxes_xyxy=frame['original_boxes_xyxy'], **matching))
    assert abs(sum(row['end'] - row['start'] for row in intervals) - len(frames) / fps) < 1e-7
    return intervals, matches


def persistent_events(intervals, records, completed, persistence):
    groups, active = [], []
    for interval in intervals:
        contiguous = bool(active) and interval['segment'] == active[-1]['segment'] and abs(interval['start'] - active[-1]['end']) < 1e-8
        if not (interval['negative'] and interval['alarm']) or (active and not contiguous):
            if active:
                groups.append(active)
                active = []
        if interval['negative'] and interval['alarm']:
            active.append(interval)
    if active:
        groups.append(active)
    events = []
    for number, group in enumerate(groups):
        start, end = group[0]['start'], group[-1]['end']
        candidates = np.flatnonzero((completed >= start + persistence - 1e-9) & (completed < end - 1e-9))
        event = dict(event_index=number, start=start, end=end, duration=end - start, segment=group[0]['segment'],
                     sustained=end - start >= persistence - 1e-9, opportunity=bool(len(candidates)))
        if len(candidates):
            output = int(candidates[0])
            source = records[output]
            selected = max(range(len(source['detections'])), key=lambda index: (source['detections'][index]['score'], -index))
            event.update(feedback_time=float(completed[output]), source_output=output,
                source_input_frames=[source['frame_index']], source_annotation_frame=source['frame_index'],
                source_input_available_time=source['acquired_time_seconds'], source_detection_index=selected,
                source_score=source['detections'][selected]['score'], source_box=source['detections'][selected]['xyxy'],
                source_detections=source['detections'],
                feedback_scope='Highest-confidence displayed bounding box; no negative label is supplied outside that box.')
        events.append(event)
    return events


def lesion_results(intervals, receipt, first_annotation, overlap_rule=False):
    rows = intervals
    if overlap_rule:
        rows = [dict(row, detected_lesion_ids=row['overlap_lesion_ids']) for row in intervals]
    result = temporal.lesion_followup(rows, receipt['start_frame'] / receipt['fps'],
        receipt['end_frame_exclusive'] / receipt['fps'], 0., first_annotation, receipt['fps'])
    for lesion in result:
        alert = lesion['first_correct_alert_time']
        lesion['visible_time_to_first_detection_seconds'] = None if alert is None else sum(
            max(0., min(row['end'], alert) - row['start']) for row in rows
            if lesion['lesion_id'] in row['visible_lesion_ids'] and row['start'] < alert)
        lesion['designated_lesion'] = lesion['lesion_id'] == receipt['designated_lesion_id']
    return result


def evaluate_video(video, metadata, prediction_path, annotation, output, camera_rate=False, smoke=False):
    scope = None
    if smoke:
        assert camera_rate
        completion = temporal.read(prediction_path.parent / 'complete.json')
        assert completion['is_smoke']
        scope = (completion['start_frame'], completion['end_frame_exclusive'])
    clips, receipt, frames, segments, ends = full_frame_metadata(metadata, video, scope)
    records, completed = load_predictions(prediction_path, clips, receipt, camera_rate, smoke)
    intervals, matches = displayed_intervals(receipt, frames, segments, records, completed)
    config = dict(threshold=.2, persistence_seconds=1., followup_seconds=60., mute_seconds=[1., 2., 5.],
                  temporal_bins=[[0., 2.], [2., 10.], [10., 60.]])
    first_annotation = {row['lesion_id']: row['first_frame'] for row in annotation['lesions']}
    events = persistent_events(intervals, records, completed, config['persistence_seconds'])
    for event in events:
        if event['opportunity']:
            index = event['source_annotation_frame']
            event['within_same_identity_annotation_gaps'] = [dict(lesion_id=gap['lesion_id'],
                seconds_since_previous_annotation=(index - gap['pre_end_frame']) / receipt['fps'],
                seconds_until_next_annotation=(gap['post_start_frame'] - index) / receipt['fps'])
                for gap in annotation['gaps'] if gap['gap_start_frame'] <= index <= gap['gap_end_frame']]
    opportunities = [temporal.replay(event, intervals, ends[event['segment']],
        receipt['end_frame_exclusive'] / receipt['fps'], first_annotation, receipt['fps'], config)
        for event in events if event['opportunity']]
    negative = sum(row['end'] - row['start'] for row in intervals if row['negative'])
    positive = len(frames) / receipt['fps'] - negative
    false_seconds = sum(row['duration'] for row in events)
    target = sum(row['end'] - row['start'] for row in intervals if row['target_detected'])
    overlap_target = sum(row['end'] - row['start'] for row in intervals if row['overlap_target_detected'])
    result = dict(video=video, split=receipt['split'], fps=receipt['fps'], inference_inputs=len(records),
        is_smoke=smoke, input_schedule='every_camera_frame' if camera_rate else 'eight_frame_endpoint',
        start_frame=receipt['start_frame'], end_frame_exclusive=receipt['end_frame_exclusive'],
        full_frame_known_frames=len(frames), unknown_annotation_frames=receipt['requested_frames'] - len(frames),
        crop_out_frames_restored=sum(frame['annotation_status'] == 'target_outside_crop' for frame in frames.values()),
        observed_display_seconds=len(frames) / receipt['fps'], negative_display_seconds=negative,
        positive_display_seconds=positive, negative_alarm_seconds=false_seconds,
        negative_display_alarm_fraction=false_seconds / negative if negative else None,
        target_detected_display_seconds=target,
        positive_display_target_detection_fraction=target / positive if positive else None,
        overlap_target_detected_display_seconds=overlap_target,
        positive_display_overlap_detection_fraction=overlap_target / positive if positive else None,
        positive_frame_unmatched_box_seconds=sum(row['end'] - row['start'] for row in intervals
                                                if not row['negative'] and row['unmatched_detection_count'] > 0),
        threshold=.2, primary_matching='Confidence-ordered one-to-one IoU >= 0.5, original image coordinates.',
        overlap_descriptor='Any strictly positive intersection area; independent of primary matching.',
        display_timing=('Every camera image' if camera_rate else 'Every eight-camera-frame last image') +
                       ', zero extra latency; hold the most recent completed boxes.',
        processing_seconds_mean=float(np.mean([row['processing_seconds'] for row in records])),
        negative_alarm_intervals=len(events), opportunities=len(opportunities),
        sources_within_same_identity_annotation_gaps=sum(bool(row['within_same_identity_annotation_gaps']) for row in opportunities),
        designated_lesion_id=receipt['designated_lesion_id'],
        designated_lesion_time=receipt['designated_lesion_first_frame'] / receipt['fps'],
        before_designated_lesion_opportunities=sum(row['source_annotation_frame'] < receipt['designated_lesion_first_frame'] for row in opportunities),
        window_lesions=lesion_results(intervals, receipt, first_annotation),
        overlap_window_lesions=lesion_results(intervals, receipt, first_annotation, True),
        methods={name: temporal.summarize_opportunities(opportunities, name)
                 for name in ['baseline', 'mute_1s', 'mute_2s', 'mute_5s']},
        input_identity={str(path): digest(path) for path in [prediction_path, prediction_path.parent / 'complete.json',
            prediction_path.parent / 'identity.json', metadata / (video + '.jsonl'), metadata / (video + '.json')]})
    destination = output / video
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in [('events', events), ('opportunities', opportunities), ('box_matches', matches)]:
        write_json(destination / (name + '.json'), value)
    np.savez_compressed(destination / 'display_timeline.npz', **{key: np.asarray([row[key] for row in intervals])
        for key in ['start', 'end', 'frame', 'output', 'segment', 'negative', 'alarm', 'target_detected', 'overlap_target_detected']},
        completion_times=completed)
    write_json(destination / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--annotations', type=Path, default=Path('results/runs/20260901_realcolon_visibility_a0_v3/summary.json'))
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--camera-rate', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.metadata_only:
        results = []
        for video in args.videos:
            clips, receipt, frames, segments, ends = full_frame_metadata(args.metadata, video)
            results.append(dict(video=video, clips=len(clips), known_full_frame=len(frames),
                unknown_full_frame=receipt['requested_frames'] - len(frames),
                restored_crop_out=sum(frame['annotation_status'] == 'target_outside_crop' for frame in frames.values())))
        write_json(args.output / 'summary.json', dict(status='METADATA_COMPLETED', videos=results))
        print(json.dumps(results), flush=True)
        return
    assert args.detections is not None
    assert not args.smoke or (args.camera_rate and len(args.videos) == 1)
    inference_config = temporal.read(args.detections.parent / 'config.json')
    identity = {str(path): digest(path) for path in [Path(__file__), Path(temporal.__file__), args.annotations,
                args.detections.parent / 'config.json']}
    identity_path = args.output / 'identity.json'
    if identity_path.exists():
        assert args.resume and temporal.read(identity_path) == identity
    else:
        write_json(identity_path, identity)
        write_json(args.output / 'config.json', dict(inference_config=inference_config,
            metadata_directory=str(args.metadata), detections_directory=str(args.detections),
            annotation_summary=str(args.annotations), camera_rate=args.camera_rate, is_smoke=args.smoke,
            confidence=.2, matching_iou=.5, positive_intersection_descriptor=True,
            persistence_seconds=1., followup_seconds=60., mute_seconds=[1., 2., 5.],
            temporal_bins=[[0., 2.], [2., 10.], [10., 60.]], display_extra_latency_seconds=0.))
        shutil.copyfile(__file__, args.output / 'evaluator_source.py')
        shutil.copyfile(temporal.__file__, args.output / 'temporal_source.py')
    saved_config = temporal.read(args.output / 'config.json')
    assert saved_config['camera_rate'] == args.camera_rate and saved_config['is_smoke'] == args.smoke
    planned_videos = args.videos if args.smoke else [row if isinstance(row, str) else row['video']
                                                    for row in inference_config['videos']]
    assert set(args.videos).issubset(planned_videos)
    annotations = {row['video_id']: row for row in temporal.read(args.annotations)['videos']}
    for video in args.videos:
        saved = args.output / video / 'summary.json'
        if saved.exists():
            assert args.resume
            result = temporal.read(saved)
            assert all(digest(path) == expected for path, expected in result['input_identity'].items())
        else:
            if args.smoke:
                prediction_path = args.detections / 'detections.jsonl'
            elif 'prediction_directories' in inference_config:
                prediction_path = Path(inference_config['prediction_directories'][video]) / 'detections.jsonl'
            else:
                prediction_path = args.detections / video / 'detections.jsonl'
            result = evaluate_video(video, args.metadata, prediction_path, annotations[video], args.output, args.camera_rate, args.smoke)
        print('ENDOMIND_EVALUATED', video, result['opportunities'], 'sources', result['negative_alarm_seconds'], 'false seconds', flush=True)
    results = [temporal.read(args.output / video / 'summary.json') for video in planned_videos
               if (args.output / video / 'summary.json').exists()]
    status = 'SMOKE_COMPLETED' if args.smoke else ('COMPLETED' if len(results) == len(planned_videos) else 'PARTIAL')
    write_json(args.output / 'summary.json', dict(status=status,
        planned_videos=planned_videos, videos=results, identity=identity,
        interpretation='Full native-field-of-view detector observation at fixed confidence 0.2. Replays share videos and lesions.'))


if __name__ == '__main__':
    main()
