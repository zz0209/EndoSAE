import argparse
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.single_object_track import STrack

import evaluate_acknowledged_prompting as acknowledgement
from src.evaluation.realcolon_task import digest, write_json


def initialize_clicked_track(tracker, box, assigned_id):
    if assigned_id >= 0:
        return assigned_id, 'bound_existing_assigned_track'
    xyxy = np.asarray(box['xyxy'], dtype=np.float32)
    candidates = [track for track in tracker.tracked_tracks
                  if not track.is_activated and track.frame_id == tracker.frame_id
                  and np.allclose(track.tlbr, xyxy, rtol=0., atol=1e-4)
                  and np.isclose(track.score, box['score'], rtol=0., atol=1e-7)]
    assert len(candidates) <= 1
    if candidates:
        track = candidates[0]
        action = 'confirmed_current_unconfirmed_track'
    else:
        track = STrack(STrack.tlbr_to_tlwh(xyxy), float(box['score']),
            tracker.minimum_consecutive_frames, tracker.shared_kalman,
            tracker.internal_id_counter, tracker.external_id_counter)
        track.activate(tracker.kalman_filter, tracker.frame_id)
        tracker.tracked_tracks.append(track)
        action = 'initialized_from_clicked_detection'
    track.is_activated = True
    assert track.external_track_id >= 0
    return int(track.external_track_id), action


def track_records(records, fps, parameters, click_event=None):
    tracker = sv.ByteTrack(frame_rate=fps, **parameters)
    offsets, identities, seconds = [0], [], []
    started = time.perf_counter()
    initialization = None
    for output, record in enumerate(records):
        boxes = record['detections']
        detections = sv.Detections(
            xyxy=np.asarray([box['xyxy'] for box in boxes], dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray([box['score'] for box in boxes], dtype=np.float32),
            class_id=np.asarray([box['class_id'] for box in boxes], dtype=int),
            data={'detection_index': np.arange(len(boxes))})
        begin = time.perf_counter()
        tracked = tracker.update_with_detections(detections)
        ids = np.full(len(boxes), -1, dtype=np.int64)
        if len(tracked):
            indices = tracked.data['detection_index']
            assert len(np.unique(indices)) == len(indices)
            ids[indices] = tracked.tracker_id
        if click_event is not None and output == click_event['output_index']:
            selected = click_event['detection_index']
            identity, action = initialize_clicked_track(tracker, boxes[selected], int(ids[selected]))
            ids[selected] = identity
            initialization = dict(action=action, track_id=identity, frame=record['frame_index'],
                                  detection_index=selected, detection=boxes[selected])
        seconds.append(time.perf_counter() - begin)
        identities.extend(ids.tolist())
        offsets.append(len(identities))
        if (output + 1) % 1000 == 0 or output + 1 == len(records):
            print('TRACKED', record['video_id'], output + 1, '/', len(records),
                  'frames', round(time.perf_counter() - started, 3), 'seconds', flush=True)
    return dict(offsets=np.asarray(offsets, dtype=np.int64), track_ids=np.asarray(identities, dtype=np.int64),
                frame_indices=np.asarray([row['frame_index'] for row in records], dtype=np.int64),
                processing_seconds=np.asarray(seconds)), dict(seconds=time.perf_counter() - started,
                    max_time_lost=tracker.max_time_lost, new_track_threshold=tracker.det_thresh,
                    unique_assigned_ids=int(len(set(identities) - {-1})),
                    raw_detections=len(identities), unassigned_detections=identities.count(-1),
                    click_initialization=initialization)


def keep_confirmed_track(records, tracks, click):
    keep = np.ones(len(tracks['track_ids']), dtype=bool)
    binding = dict(available=False, track_id=None, reason='No acknowledgement')
    if click['available']:
        position = tracks['offsets'][click['output_index']] + click['detection_index']
        identity = int(tracks['track_ids'][position])
        binding = dict(available=identity >= 0, track_id=identity if identity >= 0 else None,
                       reason='Assigned clicked track' if identity >= 0 else 'Clicked detection has no assigned track')
        if identity >= 0:
            for output, record in enumerate(records):
                if record['available_time_seconds'] + 1e-9 >= click['time']:
                    start, end = tracks['offsets'][output:output + 2]
                    keep[start:end] = tracks['track_ids'][start:end] != identity
    assert np.all(keep[tracks['track_ids'] < 0])
    return keep, binding


def track_observation_groups(data, matches, source, tracks, click, binding, fps):
    groups = np.full(len(data['start']), 'not_post_click_source_prompt', dtype='<U36')
    matching_by_frame = {row['frame_index']: row for row in matches}
    if click['available']:
        for index in np.flatnonzero(data['post_click']):
            frame = int(data['frame'][index])
            if frame not in matching_by_frame:
                continue
            matched = [row for row in matching_by_frame[frame]['matches'] if row['lesion_id'] == source]
            if not matched:
                continue
            original_index = matched[0]['prediction_index']
            identity = int(tracks['track_ids'][tracks['offsets'][data['output'][index]] + original_index])
            groups[index] = ('unassigned' if identity < 0 else
                             'clicked_track' if identity == binding['track_id'] else 'other_track')
    data['source_track_group'] = groups
    dt = data['end'] - data['start']
    columns = list(data['lesion_ids'])
    summaries = {}
    if source in columns:
        column = columns.index(source)
        for group in ['clicked_track', 'other_track', 'unassigned']:
            selected = groups == group
            summaries[group] = dict(baseline_correct_seconds=float(dt[selected].sum()),
                retained_seconds=float(dt[selected & data['method_detected'][-1, :, column]].sum()))
    gaps, active = [], None
    if binding['available']:
        for output in range(click['output_index'], len(tracks['frame_indices'])):
            start, end = tracks['offsets'][output:output + 2]
            present = np.any(tracks['track_ids'][start:end] == binding['track_id'])
            current = int(tracks['frame_indices'][output])
            if not present and active is None:
                active = current
            if present and active is not None:
                gaps.append(dict(first_absent_frame=active, return_frame=current,
                                 seconds=(current - active) / fps, window_censored=False))
                active = None
        if active is not None:
            gaps.append(dict(first_absent_frame=active, return_frame=None,
                seconds=(int(tracks['frame_indices'][-1]) + 1 - active) / fps, window_censored=True))
    return dict(prompt_groups=summaries, clicked_track_absence_intervals=gaps,
        interpretation='Groups use the baseline matched source detection and its actual assigned ID. Track absence uses every camera output independent of annotations; annotation reappearance remains a separate description.')


def evaluate_video(video, config, destination, annotations, source_override=None):
    detection = acknowledgement.detection
    metadata = Path(config['metadata'])
    prediction = Path(config['predictions']) / video / 'detections.jsonl'
    clips, receipt, frames, segments, _ = detection.full_frame_metadata(metadata, video)
    records, completed = detection.load_predictions(prediction, clips, receipt, camera_rate=True)
    intervals, matches = detection.displayed_intervals(receipt, frames, segments, records, completed)
    first = {row['lesion_id']: row['first_frame'] for row in annotations['lesions']}
    source = source_override or (receipt['designated_lesion_id'] if config['source_selection'] == 'window_designated'
              else min(first, key=lambda key: (first[key], key)))
    assert source in first
    click = acknowledgement.choose_click(intervals, matches, records, source, config['reaction_seconds'])
    click_event = (dict(output_index=click['output_index'], detection_index=click['detection_index'])
                   if config.get('click_initialization', False) and click['available'] else None)
    tracks, runtime = track_records(records, receipt['fps'], config['tracker_parameters'], click_event)
    keep, binding = keep_confirmed_track(records, tracks, click)
    data = acknowledgement.decision_timeline(receipt, frames, intervals, records, source, click)
    ids = list(data['lesion_ids'])
    retained = np.zeros_like(data['baseline'])
    for row, output in enumerate(data['output']):
        frame_index = int(data['frame'][row])
        if frame_index not in frames:
            continue
        start, end = tracks['offsets'][output:output + 2]
        boxes = [box for box, flag in zip(records[output]['detections'], keep[start:end]) if flag]
        matching = detection.overlap(boxes, frames[frame_index]['original_boxes_xyxy'])
        retained[row] = [lesion in matching['detected_lesion_ids'] for lesion in ids]
    data['method_detected'] = np.concatenate([data['method_detected'], retained[None]], axis=0)
    names = list(acknowledgement.METHODS) + ['bytetrack_acknowledgement']
    data['method_names'] = np.asarray(names)
    track_groups = track_observation_groups(data, matches, source, tracks, click, binding, receipt['fps'])
    scope = data['post_click'] if click['available'] else np.ones(len(data['start']), dtype=bool)
    methods = {name: acknowledgement.summarize_policy(data, index, scope, first, receipt['fps'], source, click)
               for index, name in enumerate(names)}
    result = dict(video=video, split=receipt['split'], source_lesion_id_for_evaluation=source,
        camera_frames=len(records), fps=receipt['fps'], click=click, track_binding=binding, runtime=runtime,
        known_camera_frames=len(frames), unknown_camera_frames=len(records) - len(frames),
        hidden_detections=int((~keep).sum()), methods=methods, track_observation_groups=track_groups,
        input_identity={str(path): digest(path) for path in [prediction, prediction.parent / 'complete.json',
            prediction.parent / 'identity.json', metadata / (video + '.json'), metadata / (video + '.jsonl')]})
    destination.mkdir(exist_ok=False)
    np.savez_compressed(destination / 'tracks.npz', **tracks)
    np.savez_compressed(destination / 'keep_masks.npz', offsets=tracks['offsets'], keep=keep,
                        frame_indices=tracks['frame_indices'])
    np.savez_compressed(destination / 'decisions.npz', **data)
    write_json(destination / 'click_input.json', dict(**click, track_binding=binding))
    write_json(destination / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert sv.__version__ == '0.27.0'
    cv2.setNumThreads(1)
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
        write_json(args.output / 'environment.json', dict(
            packages={name: importlib.metadata.version(name) for name in
                      ['supervision', 'numpy', 'scipy', 'opencv-python', 'matplotlib', 'defusedxml', 'requests']},
            threads={key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
            cv2_threads=cv2.getNumThreads(), supervision_path=sv.__file__))
        for path in files[:-1]:
            shutil.copyfile(path, args.output / path.name)
    annotations = {row['video_id']: row for row in acknowledgement.detection.temporal.read(Path(config['annotations']))['videos']}
    for video in args.videos:
        saved = args.output / video / 'summary.json'
        if saved.exists():
            assert args.resume
            result = acknowledgement.detection.temporal.read(saved)
            assert all(digest(path) == value for path, value in result['input_identity'].items())
        else:
            result = evaluate_video(video, config, args.output / video, annotations[video])
        print('TRACKER_EVALUATED', video, json.dumps(result['track_binding']), flush=True)
    results = [acknowledgement.detection.temporal.read(args.output / video / 'summary.json')
               for video in config['videos'] if (args.output / video / 'summary.json').exists()]
    write_json(args.output / 'summary.json', dict(status='COMPLETED' if len(results) == len(config['videos']) else 'PARTIAL',
        independent_procedures=len(results), videos=results))


if __name__ == '__main__':
    main()
