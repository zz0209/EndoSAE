import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.realcolon_task import digest, project_boxes, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def load_metadata(directory, video):
    clips = rows(directory / (video + '.jsonl'))
    receipt = read(directory / (video + '.json'))
    frames, segments = {}, {}
    for clip in clips:
        for frame in clip['frames']:
            index = frame['frame_index']
            assert index not in frames and frame['eligible'] and frame['class'] in (0, 1)
            frames[index] = frame
    for excluded in receipt['excluded_clips']:
        for frame in excluded['frames']:
            index = frame['frame_index']
            if frame['eligible']:
                assert index not in frames and frame['class'] in (0, 1)
                frames[index] = frame
    for frame in receipt['terminal_frame_records']:
        index = frame['frame_index']
        if frame['eligible']:
            assert index not in frames and frame['class'] in (0, 1)
            frames[index] = frame
    segment, previous = -1, None
    for index in sorted(frames):
        if previous is None or index != previous + 1:
            segment += 1
        segments[index] = segment
        previous = index
    for clip in clips:
        clip['input_metadata_continuity_segment'] = clip['continuity_segment']
        clip['continuity_segment'] = segments[clip['frames'][-1]['frame_index']]
    segment_ends = {segment: (max(index for index in segments if segments[index] == segment) + 1) / receipt['fps']
                    for segment in set(segments.values())}
    assert len(clips) == receipt['clips']
    return clips, receipt, frames, segments, segment_ends


def completion_times(clips, directory, config):
    available = np.asarray([clip['available_time_seconds'] for clip in clips])
    latency = config['latency']
    if latency['mode'] == 'recorded':
        durations = np.load(directory / latency['filename'])
        assert durations.shape == available.shape
    else:
        assert latency['mode'] == 'fixed'
        durations = np.full(len(clips), latency['seconds'])
    assert np.isfinite(durations).all() and (durations >= 0).all()
    completed, previous = [], -np.inf
    for availability, duration in zip(available, durations):
        previous = max(float(availability), previous) + float(duration)
        completed.append(previous)
    return np.asarray(completed), durations


def displayed_intervals(clips, receipt, frames, segments, completed, maps, threshold):
    fps = receipt['fps']
    intervals = []
    for index, frame in sorted(frames.items()):
        left, right = index / fps, (index + 1) / fps
        interior = completed[(completed > left) & (completed < right)]
        boundaries = np.unique(np.concatenate(([left], interior, [right])))
        support = project_boxes(frame)[0]
        lesion_support = {identity: project_boxes(dict(frame, boxes_xyxy=[box for box in frame['boxes_xyxy']
                            if box['lesion_id'] == identity]))[0] for identity in frame['visible_lesion_ids']}
        assert bool(support.any()) == bool(frame['class'])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output = int(np.searchsorted(completed, start, side='right') - 1)
            alarm, detected = False, []
            if output >= 0:
                scores = maps[output]
                alarm = bool(scores.max() >= threshold)
                detected = [identity for identity, mask in lesion_support.items() if np.any(scores[mask] >= threshold)]
            intervals.append(dict(start=float(start), end=float(end), frame=index, output=output,
                segment=int(segments[index]), negative=frame['class'] == 0, alarm=alarm,
                target_detected=bool(detected), visible_lesion_ids=sorted(lesion_support), detected_lesion_ids=detected))
    assert intervals
    assert abs(sum(row['end'] - row['start'] for row in intervals) - len(frames) / fps) < 1e-7
    return intervals


def persistent_events(intervals, completed, clips, maps, config):
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
        ready = start + config['persistence_seconds']
        indices = np.flatnonzero((completed >= ready - 1e-9) & (completed < end - 1e-9))
        indices = [int(index) for index in indices if clips[index]['continuity_segment'] == group[0]['segment']]
        record = dict(event_index=number, start=start, end=end, duration=end - start,
                      segment=group[0]['segment'], sustained=end - start >= config['persistence_seconds'] - 1e-9,
                      opportunity=bool(indices))
        if indices:
            output = indices[0]
            query = int(maps[output].argmax())
            feedback = float(completed[output])
            assert maps[output, query] >= config['threshold']
            record.update(feedback_time=feedback, source_output=output, source_clip_id=clips[output]['clip_id'],
                source_input_frames=[frame['frame_index'] for frame in clips[output]['frames']],
                source_input_available_time=clips[output]['available_time_seconds'], source_query=query,
                source_query_row=query // 14, source_query_column=query % 14,
                source_score=float(maps[output, query]), source_annotation_frame=int(np.floor(feedback * clips[output]['fps'] + 1e-8)))
        events.append(record)
    return events


def duration(intervals, start, end, key, mute_end=-np.inf):
    return float(sum(max(0., min(row['end'], end) - max(row['start'], start, mute_end))
                     for row in intervals if row[key]))


def lesion_followup(intervals, feedback, end, mute_seconds, first_annotation, fps):
    future = [row for row in intervals if row['end'] > feedback and row['start'] < end]
    identities = sorted({identity for row in future for identity in row['visible_lesion_ids']})
    results = []
    for identity in identities:
        visible = [row for row in future if identity in row['visible_lesion_ids']]
        first_visible = max(feedback, visible[0]['start'])
        alerts = [max(feedback, row['start'], feedback + mute_seconds) for row in visible
                  if identity in row['detected_lesion_ids'] and row['end'] > max(feedback, row['start'], feedback + mute_seconds)]
        first_alert = min(alerts) if alerts else None
        results.append(dict(lesion_id=identity, scope='previously_visible' if first_annotation[identity] / fps < feedback else 'first_appearance_after_feedback',
            first_annotated_time=first_annotation[identity] / fps, first_observed_visible_time=first_visible,
            source_to_first_observed_visible_seconds=first_visible - feedback,
            first_observed_visibility_outside_60_seconds=first_visible - feedback >= 60.,
            detected=first_alert is not None, first_correct_alert_time=first_alert,
            first_correct_alert_delay_seconds=None if first_alert is None else first_alert - first_visible,
            visible_seconds=sum(min(row['end'], end) - max(row['start'], feedback) for row in visible),
            observation_end=end, outcome='detected' if first_alert is not None else 'no_correct_alert_during_observed_visibility'))
    if results:
        earliest = min(row['first_observed_visible_time'] for row in results)
        new_times = [row['first_observed_visible_time'] for row in results if row['scope'] == 'first_appearance_after_feedback']
        for row in results:
            row['next_visible_lesion'] = abs(row['first_observed_visible_time'] - earliest) < 1e-8
            row['next_new_lesion'] = bool(new_times) and row['scope'] == 'first_appearance_after_feedback' and abs(row['first_observed_visible_time'] - min(new_times)) < 1e-8
    return results


def replay(event, intervals, segment_end, window_end, first_annotation, fps, config):
    feedback = event['feedback_time']
    horizon = min(feedback + config['followup_seconds'], window_end)
    followup = [dict(row, false_alarm=row['negative'] and row['alarm']) for row in intervals
                if row['end'] > feedback and row['start'] < window_end]
    observed = sum(max(0., min(row['end'], horizon) - max(row['start'], feedback)) for row in followup)
    elapsed = horizon - feedback
    unknown = max(0., elapsed - observed)
    negative = duration(followup, feedback, horizon, 'negative')
    positive = observed - negative
    methods = {}
    for mute in [0.] + config['mute_seconds']:
        name = 'baseline' if mute == 0 else 'mute_' + format(mute, 'g') + 's'
        false_seconds = duration(followup, feedback, horizon, 'false_alarm', feedback + mute)
        target_seconds = duration(followup, feedback, horizon, 'target_detected', feedback + mute)
        methods[name] = dict(false_alarm_seconds=false_seconds,
            false_alarm_fraction_of_observed_seconds=false_seconds / observed if observed else None,
            false_alarm_fraction_of_negative_seconds=false_seconds / negative if negative else None,
            target_detected_seconds=target_seconds, target_detection_fraction=target_seconds / positive if positive > 1e-8 else None,
            bins=[dict(start=start, end=end,
                elapsed_seconds=max(0., min(horizon, feedback + end) - (feedback + start)),
                observed_seconds=sum(max(0., min(row['end'], horizon, feedback + end) - max(row['start'], feedback + start)) for row in followup),
                false_alarm_seconds=duration(followup, feedback + start, min(horizon, feedback + end), 'false_alarm', feedback + mute))
                for start, end in config['temporal_bins']],
            lesions=lesion_followup(followup, feedback, window_end, mute, first_annotation, fps))
    return dict(event, observed_seconds=observed, elapsed_seconds=elapsed, unknown_seconds=unknown,
        observation_coverage=observed / elapsed if elapsed else None,
        negative_seconds=negative, positive_seconds=positive, zero_observed_time=observed == 0,
        complete_followup=elapsed >= config['followup_seconds'] - 1e-8,
        complete_observation=unknown < 1e-8,
        followup_end=horizon, remaining_window_observation_end=window_end,
        first_gap_time=segment_end if segment_end < window_end - 1e-8 else None,
        followup_censor_reason=None if elapsed >= config['followup_seconds'] - 1e-8 else 'window_end',
        censor_reason='window_end', methods=methods)


def summarize_opportunities(opportunities, method):
    if not opportunities:
        return dict(opportunities=0)
    full = [row for row in opportunities if row['complete_followup']]
    fully_observed = [row for row in full if row['complete_observation']]
    nonzero = [row for row in opportunities if not row['zero_observed_time']]
    observed = sum(row['observed_seconds'] for row in opportunities)
    negative = sum(row['negative_seconds'] for row in opportunities)
    false = sum(row['methods'][method]['false_alarm_seconds'] for row in opportunities)
    positive = sum(row['positive_seconds'] for row in opportunities)
    detected = sum(row['methods'][method]['target_detected_seconds'] for row in opportunities)
    lesion_summary = {}
    for scope in ['previously_visible', 'first_appearance_after_feedback', 'next_visible_lesion', 'next_new_lesion']:
        lesions = [lesion for row in opportunities for lesion in row['methods'][method]['lesions']
                   if (lesion[scope] if scope in ('next_visible_lesion', 'next_new_lesion') else lesion['scope'] == scope)]
        delays = [lesion['first_correct_alert_delay_seconds'] for lesion in lesions if lesion['detected']]
        lesion_summary[scope] = dict(episode_lesion_observations=len(lesions), detected=sum(lesion['detected'] for lesion in lesions),
            missed=sum(not lesion['detected'] for lesion in lesions), detected_delay_mean_seconds=float(np.mean(delays)) if delays else None)
    return dict(opportunities=len(opportunities), complete_followups=len(full), truncated_followups=len(opportunities) - len(full),
        fully_observed_complete_followups=len(fully_observed), zero_observed_followups=len(opportunities) - len(nonzero),
        unknown_seconds=sum(row['unknown_seconds'] for row in opportunities),
        observed_seconds=observed, negative_seconds=negative,
        fully_observed_complete_mean_false_alarm_seconds=float(np.mean([row['methods'][method]['false_alarm_seconds'] for row in fully_observed])) if fully_observed else None,
        complete_followup_mean_false_alarm_seconds=float(np.mean([row['methods'][method]['false_alarm_seconds'] for row in full])) if full else None,
        complete_followup_mean_false_alarm_reduction_seconds=float(np.mean([row['methods']['baseline']['false_alarm_seconds'] - row['methods'][method]['false_alarm_seconds'] for row in full])) if full else None,
        mean_opportunity_false_alarm_fraction_of_observed_seconds=float(np.mean([row['methods'][method]['false_alarm_fraction_of_observed_seconds'] for row in nonzero])) if nonzero else None,
        mean_opportunity_false_alarm_reduction_fraction=float(np.mean([row['methods']['baseline']['false_alarm_fraction_of_observed_seconds'] - row['methods'][method]['false_alarm_fraction_of_observed_seconds'] for row in nonzero])) if nonzero else None,
        all_followup_false_alarm_fraction_of_observed_seconds=false / observed if observed else None,
        all_followup_false_alarm_fraction_of_negative_seconds=false / negative if negative else None,
        target_detection_fraction=detected / positive if positive > 1e-8 else None, lesions=lesion_summary)


def evaluate_video(video, config, output, annotation):
    directory = Path(config['cache_root']) / 'videos' / video
    assert (directory / 'complete.json').exists(), 'Encoder has not completed this video'
    clips, receipt, frames, segments, segment_ends = load_metadata(Path(config['metadata_directory']), video)
    completion = read(directory / 'complete.json')
    assert completion['status'] == 'COMPLETE' and completion['clips'] == len(clips) and completion['video_id'] == video
    maps = np.load(directory / 'maps.npy', mmap_mode='r')
    assert maps.shape == (len(clips), 196) and np.isfinite(maps).all()
    indices = np.load(directory / 'frame_indices.npy', mmap_mode='r')
    assert np.array_equal(indices, np.asarray([[frame['frame_index'] for frame in clip['frames']] for clip in clips]))
    completed, durations = completion_times(clips, directory, config)
    intervals = displayed_intervals(clips, receipt, frames, segments, completed, maps, config['threshold'])
    events = persistent_events(intervals, completed, clips, maps, config)
    first_annotation = {row['lesion_id']: row['first_frame'] for row in annotation['lesions']}
    window_end = receipt['end_frame_exclusive'] / receipt['fps']
    opportunities = [replay(event, intervals, segment_ends[event['segment']], window_end, first_annotation,
                            receipt['fps'], config) for event in events if event['opportunity']]
    total_false = sum(event['duration'] for event in events)
    covered = sum(event['duration'] for event in events if event['opportunity'])
    residual = sum(event['end'] - event['feedback_time'] for event in events if event['opportunity'])
    union = []
    for event in opportunities:
        start, end = event['feedback_time'], event['followup_end']
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    union_false = sum(max(0., min(end, event['end']) - max(start, event['start']))
                      for start, end in union for event in events)
    assert 0 <= union_false <= total_false + 1e-7
    negative_display = sum(row['end'] - row['start'] for row in intervals if row['negative'])
    positive_display = sum(row['end'] - row['start'] for row in intervals if not row['negative'])
    target_display = sum(row['end'] - row['start'] for row in intervals if row['target_detected'])
    names = ['baseline'] + ['mute_' + format(value, 'g') + 's' for value in config['mute_seconds']]
    result = dict(video=video, split=receipt['split'], clips=len(clips), fps=receipt['fps'],
        eligible_camera_frames=len(frames), unknown_annotation_frames=receipt['requested_frames'] - len(frames),
        unknown_annotation_seconds=(receipt['requested_frames'] - len(frames)) / receipt['fps'],
        observation_segments=len(segment_ends), annotation_status_counts=receipt['frame_status_counts'],
        encoded_input_clips=len(clips), excluded_input_clips=len(receipt['excluded_clips']),
        threshold=config['threshold'], latency=config['latency'], mean_declared_processing_seconds=float(durations.mean()),
        observed_display_seconds=sum(row['end'] - row['start'] for row in intervals),
        negative_display_seconds=negative_display, positive_display_seconds=positive_display,
        target_detected_display_seconds=target_display,
        negative_display_alarm_fraction=total_false / negative_display if negative_display else None,
        positive_display_target_detection_fraction=target_display / positive_display if positive_display else None,
        negative_alarm_seconds=total_false, negative_alarm_intervals=len(events),
        sustained_intervals=sum(event['sustained'] for event in events), opportunities=len(opportunities),
        opportunity_interval_seconds=covered, opportunity_interval_coverage=covered / total_false if total_false else None,
        own_interval_remaining_alarm_seconds=residual, own_interval_remaining_coverage=residual / total_false if total_false else None,
        union_followup_false_alarm_seconds=union_false,
        union_followup_false_alarm_coverage=union_false / total_false if total_false else None,
        interpretation='Independent opportunity replays can overlap. Opportunity reductions are averaged, never added as procedure-level time saved.',
        before_first_lesion_opportunities=sum(event['feedback_time'] < receipt['first_lesion_frame'] / receipt['fps'] for event in events if event['opportunity']),
        after_first_lesion_opportunities=sum(event['feedback_time'] >= receipt['first_lesion_frame'] / receipt['fps'] for event in events if event['opportunity']),
        methods={name: summarize_opportunities(opportunities, name) for name in names},
        input_identity={str(path): digest(path) for path in [directory / 'maps.npy', directory / 'frame_indices.npy', directory / 'complete.json',
            Path(config['metadata_directory']) / (video + '.jsonl'), Path(config['metadata_directory']) / (video + '.json')]})
    destination = output / video
    destination.mkdir(exist_ok=True)
    write_json(destination / 'summary.json', result)
    write_json(destination / 'events.json', events)
    write_json(destination / 'opportunities.json', opportunities)
    np.savez_compressed(destination / 'display_timeline.npz', start=np.asarray([row['start'] for row in intervals]),
        end=np.asarray([row['end'] for row in intervals]), frame=np.asarray([row['frame'] for row in intervals]),
        output=np.asarray([row['output'] for row in intervals]), segment=np.asarray([row['segment'] for row in intervals]),
        negative=np.asarray([row['negative'] for row in intervals]), alarm=np.asarray([row['alarm'] for row in intervals]),
        target_detected=np.asarray([row['target_detected'] for row in intervals]), completion_times=completed)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config, output = read(args.config), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {str(path): digest(path) for path in [args.config, __file__, 'src/evaluation/realcolon_task.py']}
    if (output / 'identity.json').exists():
        assert args.resume and read(output / 'identity.json') == identity
    else:
        write_json(output / 'identity.json', identity)
        write_json(output / 'config.json', config)
        shutil.copyfile(__file__, output / 'source.py')
    started = time.perf_counter()
    if args.metadata_only:
        result = []
        for video in config['videos']:
            clips, receipt, frames, segments, segment_ends = load_metadata(Path(config['metadata_directory']), video)
            result.append(dict(video=video, split=receipt['split'], clips=len(clips), eligible_display_frames=len(frames),
                unknown_annotation_frames=receipt['requested_frames'] - len(frames),
                continuity_segments=len(segment_ends), fps=receipt['fps'], maps_available=(Path(config['cache_root']) / 'videos' / video / 'maps.npy').exists()))
        write_json(output / 'summary.json', dict(status='METADATA_READY_PREDICTIONS_NOT_EVALUATED', videos=result))
        print(json.dumps(result, indent=2), flush=True)
        return
    annotations = {row['video_id']: row for row in read(config['annotation_summary'])['videos']}
    results = []
    for video in config['videos']:
        saved = output / video / 'summary.json'
        result = read(saved) if args.resume and saved.exists() else evaluate_video(video, config, output, annotations[video])
        results.append(result)
        write_json(output / 'progress.json', dict(completed_videos=len(results), videos=len(config['videos']), current_video=video))
        print(video, 'opportunities', result['opportunities'], 'false_alarm_seconds', round(result['negative_alarm_seconds'], 3), flush=True)
    aggregate = {}
    for split in sorted({row['split'] for row in results}):
        selected = [row for row in results if row['split'] == split]
        methods = {}
        for name in selected[0]['methods']:
            keys = ['complete_followup_mean_false_alarm_seconds', 'all_followup_false_alarm_fraction_of_observed_seconds',
                    'all_followup_false_alarm_fraction_of_negative_seconds', 'target_detection_fraction',
                    'complete_followup_mean_false_alarm_reduction_seconds', 'mean_opportunity_false_alarm_fraction_of_observed_seconds',
                    'mean_opportunity_false_alarm_reduction_fraction']
            methods[name] = {key: dict(mean=float(np.mean(values)) if values else None, videos=len(values))
                for key in keys for values in [[row['methods'][name].get(key) for row in selected if row['methods'][name].get(key) is not None]]}
        aggregate[split] = dict(videos=len(selected), opportunities=sum(row['opportunities'] for row in selected),
            methods_video_equal_mean=methods)
    write_json(output / 'summary.json', dict(status='COMPLETED', label='Idealized causal offline evaluation' if config['latency'] == dict(mode='fixed', seconds=0.) else 'Causal offline replay with declared latency',
        per_video=results, aggregate=aggregate, seconds=time.perf_counter() - started, python=platform.python_version(), numpy=np.__version__,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
