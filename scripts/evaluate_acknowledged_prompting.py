import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import evaluate_endomind_observation as detection
from src.evaluation.realcolon_task import digest, write_json


METHODS = ['baseline', 'mute_1s', 'mute_5s', 'mute_30s', 'mute_120s', 'mute_to_end']
DOSES = [0., 1., 5., 30., 120., None]


def choose_click(intervals, matches, records, source, reaction):
    first = next((row['start'] for row in intervals if source in row['detected_lesion_ids']), None)
    if first is None:
        return dict(available=False, first_correct_prompt_time=None, reason='Source never correctly prompted')
    for row, matching in zip(intervals, matches):
        if row['start'] + 1e-9 < first + reaction or source not in row['detected_lesion_ids']:
            continue
        candidates = [item for item in matching['matches'] if item['lesion_id'] == source]
        chosen = max(candidates, key=lambda item: (
            records[row['output']]['detections'][item['prediction_index']]['score'], -item['prediction_index']))
        record = records[row['output']]
        return dict(available=True, time=row['start'], first_correct_prompt_time=first,
                    reaction_seconds=reaction, elapsed_seconds=row['start'] - first,
                    additional_wait_seconds=max(0., row['start'] - first - reaction),
                    annotation_frame=row['frame'], input_frame=record['frame_index'],
                    output_index=row['output'], detection_index=chosen['prediction_index'],
                    detection=record['detections'][chosen['prediction_index']],
                    input_available_time=record['available_time_seconds'])
    return dict(available=False, first_correct_prompt_time=first,
                reason='No correctly displayed source box available after reaction time')


def decision_timeline(receipt, frames, intervals, records, source, click):
    ids = sorted({item['lesion_id'] for frame in frames.values() for item in frame['original_boxes_xyxy']})
    lookup = {row['frame']: row for row in intervals}
    assert len(lookup) == len(intervals)
    end = receipt['end_frame_exclusive'] / receipt['fps']
    time = click.get('time', end)
    expiry = [time + dose if dose is not None else end for dose in DOSES]
    boundaries = [time] + expiry
    rows = []
    known_absence, crossed_unknown = False, False
    for output, record in enumerate(records):
        index = record['frame_index']
        begin, finish = index / receipt['fps'], (index + 1) / receipt['fps']
        known = index in frames
        base = lookup.get(index)
        points = [begin] + sorted({x for x in boundaries if begin + 1e-10 < x < finish - 1e-10}) + [finish]
        for start, stop in zip(points[:-1], points[1:]):
            post = click['available'] and start >= time - 1e-9
            visible = [known and lesion in base['visible_lesion_ids'] for lesion in ids]
            detected = [known and lesion in base['detected_lesion_ids'] for lesion in ids]
            if post:
                if not known:
                    crossed_unknown = True
                elif source not in base['visible_lesion_ids']:
                    known_absence = True
            stratum = ('later_reappearance' if known_absence else
                       'unknown_continuity' if crossed_unknown else 'current_visibility') if post else 'before_click'
            keep = [not (post and start < limit - 1e-9) for limit in expiry]
            rows.append(dict(start=start, end=stop, frame=index, output=output, known=known,
                             post_click=post, stratum=stratum, visible=visible,
                             baseline=detected, keep=keep))
    arrays = {key: np.asarray([row[key] for row in rows]) for key in
              ['start', 'end', 'frame', 'output', 'known', 'post_click', 'stratum', 'visible', 'baseline', 'keep']}
    arrays['method_detected'] = arrays['baseline'][None, :, :] & arrays['keep'].T[:, :, None]
    arrays['lesion_ids'] = np.asarray(ids)
    arrays['method_names'] = np.asarray(METHODS)
    assert abs(np.sum(arrays['end'] - arrays['start']) - receipt['requested_frames'] / receipt['fps']) < 1e-7
    return arrays


def bouts(data, active, scope):
    indices = np.flatnonzero(scope)
    result, current = [], []

    def finish(reason):
        if not current:
            return
        first, last = current[0], current[-1]
        preceding_unknown = first > 0 and not data['known'][first - 1]
        result.append(dict(start=float(data['start'][first]), end=float(data['end'][last]),
            duration_seconds=float(data['end'][last] - data['start'][first]),
            first_frame=int(data['frame'][first]), last_frame=int(data['frame'][last]),
            camera_frames=int(len(np.unique(data['frame'][current]))),
            one_camera_frame=len(np.unique(data['frame'][current])) == 1,
            left_censored=bool(preceding_unknown or first == indices[0]),
            right_censored=reason in ['unknown_annotation', 'observation_end', 'scope_end'], end_reason=reason))
        current.clear()

    for index in indices:
        if current and index != current[-1] + 1:
            finish('scope_end')
        if not data['known'][index]:
            finish('unknown_annotation')
        elif active[index]:
            current.append(index)
        else:
            finish('no_correct_prompt')
    if current:
        following = current[-1] + 1
        reason = ('observation_end' if following == len(active) else
                  'unknown_annotation' if not data['known'][following] else
                  'no_correct_prompt' if not active[following] else 'scope_end')
        finish(reason)
    return dict(intervals=result, count=len(result), one_camera_frame_count=sum(row['one_camera_frame'] for row in result),
                longest_seconds=max((row['duration_seconds'] for row in result), default=0.))


def summarize_lesion(data, column, method, scope, first_annotation, fps):
    dt = data['end'] - data['start']
    visible = data['visible'][:, column]
    base = data['baseline'][:, column]
    retained = data['method_detected'][method, :, column]
    eligible = scope & data['known']
    baseline_seconds = float(dt[eligible & base].sum())
    intersection_seconds = float(dt[eligible & base & retained].sum())
    actual_seconds = float(dt[eligible & retained].sum())
    visible_seconds = float(dt[eligible & visible].sum())
    first_visible = np.flatnonzero(visible & data['known'])
    first_alert = np.flatnonzero(retained & data['known'])
    origin = float(data['start'][first_visible[0]]) if len(first_visible) else None
    alert = float(data['start'][first_alert[0]]) if len(first_alert) else None
    assert intersection_seconds <= baseline_seconds + 1e-8
    lesion = str(data['lesion_ids'][column])
    return dict(lesion_id=lesion, known_visible_seconds=visible_seconds,
        baseline_correct_seconds=baseline_seconds, baseline_qualified_retained_seconds=intersection_seconds,
        hidden_baseline_seconds=baseline_seconds - intersection_seconds,
        baseline_qualified_retention=intersection_seconds / baseline_seconds if baseline_seconds else None,
        actual_correct_seconds=actual_seconds,
        actual_visible_coverage=actual_seconds / visible_seconds if visible_seconds else None,
        baseline_visible_coverage=baseline_seconds / visible_seconds if visible_seconds else None,
        baseline_missed_in_scope=bool(visible_seconds and not baseline_seconds),
        completely_hidden_in_scope=bool(baseline_seconds and not intersection_seconds),
        actual_missed_in_scope=bool(visible_seconds and not actual_seconds),
        actual_missed_entire_window=bool(len(first_visible) and not len(first_alert)),
        completely_hidden_entire_window=bool(np.any(base & data['known']) and not len(first_alert)),
        first_annotation_time=first_annotation[lesion] / fps,
        first_observed_visibility_time=origin, first_correct_prompt_time=alert,
        first_prompt_delay_seconds=alert - origin if alert is not None and origin is not None else None,
        first_prompt_delay_from_annotation_seconds=alert - first_annotation[lesion] / fps if alert is not None else None,
        prompt_bouts=bouts(data, retained, scope))


def summarize_policy(data, method, scope, first_annotation, fps, source, click):
    lesions = [summarize_lesion(data, column, method, scope, first_annotation, fps)
               for column in range(len(data['lesion_ids']))]
    protected = [row for row in lesions if not click['available'] or row['lesion_id'] != source]
    defined = [row['baseline_qualified_retention'] for row in protected if row['baseline_qualified_retention'] is not None]
    baseline_seconds = sum(row['baseline_correct_seconds'] for row in protected)
    retained_seconds = sum(row['baseline_qualified_retained_seconds'] for row in protected)
    acknowledged = next((row for row in lesions if row['lesion_id'] == source), None) if click['available'] else None
    strata = {}
    if acknowledged is not None:
        column = list(data['lesion_ids']).index(source)
        for stratum in ['current_visibility', 'later_reappearance', 'unknown_continuity']:
            strata[stratum] = summarize_lesion(data, column, method, scope & (data['stratum'] == stratum), first_annotation, fps)
    return dict(lesions=lesions, acknowledged=acknowledged, acknowledged_observation_strata=strata,
        acknowledged_suppression_fraction=(1. - acknowledged['baseline_qualified_retention'])
            if acknowledged is not None and acknowledged['baseline_qualified_retention'] is not None else None,
        unacknowledged=dict(lesion_count=len(protected), baseline_detected_lesion_count=len(defined),
            macro_lesion_retention=float(np.mean(defined)) if defined else None,
            baseline_correct_seconds=baseline_seconds, retained_seconds=retained_seconds,
            hidden_seconds=baseline_seconds - retained_seconds,
            time_weighted_retention=retained_seconds / baseline_seconds if baseline_seconds else None,
            completely_hidden_lesions=sum(row['completely_hidden_in_scope'] for row in protected),
            actual_missed_lesions=sum(row['actual_missed_in_scope'] for row in protected)))


def evaluate(video, metadata, predictions, annotations, config, output, source_override=None):
    clips, receipt, frames, segments, _ = detection.full_frame_metadata(metadata, video)
    records, completed = detection.load_predictions(predictions, clips, receipt, camera_rate=True)
    intervals, matches = detection.displayed_intervals(receipt, frames, segments, records, completed)
    first_annotation = {row['lesion_id']: row['first_frame'] for row in annotations['lesions']}
    source = source_override or min(first_annotation, key=lambda key: (first_annotation[key], key))
    click = choose_click(intervals, matches, records, source, config['reaction_seconds'])
    data = decision_timeline(receipt, frames, intervals, records, source, click)
    all_scope = np.ones(len(data['start']), dtype=bool)
    scope = data['post_click'] if click['available'] else all_scope
    dt = data['end'] - data['start']
    methods = {name: summarize_policy(data, method, scope, first_annotation, receipt['fps'], source, click)
               for method, name in enumerate(METHODS)}
    destination = output / video
    destination.mkdir(exist_ok=False)
    np.savez_compressed(destination / 'decisions.npz', **data)
    write_json(destination / 'click_input.json', {key: value for key, value in click.items()})
    write_json(destination / 'box_matches.json', matches)
    result = dict(video=video, split=receipt['split'], source_lesion_id_for_evaluation=source,
        click=click, fps=receipt['fps'], camera_frames=len(records),
        window_seconds=float(dt.sum()), known_seconds=float(dt[data['known']].sum()),
        unknown_seconds=float(dt[~data['known']].sum()),
        post_click_known_seconds=float(dt[data['known'] & data['post_click']].sum()),
        post_click_unknown_seconds=float(dt[~data['known'] & data['post_click']].sum()),
        evaluation_scope='Post-click; entire window if no acknowledgement is available',
        methods=methods, input_identity={str(path): digest(path) for path in
            [predictions, predictions.parent / 'complete.json', predictions.parent / 'identity.json',
             metadata / (video + '.jsonl'), metadata / (video + '.json')]})
    write_json(destination / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=Path('results/runs/20260901_realcolon_visibility_a0_v3/summary.json'))
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-source-lesion')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.smoke_source_lesion is None or len(args.videos) == 1
    args.output.mkdir(parents=True, exist_ok=args.resume)
    identity_paths = [Path(__file__), Path(detection.__file__), Path(detection.temporal.__file__), args.protocol, args.annotations]
    identity = {str(path): digest(path) for path in identity_paths}
    config = dict(protocol=detection.temporal.read(args.protocol), reaction_seconds=1., methods=METHODS,
        mute_seconds=DOSES, metadata=str(args.metadata), predictions=str(args.predictions), videos=args.videos,
        smoke_source_lesion=args.smoke_source_lesion, matching_iou=.5,
        retention='Time intersection of baseline and retained method-qualified detections divided by baseline-qualified time',
        scope='Offline replay at camera acquisition times with zero added latency; processing time remains in prediction records. '
              'GT simulates one correctly located acknowledgement and evaluates lesion identity. Display seconds measure prompts; '
              'clinical attention and effectiveness require separate clinical evaluation.')
    if (args.output / 'identity.json').exists():
        assert args.resume and detection.temporal.read(args.output / 'identity.json') == identity
        assert detection.temporal.read(args.output / 'config.json') == config
    else:
        write_json(args.output / 'identity.json', identity)
        write_json(args.output / 'config.json', config)
        shutil.copyfile(__file__, args.output / 'evaluator_source.py')
        shutil.copyfile(detection.__file__, args.output / 'detection_helpers_source.py')
        shutil.copyfile(detection.temporal.__file__, args.output / 'temporal_helpers_source.py')
    annotations = {row['video_id']: row for row in detection.temporal.read(args.annotations)['videos']}
    results = []
    for video in args.videos:
        summary = args.output / video / 'summary.json'
        if summary.exists():
            assert args.resume
            result = detection.temporal.read(summary)
            assert all(digest(path) == value for path, value in result['input_identity'].items())
        else:
            prediction = args.predictions / video / 'detections.jsonl'
            result = evaluate(video, args.metadata, prediction, annotations[video], config, args.output, args.smoke_source_lesion)
        results.append(result)
        print('ACKNOWLEDGEMENT_EVALUATED', video, 'click', result['click'].get('time'),
              'protected lesions', result['methods']['baseline']['unacknowledged']['lesion_count'], flush=True)
    write_json(args.output / 'summary.json', dict(status='SMOKE_COMPLETED' if args.smoke_source_lesion else 'COMPLETED',
        independent_procedures=len(results), videos=results, identity=identity))


if __name__ == '__main__':
    main()
