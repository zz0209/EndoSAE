import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import evaluate_bytetrack_acknowledgement as tracking
from src.evaluation.realcolon_task import digest, write_json


read = tracking.acknowledgement.detection.temporal.read
TRACKER_CONFIG = Path('results/runs/20260927T1124Z_acknowledgement_pilot_v1/bytetrack_config.json')


def enumerate_sources(receipt, annotations):
    first = {row['lesion_id']: row['first_frame'] for row in annotations['lesions']
             if receipt['start_frame'] <= row['first_frame'] < receipt['end_frame_exclusive']}
    ordered = sorted(first, key=lambda key: (first[key], key))
    return [dict(source_lesion_id=source, first_annotation_frame=first[source],
                 earliest_source=source == ordered[0],
                 later_identities=[other for other in ordered if first[other] > first[source]])
            for source in ordered if any(first[other] > first[source] for other in ordered)]


def mean_defined(values):
    defined = [value for value in values if value is not None]
    return dict(mean=float(np.mean(defined)) if defined else None,
                defined=len(defined), total=len(values))


def episode_metrics(result, source_record):
    click = result['click']
    source = source_record['source_lesion_id']
    methods = {}
    for name, policy in result['methods'].items():
        other = [row for row in policy['lesions'] if row['lesion_id'] != source
                 and row['known_visible_seconds'] > 0] if click['available'] else []
        groups = {
            'all_other': other,
            'later_first_appearance_after_click': [row for row in other
                if row['first_annotation_time'] > click['time']],
            'earlier_seen_at_click': [row for row in other
                if row['first_annotation_time'] <= click['time']],
        } if click['available'] else {key: [] for key in
            ['all_other', 'later_first_appearance_after_click', 'earlier_seen_at_click']}
        group_results = {}
        for group, lesions in groups.items():
            group_results[group] = dict(lesion_ids=[row['lesion_id'] for row in lesions],
                retention=mean_defined([row['baseline_qualified_retention'] for row in lesions]),
                baseline_missed_lesions=[row['lesion_id'] for row in lesions if row['baseline_missed_in_scope']],
                completely_hidden_lesions=[row['lesion_id'] for row in lesions if row['completely_hidden_in_scope']])
        methods[name] = dict(suppression=policy['acknowledged_suppression_fraction'],
            protection=group_results['all_other']['retention']['mean'], groups=group_results)
    return dict(episode_id=result['video'] + '__' + source, source_lesion_id=source,
        earliest_source=source_record['earliest_source'], click_available=click['available'],
        evaluation_scope='post_click' if click['available'] else 'No click; post-click metrics undefined; saved policy outputs retain unchanged full-window display',
        methods=methods)


def aggregate_episodes(episodes):
    names = list(episodes[0]['methods']) if episodes else []
    return dict(episodes=len(episodes), clicks_available=sum(row['click_available'] for row in episodes),
        methods={name: dict(
            suppression=mean_defined([row['methods'][name]['suppression'] for row in episodes]),
            protection=mean_defined([row['methods'][name]['protection'] for row in episodes]),
            protection_by_onset={group: mean_defined([
                row['methods'][name]['groups'][group]['retention']['mean'] for row in episodes])
                for group in ['later_first_appearance_after_click', 'earlier_seen_at_click']}) for name in names})


def aggregate_procedures(procedures, stratum):
    rows = [row[stratum] for row in procedures]
    names = list(rows[0]['methods']) if rows else []
    return dict(independent_procedures=len(rows), episodes=sum(row['episodes'] for row in rows),
        clicks_available=sum(row['clicks_available'] for row in rows),
        methods={name: dict(
            suppression=mean_defined([row['methods'][name]['suppression']['mean'] for row in rows]),
            protection=mean_defined([row['methods'][name]['protection']['mean'] for row in rows]),
            protection_by_onset={group: mean_defined([
                row['methods'][name]['protection_by_onset'][group]['mean'] for row in rows])
                for group in ['later_first_appearance_after_click', 'earlier_seen_at_click']}) for name in names})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert tracking.sv.__version__ == '0.27.0'
    cv2.setNumThreads(1)
    protocol = read(args.protocol)
    population = list(read(Path(protocol['metadata_config']))['videos'])
    assert set(args.videos) <= set(population)
    config = read(TRACKER_CONFIG)
    config.update(metadata=str(args.metadata), predictions=str(args.predictions),
                  annotations=str(args.annotations), videos=population, source_selection='all_eligible')
    files = [Path(__file__), Path(tracking.__file__), Path(tracking.acknowledgement.__file__),
             Path(tracking.acknowledgement.detection.__file__),
             Path(tracking.acknowledgement.detection.temporal.__file__), args.protocol, args.annotations,
             Path(protocol['metadata_config']), TRACKER_CONFIG]
    identity = {str(path): digest(path) for path in files}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output / 'identity.json').exists():
        assert args.resume and read(args.output / 'identity.json') == identity
        assert read(args.output / 'config.json') == config
    else:
        write_json(args.output / 'identity.json', identity)
        write_json(args.output / 'config.json', config)
        write_json(args.output / 'environment.json', {name: importlib.metadata.version(name)
                   for name in ['numpy', 'supervision', 'scipy', 'opencv-python']})
        for path in files[:5]:
            shutil.copyfile(path, args.output / path.name)
    annotations = {row['video_id']: row for row in read(args.annotations)['videos']}
    planned = {video: enumerate_sources(read(args.metadata / (video + '.json')), annotations[video])
               for video in population}
    write_json(args.output / 'planned_episodes.json', planned)
    for video in args.videos:
        destination = args.output / video
        (destination / 'sources').mkdir(parents=True, exist_ok=True)
        episodes, records = [], []
        for source in planned[video]:
            source_id = source['source_lesion_id']
            target = destination / 'sources' / source_id
            if (target / 'summary.json').exists():
                assert args.resume
                result = read(target / 'summary.json')
                assert all(digest(path) == value for path, value in result['input_identity'].items())
            else:
                result = tracking.evaluate_video(video, config, target, annotations[video], source_id)
            episode_id = video + '__' + source_id
            records.append(dict(episode_id=episode_id, **source, click=result['click'],
                                summary=str(target / 'summary.json')))
            episodes.append(episode_metrics(result, source))
            write_json(destination / 'episodes.json', records)
            print('EPISODE_EVALUATED', episode_id, 'click', result['click'].get('time'), flush=True)
        write_json(destination / 'summary.json', dict(video=video,
            planned_episodes=len(planned[video]), completed_episodes=len(episodes),
            episodes=episodes, all_sources=aggregate_episodes(episodes),
            earliest_source=aggregate_episodes([row for row in episodes if row['earliest_source']])))
    procedures = [read(args.output / video / 'summary.json') for video in population
                  if (args.output / video / 'summary.json').exists()]
    write_json(args.output / 'summary.json', dict(
        status='COMPLETED' if len(procedures) == len(population) else 'PARTIAL',
        planned_videos=population, pending_videos=[video for video in population
            if not (args.output / video / 'summary.json').exists()],
        planned_episodes=sum(map(len, planned.values())),
        all_sources=aggregate_procedures(procedures, 'all_sources'),
        earliest_source=aggregate_procedures(procedures, 'earliest_source'), procedures=procedures,
        interpretation='Independent single-click replays. Means are episode-within-procedure then equal procedure. Overlapping prompt seconds are not added as procedure time saved. Undefined post-click values remain null.'))


if __name__ == '__main__':
    main()
