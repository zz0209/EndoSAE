import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_adaptive_causal_events import specification, event_rows, boundary
from train_acknowledgement_sae import now, json_digest
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def evaluate(run, smoke):
    config = read_json(run / 'config.json')
    detector, preparation, _, _ = specification(Path(config['detector_run']))
    annotated = read_json(Path(config['annotated_run']) / 'config.json')
    output = run / ('smoke' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    identity = dict(config=digest(run / 'config.json'), source=digest(__file__),
        evaluation=digest('scripts/evaluate_adaptive_causal_events.py'),
        event_matching=digest('scripts/evaluate_prompt_event_components.py'))
    banks, locations, manifests = {}, {}, {}
    started = time.perf_counter()
    for support, settings in [('detector', detector), ('annotated', annotated)]:
        root = Path(settings['storage_root']) / 'encoded'
        summary = read_json(root / 'summary.json')
        assert summary['status'] == 'COMPLETE'
        identity[support] = digest(root / 'summary.json')
        keys = summary['keys'][:1] if smoke else summary['keys']
        banks[support], locations[support] = {}, {}
        for video in sorted({r['video'] for r in summary['receipts']}):
            chunks = {key: [] for key in keys}
            positions = []
            for item in [r for r in summary['receipts'] if r['video'] == video]:
                path = root / video / f"{item['output']:06d}" / 'codes.npz'
                assert digest(path) == item['codes_sha256']
                with np.load(path, allow_pickle=False) as saved:
                    positions.append(saved['positions'].copy())
                    for key in keys:
                        chunks[key].append(saved[key].copy())
            banks[support][video] = {}
            for key, values in chunks.items():
                unit = np.concatenate(values).astype(np.float64)
                unit /= np.linalg.norm(unit, axis=1, keepdims=True)
                banks[support][video][key] = unit
            locations[support][video] = np.concatenate(positions)
            assert len(np.unique(locations[support][video])) == len(locations[support][video])
            manifests[video] = read_json(Path(preparation['event_reference']) / 'inputs' / video / 'events.json')
            print('LOADED', support, video, len(positions), flush=True)
    for video in manifests:
        np.testing.assert_array_equal(locations['detector'][video], locations['annotated'][video])
    signature = json_digest(identity)
    references = {}
    for support, folder in [('detector', config['detector_run']), ('annotated', config['annotated_run'])]:
        table = pd.read_csv(Path(folder) / 'analysis/events.csv')
        references[support] = table[table.variant == 'budget1024'].set_index(['video', 'method', 'seed', 'event_id'])
    variants = [('source_annotated', 'annotated', 'detector'), ('query_annotated', 'detector', 'annotated')]
    if smoke:
        variants += [('detector_replay', 'detector', 'detector'), ('annotated_replay', 'annotated', 'annotated')]
    receipts = []
    for key in keys:
        for variant, source_support, query_support in variants:
            target = output / f'{key}_{variant}'
            target.mkdir(exist_ok=True)
            receipt_path = target / 'complete.json'
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert receipt['signature'] == signature and digest(target / 'events.csv') == receipt['events_sha256']
            else:
                scores, matched, calibration, rows = {}, [], [], []
                for video, manifest in manifests.items():
                    positions = locations['detector'][video]
                    lookup = {int(p): i for i, p in enumerate(positions)}
                    size = max(max(max(f['positions']) for f in manifest['frames'].values()), int(positions.max())) + 1
                    scores[video] = {}
                    for episode in manifest['episodes']:
                        source = banks[source_support][video][key][lookup[episode['source_position']]]
                        score = np.full(size, np.nan)
                        score[positions] = banks[query_support][video][key] @ source
                        scores[video][episode['episode_id']] = score
                    for event in manifest['events']:
                        frame = manifest['frames'][str(event['output'])]
                        match = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
                        position = frame['positions'][match['prediction_index']]
                        matched.append(dict(video=video, label=event['same_identity'],
                            score=scores[video][event['episode']][position]))
                table = pd.DataFrame(matched)
                for video in manifests:
                    fit = table[(table.video != video) & np.isfinite(table.score)]
                    threshold = boundary(fit.score.to_numpy(), fit.video.to_numpy(), fit.label.to_numpy(), .99)
                    assert video not in set(fit.video)
                    calibration.append(dict(video=video, threshold=threshold, fit_procedures=sorted(fit.video.unique())))
                    rows.extend(event_rows(manifests[video], scores[video], scores[video], threshold, key, variant))
                events = pd.DataFrame(rows)
                assert len(events) == 5394
                replay_checks = 0
                if source_support == query_support:
                    local = events.set_index(['video', 'method', 'seed', 'event_id'])
                    original = references[source_support].loc[local.index]
                    np.testing.assert_allclose(local.matched_score, original.matched_score, atol=2e-14, equal_nan=True)
                    np.testing.assert_allclose(local.threshold, original.threshold, atol=2e-14)
                    assert local.retained.equals(original.retained)
                    replay_checks = len(events)
                events.to_csv(target / 'events.csv', index=False)
                atomic_write_json(target / 'calibration.json', calibration)
                receipt = dict(signature=signature, key=key, variant=variant, source_support=source_support,
                    query_support=query_support, events_sha256=digest(target / 'events.csv'),
                    replay_checks=replay_checks, completed_at=now())
                atomic_write_json(receipt_path, receipt)
            receipts.append(receipt)
            progress = dict(status='RUNNING', completed=len(receipts), total=len(keys) * len(variants),
                seconds=time.perf_counter() - started, updated_at=now())
            atomic_write_json(output / 'progress.json', progress)
            print('REGION_FACTORIAL', progress, flush=True)
            pause_after_checkpoint(receipt_path)
    events = pd.concat([pd.read_csv(output / f"{r['key']}_{r['variant']}" / 'events.csv') for r in receipts])
    events.to_csv(output / 'events.csv', index=False)
    rows = []
    for key, group in events.groupby(['video', 'method', 'seed', 'variant']):
        same, other = group[group.same_identity], group[~group.same_identity]
        rows.append(dict(zip(['video', 'method', 'seed', 'variant'], key),
            repeat_removal=(~same.retained).mean(), other_retention=other.retained.mean(),
            first_retention=other[other.first_prompt].retained.mean()))
    procedures = pd.DataFrame(rows)
    procedures.to_csv(output / 'procedures.csv', index=False)
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    procedures.groupby(['method', 'variant', 'seed'])[metrics].mean().to_csv(output / 'seeds.csv')
    summary = procedures.groupby(['method', 'variant'])[metrics].mean()
    summary.to_csv(output / 'summary.csv')
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', identity=identity, receipts=receipts,
        events=len(events), replay_checks=sum(r['replay_checks'] for r in receipts),
        seconds=time.perf_counter() - started, completed_at=now()))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    evaluate(args.run, args.smoke)
