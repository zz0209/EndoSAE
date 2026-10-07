import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_prompt_event_components import event_rows
from evaluate_query_conditioned_components import boundary
from train_acknowledgement_sae import now
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def load_codes(run):
    config = read_json(run / 'config.json')
    root = Path(config['storage_root']) / 'encoded'
    ready = read_json(root / 'summary.json')
    assert ready['status'] == 'COMPLETE'
    result = {}
    for item in ready['receipts']:
        path = root / item['video'] / f"{item['output']:06d}" / 'codes.npz'
        assert digest(path) == item['codes_sha256']
        with np.load(path, allow_pickle=False) as saved:
            local = result.setdefault(item['video'], {k: [] for k in ['positions'] + ready['keys']})
            for key in local:
                local[key].append(saved[key].copy())
    return {video: {k: np.concatenate(v) for k, v in local.items()} for video, local in result.items()}, ready['keys']


def evaluate(run, smoke):
    config = read_json(run / 'config.json')
    original = Path(config['reference_run'])
    selected = read_json(run / 'history_selection.json')
    preparation = read_json(Path(config['prefix_run']) / 'config.json')
    current, keys = load_codes(original)
    additional = None
    if not smoke:
        additional, extra_keys = load_codes(Path(config['history_encoding_run']))
        assert keys == extra_keys
    output = run / ('smoke' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    manifests = {v: read_json(Path(preparation['event_reference']) / 'inputs' / v / 'events.json') for v in current}
    reference_events = pd.read_csv(original / 'analysis/events.csv')
    reference_events = reference_events[reference_events.variant == 'budget1024']
    reference_calibration = read_json(original / 'analysis/calibration.json')
    variants = ['current'] if smoke else config['variants']
    all_rows, calibration, counts, checks = [], [], [], []
    started = time.perf_counter()
    for key in keys[:1] if smoke else keys:
        method, seed = key.rsplit('_seed', 1)
        finished = output / f'{key}.json'
        event_path = output / f'{key}.csv'
        if finished.exists():
            receipt = read_json(finished)
            assert receipt['config_sha256'] == digest(run / 'config.json')
            assert receipt['source_sha256'] == digest(__file__)
            all_rows.extend(pd.read_csv(event_path).to_dict('records'))
            calibration.extend(receipt['calibration'])
            counts.extend(receipt['counts'])
            checks.extend(receipt['checks'])
            continue
        scores = {variant: {} for variant in variants}
        local_counts = []
        for video, manifest in manifests.items():
            positions = current[video]['positions']
            values = current[video][key].astype(np.float64)
            if additional is not None:
                positions = np.concatenate([positions, additional[video]['positions']])
                values = np.concatenate([values, additional[video][key].astype(np.float64)])
            assert len(positions) == len(np.unique(positions))
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            lookup = {int(p): i for i, p in enumerate(positions)}
            length = max(int(positions.max()), max(max(f['positions']) for f in manifest['frames'].values())) + 1
            mapping = {r['position']: r for r in selected['videos'][video]['rows']}
            for variant in variants:
                scores[variant][video] = {}
            for episode in manifest['episodes']:
                source = values[lookup[episode['source_position']]]
                cosine = values @ source
                base = np.full(length, np.nan)
                base[current[video]['positions']] = cosine[:len(current[video]['positions'])]
                scores['current'][video][episode['episode_id']] = base
                if smoke:
                    continue
                gated, mean, minimum = [np.full(length, np.nan) for _ in range(3)]
                for position, row in mapping.items():
                    if not row['complete']:
                        continue
                    support = [position] + row['previous']
                    indices = np.array([lookup[p] for p in support])
                    local = cosine[indices]
                    direct = np.sum(values[indices] * source, axis=1)
                    np.testing.assert_allclose(local, direct, atol=2e-14, rtol=0)
                    gated[position], mean[position], minimum[position] = local[0], local.mean(), local.min()
                    assert minimum[position] <= gated[position] + 1e-14
                for variant, value in [('history_available_current', gated), ('history_mean', mean), ('history_minimum', minimum)]:
                    scores[variant][video][episode['episode_id']] = value
            for event in manifest['events']:
                frame = manifest['frames'][str(event['output'])]
                match = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
                position = frame['positions'][match['prediction_index']]
                row = mapping.get(position)
                local_counts.append(dict(key=key, video=video, same_identity=event['same_identity'],
                    first_prompt=event['first_prompt'], history_available=bool(row and row['complete'])))
        new_rows, new_calibration, new_checks = [], [], []
        for variant in variants:
            matched = []
            for video, manifest in manifests.items():
                for event in manifest['events']:
                    frame = manifest['frames'][str(event['output'])]
                    match = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
                    position = frame['positions'][match['prediction_index']]
                    matched.append(dict(video=video, label=event['same_identity'],
                        score=scores[variant][video][event['episode']][position]))
            table = pd.DataFrame(matched)
            for video, manifest in manifests.items():
                used = table[(table.video != video) & np.isfinite(table.score)]
                assert len(used.video.unique()) == len(manifests) - 1
                threshold = boundary(used.score.to_numpy(), used.video.to_numpy(), used.label.to_numpy(), .99)
                new_calibration.append(dict(key=key, variant=variant, video=video, threshold=threshold,
                    fit_procedures=sorted(used.video.unique()), fit_events=len(used)))
                rows = event_rows(manifest, scores[variant][video], scores['current'][video], threshold, key, variant)
                if variant == 'current':
                    expected_threshold = next(r['threshold'] for r in reference_calibration
                        if r['key'] == key and r['budget'] == 1024 and r['video'] == video)
                    np.testing.assert_allclose(threshold, expected_threshold, atol=2e-14, rtol=0)
                    old = reference_events[(reference_events.method == method) &
                        (reference_events.seed == int(seed)) & (reference_events.video == video)].sort_values('event_id')
                    actual = pd.DataFrame(rows).sort_values('event_id')
                    np.testing.assert_array_equal(actual.event_id, old.event_id)
                    np.testing.assert_array_equal(actual.retained, old.retained)
                    np.testing.assert_allclose(actual.matched_score, old.matched_score, atol=2e-14, rtol=0, equal_nan=True)
                    new_checks.append(dict(key=key, video=video, baseline_events=len(rows), passed=True))
                new_rows.extend(rows)
        pd.DataFrame(new_rows).to_csv(event_path, index=False)
        atomic_write_json(finished, dict(status='COMPLETE', config_sha256=digest(run / 'config.json'),
            source_sha256=digest(__file__), calibration=new_calibration, counts=local_counts, checks=new_checks,
            events_sha256=digest(event_path), completed_at=now()))
        all_rows.extend(new_rows)
        calibration.extend(new_calibration)
        counts.extend(local_counts)
        checks.extend(new_checks)
        progress = dict(status='RUNNING', completed=len(list(output.glob('*_seed*.csv'))), total=len(keys),
            seconds=time.perf_counter() - started, key=key)
        atomic_write_json(output / 'progress.json', progress)
        print('QUERY_HISTORY_EVALUATION', progress, flush=True)
        pause_after_checkpoint(finished)
    pd.DataFrame(all_rows).to_csv(output / 'events.csv', index=False)
    pd.DataFrame(counts).to_csv(output / 'availability.csv', index=False)
    atomic_write_json(output / 'calibration.json', calibration)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', events=len(all_rows),
        checks=checks, variants=variants, seconds=time.perf_counter() - started,
        selection_sha256=digest(run / 'history_selection.json'), config_sha256=digest(run / 'config.json'),
        source_sha256=digest(__file__), completed_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    evaluate(args.run, args.smoke)
