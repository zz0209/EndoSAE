import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_single_component_capacity import counts, replay, application
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


METRICS = ['corrected', 'repeat_damage', 'other_damage', 'repeat_gain']


def event_lookup(manifest, episode, positions):
    lookup = {int(p): i for i, p in enumerate(positions)}
    events = [(i, e) for i, e in enumerate(manifest['events']) if e['episode'] == episode]
    maximum = max(len(manifest['frames'][str(e['output'])]['positions']) for _, e in events)
    if maximum > 12:
        raise ValueError('Observed detection count exceeds the exact replay budget')
    indices = np.full((len(events), maximum), len(positions), dtype=int)
    tables = np.zeros((len(events), 2 ** maximum), dtype=bool)
    for row, (_, event) in enumerate(events):
        frame = manifest['frames'][str(event['output'])]
        size = len(frame['positions'])
        indices[row, :size] = [lookup.get(int(p), len(positions)) for p in frame['positions']]
        outcomes = []
        for mask in range(2 ** size):
            boxes = [box for j, box in enumerate(frame['detections']) if mask & (1 << j)]
            state = application.shared.memory.acknowledgement.detection.overlap(
                boxes, frame['frame']['original_boxes_xyxy'])
            outcomes.append(event['lesion'] in state['detected_lesion_ids'])
        tables[row] = np.asarray(outcomes)[np.arange(2 ** maximum) % (2 ** size)]
    return np.array([i for i, _ in events]), indices, tables


def lookup_replay(scores, threshold, indices, tables):
    keep = np.column_stack([scores < threshold, np.ones(len(scores), dtype=bool)])
    patterns = (keep[:, indices] * (1 << np.arange(indices.shape[1]))).sum(2)
    return tables[np.arange(len(indices))[None], patterns]


def deletion_scores(unit, source, coordinates, original):
    product = unit * unit[source]
    norm = np.square(unit).sum(1)
    removed_product = product[:, coordinates].sum(2).T
    removed_query = np.square(unit[:, coordinates]).sum(2).T
    removed_source = np.square(unit[source, coordinates]).sum(1)
    denominator = np.sqrt((norm[None] - removed_query) * (norm[source] - removed_source[:, None]))
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError('A deletion has a zero or invalid aggregate')
    result = np.clip((product.sum(1)[None] - removed_product) / denominator, -1, 1)
    unaffected = (removed_query == 0) & (removed_source[:, None] == 0)
    result[unaffected] = np.broadcast_to(original, result.shape)[unaffected]
    return result


def direct_scores(unit, source, coordinates, original):
    changed = unit.copy()
    changed[:, coordinates] = 0
    changed /= np.linalg.norm(changed, axis=1, keepdims=True)
    scores = np.clip(changed @ changed[source], -1, 1)
    unaffected = (np.square(unit[:, coordinates]).sum(1) == 0) & (np.square(unit[source, coordinates]).sum() == 0)
    scores[unaffected] = original[unaffected]
    return scores


def choose(metrics, pairs):
    safe = (metrics[:, 1] == 0) & (metrics[:, 2] == 0)
    candidates = np.flatnonzero(safe)
    if not len(candidates):
        return None
    order = np.lexsort((pairs[candidates, 1], pairs[candidates, 0],
                        -metrics[candidates, 3].astype(int), -metrics[candidates, 0].astype(int)))
    return int(candidates[order[0]])


def evaluate(run, smoke, resume):
    config = read_json(run / 'config.json')
    parent = Path(config['single_run'])
    event_run = Path(read_json(parent / 'config.json')['event_run'])
    root = run / ('smoke' if smoke else 'evaluation')
    root.mkdir(exist_ok=True)
    folders = sorted((parent / 'evaluation').glob('*/*/complete.json'))
    if smoke:
        folders = [p for p in folders if p.parent.parent.name == config['smoke_video']
                   and p.parent.name.endswith('seed' + str(config['smoke_seed']))]
    started, jobs = time.perf_counter(), 0
    total = sum(len(read_json(event_run / 'inputs' / p.parent.parent.name / 'events.json')['episodes'])
                for p in folders)
    for receipt in folders:
        folder, video, key = receipt.parent, receipt.parent.parent.name, receipt.parent.name
        manifest = read_json(event_run / 'inputs' / video / 'events.json')
        with np.load(folder / 'effects.npz', allow_pickle=False) as saved:
            unit, positions = saved['unit_codes'].copy(), saved['positions'].copy()
            before, same, singles = saved['before'].copy(), saved['same'].copy(), saved['retained'].copy()
        original_folder = event_run / 'evaluation' / video / key
        threshold = float(pd.read_csv(original_folder / 'events.csv').threshold.iloc[0])
        dimension = unit.shape[1]
        features = np.arange(dimension) if not smoke else np.unique(np.linspace(0, dimension - 1, 64, dtype=int))
        pairs = np.array(list(itertools.combinations(features, 2)), dtype=np.int32)
        for episode in manifest['episodes']:
            identifier = episode['episode_id']
            target = root / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(config=digest(run / 'config.json'), source=digest(__file__),
                parent=digest(receipt), effects=digest(folder / 'effects.npz'),
                events=digest(event_run / 'inputs' / video / 'events.json'), smoke=smoke)
            if (target / 'complete.json').exists():
                previous = read_json(target / 'complete.json')
                assert resume and previous['identity'] == identity
                for name, signature in previous['assets'].items():
                    assert digest(target / name) == signature
                jobs += 1
                continue
            begin = time.perf_counter()
            with np.load(original_folder / 'scores.npz', allow_pickle=False) as saved:
                original = saved[identifier + '__before'].copy()
            source = int(np.flatnonzero(positions == episode['source_position'])[0])
            event_ids, indices, tables = event_lookup(manifest, identifier, positions)
            old, labels = before[event_ids], same[event_ids]
            np.testing.assert_array_equal(lookup_replay(original[None], threshold, indices, tables)[0], old)
            one = deletion_scores(unit, source, np.arange(dimension)[:, None], original)
            single_retained = lookup_replay(one, threshold, indices, tables)
            np.testing.assert_array_equal(single_retained, singles[1:, event_ids])
            single_metrics = np.column_stack([counts(single_retained, old, labels)[m] for m in METRICS])
            single_index = choose(single_metrics, np.column_stack([np.arange(dimension), -np.ones(dimension, int)]))
            stats = np.zeros((2, len(pairs), len(METRICS)), dtype=np.uint16)
            differences = np.zeros(len(pairs), dtype=np.uint16)
            direct_error = 0.
            for start in range(0, len(pairs), config['pair_batch']):
                stop = min(start + config['pair_batch'], len(pairs))
                selected = pairs[start:stop]
                exact = deletion_scores(unit, source, selected, original)
                additive = np.clip(one[selected[:, 0]] + one[selected[:, 1]] - original[None], -1, 1)
                predicted = []
                for mode, scores in enumerate([exact, additive]):
                    retained = lookup_replay(scores, threshold, indices, tables)
                    predicted.append(retained)
                    metrics = counts(retained, old, labels)
                    stats[mode, start:stop] = np.column_stack([metrics[m] for m in METRICS])
                differences[start:stop] = (predicted[0] != predicted[1]).sum(1)
                direct = direct_scores(unit, source, selected[0], original)
                error = float(np.max(np.abs(direct - exact[0])))
                direct_error = max(direct_error, error)
                np.testing.assert_allclose(direct, exact[0], atol=2e-12, rtol=0)
                check_ids, check = replay(manifest, identifier, positions, exact[:1], threshold)
                np.testing.assert_array_equal(check_ids, event_ids)
                np.testing.assert_array_equal(check, predicted[0][:1])
                if start % (config['pair_batch'] * 16) == 0 or stop == len(pairs):
                    print('PAIR_CAPACITY', video, key, identifier, stop, '/', len(pairs), flush=True)
                    atomic_write_json(root / 'progress.json', dict(completed=jobs, total=total,
                        video=video, model=key, episode=identifier, pairs=stop, total_pairs=len(pairs)))
            rows = []
            selected_outputs = dict(event_ids=event_ids, before=old, same=labels)
            candidates = [('unchanged', None, None), ('single', single_index, single_metrics)]
            candidates.extend((name, choose(stats[i], pairs), stats[i])
                              for i, name in enumerate(['pair_exact', 'pair_additive']))
            for mode, index, metrics in candidates:
                coordinates = [] if index is None else ([index] if mode == 'single' else pairs[index].tolist())
                values = [0, 0, 0, 0] if index is None else metrics[index].astype(int).tolist()
                if mode.startswith('pair_') and single_index is not None:
                    one_values = single_metrics[single_index].astype(int).tolist()
                    if (one_values[0], one_values[3]) >= (values[0], values[3]):
                        coordinates, values = [single_index], one_values
                if values[0] == 0 and values[3] == 0:
                    coordinates, values = [], [0, 0, 0, 0]
                scores = original[None]
                if coordinates:
                    scores = deletion_scores(unit, source, np.array([coordinates]), original)
                    if mode == 'pair_additive' and len(coordinates) == 2:
                        scores = np.clip(one[coordinates[0]][None] + one[coordinates[1]][None] - original[None], -1, 1)
                retained = lookup_replay(scores, threshold, indices, tables)[0]
                actual = counts(retained[None], old, labels)
                assert [int(actual[m][0]) for m in METRICS] == values
                unique = 0
                if len(coordinates) == 2:
                    unique = int((~old & ~labels & retained &
                        ~single_retained[coordinates[0]] & ~single_retained[coordinates[1]]).sum())
                method, seed = key.rsplit('_seed', 1)
                rows.append(dict(video=video, model=method, seed=int(seed), episode=identifier, mode=mode,
                    features=coordinates, original_wrong=int((~old & ~labels).sum()),
                    original_correct_repeat=int((~old & labels).sum()), events=len(old),
                    joint_only_corrected=unique,
                    **dict(zip(METRICS, values))))
                selected_outputs[mode + '_scores'] = scores[0]
                selected_outputs[mode + '_retained'] = retained
            np.savez_compressed(target / 'statistics.npz', pairs=pairs, metrics=stats,
                interaction_changed_events=differences, single_metrics=single_metrics)
            np.savez_compressed(target / 'selected_predictions.npz', **selected_outputs)
            atomic_write_json(target / 'selected.json', rows)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity,
                pairs=len(pairs), events=len(old), maximum_detections=indices.shape[1],
                singleton_decisions_exact=True, direct_error=direct_error,
                seconds=time.perf_counter() - begin,
                assets={name: digest(target / name) for name in ['statistics.npz', 'selected_predictions.npz', 'selected.json']}))
            jobs += 1
            atomic_write_json(root / 'progress.json', dict(completed=jobs, total=total, video=video, model=key))
            pause_after_checkpoint(target / 'complete.json')
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', jobs=jobs, total=total,
        seconds=time.perf_counter() - started, numpy=np.__version__, pandas=pd.__version__))


def summarize(run, smoke):
    root = run / ('smoke' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    rows = [row for path in sorted(root.glob('*/*/*/selected.json')) for row in read_json(path)]
    table = pd.DataFrame(rows)
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'sources.csv', index=False)
    metrics = ['original_wrong', 'original_correct_repeat'] + METRICS
    for name, keys in [('procedures', ['model', 'seed', 'video', 'mode']),
                       ('seeds', ['model', 'seed', 'mode']), ('summary', ['model', 'mode'])]:
        result = table.groupby(keys)[metrics].sum().reset_index()
        result.to_csv(output / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, (model, group) in zip(axes, result.groupby('model')):
        order = ['unchanged', 'single', 'pair_exact', 'pair_additive']
        group = group.set_index('mode').loc[order]
        ax.bar(np.arange(4), group.corrected, color=['#999999', '#4477aa', '#228833', '#ccbb44'])
        ax.set_xticks(np.arange(4), ['Unchanged', 'One coordinate', 'Two: actual', 'Two: additive'], rotation=15)
        ax.set(title=model, ylabel='Corrected original errors; summed across seeds')
        ax.text(.02, .95, f'Original errors: {int(group.original_wrong.iloc[0])}', transform=ax.transAxes, va='top')
        assert (group.repeat_damage == 0).all() and (group.other_damage == 0).all()
    fig.suptitle('Label-informed capacity on examined development events; zero observed damage')
    fig.savefig(output / 'capacity.png', dpi=170)
    fig.savefig(output / 'capacity.pdf')
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', sources=len(table),
        interpretation='Label-informed exhaustive two-coordinate capacity; not a deployed policy or an independent test.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.run, args.smoke)
    else:
        evaluate(args.run, args.smoke, args.resume)
