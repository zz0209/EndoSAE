import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import itertools
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_pair_component_capacity as pair
import evaluate_source_component_transfer as transfer
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def bank(run, resume):
    config, _, _, _, original = transfer.inputs(run)
    training = read_json(Path(config['training_run']) / 'config.json')
    cohort = read_json(training['cohort_config'])
    records, files = [], [run / 'config.json', run / 'protocol.json', Path(__file__)]
    for split in ['train', 'val']:
        for video in cohort['fit_video_ids'][split]:
            path = Path(training['original_prepared_run']) / 'tokens' / video / 'records.json'
            files.append(path)
            records.extend(read_json(path))
    assert len(records) == 336
    for video in training['added_training_videos']:
        path = Path(training['added_training_storage']) / 'tokens' / video / 'records.json'
        files.append(path)
        records.extend(r for r in read_json(path) if r['original_observation'])
    assert len(records) == 432
    selected = np.array([i for i, r in enumerate(records) if r['split'] == 'train'])
    rows = [records[i] for i in selected]
    videos = {r['video_id'] for r in rows}
    counts = Counter((r['video_id'], r['lesion_id']) for r in rows)
    assert len(videos) == 27 and len(counts) == 85 and len(rows) == 340 and set(counts.values()) == {4}
    assert not videos & set(original['development_videos'] + original['extension_videos'])
    lesions = Counter(v for v, _ in counts)
    weights = np.array([1 / 27 / lesions[r['video_id']] / 4 for r in rows])
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    root = run / 'bank'
    root.mkdir(exist_ok=True)
    for key, model in transfer.model_specs(original):
        method = key.rsplit('_seed', 1)[0].split('_', 1)[1]
        seed = key.rsplit('_seed', 1)[1]
        directory = Path(config['training_run']) / 'components/original_four' / method / ('seed' + seed) / 'foldfull'
        receipt = read_json(directory / 'complete.json')
        assert receipt['status'] == 'COMPLETE'
        assert digest(directory / 'effects.npz') == receipt['assets']['effects.npz']
        assert digest(model / 'model.npz') == receipt['identity']['model_sha256']
        files.extend([directory / 'complete.json', model / 'model.npz'])
        with np.load(directory / 'effects.npz') as saved:
            lookup = {int(index): i for i, index in enumerate(saved['indices'])}
            vectors = saved['unit_codes'][[lookup[int(i)] for i in selected]]
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-7)
        target = root / (key + '.npz')
        if target.exists():
            assert resume
            with np.load(target) as saved:
                np.testing.assert_array_equal(saved['vectors'], vectors)
                np.testing.assert_array_equal(saved['weights'], weights)
        else:
            np.savez_compressed(target, vectors=vectors, weights=weights)
    identity = {str(p): digest(p) for p in files}
    value = dict(status='COMPLETE', identity=identity, observations=len(rows), procedures=27,
                 lesions=85, files={p.name: digest(p) for p in root.glob('*.npz')}, records=rows)
    if (root / 'summary.json').exists():
        assert resume and read_json(root / 'summary.json') == value
    else:
        atomic_write_json(root / 'summary.json', value)
    print('BANK_COMPLETE', 340, 'observations', 27, 'procedures', flush=True)


def optimize(source, negatives, weights, positives, prefix_unit, prefix_source, prefix_original,
             prefix_indices, prefix_tables, prefix_before, threshold, features, batch, tolerance, progress):
    unit = np.vstack([source, negatives, positives])
    original = np.clip(unit @ source, -1, 1)
    negative_end = 1 + len(negatives)
    preserve = original[negative_end:] >= threshold
    base_loss = float(np.maximum(original[1:negative_end] - threshold, 0) @ weights)
    best = dict(coordinates=[], objective=base_loss)
    selected, measures = {'unchanged': dict(best)}, {}
    for policy, candidates in [('bank_single', features[:, None]),
                               ('bank_pair', np.array(list(itertools.combinations(features, 2)), dtype=int))]:
        losses, feasible_all = [], []
        started = time.perf_counter()
        for start in range(0, len(candidates), batch):
            coordinates = candidates[start:start + batch]
            scores = pair.deletion_scores(unit, 0, coordinates, original)
            feasible = np.all(scores[:, negative_end:][:, preserve] >= threshold, axis=1)
            prefix_scores = pair.deletion_scores(prefix_unit, prefix_source, coordinates, prefix_original)
            retained = pair.lookup_replay(prefix_scores, threshold, prefix_indices, prefix_tables)
            feasible &= np.all(~retained[:, ~prefix_before], axis=1)
            objective = np.maximum(scores[:, 1:negative_end] - threshold, 0) @ weights
            losses.append(objective)
            feasible_all.append(feasible)
            index = int(np.argmin(np.where(feasible, objective, np.inf)))
            if feasible[index] and objective[index] < best['objective'] - tolerance:
                best = dict(coordinates=coordinates[index].tolist(), objective=float(objective[index]))
            if start % (batch * 32) == 0 or start + batch >= len(candidates):
                progress(policy, min(start + batch, len(candidates)), len(candidates))
        selected[policy] = dict(best)
        measures[policy] = dict(loss=np.concatenate(losses), feasible=np.concatenate(feasible_all),
                                seconds=time.perf_counter() - started)
        assert best['objective'] <= base_loss + 1e-12
    return selected, measures


def select(run, smoke, resume):
    config, capacity, event, _, original = transfer.inputs(run)
    root = run / ('smoke_selection' if smoke else 'selection')
    root.mkdir(exist_ok=True)
    bank_receipt = read_json(run / 'bank/summary.json')
    assert bank_receipt['status'] == 'COMPLETE'
    history = Path(config['history_run'])
    old_choices = read_json(history / 'selection.json')['choices']
    choices, completed, started = [], 0, time.perf_counter()
    folders = sorted((capacity / 'evaluation').glob('*/*/complete.json'))
    if smoke:
        folders = [p for p in folders if p.parent.parent.name == config['smoke_video']
                   and p.parent.name.endswith(str(config['smoke_seed']))]
    total = sum(len(read_json(event / 'inputs' / p.parent.parent.name / 'events.json')['episodes']) for p in folders)
    for receipt in folders:
        video, key = receipt.parent.parent.name, receipt.parent.name
        manifest = read_json(event / 'inputs' / video / 'events.json')
        bank_path = run / 'bank' / (key + '.npz')
        assert digest(bank_path) == bank_receipt['files'][bank_path.name]
        with np.load(bank_path) as saved:
            negatives, weights = saved['vectors'], saved['weights']
        with np.load(receipt.parent / 'effects.npz') as saved:
            unit, positions, before = saved['unit_codes'], saved['positions'], saved['before']
        with np.load(Path(config['positive_run']) / 'selection' / video / key / 'vectors.npz') as saved:
            positive_arrays = {k: saved[k] for k in saved.files}
        with np.load(history / 'selection' / video / key / 'vectors.npz') as saved:
            history_arrays = {k: saved[k] for k in saved.files}
        features = np.unique(np.linspace(0, unit.shape[1] - 1, 64, dtype=int)) if smoke else np.arange(unit.shape[1])
        for episode in manifest['episodes']:
            identifier = episode['episode_id']
            target = root / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'), source=digest(__file__),
                            bank=digest(bank_path), capacity=digest(receipt), history=digest(history / 'selection.json'),
                            positives=digest(Path(config['positive_run']) / 'selection' / video / key / 'vectors.npz'),
                            events=digest(event / 'inputs' / video / 'events.json'), smoke=smoke)
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity
                choices.extend(saved['choices'])
                completed += 1
                continue
            row = next(r for r in old_choices if r['video'] == video and r['model'] == key
                       and r['episode'] == identifier and r['policy'] == 'unchanged')
            prefix_ids = history_arrays[identifier + '__prefix_indices']
            prefix_events = [manifest['events'][int(i)] for i in prefix_ids]
            assert all(e['same_identity'] and e['frame'] <= row['activation_frame'] for e in prefix_events)
            used_positions = sorted({p for e in prefix_events for p in manifest['frames'][str(e['output'])]['positions']
                                     if p in positions} | {episode['source_position']})
            mapping = [np.flatnonzero(positions == p).item() for p in used_positions]
            prefix_unit = unit[mapping]
            prefix_source = used_positions.index(episode['source_position'])
            source = prefix_unit[prefix_source]
            np.testing.assert_array_equal(source, positive_arrays[identifier + '__source'])
            with np.load(event / 'evaluation' / video / key / 'scores.npz') as saved:
                prefix_original = saved[identifier + '__before'][mapping]
            _, lookup, tables = pair.event_lookup(dict(manifest, events=prefix_events), identifier, np.array(used_positions))
            np.testing.assert_array_equal(pair.lookup_replay(prefix_original[None], row['threshold'], lookup, tables)[0], before[prefix_ids])
            def progress(policy, count, size):
                atomic_write_json(root / 'progress.json', dict(completed=completed, total=total, video=video, model=key,
                    episode=identifier, policy=policy, candidates=count, total_candidates=size))
                print('BANK_SELECTION', completed, '/', total, video, key, identifier, policy, count, '/', size, flush=True)
            selected, measures = optimize(source, negatives, weights, positive_arrays[identifier + '__positives'],
                prefix_unit, prefix_source, prefix_original, lookup, tables, before[prefix_ids], row['threshold'],
                features, config['pair_batch'], config['improvement_tolerance'], progress)
            rows = [dict(model=key, video=video, episode=identifier, policy=policy, **value,
                         threshold=row['threshold'], activation_frame=row['activation_frame'],
                         feedback_available=row['feedback_available'], prefix_events=len(prefix_ids))
                    for policy, value in selected.items()]
            arrays = {policy + '_' + field: value[field] for policy, value in measures.items() for field in ['loss', 'feasible']}
            arrays['features'] = features
            np.savez_compressed(target / 'objectives.npz', **arrays)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, choices=rows,
                seconds={p: value['seconds'] for p, value in measures.items()}, objectives_sha256=digest(target / 'objectives.npz')))
            choices.extend(rows)
            completed += 1
            pause_after_checkpoint(root / 'progress.json')
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', choices=choices, jobs=completed,
        seconds=time.perf_counter() - started, future_outcomes_used_for_selection=False,
        numpy=np.__version__, python=platform.python_version()))


def evaluate(run, smoke):
    config, capacity, event, _, _ = transfer.inputs(run)
    root = run / ('smoke_selection' if smoke else 'selection')
    selection = read_json(root / 'summary.json')
    assert selection['status'] == 'COMPLETE' and not selection['future_outcomes_used_for_selection']
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    rows, errors = [], []
    for key, local in itertools.groupby(selection['choices'], key=lambda r: (r['video'], r['model'])):
        video, model = key
        manifest = read_json(event / 'inputs' / video / 'events.json')
        with np.load(capacity / 'evaluation' / video / model / 'effects.npz') as saved:
            unit, positions, before, same = saved['unit_codes'], saved['positions'], saved['before'], saved['same']
        for choice in local:
            identifier, coordinates = choice['episode'], choice['coordinates']
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
            source = np.flatnonzero(positions == episode['source_position']).item()
            with np.load(event / 'evaluation' / video / model / 'scores.npz') as saved:
                original = saved[identifier + '__before']
            original_ids, original_retained = pair.replay(manifest, identifier, positions, original[None], choice['threshold'])
            np.testing.assert_array_equal(original_retained[0], before[original_ids])
            scores = original.copy()
            if coordinates:
                scores = pair.deletion_scores(unit, source, np.array([coordinates]), original)[0]
                direct = pair.direct_scores(unit, source, coordinates, original)
                errors.append(float(np.max(np.abs(scores - direct))))
                np.testing.assert_allclose(scores, direct, atol=2e-12, rtol=0)
            ids, retained = pair.replay(manifest, identifier, positions, scores[None], choice['threshold'])
            causal = np.array([manifest['events'][i]['frame'] > choice['activation_frame'] for i in ids])
            retained[0, ~causal] = before[ids[~causal]]
            prefix = np.array([manifest['events'][i]['first_prompt'] for i in ids])
            values = pair.counts(retained[:, causal], before[ids[causal]], same[ids[causal]])
            method, seed = model.rsplit('_seed', 1)
            rows.append(dict(**choice, method=method, seed=int(seed), after_events=int(causal.sum()),
                original_wrong=int((~same[ids[causal]] & ~before[ids[causal]]).sum()),
                first_damage=int((~retained[0] & before[ids] & ~same[ids] & prefix).sum()),
                **{k: int(v[0]) for k, v in values.items()}))
            target = output / video / model / identifier
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / (choice['policy'] + '.npz'), ids=ids, retained=retained[0],
                                before=before[ids], same=same[ids], after_activation=causal, scores=scores, positions=positions)
    data = pd.DataFrame(rows)
    data.to_csv(output / 'sources.csv', index=False)
    metrics = ['original_wrong', 'corrected', 'repeat_damage', 'other_damage', 'repeat_gain', 'first_damage']
    for name, groups in [('procedures', ['method', 'seed', 'video', 'policy']),
                         ('seeds', ['method', 'seed', 'policy']), ('summary', ['method', 'policy'])]:
        data.groupby(groups)[metrics].sum().reset_index().to_csv(output / (name + '.csv'), index=False)
    summary = pd.read_csv(output / 'summary.csv')
    fig, axes = pair.plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, metric in zip(axes, ['corrected', 'repeat_damage', 'other_damage']):
        for method, local in summary.groupby('method'):
            local = local.set_index('policy').loc[config['policies']]
            ax.plot(range(3), local[metric], marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
        ax.set_xticks(range(3), ['Original', 'One component', 'Two components'], rotation=15)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Events summed across seeds')
        ax.legend()
    fig.suptitle('Training-bank selection | After initial visibility | Examined development')
    fig.savefig(output / 'effects.png', dpi=170)
    pair.plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', source_rows=len(rows),
        direct_score_max_error=max(errors, default=0), selection_sha256=digest(root / 'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['bank', 'select', 'evaluate'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'bank':
        bank(args.run, args.resume)
    elif args.phase == 'select':
        select(args.run, args.smoke, args.resume)
    else:
        evaluate(args.run, args.smoke)
