import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_query_conditioned_components import boundary, load_data
from summarize_adaptive_identity import predictions, bootstrap
from src.checkpoint_io import atomic_write_json, read_json


def evidence(data):
    a, b = data['unit'][data['source']], data['unit'][data['query']]
    assert np.all(a >= 0) and np.all(b >= 0)
    products = a * b
    scores = products.sum(1)
    np.testing.assert_allclose(scores, data['before'], atol=3e-7, rtol=1e-6)
    order = np.argsort(-products, axis=1, kind='stable')
    cumulative = np.take_along_axis(products, order, axis=1).cumsum(1)
    nonzero = scores > 0
    count90 = np.zeros(len(scores), dtype=np.int64)
    effective = np.zeros(len(scores))
    count90[nonzero] = (cumulative[nonzero] < .9 * scores[nonzero, None]).sum(1) + 1
    effective[nonzero] = scores[nonzero] ** 2 / np.square(products[nonzero]).sum(1)
    eligible = [v for v in np.unique(data['videos']) if len(set(data['labels'][data['videos'] == v])) == 2]
    rows = []
    for video in eligible:
        selected = np.flatnonzero(data['videos'] == video)
        calibration = np.isin(data['videos'], [v for v in eligible if v != video])
        threshold = boundary(scores[calibration], data['videos'][calibration], data['labels'][calibration], .99)
        for i in selected:
            accepted = bool(scores[i] >= threshold)
            witness = int(np.searchsorted(cumulative[i], threshold, side='left') + 1) if accepted else None
            if accepted:
                assert 1 <= witness <= products.shape[1]
                assert cumulative[i, witness - 1] >= threshold
                assert witness == 1 or cumulative[i, witness - 2] < threshold
            row = dict(video=video, pair=int(i), same_identity=bool(data['labels'][i]), score=float(scores[i]),
                threshold=threshold, matched=accepted, witness_components=witness, mass90_components=int(count90[i]),
                effective_components=float(effective[i]), shared_components=int((products[i] > 0).sum()),
                source_components=int((a[i] > 0).sum()), query_components=int((b[i] > 0).sum()))
            for budget in [8, 16, 32, 64, 128]:
                row[f'mass_at_{budget}'] = float(cumulative[i, budget - 1] / scores[i]) if nonzero[i] else None
                kept_a, kept_b = a[i].copy(), b[i].copy()
                kept_a[order[i, :budget]] = 0
                kept_b[order[i, :budget]] = 0
                denominator = np.linalg.norm(kept_a) * np.linalg.norm(kept_b)
                changed = float(np.dot(kept_a, kept_b) / denominator) if denominator > 0 else None
                row[f'score_after_removing_{budget}'] = changed
            rows.append(row)
    return rows


def main(args):
    config = read_json(args.run / 'config.json')
    root = args.run if args.legacy else Path(config['storage_root'])
    training = read_json(root / 'training_summary.json')
    items = training['outputs']
    for extra in args.additional_run:
        assert not args.legacy
        extra_config = read_json(extra / 'config.json')
        extra_training = read_json(Path(extra_config['storage_root']) / 'training_summary.json')
        assert extra_training['status'] == 'COMPLETE' and not extra_training['smoke']
        assert not {r['seed'] for r in items}.intersection(extra_config['seeds'])
        items.extend(extra_training['outputs'])
    if args.legacy:
        selection = read_json(args.run / 'checkpoint_selection.json')
        items = [item for item in items if item['condition'] == 'expanded_real_views'
                 and item['method'] in ['token_sparse', 'token_dense'] and item['fold'] != 'full']
    if args.smoke:
        items = items[:1]
    rows = []
    for item in items:
        if args.legacy:
            chosen = [r for r in selection if r['condition'] == item['condition']
                      and r['method'] == item['method'] and r['seed'] == item['seed']]
            assert len(chosen) == 1
            data = load_data(item, chosen[0]['step'])
            condition = 'frozen'
        else:
            _, saved, _ = predictions(Path(item['directory']))
            unit = saved['pooled_codes'].astype(np.float64)
            unit /= np.linalg.norm(unit, axis=1, keepdims=True)
            data = dict(unit=unit, source=saved['source'], query=saved['query'], before=saved['scores'],
                        videos=saved['videos'], labels=saved['same_identity'].astype(bool))
            condition = 'adaptive' if item['adaptive'] else 'frozen'
        metadata = dict(condition=condition, method=item['method'], seed=item['seed'], fold=item['fold'])
        rows.extend(dict(metadata, **row) for row in evidence(data))
        print('MATCH_EVIDENCE', condition, item['method'], item['seed'], item['fold'], len(rows), flush=True)
    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'pairs.csv', index=False)
    metrics = ['mass90_components', 'effective_components', 'shared_components', 'source_components',
               'query_components', 'witness_components'] + [f'mass_at_{v}' for v in [8, 16, 32, 64, 128]]
    groups = ['condition', 'method', 'seed', 'video', 'same_identity']
    procedures = table.groupby(groups)[metrics].mean().reset_index()
    procedures.to_csv(args.output / 'procedures.csv', index=False)
    summary = procedures.groupby(['condition', 'method', 'same_identity'])[metrics].mean().reset_index()
    summary.to_csv(args.output / 'summary.csv', index=False)
    decisions = table.groupby(['condition', 'method', 'seed', 'video', 'same_identity', 'matched'])[metrics].mean().reset_index()
    decisions.to_csv(args.output / 'decision_groups.csv', index=False)
    effects = []
    if not args.smoke:
        averaged = procedures.groupby(['condition', 'method', 'video', 'same_identity'])[metrics].mean().reset_index()
        for (condition, same), subset in averaged.groupby(['condition', 'same_identity']):
            for metric in ['mass90_components', 'effective_components', 'shared_components', 'mass_at_32']:
                pivot = subset.pivot(index='video', columns='method', values=metric)
                assert not pivot.isna().any().any()
                effects.append(dict(condition=condition, same_identity=bool(same), metric=metric,
                    **bootstrap(pivot.token_sparse - pivot.token_dense)))
        pd.DataFrame(effects).to_csv(args.output / 'paired_effects.csv', index=False)
    atomic_write_json(args.output / 'verification.json', dict(status='PASS', source=str(args.run),
        additional_sources=[str(path) for path in args.additional_run],
        models=len(items), pairs=len(table), procedures=int(table.video.nunique()), smoke=args.smoke,
        interpretation='Descriptive contribution concentration and exact positive match witnesses. No semantic labels, model selection, or runtime reduction is inferred.',
        checks='Saved cosine scores equal component sums. Positive witnesses cross the calibrated threshold and their preceding subsets do not. Procedure-paired comparisons use label-separated means.'))
    print(summary[['condition', 'method', 'same_identity', 'mass90_components', 'effective_components', 'mass_at_32']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--additional-run', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
