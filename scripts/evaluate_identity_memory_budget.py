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

from evaluate_query_conditioned_components import boundary, measurements, load_data
from summarize_adaptive_identity import predictions, bootstrap
from src.checkpoint_io import atomic_write_json, read_json


def summarize(data, budget):
    codes = data['unit'].copy()
    assert np.all(codes >= 0)
    if budget < codes.shape[1]:
        selected = np.argsort(-codes, axis=1, kind='stable')[:, :budget]
        kept = np.zeros_like(codes)
        np.put_along_axis(kept, selected, np.take_along_axis(codes, selected, axis=1), axis=1)
        codes = kept
    norm = np.linalg.norm(codes, axis=1, keepdims=True)
    assert (norm > 0).all()
    codes /= norm
    scores = (codes[data['source']] * codes[data['query']]).sum(1)
    if budget == codes.shape[1]:
        np.testing.assert_allclose(scores, data['before'], atol=3e-7, rtol=1e-6)
        scores = data['before'].copy()
    counts = (codes > 0).sum(1)
    eligible = [v for v in np.unique(data['videos']) if len(set(data['labels'][data['videos'] == v])) == 2]
    rows = []
    for video in eligible:
        mask = data['videos'] == video
        calibration = np.isin(data['videos'], [v for v in eligible if v != video])
        original_threshold = boundary(data['before'][calibration], data['videos'][calibration], data['labels'][calibration], .99)
        threshold = boundary(scores[calibration], data['videos'][calibration], data['labels'][calibration], .99)
        metrics = measurements(data['before'][mask], scores[mask], data['labels'][mask], threshold, original_threshold)
        observations = np.unique(np.concatenate([data['source'][mask], data['query'][mask]]))
        rows.append(dict(video=video, budget=budget, **metrics,
            score_mae=float(np.abs(data['before'][mask] - scores[mask]).mean()),
            active_components=float(counts[observations].mean()),
            indexed_fp32_bytes=float((counts[observations] * 6).mean())))
    return rows, scores


def main(args):
    config = read_json(args.run / 'config.json')
    root = args.run if args.legacy else Path(config['storage_root'])
    items = read_json(root / 'training_summary.json')['outputs']
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
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions_output = {}
    for index, item in enumerate(items):
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
        for budget in [32, 1024]:
            result, scores = summarize(data, budget)
            if budget == 1024:
                np.testing.assert_allclose(scores, data['before'], atol=3e-7, rtol=1e-6)
            metadata = dict(condition=condition, method=item['method'], seed=item['seed'], fold=item['fold'])
            rows.extend(dict(metadata, **row) for row in result)
            predictions_output[f'model{index}_budget{budget}'] = scores
        print('MEMORY_BUDGET', index + 1, len(items), condition, item['method'], flush=True)
    np.savez_compressed(args.output / 'scores.npz', **predictions_output)
    atomic_write_json(args.output / 'models.json', items)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'procedures.csv', index=False)
    metrics = ['auroc', 'recall', 'protection', 'matched_capacity', 'score_mae', 'active_components', 'indexed_fp32_bytes']
    summary = table.groupby(['condition', 'method', 'budget'])[metrics].mean().reset_index()
    summary.to_csv(args.output / 'summary.csv', index=False)
    effects = []
    averaged = table.groupby(['condition', 'method', 'budget', 'video'])[metrics].mean().reset_index()
    for (condition, method), subset in averaged.groupby(['condition', 'method']):
        for metric in ['auroc', 'recall', 'protection', 'matched_capacity']:
            pivot = subset.pivot(index='video', columns='budget', values=metric)
            effects.append(dict(condition=condition, method=method, metric=metric,
                **bootstrap(pivot[32] - pivot[1024])))
    pd.DataFrame(effects).to_csv(args.output / 'paired_effects.csv', index=False)
    atomic_write_json(args.output / 'verification.json', dict(status='PASS', source=str(args.run),
        additional_sources=[str(path) for path in args.additional_run],
        models=len(items), procedures=int(table.video.nunique()), smoke=args.smoke,
        scope='Own-observation largest32pooled activations, renormalized before matching. Same other-procedure threshold calibration for full and compressed codes. Full codes reproduce saved scores. Indexed storage assumes uint16 indices and FP32 values; no runtime claim.'))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--additional-run', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
