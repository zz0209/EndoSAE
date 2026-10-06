import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from datetime import datetime
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


FEATURES = ['delta', 'score', 'score_squared', 'product', 'squared_difference',
            'coordinate_mass', 'concentration_mean', 'concentration_difference',
            'active_overlap', 'log_support_mean', 'log_support_difference']
POLICIES = ['unchanged', 'fixed', 'score_only', 'conditioned', 'label_capacity']


def now():
    return datetime.now().astimezone().isoformat()


def load_data(item, step):
    folder = Path(item['directory'])
    records = read_json(folder / 'records.json')
    path = folder / f'held_{step:04d}.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    indices = arrays['indices']
    lookup = {int(index): position for position, index in enumerate(indices)}
    source = np.array([lookup[int(index)] for index in arrays['source']])
    query = np.array([lookup[int(index)] for index in arrays['query']])
    unit = arrays['pooled_codes'].astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    before = np.sum(unit[source] * unit[query], axis=1)
    np.testing.assert_allclose(before, arrays['scores'], atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(unit, arrays['embeddings'], atol=3e-7, rtol=1e-6)
    labels = arrays['same_identity'].astype(bool)
    videos = np.array([records[int(i)]['video_id'] for i in arrays['source']])
    for i, j, video, same in zip(arrays['source'], arrays['query'], videos, labels, strict=True):
        a, b = records[int(i)], records[int(j)]
        assert a['video_id'] == b['video_id'] == video
        assert a['end_frame'] < b['start_frame']
        assert same == (a['lesion_id'] == b['lesion_id'])
    assert set(videos) <= set(item['held_videos'])
    assert not set(videos) & set(item['fit_videos'])
    support = np.array([sum(records[int(i)]['roi_tokens_per_frame']) for i in indices])
    assert np.all(support > 0)
    return dict(unit=unit, source=source, query=query, before=before, labels=labels,
                videos=videos, support=support, raw_source=arrays['source'], raw_query=arrays['query'],
                input_sha256=digest(path), records_sha256=digest(folder / 'records.json'),
                saved_score_error=float(np.max(np.abs(before - arrays['scores']))))


def candidates(data):
    a, b = data['unit'][data['source']], data['unit'][data['query']]
    original = data['before']
    an, bn = np.square(a).sum(1), np.square(b).sum(1)
    remaining = (an[:, None] - a * a) * (bn[:, None] - b * b)
    assert np.all(remaining > 0)
    changed = (original[:, None] - a * b) / np.sqrt(remaining)
    delta = changed - original[:, None]
    shape = delta.shape
    concentration_a, concentration_b = np.sum(a ** 4, 1), np.sum(b ** 4, 1)
    overlap = ((a > 0) & (b > 0)).sum(1) / ((a > 0) | (b > 0)).sum(1)
    count_a, count_b = np.log1p(data['support'][data['source']]), np.log1p(data['support'][data['query']])
    terms = [np.ones(shape), np.broadcast_to(original[:, None], shape),
             np.broadcast_to(original[:, None] ** 2, shape), a * b, (a - b) ** 2, a * a + b * b,
             np.broadcast_to(((concentration_a + concentration_b) / 2)[:, None], shape),
             np.broadcast_to(np.abs(concentration_a - concentration_b)[:, None], shape),
             np.broadcast_to(overlap[:, None], shape),
             np.broadcast_to(((count_a + count_b) / 2)[:, None], shape),
             np.broadcast_to(np.abs(count_a - count_b)[:, None], shape)]
    features = np.stack(terms, axis=-1) * delta[..., None]
    assert features.shape[-1] == len(FEATURES) and np.isfinite(features).all()
    return changed, delta, features


def pair_weights(videos, labels):
    weights = np.zeros(len(videos))
    unique = np.unique(videos)
    for video in unique:
        for label in [False, True]:
            mask = (videos == video) & (labels == label)
            assert mask.any()
            weights[mask] = 1 / len(unique) / 2 / mask.sum()
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    return weights


def boundary(scores, videos, labels, quantile):
    weights = pair_weights(videos, labels)
    return float(np.quantile(scores[~labels], quantile, method='inverted_cdf', weights=weights[~labels]))


def fit_predict(features, utility, weights, fit, alpha, dimensions):
    x = features[fit, :, :dimensions].reshape(-1, dimensions)
    y = utility[fit].ravel()
    w = np.repeat(weights / features.shape[1], features.shape[1])
    scaler = StandardScaler(with_mean=False).fit(x, sample_weight=w)
    model = Ridge(alpha=alpha, fit_intercept=False, solver='cholesky')
    model.fit(scaler.transform(x), y, sample_weight=w)
    coefficient = model.coef_ / scaler.scale_
    prediction = features[:, :, :dimensions] @ coefficient
    np.testing.assert_allclose(prediction[fit].ravel(), model.predict(scaler.transform(x)), atol=1e-12)
    return prediction, dict(coefficient=coefficient.tolist(), scale=scaler.scale_.tolist(),
                           alpha=alpha, features=FEATURES[:dimensions])


def choose(prediction, changed, before):
    best = np.argmax(prediction, axis=1)
    positive = prediction[np.arange(len(best)), best] > 0
    selected = np.where(positive, best, -1)
    score = np.where(positive, changed[np.arange(len(best)), best], before)
    return score, selected


def direct_check(data, scores, choices):
    largest = 0.
    for name, selected in choices.items():
        a = data['unit'][data['source']].copy()
        b = data['unit'][data['query']].copy()
        used = np.flatnonzero(selected >= 0)
        a[used, selected[used]], b[used, selected[used]] = 0, 0
        direct = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        error = float(np.max(np.abs(direct - scores[name])))
        largest = max(largest, error)
        np.testing.assert_allclose(direct, scores[name], atol=2e-14, rtol=2e-14)
    return largest


def measurements(before, scores, labels, threshold, original_threshold):
    original = before > original_threshold
    after = scores > threshold
    negative_correct = ~labels & ~original
    positive_correct = labels & original
    negative_error = ~labels & original
    positive_error = labels & ~original
    matched = np.unique(np.r_[scores, np.nextafter(scores.min(), -np.inf)])
    protection = np.mean(scores[~labels, None] <= matched[None, :], axis=0)
    recall = np.mean(scores[labels, None] > matched[None, :], axis=0)
    required = float(np.mean(before[~labels] <= original_threshold))
    return dict(recall=float(np.mean(after[labels])), protection=float(np.mean(~after[~labels])),
        auroc=float(roc_auc_score(labels, scores)),
        signed_utility=float(.5 * (np.mean(scores[labels] - before[labels]) - np.mean(scores[~labels] - before[~labels]))),
        corrected=int(np.sum(negative_error & ~after)), repeat_damage=int(np.sum(positive_correct & ~after)),
        other_damage=int(np.sum(negative_correct & after)), repeat_gain=int(np.sum(positive_error & after)),
        original_wrong=int(negative_error.sum()), original_correct_repeat=int(positive_correct.sum()),
        matched_capacity=float(np.max(recall[protection >= required])),
        protected99_capacity=float(np.max(recall[protection >= .99])),
        positives=int(labels.sum()), negatives=int((~labels).sum()), threshold=threshold)


def run_batch(run, smoke, stop_after):
    config = read_json(run / 'config.json')
    parent = Path(config['training_run'])
    summary_path = parent / 'components/summary.json'
    summary = read_json(summary_path)
    assert summary['status'] == 'COMPLETE'
    items = [item for item in summary['models'] if item['condition'] == config['condition']
             and item['method'] in config['methods'] and item['fold'] != 'full'
             and item['seed'] in config['seeds']]
    if smoke:
        items = [item for item in items if item['seed'] == config['seeds'][0] and item['fold'] in config['smoke_folds']]
    root = run / 'smoke' if smoke else run
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'),
                    source=digest(__file__), parent_summary=digest(summary_path), smoke=smoke,
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
    identity_path = root / 'identity.json'
    if identity_path.exists():
        assert read_json(identity_path) == identity
    atomic_write_json(identity_path, identity)
    started, completed, outputs, exclusions = time.perf_counter(), 0, [], []
    for item in items:
        data = load_data(item, config['step'])
        videos = sorted(set(data['videos']))
        eligible = [v for v in videos if len(set(data['labels'][data['videos'] == v])) == 2]
        exclusions.append(dict(method=item['method'], seed=item['seed'], fold=item['fold'],
                               procedures=sorted(set(videos) - set(eligible))))
        changed, delta, features = candidates(data)
        utility = (2 * data['labels'].astype(float) - 1)[:, None] * delta
        targets = eligible[:1] if smoke else eligible
        for target in targets:
            directory = root / item['method'] / f"seed{item['seed']}" / f"fold{item['fold']}" / target
            directory.mkdir(parents=True, exist_ok=True)
            receipt_path = directory / 'complete.json'
            signature = dict(identity=identity, input=data['input_sha256'], records=data['records_sha256'], target=target)
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert receipt['signature'] == signature
                assert digest(directory / 'predictions.npz') == receipt['predictions_sha256']
                outputs.append(dict(directory=str(directory), **receipt))
                completed += 1
                continue
            fit_videos = sorted(set(eligible) - {target})
            assert len(fit_videos) >= 2 and target not in item['fit_videos']
            assert not set(fit_videos) & set(item['fit_videos'])
            fit, held = np.isin(data['videos'], fit_videos), data['videos'] == target
            weights = pair_weights(data['videos'][fit], data['labels'][fit])
            scores, choices, policies = {'unchanged': data['before']}, {}, {}
            fixed_gain = weights @ utility[fit]
            fixed = int(np.argmax(fixed_gain)) if fixed_gain.max() > 0 else -1
            choices['fixed'] = np.full(len(changed), fixed)
            scores['fixed'] = changed[:, fixed] if fixed >= 0 else data['before']
            policies['fixed'] = dict(coordinate=fixed, fitting_utility=float(max(fixed_gain.max(), 0)))
            for name, dimensions in [('score_only', 3), ('conditioned', len(FEATURES))]:
                prediction, policies[name] = fit_predict(features, utility, weights, fit, config['ridge_alpha'], dimensions)
                scores[name], choices[name] = choose(prediction, changed, data['before'])
            scores['label_capacity'], choices['label_capacity'] = choose(utility, changed, data['before'])
            direct_error = direct_check(data, scores, choices)
            original_threshold = boundary(data['before'][fit], data['videos'][fit], data['labels'][fit], config['negative_quantile'])
            rows = []
            for scope, mask in [('fit', fit), ('held', held)]:
                for name in POLICIES:
                    own_threshold = boundary(scores[name][fit], data['videos'][fit], data['labels'][fit], config['negative_quantile'])
                    for boundary_name, threshold in [('original_fit', original_threshold), ('own_fit', own_threshold)]:
                        for video in sorted(set(data['videos'][mask])):
                            local = mask & (data['videos'] == video)
                            rows.append(dict(scope=scope, method=item['method'], seed=item['seed'], fold=item['fold'],
                                target=target, video=video, policy=name, boundary=boundary_name,
                                **measurements(data['before'][local], scores[name][local], data['labels'][local], threshold, original_threshold)))
            saved = dict(before=data['before'], same=data['labels'], videos=data['videos'],
                         source=data['raw_source'], query=data['raw_query'], fit=fit, held=held)
            saved.update({name + '_score': value for name, value in scores.items()})
            saved.update({name + '_coordinate': value for name, value in choices.items()})
            np.savez_compressed(directory / 'predictions.npz', **saved)
            atomic_write_json(directory / 'policies.json', policies)
            atomic_write_json(directory / 'procedures.json', rows)
            receipt = dict(status='COMPLETE', signature=signature, completed_at=now(),
                method=item['method'], seed=item['seed'], fold=item['fold'], target=target,
                encoder_fit_videos=item['fit_videos'], policy_fit_videos=fit_videos,
                direct_error=direct_error, saved_score_error=data['saved_score_error'],
                predictions_sha256=digest(directory / 'predictions.npz'))
            atomic_write_json(receipt_path, receipt)
            outputs.append(dict(directory=str(directory), **receipt))
            completed += 1
            elapsed = time.perf_counter() - started
            atomic_write_json(root / 'progress.json', dict(status='RUNNING', completed=completed,
                model=item['method'], seed=item['seed'], fold=item['fold'], target=target, elapsed_seconds=elapsed, updated_at=now()))
            print('POLICY_TRANSFER', completed, item['method'], item['seed'], item['fold'], target, round(elapsed, 2), 'seconds', flush=True)
            pause_after_checkpoint(root / 'progress.json')
            if stop_after and completed >= stop_after:
                return
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', outputs=outputs, exclusions=exclusions,
        jobs=completed, elapsed_seconds=time.perf_counter() - started, completed_at=now(), identity=identity))
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=completed, updated_at=now()))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = run / 'smoke' if smoke else run
    summary = read_json(root / 'summary.json')
    assert summary['status'] == 'COMPLETE'
    rows = [row for item in summary['outputs'] for row in read_json(Path(item['directory']) / 'procedures.json')]
    table = pd.DataFrame(rows)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'procedures.csv', index=False)
    held = table[table.scope == 'held']
    means = ['recall', 'protection', 'auroc', 'signed_utility', 'matched_capacity', 'protected99_capacity']
    counts = ['corrected', 'repeat_damage', 'other_damage', 'repeat_gain', 'original_wrong', 'original_correct_repeat', 'positives', 'negatives']
    aggregation = {**{k: 'mean' for k in means}, **{k: 'sum' for k in counts}}
    held.groupby(['method', 'seed', 'policy', 'boundary']).agg(aggregation).reset_index().to_csv(output / 'seeds.csv', index=False)
    overall = held.groupby(['method', 'policy', 'boundary']).agg(aggregation).reset_index()
    overall.to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, method in enumerate(['token_dense', 'token_sparse']):
        subset = held[(held.method == method) & (held.boundary == 'original_fit')]
        for j, metric in enumerate(['auroc', 'signed_utility', 'matched_capacity']):
            axis = axes[i, j]
            for seed, local in subset.groupby('seed'):
                values = local.groupby('policy')[metric].mean().reindex(POLICIES)
                axis.plot(range(len(POLICIES)), values, 'o-', label=str(seed))
            axis.set_xticks(range(len(POLICIES)), POLICIES, rotation=25, ha='right')
            axis.set_title(f'{method} | {metric}')
            axis.grid(alpha=.2)
            if j == 0:
                axis.legend(fontsize=8)
    fig.suptitle('Procedure-held-out conditional component selection | Matched capacity uses held labels')
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', jobs=summary['jobs'],
        held_rows=len(held), max_direct_error=max(r['direct_error'] for r in summary['outputs']),
        max_saved_score_error=max(r['saved_score_error'] for r in summary['outputs'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.phase == 'evaluate':
        run_batch(args.run, args.smoke, args.stop_after)
    else:
        summarize(args.run, args.smoke)
