import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC

import evaluate_source_component_transfer as transfer
import evaluate_feedback_tradeoff as tradeoff
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def selection_name(smoke):
    return 'smoke_selection.json' if smoke else 'selection.json'


def fit(run, smoke, resume):
    config, capacity, event, _, original = transfer.inputs(run)
    prior = Path(config['bank_run'])
    bank_receipt = read_json(prior / 'bank/summary.json')
    assert bank_receipt['status'] == 'COMPLETE' and bank_receipt['observations'] == 340
    history = Path(config['history_run'])
    choices = [r for r in read_json(history / 'selection.json')['choices'] if r['policy'] == 'unchanged']
    if smoke:
        choices = [r for r in choices if r['video'] == config['smoke_video'] and r['model'].endswith(str(config['smoke_seed']))]
    root = run / ('smoke_fit' if smoke else 'fit')
    root.mkdir(exist_ok=True)
    rows, begin = [], time.perf_counter()
    for completed, choice in enumerate(choices, 1):
        video, key, identifier = (choice[k] for k in ['video', 'model', 'episode'])
        target = root / video / key / identifier
        target.mkdir(parents=True, exist_ok=True)
        bank_path = prior / 'bank' / (key + '.npz')
        history_path = history / 'selection' / video / key / 'vectors.npz'
        codes_path = capacity / 'evaluation' / video / key / 'effects.npz'
        identity = {str(p): digest(p) for p in [run / 'config.json', run / 'protocol.json', Path(__file__),
                    bank_path, history_path, codes_path, event / 'inputs' / video / 'events.json']}
        if (target / 'complete.json').exists():
            saved = read_json(target / 'complete.json')
            assert resume and saved['identity'] == identity
            rows.extend(saved['choices'])
            continue
        assert identity[str(bank_path)] == bank_receipt['files'][bank_path.name]
        manifest = read_json(event / 'inputs' / video / 'events.json')
        episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
        with np.load(history_path) as saved:
            positions = np.unique(np.r_[episode['source_position'], saved[identifier + '__prefix_positions']])
            prefix = saved[identifier + '__prefix_indices']
        assert all(manifest['events'][i]['same_identity'] and manifest['events'][i]['frame'] <= choice['activation_frame'] for i in prefix)
        with np.load(codes_path) as saved:
            indices = [np.flatnonzero(saved['positions'] == p).item() for p in positions]
            positives = saved['unit_codes'][indices].astype(float)
        with np.load(bank_path) as saved:
            negatives, weights = saved['vectors'].astype(float), saved['weights']
        x = np.vstack([positives, negatives])
        y = np.r_[np.ones(len(positives)), -np.ones(len(negatives))]
        sample_weight = np.r_[np.full(len(positives), 1 / len(positives)), weights] * len(x) / 2
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-7)
        estimator = SVC(kernel='linear', C=config['C'], tol=config['tolerance'], cache_size=64,
                        probability=False, max_iter=config['max_iter'])
        estimator.fit(x, y, sample_weight=sample_weight)
        assert estimator.fit_status_ == 0 and np.array_equal(estimator.classes_, [-1, 1])
        w, bias = estimator.coef_[0], float(estimator.intercept_[0])
        linear = x @ w + bias
        np.testing.assert_allclose(linear, estimator.decision_function(x), atol=1e-9, rtol=0)
        directions = dict(positive_mean=positives.mean(0), exemplar_svm=w)
        local = [dict(choice, feature=-1)]
        for policy, direction in directions.items():
            assert np.isfinite(direction).all() and np.linalg.norm(direction) > 0
            direction = direction / np.linalg.norm(direction)
            path = target / (policy + '.npy')
            np.save(path, direction)
            local.append(dict(choice, policy=policy, feature=-1,
                              direction_file=str(path.relative_to(run)), direction_sha256=digest(path)))
        np.savez_compressed(target / 'fit.npz', positives=positives, positive_positions=positions,
                            coefficient=w, intercept=bias, sample_weight=sample_weight,
                            train_decision=linear, support_indices=estimator.support_, dual=estimator.dual_coef_)
        atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, choices=local,
            positives=len(positives), negatives=len(negatives), iterations=int(estimator.n_iter_[0]),
            fit_sha256=digest(target / 'fit.npz'), positive_accuracy=float(np.mean(linear[:len(positives)] > 0)),
            negative_accuracy=float(np.sum(weights * (linear[len(positives):] < 0)))))
        rows.extend(local)
        atomic_write_json(root / 'progress.json', dict(completed=completed, total=len(choices)))
        print('EXEMPLAR_FIT', completed, '/', len(choices), video, key, identifier, 'positives', len(positives), flush=True)
        pause_after_checkpoint(root / 'progress.json')
    atomic_write_json(run / selection_name(smoke), dict(status='COMPLETE', choices=rows,
        target_outcomes_used_for_selection=True, future_outcomes_used_for_selection=False,
        python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
        seconds=time.perf_counter() - begin, source_sha256=digest(__file__)))


def evaluate(run, smoke, resume):
    config, _, event, _, original = transfer.inputs(run)
    choices = [r for r in read_json(run / selection_name(smoke))['choices'] if r['policy'] == 'unchanged']
    score_root = run / ('smoke_scores' if smoke else 'scores')
    assert read_json(score_root / 'summary.json')['status'] == 'COMPLETE'
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    root.mkdir(exist_ok=True)
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    prior = Path(config['baseline_run'])
    rows, completed, begin = [], 0, time.perf_counter()
    for video in sorted({r['video'] for r in choices}):
        receipt, frames, records, _, offsets, _, _, first = transfer.application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for choice in [r for r in choices if r['video'] == video]:
            key, identifier = choice['model'], choice['episode']
            target = root / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            paths = [run / 'config.json', run / 'protocol.json', Path(__file__), run / selection_name(smoke),
                     score_root / video / 'complete.json']
            identity = {str(p): digest(p) for p in paths}
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity
                rows.extend(saved['rows'])
                completed += 1
                continue
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
            with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                data = {k: saved[k] for k in saved.files}
            data['known'] &= data['frame'] > choice['activation_frame']
            columns = np.asarray(episode['groups']['all_other'], dtype=int)
            with np.load(prior / 'matched' / video / key / identifier / 'unchanged_curve.npz') as saved:
                old_curve = {k: saved[k] for k in saved.files}
            old = tradeoff.measures(old_curve, columns)
            index = int(np.searchsorted(old_curve['threshold'], choice['threshold'], side='right') - 1)
            floor = float(old['other'][index])
            local, checks = [], {}
            for policy in config['policies']:
                scores = np.load(score_root / video / key / identifier / (policy + '.npy'))
                curve, verification = transfer.application.shared.episode_curve(scores, offsets, records, frames,
                    data, episode['source_lesion_id'], first, receipt['fps'])
                if policy == 'unchanged':
                    assert set(curve) == set(old_curve)
                    for field in curve:
                        if curve[field].dtype.kind in 'USb':
                            np.testing.assert_array_equal(curve[field], old_curve[field])
                        else:
                            np.testing.assert_allclose(curve[field], old_curve[field], atol=1e-9, rtol=0, equal_nan=True)
                    np.testing.assert_array_equal(scores, np.load(prior / 'scores' / video / key / identifier / 'unchanged.npy'))
                values = tradeoff.measures(curve, columns)
                points = {}
                for label, limit in [('matched_retention', floor), ('zero_loss', 1.)]:
                    point = tradeoff.choose(values, values['protected'] & (values['other'] >= limit - 1e-12), 'repeat', 'other')
                    assert point is not None
                    points[label] = point
                    fixed, _, _ = transfer.application.fixed_curve(scores, float(curve['threshold'][point]), offsets,
                        records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                    for field in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                  'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                        np.testing.assert_allclose(curve[field][point], fixed[field][0], atol=1e-9, rtol=0, equal_nan=True)
                method, seed = key.rsplit('_seed', 1)
                row = dict(method=method, seed=int(seed), video=video, episode=identifier, policy=policy, protection_floor=floor)
                for label, point in points.items():
                    row[label + '_repeat'] = float(values['repeat'][point])
                    row[label + '_other'] = float(values['other'][point])
                    row[label + '_threshold'] = float(curve['threshold'][point])
                local.append(row)
                np.savez_compressed(target / (policy + '_curve.npz'), **curve)
                checks[policy] = verification
                print('EXEMPLAR_CURVE', completed * 3 + len(local), '/', len(choices) * 3, video, key, identifier, policy, flush=True)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, rows=local,
                checks=checks, original_curve_exact=True, direct_fixed_checks=6))
            rows.extend(local)
            completed += 1
            atomic_write_json(root / 'progress.json', dict(completed=completed, total=len(choices)))
            pause_after_checkpoint(root / 'progress.json')
    pd.DataFrame(rows).to_csv(root / 'sources.csv', index=False)
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', source_models=completed, curves=len(rows),
        seconds=time.perf_counter() - begin, original_curves_exact=completed, direct_fixed_checks=completed * 6))


def summarize(run, smoke):
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    data = pd.read_csv(root / 'sources.csv')
    metrics = ['matched_retention_repeat', 'matched_retention_other', 'zero_loss_repeat', 'zero_loss_other']
    procedures = data.groupby(['method', 'seed', 'video', 'policy'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed', 'policy'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy'])[metrics].mean().reset_index()
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    for name, table in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    fig, axes = transfer.plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    policies = read_json(run / 'config.json')['policies']
    for ax, metric, title in zip(axes, ['matched_retention_repeat', 'zero_loss_repeat'],
                               ['At original other-prompt retention', 'All other prompts retained']):
        for method, local in summary.groupby('method'):
            values = local.set_index('policy').loc[policies, metric] * 100
            ax.plot(range(len(policies)), values, marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
        ax.set_xticks(range(len(policies)), ['Original', 'Positive mean', 'Exemplar SVM'])
        ax.set_ylabel('Attainable repeat removal (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('Initial observations and training negatives | Label-informed threshold diagnosis')
    fig.savefig(output / 'directions.png', dpi=170)
    transfer.plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', results=summary.to_dict('records')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['fit', 'score', 'evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'fit':
        fit(args.run, args.smoke, args.resume)
    elif args.phase == 'score':
        transfer.score(args.run, args.smoke, args.resume, selection_name(args.smoke))
    elif args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
