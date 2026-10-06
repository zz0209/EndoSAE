import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_source_component_transfer as transfer
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def verify(run, smoke):
    config, _, _, _, original = transfer.inputs(run)
    prefix = 'smoke_' if smoke else ''
    choices = read_json(run / (prefix + 'selection.json'))['choices']
    rows, gaps, errors, count = [], [], [], 0
    for choice in [r for r in choices if r['policy'] == 'unchanged']:
        video, key, identifier = (choice[k] for k in ['video', 'model', 'episode'])
        fit = run / (prefix + 'fit') / video / key / identifier
        receipt = read_json(fit / 'complete.json')
        assert digest(fit / 'fit.npz') == receipt['fit_sha256']
        with np.load(fit / 'fit.npz') as saved:
            positives, w, bias, sw = (saved[k] for k in ['positives', 'coefficient', 'intercept', 'sample_weight'])
            saved_score, dual, support = (saved[k] for k in ['train_decision', 'dual', 'support_indices'])
            positive_positions = saved['positive_positions']
        native_fit = Path(config['native_direction_run']) / 'fit' / video / key / identifier / 'fit.npz'
        with np.load(native_fit) as saved:
            np.testing.assert_array_equal(positive_positions, saved['positive_positions'])
            np.testing.assert_array_equal(sw, saved['sample_weight'])
        with np.load(run / 'bank' / (key + '.npz')) as saved:
            negatives = saved['vectors']
        x = np.vstack([positives, negatives])
        y = np.r_[np.ones(len(positives)), -np.ones(len(negatives))]
        direct = x @ w + bias
        np.testing.assert_allclose(direct, saved_score, atol=1e-12, rtol=0)
        np.testing.assert_allclose(dual[0] @ x[support], w, atol=1e-12, rtol=0)
        np.testing.assert_allclose(dual.sum(), 0, atol=1e-7)
        alpha = np.abs(dual[0])
        assert np.all(alpha <= config['C'] * sw[support] + 1e-7)
        primal = .5 * (w @ w) + config['C'] * np.sum(sw * np.maximum(1 - y * direct, 0))
        gap = float(primal - alpha.sum() + .5 * (w @ w))
        assert -1e-5 <= gap <= 1e-3 * max(1, abs(primal)), gap
        gaps.append(gap)
        with np.load(Path(config['fitting_vectors_root']) / video / (key + '.npz')) as saved:
            vectors, positions = saved['unit_codes'], saved['positions']
        _, _, _, tracks, _, _, _, _ = transfer.video_inputs(original, video)
        frames = np.repeat(tracks['frame_indices'], np.diff(tracks['offsets']))
        before = np.load(Path(config['baseline_run']) / 'scores' / video / key / identifier / 'unchanged.npy')
        for policy, expected in [('positive_mean', positives.mean(0)), ('exemplar_svm', w)]:
            direction = np.load(fit / (policy + '.npy'))
            np.testing.assert_allclose(direction, expected / np.linalg.norm(expected), atol=1e-12, rtol=0)
            score = np.load(run / (prefix + 'scores') / video / key / identifier / (policy + '.npy'))
            early = frames <= choice['activation_frame']
            np.testing.assert_array_equal(score[early], before[early])
            expected_score = np.clip(vectors @ direction, -1, 1)
            active = frames[positions] > choice['activation_frame']
            np.testing.assert_allclose(score[positions[active]], expected_score[active], atol=2e-12, rtol=0)
            errors.append(float(np.max(np.abs(score[positions[active]] - expected_score[active]))) if active.any() else 0.)
            count += int(active.sum())
        saved = read_json(run / (prefix + 'evaluation') / video / key / identifier / 'complete.json')
        assert saved['original_curve_exact'] and saved['direct_fixed_checks'] == 6
        rows.extend(saved['rows'])
    data = pd.DataFrame(rows)
    metrics = ['matched_retention_repeat', 'matched_retention_other', 'zero_loss_repeat', 'zero_loss_other']
    procedures = data.groupby(['method', 'seed', 'video', 'policy'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed', 'policy'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy'])[metrics].mean().reset_index()
    for name, table in [('sources', data), ('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        folder = prefix + ('evaluation' if name == 'sources' else 'analysis')
        saved = pd.read_csv(run / folder / (name + '.csv'))
        keys = [c for c in ['method', 'seed', 'video', 'episode', 'policy'] if c in table]
        pd.testing.assert_frame_equal(table.sort_values(keys).reset_index(drop=True),
            saved[table.columns].sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
        native = pd.read_csv(Path(config['native_direction_run']) / ('evaluation' if name == 'sources' else 'analysis') / (name + '.csv'))
        if smoke:
            continue
        paired = table.merge(native, on=keys, suffixes=('_changed', '_native'), validate='one_to_one')
        assert len(paired) == len(table)
        for metric in metrics:
            paired[metric + '_delta'] = paired[metric + '_changed'] - paired[metric + '_native']
        paired.to_csv(run / 'analysis' / ('paired_' + name + '.csv'), index=False)
    prep = read_json(run / 'preparation/summary.json')
    assert prep['status'] == 'COMPLETE' and all(r['native_exact'] for r in prep['receipts'])
    diagnostic_rows = [dict(stage=r['stage'], video=r['video'], model=r['model'], **d)
                       for r in prep['receipts'] for d in r['diagnostics']]
    pd.DataFrame(diagnostic_rows).to_csv(run / 'preparation/diagnostics.csv', index=False)
    if not smoke:
        paired = pd.read_csv(run / 'analysis/paired_seeds.csv')
        fig, axes = transfer.plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        for ax, metric, title in zip(axes, ['matched_retention_repeat', 'zero_loss_repeat'],
                                    ['Original protection', 'Complete other-prompt retention']):
            labels = []
            for index, ((method, policy), local) in enumerate(paired[paired.policy != 'unchanged'].groupby(['method', 'policy'])):
                values = local[metric + '_delta'] * 100
                ax.scatter(np.full(len(values), index), values, s=30)
                ax.scatter(index, values.mean(), marker='_', s=200, color='black')
                labels.append(('SAE full' if method.endswith('sparse') else 'Dense K64') + '\n' + policy.replace('_', ' '))
            ax.axhline(0, color='gray', lw=1)
            ax.set_xticks(range(len(labels)), labels, fontsize=8)
            ax.set_ylabel('Repeat-removal change (percentage points)')
            ax.set_title(title)
        fig.suptitle('Frozen encoders: inference cutoff intervention | three seeds')
        fig.savefig(run / 'analysis/budget_effect.png', dpi=170)
        transfer.plt.close(fig)
    atomic_write_json(run / (prefix + 'verification.json'), dict(status='PASS', models=len(gaps),
        maximum_duality_gap=max(gaps), direct_query_scores=count, maximum_score_error=max(errors),
        original_curves=len(gaps), fixed_curve_checks=len(gaps) * 6, aggregates_exact=True,
        positive_positions_and_weights_exact=True, preparation_groups=len(prep['receipts']), source_sha256=digest(__file__)))
    print('BUDGET_VERIFICATION_PASS', len(gaps), count, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify(args.run, args.smoke)
