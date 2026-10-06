import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_feedback_tradeoff as tradeoff
import evaluate_initial_visibility_component as initial
import evaluate_source_component_transfer as transfer
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def prepare(run, smoke, resume):
    config = read_json(run / 'config.json')
    prior = Path(config['selection_run'])
    verified = read_json(prior / 'verification.json')
    assert verified['status'] == 'PASS'
    selection = read_json(prior / 'selection/summary.json')
    assert selection['status'] == 'COMPLETE' and not selection['future_outcomes_used_for_selection']
    choices = selection['choices']
    if smoke:
        choices = [row for row in choices if row['video'] == config['smoke_video']
                   and row['model'].endswith(str(config['smoke_seed']))]
    output = run / initial.selection_name(smoke)
    saved = dict(status='COMPLETE', choices=choices, target_outcomes_used_for_selection=True,
                 future_outcomes_used_for_selection=False,
                 identity={str(p): digest(p) for p in [run / 'config.json', run / 'protocol.json',
                           prior / 'selection/summary.json', prior / 'verification.json', Path(__file__)]})
    if output.exists():
        assert resume and read_json(output) == saved
    else:
        atomic_write_json(output, saved)
    print('FIXED_BANK_ACTIONS', len(choices), flush=True)


def compare(curves, episode, choice):
    columns = np.asarray(episode['groups']['all_other'], dtype=int)
    values = {name: tradeoff.measures(curve, columns) for name, curve in curves.items()}
    indices = {name: int(np.searchsorted(curve['threshold'], choice['threshold'], side='right') - 1)
               for name, curve in curves.items()}
    before, edited = values['unchanged'], values['bank_pair']
    old, changed = indices['unchanged'], indices['bank_pair']
    matched = tradeoff.choose(before, before['protected'] & (before['repeat'] >= edited['repeat'][changed] - 1e-12), 'other', 'repeat')
    floor = before['other'][old]
    operating = {name: tradeoff.choose(value, value['protected'] & (value['other'] >= floor - 1e-12), 'repeat', 'other')
                 for name, value in values.items()}
    zero = {name: tradeoff.choose(value, value['protected'] & (value['other'] >= 1 - 1e-10), 'repeat', 'other')
            for name, value in values.items()}
    assert all(i is not None for i in operating.values()) and all(i is not None for i in zero.values())
    method, seed = choice['model'].rsplit('_seed', 1)
    row = dict(method=method, seed=int(seed), video=choice['video'], episode=choice['episode'],
        original_repeat=float(before['repeat'][old]), original_other=float(floor),
        edited_repeat=float(edited['repeat'][changed]), edited_other=float(edited['other'][changed]),
        original_first_protected=bool(before['protected'][old]), edited_first_protected=bool(edited['protected'][changed]),
        matched_feasible=matched is not None, matched_index=matched,
        matched_repeat=None if matched is None else float(before['repeat'][matched]),
        matched_other=None if matched is None else float(before['other'][matched]),
        matched_threshold=None if matched is None else float(curves['unchanged']['threshold'][matched]),
        protection_gain=None if matched is None else float(edited['other'][changed] - before['other'][matched]))
    for name in curves:
        index = operating[name]
        row[name + '_matched_retention_repeat'] = float(values[name]['repeat'][index])
        row[name + '_matched_retention_other'] = float(values[name]['other'][index])
        row[name + '_matched_retention_threshold'] = float(curves[name]['threshold'][index])
        row[name + '_zero_loss_repeat'] = float(values[name]['repeat'][zero[name]])
    row['ranking_gain'] = row['bank_pair_matched_retention_repeat'] - row['unchanged_matched_retention_repeat']
    row['zero_loss_gain'] = row['bank_pair_zero_loss_repeat'] - row['unchanged_zero_loss_repeat']
    points = {name: {indices[name], operating[name], zero[name]} for name in curves}
    if matched is not None:
        points['unchanged'].add(matched)
    return row, points


def matched(run, smoke, resume):
    _, _, event, _, original = transfer.inputs(run)
    selections = read_json(run / initial.selection_name(smoke))
    choices = [row for row in selections['choices'] if row['policy'] == 'unchanged']
    score_root = run / ('smoke_scores' if smoke else 'scores')
    fixed_root = run / ('smoke_evaluation' if smoke else 'evaluation')
    output = run / ('smoke_matched' if smoke else 'matched')
    output.mkdir(exist_ok=True)
    assert read_json(fixed_root / 'summary.json')['status'] == 'COMPLETE'
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    rows, completed, start = [], 0, time.perf_counter()
    for video in sorted({row['video'] for row in choices}):
        receipt, frames, records, _, offsets, _, _, first = transfer.application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for choice in (row for row in choices if row['video'] == video):
            key, identifier = choice['model'], choice['episode']
            target = output / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(source=digest(__file__), selection=digest(run / initial.selection_name(smoke)),
                            scores=digest(score_root / video / 'complete.json'))
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity
                rows.append(saved['row'])
                completed += 1
                continue
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
            with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                data = {k: saved[k] for k in saved.files}
            data['known'] &= data['frame'] > choice['activation_frame']
            curves, arrays, checks = {}, {}, {}
            for policy in ['unchanged', 'bank_pair']:
                arrays[policy] = np.load(score_root / video / key / identifier / (policy + '.npy'))
                curve, verification = transfer.application.shared.episode_curve(arrays[policy], offsets, records, frames,
                    data, episode['source_lesion_id'], first, receipt['fps'])
                index = int(np.searchsorted(curve['threshold'], choice['threshold'], side='right') - 1)
                with np.load(fixed_root / video / key / identifier / (policy + '_after_activation_curve.npz')) as fixed:
                    for field in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                  'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                        np.testing.assert_allclose(curve[field][index], fixed[field][0], atol=1e-9, rtol=0, equal_nan=True)
                curves[policy], checks[policy] = curve, verification
                np.savez_compressed(target / (policy + '_curve.npz'), **curve)
                print('MATCHED_BANK_CURVE', completed * 2 + len(curves), '/', len(choices) * 2, video, key, identifier, policy, flush=True)
            row, points = compare(curves, episode, choice)
            direct_checks = 0
            for policy, indices in points.items():
                for index in sorted(indices):
                    fixed, _, _ = transfer.application.fixed_curve(arrays[policy], float(curves[policy]['threshold'][index]),
                        offsets, records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                    for field in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                  'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                        np.testing.assert_allclose(curves[policy][field][index], fixed[field][0], atol=1e-9, rtol=0, equal_nan=True)
                    direct_checks += 1
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, row=row,
                checks=checks, direct_checks=direct_checks,
                curves={p.name: digest(p) for p in target.glob('*_curve.npz')}))
            rows.append(row)
            completed += 1
            atomic_write_json(output / 'progress.json', dict(completed=completed, total=len(choices)))
            pause_after_checkpoint(output / 'progress.json')
    sources = pd.DataFrame(rows)
    sources.to_csv(output / 'sources.csv', index=False)
    metrics = ['original_repeat', 'original_other', 'edited_repeat', 'edited_other', 'matched_repeat', 'matched_other',
               'protection_gain', 'ranking_gain', 'zero_loss_gain', 'unchanged_matched_retention_repeat',
               'bank_pair_matched_retention_repeat', 'unchanged_zero_loss_repeat', 'bank_pair_zero_loss_repeat']
    procedures = sources.groupby(['method', 'seed', 'video'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed'])[metrics].mean().reset_index()
    summary = seeds.groupby('method')[metrics].mean().reset_index()
    for name, table in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    fig, axes = transfer.plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for ax, metric in zip(axes, ['protection_gain', 'ranking_gain', 'zero_loss_gain']):
        for i, (method, local) in enumerate(seeds.groupby('method')):
            ax.scatter(np.full(len(local), i), local[metric] * 100)
            ax.plot([i - .2, i + .2], [local[metric].mean() * 100] * 2)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks([0, 1], ['Dense', 'SAE'])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Difference (percentage points)')
    fig.suptitle('Fixed training-bank actions | Label-informed threshold controls')
    fig.savefig(output / 'tradeoff.png', dpi=170)
    transfer.plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', sources=len(rows), curves=len(rows) * 2,
        seconds=time.perf_counter() - start, matched_feasible=int(sources.matched_feasible.sum()),
        results=summary.to_dict('records')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'score', 'evaluate', 'summarize', 'matched'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run, args.smoke, args.resume)
    elif args.phase == 'score':
        transfer.score(args.run, args.smoke, args.resume, initial.selection_name(args.smoke))
    elif args.phase == 'evaluate':
        initial.evaluate(args.run, args.smoke, args.resume)
    elif args.phase == 'summarize':
        initial.summarize(args.run, args.smoke)
    else:
        matched(args.run, args.smoke, args.resume)
