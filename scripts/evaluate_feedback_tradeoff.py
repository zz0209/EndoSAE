import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_source_component_transfer import inputs, application
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def measures(curve, columns):
    values = curve['baseline_qualified_retention'][:, columns]
    count = np.isfinite(values).sum(axis=1)
    retention = np.divide(np.nansum(values, axis=1), count,
                          out=np.full(len(count), np.nan), where=count > 0)
    first = curve['first_postclick_correct_prompt_time'][:, columns]
    baseline = curve['baseline_first_postclick_correct_prompt_time'][columns]
    protected = np.all(~np.isfinite(baseline)[None] | np.isclose(first, baseline[None], atol=1e-9, rtol=0), axis=1)
    return dict(repeat=curve['acknowledged_suppression_fraction'], other=retention, protected=protected)


def choose(values, eligible, objective, secondary):
    candidates = np.flatnonzero(eligible & np.isfinite(values[objective]))
    if not len(candidates):
        return None
    maximum = values[objective][candidates].max()
    candidates = candidates[values[objective][candidates] >= maximum - 1e-12]
    second = values[secondary][candidates]
    if np.isfinite(second).any():
        candidates = candidates[second >= np.nanmax(second) - 1e-12]
    return int(candidates[-1])


def comparison(curves, episode, choice):
    columns = np.asarray(episode['groups']['all_other'], dtype=int)
    values = {name: measures(curve, columns) for name, curve in curves.items()}
    indices = {name: int(np.searchsorted(curve['threshold'], choice['threshold'], side='right') - 1)
               for name, curve in curves.items()}
    before, edited = values['unchanged'], values['single_coordinate']
    original_index, edited_index = indices['unchanged'], indices['single_coordinate']
    target = edited['repeat'][edited_index]
    matched = choose(before, before['protected'] & (before['repeat'] >= target - 1e-12), 'other', 'repeat')
    floor = before['other'][original_index]
    operating = {name: choose(value, value['protected'] & (value['other'] >= floor - 1e-12), 'repeat', 'other')
                 for name, value in values.items()}
    zero = {name: choose(value, value['protected'] & (value['other'] >= 1 - 1e-10), 'repeat', 'other')
            for name, value in values.items()}
    assert all(i is not None for i in operating.values())
    assert all(i is not None for i in zero.values())
    assert before['protected'][original_index] and edited['protected'][edited_index]
    method, seed = choice['model'].rsplit('_seed', 1)
    row = dict(method=method, seed=int(seed), video=choice['video'], episode=choice['episode'],
        original_repeat=float(before['repeat'][original_index]), original_other=float(floor),
        edited_repeat=float(target), edited_other=float(edited['other'][edited_index]),
        matched_feasible=matched is not None, matched_index=matched,
        matched_repeat=None if matched is None else float(before['repeat'][matched]),
        matched_other=None if matched is None else float(before['other'][matched]),
        matched_threshold=None if matched is None else float(curves['unchanged']['threshold'][matched]),
        protection_gain=None if matched is None else float(edited['other'][edited_index] - before['other'][matched]))
    for name in curves:
        i = operating[name]
        row[name + '_matched_retention_repeat'] = float(values[name]['repeat'][i])
        row[name + '_matched_retention_other'] = float(values[name]['other'][i])
        row[name + '_matched_retention_threshold'] = float(curves[name]['threshold'][i])
        row[name + '_zero_loss_repeat'] = float(values[name]['repeat'][zero[name]])
    row['ranking_gain'] = row['single_coordinate_matched_retention_repeat'] - row['unchanged_matched_retention_repeat']
    row['zero_loss_gain'] = row['single_coordinate_zero_loss_repeat'] - row['unchanged_zero_loss_repeat']
    points = {name: {indices[name], operating[name], zero[name]} for name in curves}
    if matched is not None:
        points['unchanged'].add(matched)
    return row, points


def evaluate(run, smoke, resume):
    config = read_json(run / 'config.json')
    prior = Path(config['feedback_run'])
    _, _, event, _, original = inputs(prior)
    choices = [c for c in read_json(prior / 'selection/summary.json')['rows'] if c['feedback'] is not None]
    if smoke:
        choices = [c for c in choices if c['video'] == config['smoke_video'] and c['model'].endswith(str(config['smoke_seed']))]
    assert choices
    output = run / ('smoke' if smoke else 'evaluation')
    output.mkdir(exist_ok=resume)
    paths = [run / 'config.json', run / 'protocol.json', prior / 'selection/summary.json', prior / 'scores/summary.json',
             Path(__file__), Path(application.__file__), Path(application.shared.__file__), Path(application.shared.memory.__file__)]
    identity = dict(files={str(p): digest(p) for p in paths}, smoke=smoke)
    if (output / 'identity.json').exists():
        assert resume and read_json(output / 'identity.json') == identity
    if (output / 'summary.json').exists():
        assert resume
        return
    atomic_write_json(output / 'identity.json', identity)
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    rows, completed, direct_count = [], 0, 0
    start = time.perf_counter()
    for video in sorted({c['video'] for c in choices}):
        receipt, frames, records, _, offsets, _, _, first = application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for choice in (c for c in choices if c['video'] == video):
            key, identifier = choice['model'], choice['episode']
            target = output / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            score_paths = {name: prior / 'scores' / video / key / identifier / (name + '.npy') for name in config['policies']}
            data_path = base / video / 'sources' / identifier / 'frame_data.npz'
            local_identity = dict(scores={n: digest(p) for n, p in score_paths.items()}, frame_data=digest(data_path))
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == local_identity
                rows.append(saved['row'])
                direct_count += saved['direct_checks']
            else:
                episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
                with np.load(data_path, allow_pickle=False) as saved:
                    data = {name: saved[name] for name in saved.files}
                data['known'] &= data['frame'] > choice['feedback']['frame']
                curves, checks = {}, {}
                for name, path in score_paths.items():
                    curve_path = target / (name + '_curve.npz')
                    check_path = target / (name + '_checks.json')
                    if resume and check_path.exists():
                        saved_check = read_json(check_path)
                        assert saved_check['identity'] == local_identity
                        assert saved_check['curve_sha256'] == digest(curve_path)
                        with np.load(curve_path, allow_pickle=False) as saved:
                            curves[name] = {k: saved[k] for k in saved.files}
                        checks[name] = saved_check['checks']
                        continue
                    scores = np.load(path, allow_pickle=False)
                    curve, verification = application.shared.episode_curve(scores, offsets, records, frames, data,
                        episode['source_lesion_id'], first, receipt['fps'])
                    index = int(np.searchsorted(curve['threshold'], choice['threshold'], side='right') - 1)
                    with np.load(prior / 'evaluation' / video / key / identifier / (name + '_after_feedback_curve.npz')) as fixed:
                        for field in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                      'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                            np.testing.assert_allclose(curve[field][index], fixed[field][0], atol=1e-9, rtol=0, equal_nan=True)
                    curves[name], checks[name] = curve, verification
                    np.savez_compressed(curve_path, **curve)
                    atomic_write_json(check_path, dict(identity=local_identity, checks=verification, curve_sha256=digest(curve_path)))
                    print('TRADEOFF_CURVE', completed * 2 + len(curves), '/', len(choices) * 2, video, key, identifier, name, flush=True)
                row, points = comparison(curves, episode, choice)
                chosen_checks = []
                for name, indices in points.items():
                    scores = np.load(score_paths[name], allow_pickle=False)
                    for index in sorted(indices):
                        fixed, _, _ = application.fixed_curve(scores, float(curves[name]['threshold'][index]), offsets,
                            records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                        for field in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                      'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                            np.testing.assert_allclose(curves[name][field][index], fixed[field][0], atol=1e-9, rtol=0, equal_nan=True)
                        chosen_checks.append(dict(policy=name, index=index, direct_fixed_replay=True))
                count = sum(len(v) for v in checks.values()) + len(chosen_checks)
                atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=local_identity, row=row,
                    direct_checks=count, selected_checks=chosen_checks))
                rows.append(row)
                direct_count += count
            completed += 1
            atomic_write_json(output / 'progress.json', dict(completed=completed, total=len(choices), unit='source-model pairs'))
            pause_after_checkpoint(output / 'progress.json')
    assert identity['files'] == {str(p): digest(p) for p in paths}
    pd.DataFrame(rows).to_csv(output / 'sources.csv', index=False)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', seconds=time.perf_counter() - start,
        source_models=len(rows), curves=len(rows) * 2, direct_checks=direct_count,
        python=sys.version, numpy=np.__version__, threads=1, gpu_used=False, identity=identity))


def summarize(run, smoke):
    root = run / ('smoke' if smoke else 'evaluation')
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    sources = pd.read_csv(root / 'sources.csv')
    metrics = ['original_repeat', 'original_other', 'edited_repeat', 'edited_other', 'matched_repeat', 'matched_other',
               'protection_gain', 'ranking_gain', 'zero_loss_gain', 'unchanged_matched_retention_repeat',
               'single_coordinate_matched_retention_repeat', 'unchanged_zero_loss_repeat', 'single_coordinate_zero_loss_repeat']
    procedures = sources.groupby(['method', 'seed', 'video'])[metrics].mean().reset_index()
    seeds = procedures.groupby(['method', 'seed'])[metrics].mean().reset_index()
    summary = seeds.groupby('method')[metrics].mean().reset_index()
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    for name, frame in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        frame.to_csv(output / (name + '.csv'), index=False)
    feasibility = sources.groupby('method')['matched_feasible'].agg(['sum', 'count']).reset_index()
    feasibility.to_csv(output / 'feasibility.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, metric, title in zip(axes, ['protection_gain', 'ranking_gain', 'zero_loss_gain'],
        ['Fixed edit vs. original threshold control\nAt least the same repeat removal',
         'Edited vs. original attainable repeat removal\nAt least original other-prompt retention',
         'Edited vs. original attainable repeat removal\nEvery other prompt retained']):
        for i, (method, frame) in enumerate(seeds.groupby('method')):
            values = frame[metric].to_numpy() * 100
            axis.scatter(np.full(len(values), i) + np.linspace(-.1, .1, len(values)), values, color='#0072B2')
            axis.plot([i - .2, i + .2], [np.nanmean(values)] * 2, color='#D55E00', linewidth=2)
        axis.axhline(0, color='black', linewidth=.8)
        axis.set_xticks([0, 1], ['Dense', 'SAE'])
        axis.set_title(title, fontsize=10)
        axis.set_ylabel('Difference (percentage points)')
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Prior-reference component edits | Matched behavioral tradeoffs' + (' | Smoke' if smoke else ''))
    fig.text(.02, .025, 'Exact source-specific threshold points selected with exposed development labels; no interpolation.\n'
             'Dots: seeds; bars: seed mean; equal procedure weighting. First prompts must remain protected. See feasibility.csv.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(output / 'tradeoff.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', results=summary.to_dict('records'),
        feasibility=feasibility.to_dict('records'), source_sha256=digest(__file__), input_sha256=digest(root / 'summary.json'),
        interpretation='Post-hoc source-specific label-informed tradeoff diagnostic; all data previously examined.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summary'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
