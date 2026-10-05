import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_acknowledgement_sae as application
from evaluate_token_causal_prompting import methods
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def pick(removal, feasible):
    eligible = np.flatnonzero(feasible & np.isfinite(removal))
    assert len(eligible)
    maximum = np.max(removal[eligible])
    return int(eligible[np.flatnonzero(removal[eligible] >= maximum - 1e-12)[-1]])


def mean(values):
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def selected_methods(config, app_config):
    available = ['reference_supcon'] + methods(app_config)
    names = config.get('score_methods', available)
    assert len(names) == len(set(names)) and set(names) <= set(available)
    return names


def curve_path(original, replay, population, video, episode_id, name):
    base = original / 'development' if population == 'development' else replay
    return base / video / 'sources' / episode_id / (name + '__score_curve.npz')


def analyze(config, app_config, original, replay, videos, population, names, seed):
    aggregate_root = original / 'development' if population == 'development' else replay
    rows, episodes = [], []
    for name in names:
        curves = {}
        for video in videos:
            summary = read_json(aggregate_root / video / 'summary.json')
            for episode in summary['episodes']:
                if not episode['click']['available']:
                    continue
                path = curve_path(original, replay, population, video, episode['episode_id'], name)
                with np.load(path, allow_pickle=False) as saved:
                    curve = {key: saved[key].copy() for key in saved.files}
                columns = np.asarray(episode['groups']['all_other'], dtype=int)
                other = curve['baseline_qualified_retention'][:, columns]
                first = curve['first_baseline_frame_retention'][:, columns]
                zero = np.all(~np.isfinite(other) | (other >= 1 - 1e-10), axis=1)
                zero &= np.all(~np.isfinite(first) | (first >= 1 - 1e-10), axis=1)
                index = pick(curve['acknowledged_suppression_fraction'], zero)
                episodes.append(dict(seed=seed, population=population, method=name, video=video,
                    episode_id=episode['episode_id'], threshold=float(curve['threshold'][index]),
                    removal=float(curve['acknowledged_suppression_fraction'][index]),
                    retention=mean(other[index]), first_prompt=mean(first[index]),
                    source_baseline_seconds=float(curve['baseline_seconds'][list(curve['lesion_ids']).index(episode['source_lesion_id'])]),
                    defined_other_lesions=int(np.isfinite(other[index]).sum())))
                curves[episode['episode_id']] = (curve, zero)
        with np.load(aggregate_root / (name + '__score_procedure_curve.npz'), allow_pickle=False) as aggregate:
            grid = aggregate['threshold']
            feasible = np.ones(len(grid), dtype=bool)
            for curve, zero in curves.values():
                positions = np.searchsorted(curve['threshold'], grid, side='right') - 1
                assert np.all(positions >= 0)
                feasible &= zero[positions]
            index = pick(aggregate['source_suppression__procedure_mean'], feasible)
            retention = aggregate['all_other__baseline_qualified_retention__procedure_mean']
            first = aggregate['all_other__first_baseline_frame_retention__procedure_values']
            qualified = (retention >= config['protection_floor'] - 1e-9)
            qualified &= np.all(~np.isfinite(first) | (first >= 1 - 1e-9), axis=0)
            calibrated = pick(aggregate['source_suppression__procedure_mean'], qualified)
            for number, video in enumerate(aggregate['videos'].tolist()):
                local = [row for row in episodes if row['method'] == name and row['video'] == video]
                source_specific = mean(np.asarray([row['removal'] for row in local]))
                shared = float(aggregate['source_suppression__procedure_values'][number, index])
                assert source_specific is None or source_specific >= shared - 1e-9
                rows.append(dict(seed=seed, population=population, method=name, video=video,
                    shared_zero_removal=shared, source_zero_removal=source_specific,
                    calibration_gap=None if source_specific is None else source_specific - shared,
                    shared_zero_threshold=float(grid[index]), posthoc_common_threshold=float(grid[calibrated]),
                    posthoc_common_removal=float(aggregate['source_suppression__procedure_values'][number, calibrated]),
                    posthoc_common_retention=float(aggregate['all_other__baseline_qualified_retention__procedure_values'][number, calibrated])))
    return rows, episodes


def evaluate(run, seed, smoke, resume):
    config = read_json(run / 'config.json')
    app_run = Path(config['application_run'])
    app_config = read_json(app_run / 'config.json')
    assert seed in config['seeds']
    original = app_run / 'evaluation' / f'seed{seed}'
    assert read_json(original / 'summary.json')['status'] == 'COMPLETE'
    names = selected_methods(config, app_config)
    output = run / ('smoke' if smoke else 'evaluation') / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=resume)
    paths = [run / 'config.json', run / 'protocol.json', app_run / 'config.json', original / 'summary.json',
             Path(__file__), Path(application.__file__), Path(application.shared.__file__), Path(application.shared.memory.__file__)]
    identity = dict(seed=seed, smoke=smoke, files={str(p): digest(p) for p in paths})
    if (output / 'identity.json').exists():
        assert resume and read_json(output / 'identity.json') == identity
    if (output / 'summary.json').exists():
        assert resume
        return
    atomic_write_json(output / 'identity.json', identity)
    started = time.perf_counter()
    base = Path(app_config['extension_base'])
    settings = read_json(base / 'config.json')
    videos = app_config['extension_videos'][:1] if smoke else app_config['extension_videos']
    replay = output / 'extension'
    replay.mkdir(exist_ok=True)
    for number, video in enumerate(videos):
        if not (replay / video / 'summary.json').exists():
            receipt, frames, records, directory, offsets, raw, available, first = application.load_video(base, settings, video)
            episodes = []
            for episode in read_json(original / 'extension' / video / 'summary.json')['episodes']:
                target = replay / video / 'sources' / episode['episode_id']
                target.mkdir(parents=True, exist_ok=True)
                if (target / 'summary.json').exists():
                    assert resume
                    episodes.append(read_json(target / 'summary.json'))
                    continue
                if not episode['click']['available']:
                    atomic_write_json(target / 'summary.json', episode)
                    episodes.append(episode)
                    continue
                prior = original / 'extension' / video / 'sources' / episode['episode_id']
                data_path = base / video / 'sources' / episode['episode_id'] / 'frame_data.npz'
                with np.load(data_path) as saved:
                    data = {key: saved[key].copy() for key in saved.files}
                verification = {}
                with np.load(prior / 'scores.npz') as saved:
                    for name in names:
                        curve, checks = application.shared.episode_curve(saved[name], offsets, records, frames, data,
                            episode['source_lesion_id'], first, receipt['fps'])
                        for group, columns in episode['groups'].items():
                            curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                        threshold = read_json(original / 'operating_points.json')['methods'][name]['threshold']
                        index = int(np.searchsorted(curve['threshold'], threshold, side='right') - 1)
                        with np.load(prior / (name + '__score_curve.npz')) as fixed:
                            for key in ['acknowledged_suppression_fraction', 'retained_baseline_seconds',
                                        'first_postclick_correct_prompt_time', 'baseline_qualified_retention']:
                                np.testing.assert_allclose(curve[key][index], fixed[key][0], atol=1e-10, rtol=0)
                        if name == 'reference_supcon':
                            reference_path = base / video / 'sources' / episode['episode_id'] / 'track__supcon_l2_curve.npz'
                            with np.load(reference_path) as reference:
                                for key in ['threshold', 'acknowledged_removed_seconds', 'retained_baseline_seconds']:
                                    np.testing.assert_array_equal(curve[key], reference[key])
                        np.savez_compressed(target / (name + '__score_curve.npz'), **curve)
                        verification[name] = dict(original_fixed_point_reproduced=True, direct_masks=checks)
                result = dict(episode, posthoc_curve_checks=verification,
                              score_sha256=digest(prior / 'scores.npz'), frame_data_sha256=digest(data_path))
                atomic_write_json(target / 'summary.json', result)
                episodes.append(result)
                print('SCORE_DIAGNOSIS', seed, video, episode['episode_id'], flush=True)
            atomic_write_json(replay / video / 'summary.json', dict(video=video, episodes=episodes, status='COMPLETE'))
        atomic_write_json(output / 'progress.json', dict(completed=number + 1, total=len(videos), video=video))
    application.shared.aggregate(replay, dict(source_variants=names, representations=['score']), videos)
    rows, episodes = [], []
    for population in ['extension'] if smoke else ['development', 'extension']:
        selected = videos if population == 'extension' else app_config['development_videos']
        values, local = analyze(config, app_config, original, replay, selected, population, names, seed)
        rows += values
        episodes += local
    assert identity['files'] == {str(p): digest(p) for p in paths}
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', smoke=smoke, seed=seed,
        seconds=time.perf_counter() - started, procedures=rows, episodes=episodes,
        interpretation='Post-hoc label-informed diagnostic; no deployable threshold or application gain.'))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    config = read_json(run / 'config.json')
    seeds = config['seeds'][:1] if smoke else config['seeds']
    summaries = [read_json(run / ('smoke' if smoke else 'evaluation') / f'seed{seed}' / 'summary.json') for seed in seeds]
    assert all(s['status'] == 'COMPLETE' for s in summaries)
    table = pd.DataFrame([row for s in summaries for row in s['procedures']])
    episodes = pd.DataFrame([row for s in summaries for row in s['episodes']])
    metrics = ['shared_zero_removal', 'source_zero_removal', 'calibration_gap', 'posthoc_common_removal', 'posthoc_common_retention']
    per_seed = table.groupby(['population', 'method', 'seed'])[metrics].mean().reset_index()
    averages = per_seed.groupby(['population', 'method'])[metrics].mean().reset_index()
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'procedures.csv', index=False)
    episodes.to_csv(output / 'sources.csv', index=False)
    per_seed.to_csv(output / 'seeds.csv', index=False)
    averages.to_csv(output / 'summary.csv', index=False)
    populations = ['extension'] if smoke else ['development', 'extension']
    app_config = read_json(Path(config['application_run']) / 'config.json')
    names = selected_methods(config, app_config)
    default_labels = dict(zip(['reference_supcon'] + methods(app_config),
        ['Ordinary SupCon', 'Projected sparse', 'Projected dense', 'Direct sparse', 'Direct dense']))
    label_map = config.get('method_labels', app_config.get('method_labels', default_labels))
    labels = [label_map[name] for name in names]
    fig, axes = plt.subplots(len(populations), 2, figsize=(12, 4 * len(populations)), squeeze=False, sharey=True)
    for i, population in enumerate(populations):
        for j, metric in enumerate(['shared_zero_removal', 'source_zero_removal']):
            for number, name in enumerate(names):
                data = per_seed[(per_seed.population == population) & (per_seed.method == name)]
                for k, seed in enumerate(seeds):
                    value = data[data.seed == seed][metric].iloc[0]
                    axes[i, j].scatter(value * 100, number + (k - (len(seeds) - 1) / 2) * .12,
                        color=['#0072B2', '#D55E00', '#009E73'][k], label=str(seed) if i == j == number == 0 else None)
            axes[i, j].set_yticks(range(len(names)), labels)
            axes[i, j].set_xlabel('Repeat prompt removal (%)')
            axes[i, j].set_title(population + ' | ' + ('shared threshold' if j == 0 else 'source-specific thresholds'))
            axes[i, j].grid(axis='x', alpha=.2)
            axes[i, j].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Label-informed score feasibility | every other-lesion prompt retained' + (' | smoke' if smoke else ''))
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(.62, .95), ncol=len(seeds), frameon=False)
    fig.text(.02, .02, 'Post-hoc diagnostic on exposed data. Labels select all shown thresholds. Equal procedure weighting.\n'
             'These values are attainable score separation, not a deployable policy or an independent result.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .90))
    fig.savefig(output / 'score_feasibility.png', dpi=170)
    plt.close(fig)
    lines = ['# Score feasibility diagnosis', '', 'All thresholds in this report use evaluation labels. These are descriptive diagnostics.', '',
        '| Population | Method | Shared zero-loss removal (%) | Source-specific zero-loss removal (%) | Gap (pp) |',
        '|---|---|---:|---:|---:|']
    for row in averages.to_dict('records'):
        lines.append(f"| {row['population']} | {row['method']} | {row['shared_zero_removal'] * 100:.4f} | {row['source_zero_removal'] * 100:.4f} | {row['calibration_gap'] * 100:.4f} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', smoke=smoke, results=averages.to_dict('records'),
        source_sha256=digest(__file__), inputs={str(run / ('smoke' if smoke else 'evaluation') / f'seed{s}' / 'summary.json'):
            digest(run / ('smoke' if smoke else 'evaluation') / f'seed{s}' / 'summary.json') for s in seeds}))
    atomic_write_json(run / 'summary_progress.json', dict(completed=1, total=1))


def prepare(run):
    config = read_json(run / 'config.json')
    measured = read_json(run / 'smoke' / f"seed{config['seeds'][0]}" / 'summary.json')['seconds']
    python = str(ROOT / 'artifacts/environments/modern/Scripts/python.exe')
    stages = []
    for seed in config['seeds']:
        stages.append(dict(id=f'diagnose_{seed}', label=f'种子 {seed} · 固定评分的区分能力', kind='diagnose',
            command=[python, str(Path(__file__)), '--run', str(run), '--phase', 'evaluate', '--seed', str(seed), '--resume'],
            progress=str(run / 'evaluation' / f'seed{seed}' / 'progress.json'),
            output=str(run / 'evaluation' / f'seed{seed}' / 'summary.json'), units=5, unit='个扩展 procedure',
            estimate_seconds=measured * 5, resources=['disk-e-io']))
    stages.append(dict(id='summary', label='逐确认对象、procedure与全部种子图表', kind='summary',
        command=[python, str(Path(__file__)), '--run', str(run), '--phase', 'summary'],
        progress=str(run / 'summary_progress.json'), output=str(run / 'analysis/summary.json'),
        units=1, unit='组结果', estimate_seconds=10, resources=[]))
    assert not (run / 'batch_plan.json').exists()
    atomic_write_json(run / 'batch_plan.json', dict(question=read_json(run / 'protocol.json')['question'], stages=stages))
    print('BATCH_ESTIMATE_SECONDS', sum(s['estimate_seconds'] for s in stages))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summary', 'prepare'], required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'evaluate':
        evaluate(args.run, args.seed, args.smoke, args.resume)
    elif args.phase == 'summary':
        summarize(args.run, args.smoke)
    else:
        prepare(args.run)
