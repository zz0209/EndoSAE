import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_acknowledgement_sae as evaluation
import evaluate_source_support_components as support_api
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def conditions(parent_config):
    return {support + '__' + variant: dict(support=support, variant=variant, **spec)
            for support in parent_config['supports'] for variant, spec in support_api.variants(parent_config).items()}


def targeted(name, spec):
    return name if spec['random'] is None else name.rsplit('__random', 1)[0]


def verify_hash(path, identity):
    canonical = lambda value: os.path.normcase(str(Path(value).resolve()).removeprefix('\\\\?\\'))
    matches = [value for saved_path, value in identity.items() if canonical(saved_path) == canonical(path)]
    assert len(matches) == 1 and matches[0] == digest(path), path


def point_curve(curve, threshold):
    count = len(curve['threshold'])
    index = int(np.searchsorted(curve['threshold'], threshold, side='right') - 1)
    assert 0 <= index < count
    scalars = {'threshold', 'acknowledged_removed_seconds', 'acknowledged_suppression_fraction'}
    return {key: value[index:index + 1].copy() if key in scalars or (value.ndim == 2 and value.shape[0] == count)
            else value.copy() for key, value in curve.items()}


def video_data(base, consumer, video):
    detection = evaluation.shared.memory.acknowledgement.detection
    clips, receipt, frames, _, _ = detection.full_frame_metadata(Path(consumer['metadata']), video)
    records, _ = detection.load_predictions(Path(consumer['predictions']) / video / 'detections.jsonl',
                                           clips, receipt, camera_rate=True)
    with np.load(base / video / 'indices.npz', allow_pickle=False) as index:
        offsets = index['offsets'].copy()
        assert np.array_equal(index['frame_indices'], [record['frame_index'] for record in records])
        assert np.array_equal(np.diff(offsets), [len(record['detections']) for record in records])
    annotation = next(row for row in read_json(consumer['annotations'])['videos'] if row['video_id'] == video)
    first = {row['lesion_id']: row['first_frame'] for row in annotation['lesions']}
    paths = [base / 'config.json', base / video / 'indices.npz', Path(consumer['annotations']),
             Path(consumer['metadata']) / (video + '.json'), Path(consumer['metadata']) / (video + '.jsonl'),
             Path(consumer['predictions']) / video / 'detections.jsonl',
             Path(consumer['predictions']) / video / 'complete.json']
    return receipt, frames, records, offsets, first, support_api.hashes(paths)


def parent_curve(config, seed, video, episode_id, spec):
    parent = spec['parent'] or '0730'
    return (Path(config['parent_component_runs'][parent]) / 'evaluation' / f'seed{seed}' / 'development' /
            video / 'sources' / episode_id / (support_api.original_name(spec) + '__score_curve.npz'))


def progress(output, completed, total, phase, started, source=None):
    atomic_write_json(output / 'progress.json', dict(completed=completed, total=total, phase=phase,
        source=source, elapsed_seconds=time.perf_counter() - started))


def source_curves(config, seed, phase, destination, source, parent_source, specs, points,
                  video_assets, video_values, resume, global_identity):
    receipt, frames, records, offsets, first = video_values
    episode = parent_source['episode']
    episode_id, video = episode['episode_id'], source['video']
    source_path = Path(source['path'])
    score_path = source_path.parent / 'scores.npz'
    verify_hash(score_path, parent_source['outputs'])
    prior_video = {str(Path(path).resolve()): value for path, value in parent_source['identity']['video_inputs'].items()}
    assert all(prior_video[str(Path(path).resolve())] == value for path, value in video_assets.items())
    settings = read_json(config['evaluation_settings_file'])
    data_path = Path(settings[phase + '_base']) / video / 'sources' / episode_id / 'frame_data.npz'
    verify_hash(data_path, parent_source['identity']['files'])
    assets = [source_path, score_path, data_path]
    if phase == 'development':
        assets += [parent_curve(config, seed, video, episode_id, spec) for spec in specs.values()]
        for spec in specs.values():
            if spec['support'] == 'actual' and spec['random'] is None:
                verify_hash(parent_curve(config, seed, video, episode_id, spec), parent_source['identity']['files'])
    local_identity = dict(global_identity=global_identity, video_assets=video_assets, files=support_api.hashes(assets))
    target = destination / video / 'sources' / episode_id
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / 'summary.json'
    if summary_path.exists():
        saved = read_json(summary_path)
        assert resume and saved['status'] == 'COMPLETE' and saved['identity'] == local_identity
        assert saved['output_hashes'] == support_api.hashes([target / name for name in saved['output_names']])
        return saved, False
    scores = support_api.load_arrays(score_path)
    assert set(scores) == set(specs) and all(len(value) == offsets[-1] for value in scores.values())
    data = support_api.load_arrays(data_path)
    assert np.array_equal(data['frame'], [record['frame_index'] for record in records])
    cache, checks_by_condition, output_names, measures = {}, {}, [], {}
    for name, spec in specs.items():
        saved_path = target / (name + '__score_curve.npz')
        if phase == 'development':
            key = hashlib.sha256(scores[name].tobytes()).hexdigest()
            actual = 'actual__' + spec['variant']
            reused = None
            if np.array_equal(scores[name], scores[actual], equal_nan=True):
                reused = parent_curve(config, seed, video, episode_id, spec)
                shutil.copyfile(reused, saved_path)
                checks = dict(kind='PARENT_COMPLETE_CURVE', parent=str(reused), parent_sha256=digest(reused),
                              same_actual_scores_exact=True)
            elif key in cache:
                reused = cache[key]
                shutil.copyfile(reused, saved_path)
                checks = dict(kind='IDENTICAL_SAVED_SCORE_CURVE', parent=str(reused), parent_sha256=digest(reused))
            else:
                curve, verification = evaluation.shared.episode_curve(scores[name], offsets, records, frames,
                    data, episode['source_lesion_id'], first, receipt['fps'])
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                support_api.save_arrays(saved_path, curve)
                checks = dict(kind='SAVED_SCORE_FULL_CURVE', direct_mask_verification=verification)
            cache[key] = saved_path
        else:
            threshold = points[name]['threshold']
            keep = ~np.isfinite(scores[name]) | (scores[name] < threshold)
            key = hashlib.sha256(keep.tobytes()).hexdigest()
            if key in cache:
                curve, verification = cache[key]
            else:
                curve, verification, actual_keep = evaluation.fixed_curve(scores[name], threshold, offsets,
                    records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                assert np.array_equal(actual_keep, keep)
                for group, columns in episode['groups'].items():
                    curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                cache[key] = curve, verification
            support_api.save_arrays(saved_path, curve)
            checks = dict(kind='FROZEN_DEVELOPMENT_THRESHOLD', threshold=threshold, direct_mask_verification=verification)
            measures[name] = support_api.measures(curve, episode)
        checks_by_condition[name] = checks
        output_names.append(saved_path.name)
    result = dict(episode, status='COMPLETE', identity=local_identity, conditions=checks_by_condition,
                  fixed_results=measures, output_names=output_names,
                  output_hashes=support_api.hashes([target / name for name in output_names]))
    atomic_write_json(summary_path, result)
    pause_after_checkpoint(summary_path)
    return result, True


def verify_actual_aggregates(config, seed, destination, specs, smoke):
    parent = Path(config['parent_run'])
    original_identity = read_json(parent / 'evaluation' / f'seed{seed}' / 'identity.json')['files']
    checks = []
    for name, spec in specs.items():
        if spec['support'] != 'actual':
            continue
        component_run = Path(config['parent_component_runs'][spec['parent'] or '0730'])
        original_summary_path = component_run / 'summary.json'
        verify_hash(original_summary_path, original_identity)
        original_summary = read_json(original_summary_path)
        old_path = component_run / 'evaluation' / f'seed{seed}' / 'development' / (support_api.original_name(spec) + '__score_procedure_curve.npz')
        verify_hash(old_path, original_summary['inputs'])
        comparison = 'NOT_APPLICABLE_PARTIAL_SMOKE_SOURCE_SET'
        if not smoke:
            old = support_api.load_arrays(old_path)
            current = support_api.load_arrays(destination / (name + '__score_procedure_curve.npz'))
            assert set(old) == set(current)
            for key in old:
                assert np.array_equal(old[key], current[key], equal_nan=True) if old[key].dtype.kind in 'fc' else (
                    np.array_equal(old[key], current[key])), (name, key)
            comparison = 'ALL_AGGREGATE_ARRAYS_EXACT'
        checks.append(dict(condition=name, parent_curve=str(old_path), parent_curve_sha256=digest(old_path),
                           historical_aggregate_hash_verified=True, comparison=comparison))
    atomic_write_json(destination / 'actual_parent_aggregate_checks.json', dict(checks=checks,
        provenance='Target/reference source curves have0830 source-level historical hashes. Complete random curves retain their original aggregate-level historical hashes; formal actual aggregates are reproduced exactly.'))


def matched_controls(config, output, phase, sources, specs, points, video_cache, resume):
    destination = output / (phase + '_controls_at_target_threshold')
    random_names = [name for name, spec in specs.items() if spec['random'] is not None]
    videos = list(dict.fromkeys(row['video'] for row in sources))
    for video in videos:
        target_video = destination / video
        target_video.mkdir(parents=True, exist_ok=True)
        receipt, frames, records, offsets, first = video_cache[(phase, video)]
        episodes = []
        for source in [row for row in sources if row['video'] == video]:
            parent_source = read_json(source['path'])
            episode = parent_source['episode']
            episode_id = episode['episode_id']
            target = target_video / 'sources' / episode_id
            target.mkdir(parents=True, exist_ok=True)
            summary_path = target / 'summary.json'
            control_identity = dict(source_summary_sha256=digest(Path(source['path'])),
                source_curve_summary_sha256=digest(output / phase / video / 'sources' / episode_id / 'summary.json'),
                operating_points_sha256=digest(output / 'operating_points.json'))
            if summary_path.exists():
                assert resume
                saved = read_json(summary_path)
                assert saved['identity'] == control_identity
                assert saved['output_hashes'] == support_api.hashes([target / name for name in saved['output_names']])
                episodes.append(saved)
                continue
            scores = support_api.load_arrays(Path(source['path']).parent / 'scores.npz')
            settings = read_json(config['evaluation_settings_file'])
            data = support_api.load_arrays(Path(settings[phase + '_base']) / video / 'sources' / episode_id / 'frame_data.npz')
            outcomes, checks, policies = {}, {}, {}
            for name in random_names:
                threshold = points[targeted(name, specs[name])]['threshold']
                if phase == 'development':
                    full = support_api.load_arrays(output / phase / video / 'sources' / episode_id / (name + '__score_curve.npz'))
                    curve = point_curve(full, threshold)
                    curve['threshold'] = np.asarray([-2.])
                    verification = dict(kind='EXACT_STATE_FROM_COMPLETE_DEVELOPMENT_CURVE')
                else:
                    keep = ~np.isfinite(scores[name]) | (scores[name] < threshold)
                    key = hashlib.sha256(keep.tobytes()).hexdigest()
                    if key not in policies:
                        curve, verification, _ = evaluation.fixed_curve(scores[name], threshold, offsets,
                            records, frames, data, episode['source_lesion_id'], first, receipt['fps'])
                        for group, columns in episode['groups'].items():
                            curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                        policies[key] = curve, verification
                    curve, verification = policies[key]
                support_api.save_arrays(target / (name + '__score_curve.npz'), curve)
                outcomes[name] = support_api.measures(curve, episode)
                checks[name] = dict(threshold=threshold, verification=verification)
            output_names = [name + '__score_curve.npz' for name in random_names]
            result = dict(episode, status='COMPLETE', fixed_results=outcomes, conditions=checks,
                          identity=control_identity, output_names=output_names,
                          output_hashes=support_api.hashes([target / name for name in output_names]))
            atomic_write_json(summary_path, result)
            pause_after_checkpoint(summary_path)
            episodes.append(result)
        atomic_write_json(target_video / 'summary.json', dict(video=video, episodes=episodes, status='COMPLETE'))
    evaluation.shared.aggregate(destination, dict(source_variants=random_names, representations=['score']), videos)
    atomic_write_json(destination / 'summary.json', dict(status='COMPLETE', videos=videos,
        policy='Each random control evaluated at its corresponding targeted method development threshold.'))


def evaluate(config_path, seed, smoke, resume, stop_after):
    config = read_json(config_path)
    assert config['device'] == 'cpu' and config['random_controls'] == 3 and config['preserve_every_first_prompt']
    assert seed in config['seeds']
    run = support_api.output_path(config['run_dir'])
    root = run / 'smoke' if smoke else run
    output = root / 'evaluation' / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    parent = Path(config['parent_run'])
    parent_config = read_json(parent / 'config.json')
    parent_summary = read_json(parent / 'summary.json')
    assert parent_summary['status'] == 'COMPLETE'
    reference = [row for row in parent_summary['aggregates'] if row['population'] == 'development' and
                 row['support'] == 'actual' and row['variant'] == 'reference' and row['threshold_mode'] == 'fixed_reference']
    assert len(reference) == 1 and abs(config['retention_floor'] - reference[0]['retention']) < 1e-15
    assert config['supports'] == parent_config['supports']
    assert config['parent_component_runs'] == parent_config['parent_runs']
    assert config['evaluation_settings_file'] == parent_config['evaluation_settings_file']
    specs = conditions(parent_config)
    assert len(specs) == 39
    receipt_path = parent / 'evaluation' / f'seed{seed}' / 'summary.json'
    parent_receipt = read_json(receipt_path)
    assert parent_receipt['status'] == 'COMPLETE'
    sources = parent_receipt['source_summaries']
    if smoke:
        sources = [next(row for row in sources if row['population'] == phase) for phase in ('development', 'extension')]
    paths = [config_path, run / 'protocol.json', parent / 'config.json', parent / 'protocol.json', parent / 'summary.json',
        receipt_path, parent / 'evaluation' / f'seed{seed}' / 'identity.json', Path(__file__), Path(support_api.__file__),
        Path(evaluation.__file__), Path(evaluation.shared.__file__), Path(evaluation.shared.memory.__file__),
        Path(evaluation.shared.memory.acknowledgement.__file__),
        Path(evaluation.shared.memory.acknowledgement.detection.__file__), ROOT / 'src/checkpoint_io.py',
        ROOT / 'scripts/evaluate_fixed_acknowledgement_policy.py', Path(config['evaluation_settings_file'])]
    identity = dict(files=support_api.hashes(paths), seed=seed, smoke=smoke, python=sys.version,
                    numpy=np.__version__, torch=str(torch.__version__), threads=torch.get_num_threads(), device='cpu')
    identity_path = output / 'identity.json'
    if identity_path.exists():
        assert resume and read_json(identity_path) == identity
    else:
        atomic_write_json(identity_path, identity)
    identity_hash = digest(identity_path)
    settings = read_json(config['evaluation_settings_file'])
    started, completed, new_sources = time.perf_counter(), 0, 0
    points, video_cache = None, {}
    progress(output, completed, len(sources), 'DEVELOPMENT_CURVES', started)
    for phase in ('development', 'extension'):
        destination = output / phase
        phase_sources = [source for source in sources if source['population'] == phase]
        videos = list(dict.fromkeys(source['video'] for source in phase_sources))
        base = Path(settings[phase + '_base'])
        consumer = read_json(base / 'config.json')
        for video in videos:
            receipt, frames, records, offsets, first, video_assets = video_data(base, consumer, video)
            video_cache[(phase, video)] = receipt, frames, records, offsets, first
            episodes = []
            for source in [source for source in phase_sources if source['video'] == video]:
                parent_source = read_json(source['path'])
                assert parent_source['status'] == 'COMPLETE'
                result, created = source_curves(config, seed, phase, destination, source, parent_source, specs,
                    points, video_assets, video_cache[(phase, video)], resume, identity_hash)
                episodes.append(result)
                completed += 1
                new_sources += int(created)
                progress(output, completed, len(sources), phase.upper(), started, source['episode_id'])
                print('SUPPORT_TRADEOFF_SOURCE', seed, completed, len(sources), phase, source['episode_id'],
                      round(time.perf_counter() - started, 3), flush=True)
                if created and stop_after and new_sources >= stop_after:
                    raise SystemExit(75)
            atomic_write_json(destination / video / 'summary.json', dict(video=video, episodes=episodes, status='COMPLETE'))
        evaluation.shared.aggregate(destination, dict(source_variants=list(specs), representations=['score']), videos)
        atomic_write_json(destination / 'summary.json', dict(status='COMPLETE', videos=videos,
            interpretation='Previously examined development' if phase == 'development' else 'Previously examined extension'))
        if phase == 'development':
            verify_actual_aggregates(config, seed, destination, specs, smoke)
            candidate_points = evaluation.calibrate(output, list(specs), config['retention_floor'])
            points = candidate_points
            atomic_write_json(output / 'calibration_scope.json', dict(retention_floor=config['retention_floor'],
                population='development', videos=videos, all_first_prompts=True,
                tie_rule='Maximum removal among eligible points, then highest threshold within1e-12.',
                input_scores_frozen=True, fitted_parameters='One scalar threshold per support and variant.'))
        matched_controls(config, output, phase, phase_sources, specs, points, video_cache, resume)
    assert identity['files'] == support_api.hashes(paths)
    atomic_write_json(output / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE',
        seed=seed, sources=len(sources), operating_points=points, seconds=time.perf_counter() - started,
        identity_sha256=digest(identity_path), source_summaries=[dict(population=s['population'], video=s['video'],
            episode_id=s['episode_id'], path=str(output / s['population'] / s['video'] / 'sources' / s['episode_id'] / 'summary.json')) for s in sources],
        threshold_scope='Development only, one matched protection floor with every first prompt retained.',
        interpretation='Frozen-score oracle support tradeoff on previously examined procedures; no new representation or intervention selection.'))
    progress(output, len(sources), len(sources), 'COMPLETE', started)
    print('SUPPORT_PROTECTED_TRADEOFF_COMPLETE', seed, flush=True)


def summarize(config_path, smoke, resume):
    config = read_json(config_path)
    run = support_api.output_path(config['run_dir'])
    output = run / 'smoke' if smoke else run
    seeds = config['seeds'][:1] if smoke else config['seeds']
    parent_config = read_json(Path(config['parent_run']) / 'config.json')
    specs = conditions(parent_config)
    procedure_rows, source_rows, paths = [], [], [config_path, run / 'protocol.json', Path(__file__)]
    for seed in seeds:
        folder = output / 'evaluation' / f'seed{seed}'
        receipt = read_json(folder / 'summary.json')
        assert receipt['status'] == ('REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE')
        assert digest(folder / 'identity.json') == receipt['identity_sha256']
        paths += [folder / 'summary.json', folder / 'identity.json', folder / 'operating_points.json']
        points = receipt['operating_points']
        for phase in ('development', 'extension'):
            for name, spec in specs.items():
                for policy in ('own_protected_threshold', 'target_threshold') if spec['random'] is not None else ('own_protected_threshold',):
                    directory = folder / (phase + '_controls_at_target_threshold' if policy == 'target_threshold' else phase)
                    curve_path = directory / (name + '__score_procedure_curve.npz')
                    paths.append(curve_path)
                    threshold = points[targeted(name, spec) if policy == 'target_threshold' else name]['threshold']
                    with np.load(curve_path, allow_pickle=False) as curve:
                        index = int(np.searchsorted(curve['threshold'], threshold, side='right') - 1) if (
                            phase == 'development' and policy == 'own_protected_threshold') else 0
                        for video_index, video in enumerate(curve['videos'].tolist()):
                            values = {key.removesuffix('__procedure_values'): support_api.numeric(value[video_index, index])
                                      for key, value in curve.items() if key.endswith('__procedure_values')}
                            procedure_rows.append(dict(seed=seed, population=phase, video=video, support=spec['support'],
                                variant=spec['variant'], policy=policy, threshold=threshold,
                                removal=values['source_suppression'], retention=values['all_other__baseline_qualified_retention'],
                                first_prompt=values['all_other__first_baseline_frame_retention'], **values))
                    for video in read_json(directory / 'summary.json')['videos']:
                        for episode in read_json(directory / video / 'summary.json')['episodes']:
                            local = support_api.load_arrays(directory / video / 'sources' / episode['episode_id'] / (name + '__score_curve.npz'))
                            if phase == 'development' and policy == 'own_protected_threshold':
                                local = point_curve(local, threshold)
                            source_rows.append(dict(seed=seed, population=phase, video=video, episode_id=episode['episode_id'],
                                support=spec['support'], variant=spec['variant'], policy=policy, threshold=threshold,
                                **support_api.measures(local, episode)))
    inputs = support_api.hashes(paths)
    if (output / 'summary.json').exists():
        assert resume and read_json(output / 'summary.json')['inputs'] == inputs
        print('REUSE_SUPPORT_PROTECTED_TRADEOFF_SUMMARY', flush=True)
        return
    metrics = ('removal', 'retention', 'first_prompt')
    aggregates = []
    for key in dict.fromkeys((r['population'], r['support'], r['variant'], r['policy']) for r in procedure_rows):
        local = [r for r in procedure_rows if (r['population'], r['support'], r['variant'], r['policy']) == key]
        per_seed = {seed: {metric: support_api.finite_mean([r[metric] for r in local if r['seed'] == seed])
                           for metric in metrics} for seed in seeds}
        aggregates.append(dict(zip(('population', 'support', 'variant', 'policy'), key), seed_results=per_seed,
            **{metric: support_api.finite_mean([r[metric] for r in per_seed.values()]) for metric in metrics}))
    atomic_write_json(output / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE',
        inputs=inputs, seeds=seeds, retention_floor=config['retention_floor'], source_rows=source_rows,
        procedure_rows=procedure_rows, aggregates=aggregates, independent_unit='procedure',
        aggregation='Lesion then source then procedure, equal procedures within seed and equal seeds.',
        interpretation='GT support remains an oracle; threshold selection uses examined development only. Extension is previously examined.'))
    lines = ['# Matched-protection source support comparison', '',
        f"The development retention requirement is {config['retention_floor']:.16f}, with every first prompt retained. Each variant receives the same calibration rule; extension uses the selected threshold unchanged.", '',
        '| Population | Support | Variant | Threshold policy | Repeat removal | Other retention | First prompt |',
        '|---|---|---|---|---:|---:|---:|']
    for row in aggregates:
        if '__random' in row['variant']:
            continue
        values = ['NA' if row[name] is None else f'{100 * row[name]:.4f}%' for name in metrics]
        lines.append(f"| {row['population']} | {row['support']} | {row['variant']} | {row['policy']} | " + ' | '.join(values) + ' |')
    lines += ['', 'All random controls, thresholds, source/lesion outcomes and natural temporal strata are saved. GT support and both examined populations retain their original evidence scope.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_write_json(output / 'summary_progress.json', dict(completed=1, total=1, phase='COMPLETE'))
    print('SUPPORT_PROTECTED_TRADEOFF_SUMMARY_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('evaluate', 'summary'), required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-sources', type=int)
    args = parser.parse_args()
    torch.set_num_threads(read_json(args.config)['threads'])
    if args.phase == 'evaluate':
        if args.seed is None:
            parser.error('--seed is required for evaluate')
        evaluate(args.config, args.seed, args.smoke, args.resume, args.stop_after_sources)
    else:
        summarize(args.config, args.smoke, args.resume)


if __name__ == '__main__':
    main()
