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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import encode_causal_detection_identity as reference_api
import evaluate_acknowledgement_sae as evaluation
from evaluate_region_identity_memory import METRICS, finite_mean
from src import token_memory_edit as model_api
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


SETTINGS_ROOT = ROOT / 'results/runs/20260928T0515Z_sae_acknowledgement_v1'
REFERENCE_FIT = ROOT / 'results/runs/20260927T1205Z_available_identity_supcon_dev_v1/fit'
METHODS = ('sparse_edit', 'dense_edit')
CONTROL_COUNT = 3


def raw_key(raw):
    values = np.asarray(raw, dtype='<f8').reshape(-1)
    assert values.shape == (768,) and np.isfinite(values).all()
    return hashlib.sha256(values.tobytes()).hexdigest()


class FixedSourceMemory:
    def __init__(self, reference, memories):
        self.reference = reference
        self.memories = memories

    def encode(self, raw, batch_size=512):
        values = np.asarray(raw).reshape(-1, 768)
        batches = [self.reference(values[start:start + batch_size])['supcon_l2']
                   for start in range(0, len(values), batch_size)]
        return np.concatenate(batches) if batches else np.empty((0, 128), dtype=np.float32)

    def memory(self, raw):
        saved_raw, memory = self.memories[raw_key(raw)]
        assert np.array_equal(np.asarray(raw).reshape(-1), saved_raw)
        return memory.copy()

    @staticmethod
    def score_encoded(memory, codes):
        return (codes @ memory).astype(np.float32)

    def score(self, source, queries, batch_size=512):
        return self.score_encoded(self.memory(source), self.encode(queries, batch_size))


def conditions():
    return list(METHODS) + ['spatial_weighted'] + [
        f'{method}__random{number}' for method in METHODS for number in range(CONTROL_COUNT)]


def source_records(config, settings, smoke):
    records = []
    for phase in ('development', 'extension'):
        base = Path(settings[phase + '_base'])
        population = read_json(base / 'config.json')
        videos = settings[phase + '_videos'][:1] if smoke else settings[phase + '_videos']
        for video in videos:
            episodes = read_json(base / video / 'summary.json')['episodes']
            for episode in episodes[:1] if smoke else episodes:
                source = Path(config['data_root']) / 'sources' / phase / video / episode['episode_id']
                complete = read_json(source / 'complete.json')
                assert complete['status'] in ('COMPLETE', 'UNAVAILABLE')
                original = Path(population['descriptors']) / video / 'sources' / episode['episode_id'] / 'track_memory.npz'
                records.append(dict(population=phase, video=video, episode=episode, source=source,
                                    original=original, available=complete['status'] == 'COMPLETE'))
    return records


def random_delta(predictor, tokens, mask, learned, seed, source_id, method, number):
    mean_codes = np.asarray(predictor.encode_tokens(tokens[mask])).mean(axis=0, dtype=np.float64)
    gains = np.asarray(predictor.source_gains, dtype=np.float64).reshape(-1)
    eligible = np.flatnonzero(mean_codes > 0)
    assert len(gains) == len(mean_codes)
    stream = int.from_bytes(hashlib.sha256(f'{seed}:{source_id}:{method}:{number}'.encode()).digest()[:8], 'little')
    order = np.random.default_rng(stream).permutation(len(eligible))
    permuted = np.zeros_like(gains)
    permuted[eligible] = gains[eligible][order]
    assert np.array_equal(np.sort(permuted[eligible]), np.sort(gains[eligible]))
    direction = np.asarray(predictor.source_delta(tokens, mask, gain_override=permuted), dtype=np.float64)
    norm, target = float(np.linalg.norm(direction)), float(np.linalg.norm(learned))
    if target == 0:
        delta, scale, status = np.zeros_like(learned), 0., 'ZERO_LEARNED_EDIT'
    elif norm <= 1e-12:
        delta, scale, status = np.zeros_like(learned), None, 'UNDEFINED_ZERO_DIRECTION'
    else:
        scale = target / norm
        delta, status = direction * scale, 'DEFINED'
        assert np.isclose(np.linalg.norm(delta), target, rtol=1e-10, atol=1e-12)
    return delta, permuted, dict(status=status, derived_seed=stream, eligible_components=len(eligible),
        effective_nonzero_gains=int(np.count_nonzero(gains[eligible])), raw_direction_norm=norm,
        learned_delta_norm=target, applied_delta_norm=float(np.linalg.norm(delta)), raw_scale=scale,
        gain_value_multiset_preserved=True, resampled=False)


def build_memories(records, models, baseline, reference, output, seed):
    with np.load(baseline / 'normalization.npz', allow_pickle=False) as saved:
        mean, scale = saved['mean'].copy(), saved['scale'].copy()
    with np.load(baseline / 'model.npz', allow_pickle=False) as saved:
        weight, bias = saved['weight'].reshape(-1).copy(), float(saved['bias'].reshape(-1)[0])
    assert mean.shape == scale.shape == weight.shape == (768,) and np.all(scale > 0)
    mappings = {name: {} for name in conditions()}
    seen = {}
    diagnostics = []
    for index, record in enumerate(records):
        source, episode = record['source'], record['episode']
        if not episode['click']['available']:
            assert not record['available']
            diagnostics.append(dict(population=record['population'], video=record['video'],
                episode_id=episode['episode_id'], status='CLICK_UNAVAILABLE'))
            continue
        with np.load(record['original'], allow_pickle=False) as saved:
            original = saved['raw_mean'].reshape(-1).copy()
            available = bool(saved['available'])
        if not record['available']:
            assert not available, (episode['episode_id'], 'Available source lacks token cache')
            diagnostics.append(dict(population=record['population'], video=record['video'],
                episode_id=episode['episode_id'], status='SOURCE_UNAVAILABLE'))
            continue
        assert available
        with np.load(source / 'tokens.npz', allow_pickle=False) as saved:
            tokens, mask = saved['tokens'].copy(), saved['mask'].astype(bool)
            assert np.array_equal(original, saved['original_raw'].reshape(-1))
            cached_pool = saved['pooled_raw'].reshape(-1).copy()
        assert tokens.shape == (8, 196, 768) and mask.shape == (8, 196) and mask.any()
        selected = tokens[mask].astype(np.float64)
        pooled = selected.mean(axis=0)
        assert np.allclose(pooled, cached_pool, atol=1e-10, rtol=0)
        key = raw_key(original)
        content = hashlib.sha256(tokens.tobytes() + mask.tobytes()).hexdigest()
        if key in seen:
            assert seen[key] == content, 'Identical source descriptor has different token inputs'
        seen[key] = content
        arrays = dict(original_raw=original, pooled_raw=pooled, source_mask=mask)
        zero_checks, controls, raws = {}, {}, {}
        for method, predictor in models.items():
            zero = np.asarray(predictor.source_delta(tokens, mask,
                gain_override=np.zeros_like(predictor.source_gains)), dtype=np.float64)
            assert np.array_equal(zero, np.zeros(768))
            assert np.array_equal(original + zero, original)
            zero_memory = reference((original + zero)[None])['supcon_l2'][0]
            original_memory = reference(original[None])['supcon_l2'][0]
            assert np.array_equal(zero_memory, original_memory)
            zero_checks[method] = dict(raw_max_error=0., reference_memory_max_error=0.)
            delta = np.asarray(predictor.source_delta(tokens, mask), dtype=np.float64)
            assert delta.shape == (768,) and np.isfinite(delta).all()
            raws[method] = original + delta
            arrays[method + '__delta'] = delta
            arrays[method + '__gains'] = np.asarray(predictor.source_gains)
            for number in range(CONTROL_COUNT):
                name = f'{method}__random{number}'
                control_delta, gains, info = random_delta(predictor, tokens, mask, delta,
                    seed, episode['episode_id'], method, number)
                raws[name], controls[name] = original + control_delta, info
                arrays[name + '__delta'], arrays[name + '__gains'] = control_delta, gains
        logits = ((selected - mean) / scale) @ weight + bias
        weights = np.clip(expit(logits), np.finfo(np.float64).tiny, 1.)
        weights /= weights.sum()
        weighted = weights @ selected
        equal = np.ones(len(selected), dtype=np.float64)
        equal_pool = (selected * equal[:, None]).sum(axis=0) / equal.sum()
        assert np.array_equal(original + (equal_pool - pooled), original)
        raws['spatial_weighted'] = original + (weighted - pooled)
        arrays['spatial_weighted__weights'] = weights
        arrays['spatial_weighted__delta'] = weighted - pooled
        for name, raw in raws.items():
            memory = reference(raw[None])['supcon_l2'][0]
            assert np.isfinite(memory).all() and np.linalg.norm(memory) > 0
            if key in mappings[name]:
                assert np.array_equal(mappings[name][key][1], memory)
            mappings[name][key] = (original, memory)
            arrays[name + '__raw'], arrays[name + '__memory'] = raw, memory
        target = output / 'source_edits' / record['population'] / record['video'] / episode['episode_id']
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / 'edits.npz', **arrays)
        info = dict(population=record['population'], video=record['video'], episode_id=episode['episode_id'],
            status='COMPLETE', source_key=key, tokens=int(mask.sum()),
            reencoded_original_max_error=float(np.max(np.abs(pooled - original))),
            zero_edit_checks=zero_checks, spatial_equal_weight_raw_max_error=0., random_controls=controls)
        atomic_write_json(target / 'summary.json', info)
        diagnostics.append(info)
        print('SOURCE_MEMORY', index + 1, len(records), episode['episode_id'], flush=True)
    atomic_write_json(output / 'source_edits.json', dict(sources=diagnostics,
        definition='Original exact raw descriptor plus raw token residual edit; fixed original SupCon readout.',
        random_scope='Permutation within source-active components; raw-space delta norm matched; undefined directions retain identity.'))
    return {name: FixedSourceMemory(reference, memory) for name, memory in mappings.items()}


def fixed_control_curves(settings, output, phase, points, smoke, resume):
    source_output = output / phase
    target_output = output / (phase + '_controls_at_method_threshold')
    target_output.mkdir(exist_ok=True)
    base = Path(settings[phase + '_base'])
    population = read_json(base / 'config.json')
    videos = settings[phase + '_videos'][:1] if smoke else settings[phase + '_videos']
    names = [name for name in conditions() if '__random' in name]
    for video in videos:
        target_video = target_output / video
        if (target_video / 'summary.json').exists():
            assert resume
            continue
        receipt, frames, records, _, offsets, _, _, first = evaluation.load_video(base, population, video)
        original = read_json(source_output / video / 'summary.json')
        episodes = []
        for episode in original['episodes']:
            target = target_video / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=True)
            if not episode['click']['available']:
                episodes.append(episode)
                continue
            with np.load(base / video / 'sources' / episode['episode_id'] / 'frame_data.npz') as saved:
                data = {key: saved[key].copy() for key in saved.files}
            results = {}
            with np.load(source_output / video / 'sources' / episode['episode_id'] / 'scores.npz') as saved:
                for name in names:
                    threshold = points[name.split('__')[0]]['threshold']
                    curve, checks, _ = evaluation.fixed_curve(saved[name], threshold, offsets, records,
                        frames, data, episode['source_lesion_id'], first, receipt['fps'])
                    for group, columns in episode['groups'].items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    np.savez_compressed(target / (name + '__score_curve.npz'), **curve)
                    results[name] = dict(threshold=threshold, direct_mask_verification=checks)
            result = dict(episode, conditions=results)
            atomic_write_json(target / 'summary.json', result)
            episodes.append(result)
        atomic_write_json(target_video / 'summary.json', dict(video=video, episodes=episodes, status='COMPLETE'))
    evaluation.shared.aggregate(target_output, dict(source_variants=names, representations=['score']), videos)
    atomic_write_json(target_output / 'summary.json', dict(status='COMPLETE', videos=videos,
        interpretation='Random source edits at the corresponding learned-method development threshold.'))


def evaluate(run, seed, smoke, resume):
    config = read_json(run / 'config.json')
    assert seed in config['seeds']
    settings_path = Path(config['evaluation_settings_file'])
    assert config['random_controls'] == CONTROL_COUNT
    settings = read_json(settings_path)
    settings.update(include_interventions=False, protocol=str(run / 'protocol.json'))
    base = run / 'smoke' if smoke else run
    output = base / 'evaluation' / f'seed{seed}'
    fit_root = base / 'fit'
    baseline = fit_root / 'spatial_weighted'
    reference_fit = Path(config.get('reference_fit', REFERENCE_FIT))
    records = source_records(config, settings, smoke)
    assets = [run / 'config.json', run / 'protocol.json', settings_path, Path(__file__),
        Path(model_api.__file__), Path(reference_api.__file__), Path(evaluation.__file__),
        Path(evaluation.shared.__file__), Path(evaluation.shared.memory.__file__),
        ROOT / 'src/region_identity_sae.py', ROOT / 'scripts/evaluate_region_identity_memory.py']
    assets += [reference_fit / name for name in ('model.npz', 'normalization.npz')]
    assets += [baseline / name for name in ('model.npz', 'normalization.npz', 'summary.json')]
    for method in METHODS:
        assets += [fit_root / method / f'seed{seed}' / name
                   for name in ('model.npz', 'normalization.npz', 'gains.npz', 'model_config.json', 'summary.json')]
    for record in records:
        assets += [record['source'] / 'complete.json']
        if record['episode']['click']['available']:
            assets.append(record['original'])
        if record['available']:
            assets += [record['source'] / 'source.json', record['source'] / 'tokens.npz']
    identity = dict(seed=seed, smoke=smoke, files={str(path): digest(path) for path in assets},
                    python=sys.version, numpy=np.__version__, torch=str(torch.__version__))
    output.mkdir(parents=True, exist_ok=resume)
    if (output / 'identity.json').exists():
        assert resume and read_json(output / 'identity.json') == identity
    else:
        atomic_write_json(output / 'identity.json', identity)
        atomic_write_json(output / 'evaluation_settings.json', settings)
        shutil.copyfile(__file__, output / Path(__file__).name)
    expected = 'REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE'
    if (output / 'summary.json').exists():
        assert resume and read_json(output / 'summary.json')['status'] == expected
        print('REUSE_TOKEN_MEMORY_EVALUATION', output, flush=True)
        return
    started = time.perf_counter()
    total = 2 if smoke else len(settings['development_videos']) + len(settings['extension_videos'])
    atomic_write_json(output / 'progress.json', dict(completed=0, total=total, phase='SOURCE_MEMORIES'))
    reference = reference_api.Representations(reference_fit)
    models = {method: model_api.load_predictor(fit_root / method / f'seed{seed}') for method in METHODS}
    predictors = build_memories(records, models, baseline, reference, output, seed)
    for phase in ('development', 'extension'):
        directory = Path(read_json(Path(settings[phase + '_base']) / 'config.json')['descriptors']) / settings[phase + '_videos'][0]
        positions = np.flatnonzero(np.load(directory / 'available.npy'))[:16]
        raw = np.load(directory / 'raw_mean.npy', mmap_mode='r')[positions].copy()
        original = reference(raw)['supcon_l2']
        cached = np.load(directory / 'supcon_l2.npy', mmap_mode='r')[positions].copy()
        error = float(np.max(np.abs(original - cached)))
        assert error < 1e-5
        for predictor in predictors.values():
            assert np.array_equal(predictor.encode(raw, 16), original)
        atomic_write_json(output / (phase + '_query_check.json'), dict(status='PASS', queries=len(raw),
            cached_supcon_max_error=error, all_method_queries_exact=True))
    evaluation.evaluate_phase(settings, output, 'development', predictors, None, smoke, resume)
    points_path = output / 'operating_points.json'
    points = read_json(points_path)['methods'] if points_path.exists() else evaluation.calibrate(
        output, list(predictors) + ['reference_supcon'], settings['retention_floor'])
    evaluation.evaluate_phase(settings, output, 'extension', predictors, points, smoke, resume)
    atomic_write_json(output / 'progress.json', dict(completed=total, total=total, phase='MATCHED_CONTROLS'))
    for phase in ('development', 'extension'):
        fixed_control_curves(settings, output, phase, points, smoke, resume)
    atomic_write_json(output / 'summary.json', dict(status=expected, operating_points=points,
        seconds=time.perf_counter() - started, extension_population='Previously examined extension',
        random_application='Each random control has its own development operating point.',
        random_intervention='Additional controls use their learned method operating point.',
        spatial_baseline='One training-only logistic model is shared across all evaluation seeds.'))
    atomic_write_json(output / 'progress.json', dict(completed=total, total=total, phase='COMPLETE'))
    print('TOKEN_MEMORY_EVALUATION_COMPLETE', seed, round(time.perf_counter() - started, 3), flush=True)


def summarize(run, smoke, resume):
    config = read_json(run / 'config.json')
    seeds = config['seeds'][:1] if smoke else config['seeds']
    base = run / 'smoke' if smoke else run
    rows, diagnostics = [], []
    assets = [run / 'config.json', run / 'protocol.json', Path(__file__)]
    for seed in seeds:
        folder = base / 'evaluation' / f'seed{seed}'
        summary = read_json(folder / 'summary.json')
        assert summary['status'] == ('REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE')
        assets += [folder / 'summary.json', folder / 'identity.json', folder / 'source_edits.json']
        diagnostics.extend(dict(seed=seed, **row) for row in read_json(folder / 'source_edits.json')['sources'])
        for method, point in summary['operating_points'].items():
            modes = ['own_operating_point', 'learned_method_threshold'] if '__random' in method else ['own_operating_point']
            for mode in modes:
                for phase in ('development', 'extension'):
                    directory = phase if mode == 'own_operating_point' else phase + '_controls_at_method_threshold'
                    path = folder / directory / (method + '__score_procedure_curve.npz')
                    assets.append(path)
                    with np.load(path, allow_pickle=False) as curve:
                        threshold = point['threshold'] if mode == 'own_operating_point' else summary['operating_points'][method.split('__')[0]]['threshold']
                        if phase == 'development' and mode == 'own_operating_point':
                            indices = np.flatnonzero(curve['threshold'] == threshold)
                            assert len(indices) == 1
                            column = int(indices[0])
                        else:
                            assert curve[METRICS['removal']].shape[1] == 1
                            column = 0
                        for i, video in enumerate(curve['videos'].tolist()):
                            metrics = {name: float(curve[key][i, column]) for name, key in METRICS.items()}
                            rows.append(dict(seed=seed, method=method, mode=mode, video=video,
                                population=phase if phase == 'development' else 'previously_examined_extension', threshold=threshold,
                                **{name: value if np.isfinite(value) else None for name, value in metrics.items()}))
    inputs = {str(path): digest(path) for path in assets}
    if (base / 'summary.json').exists():
        assert resume and read_json(base / 'summary.json')['inputs'] == inputs
        if (base / 'report.md').exists():
            print('REUSE_TOKEN_MEMORY_SUMMARY', flush=True)
            return
    aggregates = []
    for population, method, mode in sorted({(r['population'], r['method'], r['mode']) for r in rows}):
        selected = [r for r in rows if (r['population'], r['method'], r['mode']) == (population, method, mode)]
        per_seed = {seed: {metric: finite_mean([r[metric] for r in selected if r['seed'] == seed]) for metric in METRICS} for seed in seeds}
        aggregates.append(dict(population=population, method=method, mode=mode, seed_results=per_seed,
            **{metric: finite_mean([r[metric] for r in per_seed.values()]) for metric in METRICS}))
    contrasts = []
    for population, seed, video in sorted({(r['population'], r['seed'], r['video']) for r in rows}):
        selected = {(r['method'], r['mode']): r for r in rows if (r['population'], r['seed'], r['video']) == (population, seed, video)}
        comparisons = [('sparse_minus_dense', ('sparse_edit', 'own_operating_point'), ('dense_edit', 'own_operating_point'))]
        comparisons += [(method + '_minus_' + reference, (method, 'own_operating_point'), (reference, 'own_operating_point'))
                        for method in METHODS for reference in ('reference_supcon', 'spatial_weighted')]
        comparisons += [(f'{method}_minus_random{number}__{mode}', (method, 'own_operating_point'), (f'{method}__random{number}', mode))
                        for method in METHODS for number in range(CONTROL_COUNT)
                        for mode in ('own_operating_point', 'learned_method_threshold')]
        for name, left, right in comparisons:
            values = {metric: None if selected[left][metric] is None or selected[right][metric] is None else
                      selected[left][metric] - selected[right][metric] for metric in METRICS}
            contrasts.append(dict(population=population, seed=seed, video=video, contrast=name, **values))
    atomic_write_json(base / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE',
        inputs=inputs, seeds=seeds, selected_seed=None, procedure_rows=rows, aggregates=aggregates,
        paired_procedure_contrasts=contrasts, source_edit_diagnostics=diagnostics,
        independent_unit='procedure', extension_population='Previously examined extension',
        aggregation='Lesions within source episode, episodes within procedure, equal procedures within seed, equal seeds.',
        control_policy='Three fixed gain permutations per seed; no outcome-based control selection. Undefined controls retain identity.'))
    lines = ['# Source token residual editing', '',
        'Queries and the original SupCon readout remain fixed. Source edits retain the original descriptor and add only the token-derived delta. The spatial baseline uses one training-only logistic model across seeds.', '',
        f"Evaluated seeds: {', '.join(map(str, seeds))}. The extension procedures were previously examined. Random controls report independent development calibration and the learned-method operating point.", '',
        '| Population | Method | Operating point | Repeat removal | Other retention | First prompt |',
        '|---|---|---|---:|---:|---:|']
    for row in aggregates:
        values = ['NA' if row[key] is None else f'{100 * row[key]:.2f}%' for key in METRICS]
        lines.append(f"| {row['population']} | {row['method']} | {row['mode']} | " + ' | '.join(values) + ' |')
    lines += ['', 'Per-procedure paired contrasts, source token counts, zero-edit checks, and each random edit norm are saved in summary.json and the source_edits assets.', '']
    (base / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_write_json(base / 'summary_progress.json', dict(completed=1, total=1, phase='COMPLETE'))
    print('TOKEN_MEMORY_SUMMARY_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=('evaluate', 'summary'), required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'evaluate':
        if args.seed is None:
            parser.error('--seed is required for evaluate')
        evaluate(args.run, args.seed, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke, args.resume)


if __name__ == '__main__':
    main()
