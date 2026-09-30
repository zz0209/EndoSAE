import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import encode_causal_detection_identity as reference_api
import evaluate_acknowledgement_sae as evaluation
from src import acknowledgement_sae, region_identity_sae
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


DEFAULT_SETTINGS = ROOT / 'results/runs/20260928T0515Z_sae_acknowledgement_v1'
DEFAULT_REFERENCE = ROOT / 'results/runs/20260927T1205Z_available_identity_supcon_dev_v1/fit'
METRICS = {
    'removal': 'source_suppression__procedure_values',
    'retention': 'all_other__baseline_qualified_retention__procedure_values',
    'first_prompt': 'all_other__first_baseline_frame_retention__procedure_values',
}
CONTRASTS = {
    'canonical_with_sparse': {'sparse_canonical': 1., 'sparse_self': -1.},
    'canonical_with_dense': {'dense_canonical': 1., 'dense_self': -1.},
    'sparse_with_canonical': {'sparse_canonical': 1., 'dense_canonical': -1.},
    'sparse_with_self': {'sparse_self': 1., 'dense_self': -1.},
    'canonical_main_effect': {'sparse_canonical': .5, 'dense_canonical': .5,
                              'sparse_self': -.5, 'dense_self': -.5},
    'sparse_main_effect': {'sparse_canonical': .5, 'sparse_self': .5,
                           'dense_canonical': -.5, 'dense_self': -.5},
    'interaction': {'sparse_canonical': 1., 'sparse_self': -1.,
                    'dense_canonical': -1., 'dense_self': 1.},
}


class DecodedSource:
    def __init__(self, predictor, reference):
        self.predictor = predictor
        self.reference = reference

    def encode(self, raw, batch_size=512):
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[None, :]
        result = [self.reference(values[start:start + batch_size])['supcon_l2']
                  for start in range(0, len(values), batch_size)]
        return np.concatenate(result) if result else np.empty((0, 128), dtype=np.float32)

    def memory(self, raw):
        decoded = self.predictor.decode_raw(raw)
        if decoded.shape != (1, 768):
            raise ValueError('A fixed source memory requires one decoded raw descriptor')
        memory = self.reference(decoded)['supcon_l2'][0]
        if not np.isfinite(memory).all() or np.linalg.norm(memory) <= 0:
            raise ValueError('Decoded source produced an invalid frozen-reference memory')
        return memory

    @staticmethod
    def score_encoded(memory, codes):
        return acknowledgement_sae.AcknowledgementPredictor.score_encoded(memory, codes)

    def score(self, source, queries, batch_size=512):
        return self.score_encoded(self.memory(source), self.encode(queries, batch_size))


def evaluate(run, seed, smoke, resume):
    config = read_json(run / 'config.json')
    assert seed in config['seeds']
    assert set(config['methods']) == set(region_identity_sae.METHODS)
    settings_path = Path(config.get('evaluation_settings_root', DEFAULT_SETTINGS)) / f'evaluation_seed{seed}.json'
    settings = read_json(settings_path)
    settings['include_interventions'] = False
    settings['protocol'] = str(run / 'protocol.json')
    fit = run / 'smoke' / 'fit' if smoke else run / 'fit'
    output = (run / 'smoke' / 'evaluation' if smoke else run / 'evaluation') / f'seed{seed}'
    reference_fit = Path(config.get('reference_fit', DEFAULT_REFERENCE))
    models = {method: fit / method / f'seed{seed}' for method in config['methods']}
    settings['methods'] = {method: str(directory) for method, directory in models.items()}
    identity = dict(
        seed=seed, smoke=smoke, config_sha256=digest(run / 'config.json'),
        protocol_sha256=digest(run / 'protocol.json'), settings_sha256=digest(settings_path),
        source_sha256=digest(__file__), model_source_sha256=digest(region_identity_sae.__file__),
        predictor_source_sha256=digest(acknowledgement_sae.__file__),
        reference_source_sha256=digest(reference_api.__file__),
        evaluator_sha256=digest(evaluation.__file__),
        shared_evaluator_sha256=digest(evaluation.shared.__file__),
        metric_source_sha256=digest(evaluation.shared.memory.__file__),
        reference_fit=str(reference_fit),
        reference_assets={name: digest(reference_fit / name) for name in ['model.npz', 'normalization.npz']},
        model_assets={method: {name: digest(directory / name) for name in
                              ['model.npz', 'normalization.npz', 'model_config.json']}
                      for method, directory in models.items()},
        torch=str(torch.__version__), numpy=np.__version__, python=sys.version)
    output.mkdir(parents=True, exist_ok=resume)
    if (output / 'identity.json').exists():
        assert resume and read_json(output / 'identity.json') == identity
    else:
        atomic_write_json(output / 'identity.json', identity)
        atomic_write_json(output / 'config.json', config)
        atomic_write_json(output / 'evaluation_settings.json', settings)
        shutil.copyfile(__file__, output / Path(__file__).name)
    expected_status = 'REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE'
    if (output / 'summary.json').exists():
        assert resume and read_json(output / 'summary.json')['status'] == expected_status
        print('REUSE_REGION_EVALUATION', seed, output, flush=True)
        return
    reference = reference_api.Representations(reference_fit)
    total = 2 if smoke else len(settings['development_videos']) + len(settings['extension_videos'])
    atomic_write_json(output / 'progress.json', dict(completed=0, total=total, phase='LOAD_PREDICTORS'))
    predictors = {}
    for method, directory in models.items():
        predictor = region_identity_sae.load_predictor(directory)
        assert predictor.model.method == method
        predictors[method + '__decoded_source'] = DecodedSource(predictor, reference)
        predictors[method + '__latent'] = predictor
    started = time.perf_counter()
    if smoke:
        frozen_query_checks(settings, predictors, reference, output)
    evaluation.evaluate_phase(settings, output, 'development', predictors, None, smoke, resume)
    development_count = 1 if smoke else len(settings['development_videos'])
    atomic_write_json(output / 'progress.json', dict(completed=development_count, total=total,
                                                    phase='CALIBRATE_DEVELOPMENT'))
    points_path = output / 'operating_points.json'
    points = read_json(points_path)['methods'] if points_path.exists() else evaluation.calibrate(
        output, list(predictors) + ['reference_supcon'], settings['retention_floor'])
    evaluation.evaluate_phase(settings, output, 'extension', predictors, points, smoke, resume)
    atomic_write_json(output / 'summary.json', dict(
        status=expected_status, operating_points=points, seconds=time.perf_counter() - started,
        development_videos=settings['development_videos'][:1] if smoke else settings['development_videos'],
        extension_videos=settings['extension_videos'][:1] if smoke else settings['extension_videos'],
        primary='Decoded raw source passed through the frozen original SupCon; original query encoding unchanged.',
        secondary='Cosine in each trained latent representation.',
        extension_population='Previously examined extension procedures; fixed development thresholds.'))
    atomic_write_json(output / 'progress.json', dict(completed=total, total=total, phase='COMPLETE'))
    print('REGION_EVALUATION_COMPLETE', seed, round(time.perf_counter() - started, 3), flush=True)


def frozen_query_checks(settings, predictors, reference, output):
    checks = []
    for phase in ['development', 'extension']:
        base = Path(settings[phase + '_base'])
        population_config = read_json(base / 'config.json')
        video = settings[phase + '_videos'][0]
        directory = Path(population_config['descriptors']) / video
        positions = np.flatnonzero(np.load(directory / 'available.npy'))[:16]
        assert len(positions) > 0
        raw = np.load(directory / 'raw_mean.npy', mmap_mode='r')[positions].copy()
        cached = np.load(directory / 'supcon_l2.npy', mmap_mode='r')[positions].copy()
        direct = reference(raw)['supcon_l2']
        error = float(np.max(np.abs(direct - cached)))
        assert error < 1e-5, (phase, video, error)
        method_errors = {}
        for name, predictor in predictors.items():
            if name.endswith('__decoded_source'):
                encoded = predictor.encode(raw, batch_size=16)
                method_errors[name] = float(np.max(np.abs(encoded - direct)))
                assert np.array_equal(encoded, direct), name
        checks.append(dict(population=phase, video=video, positions=positions.tolist(),
            queries=len(raw), cached_supcon_max_error=error, method_query_max_errors=method_errors,
            descriptor_identity_sha256=digest(directory / 'identity.json')))
    atomic_write_json(output / 'frozen_query_checks.json', dict(status='PASS', checks=checks,
        definition='Every decoded-source method encodes untouched queries identically to the frozen original SupCon.'))


def finite_mean(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def mechanism_results(base, config, seeds, input_paths):
    rows = []
    metrics = ['canonical_mse', 'input_to_canonical_mse', 'canonical_mse_reduction',
               'self_mse', 'sparse_support_jaccard', 'identity_auroc', 'identity_average_precision']
    for seed in seeds:
        for method in config['methods']:
            directory = base / 'fit' / method / f'seed{seed}'
            fit_path = directory / 'summary.json'
            fit = read_json(fit_path)
            assert fit['status'] == 'COMPLETE' and fit['method'] == method and fit['seed'] == seed
            input_paths.append(fit_path)
            for population, stem in [('training_fit', 'selected_train'), ('held_validation', 'selected_held')]:
                path = directory / (stem + '.json')
                report = read_json(path)
                input_paths.extend([path, directory / (stem + '_errors.npz')])
                assert np.isclose(np.mean([v['canonical_mse'] for v in report['by_video'].values()]),
                                  report['macro_procedure_canonical_mse'], rtol=0, atol=1e-12)
                for video, values in report['by_video'].items():
                    identity = report['clean_identity']['by_video'][video]
                    rows.append(dict(population=population, method=method, seed=seed, video=video,
                        selected_steps=fit['steps'], source=str(path), **values,
                        canonical_mse_reduction=values['input_to_canonical_mse'] - values['canonical_mse'],
                        identity_auroc=identity['auroc'],
                        identity_average_precision=identity['average_precision'],
                        identity_pairs=identity['pairs'], identity_positives=identity['positives'],
                        identity_negatives=identity['negatives']))
    aggregates = []
    for population, method in sorted({(r['population'], r['method']) for r in rows}):
        selected = [r for r in rows if (r['population'], r['method']) == (population, method)]
        per_seed = {seed: {metric: finite_mean([r[metric] for r in selected if r['seed'] == seed])
                          for metric in metrics} for seed in seeds}
        aggregates.append(dict(population=population, method=method, seed_results=per_seed,
            per_seed_counts={seed: dict(procedures=sum(r['seed'] == seed for r in selected),
                identity_evaluable_procedures=sum(r['seed'] == seed and r['identity_auroc'] is not None
                                                  for r in selected)) for seed in seeds},
            **{metric: finite_mean([r[metric] for r in per_seed.values()]) for metric in metrics}))
    paired = []
    for population, seed, video in sorted({(r['population'], r['seed'], r['video']) for r in rows}):
        selected = {r['method']: r for r in rows
                    if (r['population'], r['seed'], r['video']) == (population, seed, video)}
        for name, weights in CONTRASTS.items():
            values = {}
            for metric in metrics:
                terms = [(weight, selected[method][metric]) for method, weight in weights.items()]
                values[metric] = None if any(value is None for _, value in terms) else float(
                    sum(weight * value for weight, value in terms))
            paired.append(dict(population=population, seed=seed, video=video, contrast=name, **values))
    return dict(procedure_rows=rows, aggregates=aggregates, paired_procedure_contrasts=paired,
        reconstruction_units='MSE in each fitted training normalization; valid noncanonical views averaged within clip, clips within procedure, procedures equally.',
        canonical_mse_reduction='Input-to-canonical MSE minus decoded-to-canonical MSE; positive values indicate regional correction.',
        identity_scope='Chronological same/different lesion pairs from canonical GT-region descriptors; separate from detector-prompt application.',
        support_scope='Positive-code support Jaccard between canonical and noncanonical views; dense models have no saved sparse-support statistic.')


def summarize(run, smoke, resume):
    config = read_json(run / 'config.json')
    base = run / 'smoke' if smoke else run
    seeds = config['seeds'][:1] if smoke else config['seeds']
    output = base / 'summary.json'
    input_paths = [run / 'config.json', run / 'protocol.json', Path(__file__)]
    rows = []
    expected_status = 'REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE'
    for seed in seeds:
        folder = base / 'evaluation' / f'seed{seed}'
        summary = read_json(folder / 'summary.json')
        assert summary['status'] == expected_status
        input_paths.extend([folder / 'summary.json', folder / 'identity.json', folder / 'operating_points.json'])
        for name, point in summary['operating_points'].items():
            method, mode = name.split('__') if '__' in name else (name, 'unchanged')
            for phase in ['development', 'extension']:
                path = folder / phase / (name + '__score_procedure_curve.npz')
                input_paths.append(path)
                with np.load(path, allow_pickle=False) as curve:
                    if phase == 'development':
                        indices = np.flatnonzero(curve['threshold'] == point['threshold'])
                        assert len(indices) == 1, (name, point['threshold'])
                        column = int(indices[0])
                    else:
                        assert curve[METRICS['removal']].shape[1] == 1
                        column = 0
                    for index, video in enumerate(curve['videos'].tolist()):
                        values = {key: float(curve[array][index, column]) for key, array in METRICS.items()}
                        rows.append(dict(seed=seed, method=method, mode=mode, video=video,
                            population=phase if phase == 'development' else 'previously_examined_extension',
                            threshold=point['threshold'],
                            **{key: value if np.isfinite(value) else None for key, value in values.items()}))
    mechanism = mechanism_results(base, config, seeds, input_paths)
    inputs = {str(path): digest(path) for path in input_paths}
    if output.exists():
        assert resume and read_json(output)['inputs'] == inputs
        if (base / 'report.md').exists():
            print('REUSE_REGION_SUMMARY', output, flush=True)
            return
    aggregates = []
    for population, method, mode in sorted({(r['population'], r['method'], r['mode']) for r in rows}):
        selected = [r for r in rows if (r['population'], r['method'], r['mode']) == (population, method, mode)]
        per_seed = {seed: {metric: finite_mean([r[metric] for r in selected if r['seed'] == seed])
                          for metric in METRICS} for seed in seeds}
        aggregates.append(dict(population=population, method=method, mode=mode, seed_results=per_seed,
            **{metric: finite_mean([r[metric] for r in per_seed.values()]) for metric in METRICS}))
    paired = []
    for population, seed, video in sorted({(r['population'], r['seed'], r['video']) for r in rows}):
        selected = {(r['method'], r['mode']): r for r in rows
                    if (r['population'], r['seed'], r['video']) == (population, seed, video)}
        for mode in ['decoded_source', 'latent']:
            comparisons = dict(CONTRASTS)
            comparisons.update({method + '_minus_reference': {method: 1., 'reference_supcon': -1.}
                                for method in config['methods']})
            for name, weights in comparisons.items():
                values = {}
                for metric in METRICS:
                    terms = [(weight, selected[(method, 'unchanged' if method == 'reference_supcon' else mode)][metric])
                             for method, weight in weights.items()]
                    values[metric] = None if any(value is None for _, value in terms) else float(
                        sum(weight * value for weight, value in terms))
                paired.append(dict(population=population, seed=seed, video=video, mode=mode,
                                   contrast=name, **values))
    contrast_aggregates = []
    for population, mode, name in sorted({(r['population'], r['mode'], r['contrast']) for r in paired}):
        selected = [r for r in paired if (r['population'], r['mode'], r['contrast']) == (population, mode, name)]
        per_seed = {seed: {metric: finite_mean([r[metric] for r in selected if r['seed'] == seed])
                          for metric in METRICS} for seed in seeds}
        contrast_aggregates.append(dict(population=population, mode=mode, contrast=name,
            seed_results=per_seed,
            **{metric: finite_mean([r[metric] for r in per_seed.values()]) for metric in METRICS}))
    result = dict(status=expected_status, inputs=inputs, procedure_rows=rows, aggregates=aggregates,
        paired_procedure_contrasts=paired, contrast_aggregates=contrast_aggregates,
        mechanism=mechanism,
        independent_unit='procedure', seeds=seeds, selected_seed=None,
        aggregation='Equal procedure means within seed, then equal seed means. Missing values remain explicit.',
        first_prompt='Fraction of baseline-qualified other-lesion first prompt frames retained, with the existing lesion/episode/procedure aggregation.',
        extension_population='Previously examined extension; no independent confirmation claim.')
    atomic_write_json(output, result)
    lines = ['# Region identity memory', '',
        'Primary readout changes the decoded source descriptor while retaining the original frozen SupCon query representation. Latent cosine is secondary. Each operating point uses development-only protection constraints; the previously examined extension receives that threshold unchanged.', '',
        f"Values average procedures within each training seed and then average the {len(seeds)} evaluated seed(s): {', '.join(map(str, seeds))}. First prompt is the retained fraction of baseline-qualified other-lesion first prompt frames.", '',
        '| Population | Model | Readout | Repeat removal | Other-prompt retention | First prompt |',
        '|---|---|---|---:|---:|---:|']
    def percentage(value):
        return 'NA' if value is None else f'{100 * value:.2f}%'
    for row in aggregates:
        lines.append(f"| {row['population']} | {row['method']} | {row['mode']} | " +
                     ' | '.join(percentage(row[metric]) for metric in METRICS) + ' |')
    lines += ['', '## Region reconstruction and identity', '',
        'Selected-model diagnostics use the saved train and held-validation outputs. Reconstruction MSE is measured in the fitted training normalization. Positive MSE reduction means the decoded region is closer to the canonical region than its original input. Identity AUROC uses canonical GT-region descriptor pairs; its defined procedure count can differ from the reconstruction population. Dense support Jaccard remains unavailable.', '',
        '| Population | Model | Canonical MSE | Input canonical MSE | MSE reduction | Support Jaccard | Identity AUROC |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in mechanism['aggregates']:
        values = [row[key] for key in ['canonical_mse', 'input_to_canonical_mse',
                  'canonical_mse_reduction', 'sparse_support_jaccard', 'identity_auroc']]
        lines.append(f"| {row['population']} | {row['method']} | " +
                     ' | '.join('NA' if value is None else f'{value:.5f}' for value in values) + ' |')
    lines += ['', 'Paired factorial contrasts and reference comparisons are stored by procedure and seed in summary.json. Positive removal differences indicate more repeated prompts removed; positive retention differences indicate more other-lesion prompts preserved.', '']
    (base / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_write_json(base / 'summary_progress.json', dict(completed=1, total=1, phase='COMPLETE'))
    print('REGION_SUMMARY_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summary'], required=True)
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
