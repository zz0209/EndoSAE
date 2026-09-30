import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/runs/20260930T0445Z_component_memory_v1'
OUT = ROOT / 'figures/component_memory_20260930/matched_protection_posthoc'
SEEDS = [20260929, 20260930, 20261001]
METHODS = ['reference_supcon', 'sparse_edit'] + [f'sparse_edit__random{n}' for n in range(3)]
METRICS = {
    'removal': 'source_suppression__procedure_values',
    'retention': 'all_other__baseline_qualified_retention__procedure_values',
    'first_prompt': 'all_other__first_baseline_frame_retention__procedure_values',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = read_json(RUN / 'summary.json')
    assert summary['status'] == 'COMPLETE'
    floor = next(row['retention'] for row in summary['aggregates']
                 if row['population'] == 'development' and row['method'] == 'reference_supcon')
    assert np.isclose(floor, 0.9953928022740465, atol=1e-14)
    inputs = {}
    selected = []
    procedure_rows = []
    for seed in SEEDS:
        folder = RUN / 'evaluation' / f'seed{seed}'
        original_points = read_json(folder / 'summary.json')['operating_points']
        for method in METHODS:
            source = folder / 'development' / f'{method}__score_procedure_curve.npz'
            inputs[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source, allow_pickle=False) as curve:
                thresholds = curve['threshold']
                values = {metric: curve[key] for metric, key in METRICS.items()}
                assert len(thresholds) > 1 and all(value.shape == (5, len(thresholds)) for value in values.values())
                assert all(np.isfinite(value).all() for value in values.values())
                means = {metric: value.mean(axis=0) for metric, value in values.items()}
                admissible = ((means['retention'] >= floor - 1e-12) &
                              (values['first_prompt'] >= 1. - 1e-12).all(axis=0))
                eligible = np.flatnonzero(admissible)
                assert len(eligible)
                maximum = means['removal'][eligible].max()
                tied = eligible[means['removal'][eligible] >= maximum - 1e-12]
                column = int(tied[np.argmax(thresholds[tied])])
                threshold = float(thresholds[column])
                record = dict(seed=seed, method=method, threshold=threshold,
                    original_threshold=original_points[method]['threshold'],
                    development_curve_points=len(thresholds), eligible_points=len(eligible),
                    **{f'development_{metric}_percent': 100 * means[metric][column] for metric in METRICS},
                    extension_status='THRESHOLD_NOT_SAVED', extension_curve_source='',
                    **{f'extension_{metric}_percent': None for metric in METRICS})
                for i, video in enumerate(curve['videos'].tolist()):
                    procedure_rows.append(dict(seed=seed, method=method, population='development', video=video,
                        threshold=threshold, **{metric + '_percent': 100 * value[i, column] for metric, value in values.items()}))
            available = [('extension', float(original_points[method]['threshold']))]
            if '__random' in method:
                available.append(('extension_controls_at_method_threshold', float(original_points['sparse_edit']['threshold'])))
            for directory, saved_threshold in available:
                if saved_threshold != threshold:
                    continue
                source = folder / directory / f'{method}__score_procedure_curve.npz'
                inputs[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
                with np.load(source, allow_pickle=False) as curve:
                    values = {metric: curve[key] for metric, key in METRICS.items()}
                    assert all(value.shape == (5, 1) and np.isfinite(value).all() for value in values.values())
                    for metric, value in values.items():
                        record[f'extension_{metric}_percent'] = 100 * float(value.mean())
                    for i, video in enumerate(curve['videos'].tolist()):
                        procedure_rows.append(dict(seed=seed, method=method, population='previously_examined_extension',
                            video=video, threshold=threshold,
                            **{metric + '_percent': 100 * value[i, 0] for metric, value in values.items()}))
                record['extension_status'] = 'EXACT_SAVED_THRESHOLD'
                record['extension_curve_source'] = str(source.relative_to(ROOT))
                break
            selected.append(record)
            print('MATCHED_PROTECTION', seed, method, f"removal={record['development_removal_percent']:.6f}%",
                  f"retention={record['development_retention_percent']:.6f}%", record['extension_status'], flush=True)

    aggregates = []
    for method in METHODS:
        rows = [row for row in selected if row['method'] == method]
        record = dict(method=method, seeds=3, extension_available_seeds=sum(row['extension_status'] == 'EXACT_SAVED_THRESHOLD' for row in rows))
        for phase in ['development', 'extension']:
            complete = all(row[f'{phase}_removal_percent'] is not None for row in rows)
            for metric in METRICS:
                record[f'{phase}_{metric}_percent'] = float(np.mean([row[f'{phase}_{metric}_percent'] for row in rows])) if complete else None
        aggregates.append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'selected_thresholds.csv', selected)
    write_csv(OUT / 'procedure_metrics.csv', procedure_rows)
    write_csv(OUT / 'aggregate_metrics.csv', aggregates)
    (OUT / 'diagnostic.json').write_text(json.dumps(dict(status='POSTHOC_CURVE_DIAGNOSTIC_COMPLETE',
        purpose='Interpret existing results; excluded from subsequent method selection.',
        retention_floor=floor, first_prompt_requirement='Every procedure equals 1',
        selection='Maximum development removal among eligible saved points; highest threshold among removal ties.',
        numerical_tolerance=1e-12, independent_unit='procedure', seeds=SEEDS,
        extension_policy='Apply the selected development threshold only when that exact threshold exists in saved extension curves. Missing points remain unavailable.',
        selected=selected, aggregates=aggregates, inputs=inputs), indent=2) + '\n', encoding='utf-8')
    lines = ['# Post-hoc matched-protection diagnostic', '',
             f'Development other-prompt retention must be at least {100 * floor:.12f}%, with every first prompt retained. '
             'For each method and seed, maximize removal across the saved full development curve and choose the highest '
             'threshold among ties. Numerical tolerance is 1e-12 in fraction units. This diagnostic is excluded from '
             'subsequent method selection. All ten procedures were previously examined.', '',
             '| Method | Development removal (%) | Other retention (%) | First prompt (%) | Extension seeds available |',
             '|---|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['method']} | {row['development_removal_percent']:.6f} | "
                     f"{row['development_retention_percent']:.6f} | {row['development_first_prompt_percent']:.6f} | "
                     f"{row['extension_available_seeds']}/3 |")
    lines += ['', '| Seed | Method | Threshold | Development removal (%) | Other retention (%) | Extension status |',
              '|---:|---|---:|---:|---:|---|']
    for row in selected:
        lines.append(f"| {row['seed']} | {row['method']} | {row['threshold']:.17g} | "
                     f"{row['development_removal_percent']:.6f} | {row['development_retention_percent']:.6f} | {row['extension_status']} |")
    reference = next(row for row in aggregates if row['method'] == 'reference_supcon')
    sparse = next(row for row in aggregates if row['method'] == 'sparse_edit')
    lines += ['', f"SAE minus reference development removal: {sparse['development_removal_percent'] - reference['development_removal_percent']:+.6f} pp.", '',
              'The saved extension curves contain only the original operating point and, for random controls, the original '
              'SAE operating point. Changed thresholds that are absent from those curves have no reported extension metrics. '
              'No interpolation, video evaluation, or model fitting was performed. A method-level extension mean is reported '
              'only when all three seed thresholds are available.', '',
              'Reproduce with `artifacts/environments/modern/Scripts/python.exe scripts/diagnose_component_memory_matched_protection.py`.']
    (OUT / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
