import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoint_io import atomic_write_json, read_json


def inspect(run):
    output = run / 'analysis'
    config = read_json(run / 'config.json')
    training = read_json(run / 'training_summary.json')
    components = read_json(run / 'components/summary.json')
    assert training['status'] == components['status'] == 'COMPLETE'
    metrics = ['recall', 'negative_retention', 'auroc', 'cross_interval_recall']
    contrasts = pd.read_csv(output / 'cohort_procedure_contrasts.csv')
    per_video = contrasts.groupby(['partition', 'condition', 'method', 'video_id'])[metrics].mean().reset_index()
    per_video.to_csv(output / 'procedure_effects.csv', index=False)
    rows = []
    for key, group in per_video.groupby(['partition', 'condition', 'method']):
        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            rows.append(dict(zip(['partition', 'condition', 'method'], key), metric=metric,
                mean=float(values.mean()), positive=int((values > 1e-12).sum()),
                negative=int((values < -1e-12).sum()), unchanged=int((np.abs(values) <= 1e-12).sum())))
    changes = pd.DataFrame(rows)
    changes.to_csv(output / 'effect_inspection.csv', index=False)
    representations = []
    for name, root in [('reference', Path(config['comparison_run'])), ('current', run)]:
        frame = pd.read_csv(root / 'analysis/representations.csv')
        grouped = frame.groupby(['partition', 'condition', 'method', 'seed', 'video_id'])[
            ['reconstruction_nmse', 'local_active', 'pooled_active']].mean().groupby(
            ['partition', 'condition', 'method']).mean().reset_index()
        representations.append(grouped.assign(comparison=name))
    representation = pd.concat(representations, ignore_index=True)
    representation.to_csv(output / 'representation_inspection.csv', index=False)
    frame = pd.read_csv(output / 'component_procedures.csv')
    held = frame[frame.scope == 'held']
    keys = ['partition', 'condition', 'method', 'seed', 'fold', 'video_id', 'boundary']
    base = held[held.variant == 'before'].set_index(keys)
    effects = []
    for variant in ['selected', 'random0', 'random1']:
        value = held[held.variant == variant].set_index(keys)
        assert set(value.index) == set(base.index)
        delta = (value[metrics[:3]] - base[metrics[:3]]).reset_index()
        effects.append(delta.assign(variant=variant))
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(output / 'component_paired_effects.csv', index=False)
    component_summary = effects.groupby(['partition', 'condition', 'method', 'variant', 'boundary'])[metrics[:3]].mean().reset_index()
    component_summary.to_csv(output / 'component_effect_inspection.csv', index=False)
    selected = pd.read_csv(output / 'selected_component_transfer.csv').groupby(
        ['partition', 'condition', 'method'])[['fit_confusion_contribution', 'held_confusion_contribution']].mean()
    selected.to_csv(output / 'component_transfer_inspection.csv')
    errors = []
    for item in components['models']:
        receipt = read_json(Path(item['analysis_directory']) / 'complete.json')
        assert receipt['held_forward_exact'] and receipt['held_scores_exact']
        errors.append(receipt['component_sum_max_error'])
    matches = []
    for item in training['outputs']:
        folder = Path(item['directory'])
        old = Path(config['comparison_run']) / folder.relative_to(run)
        first, second = read_json(folder / 'sequence.json'), read_json(old / 'sequence.json')
        length = min(len(first), len(second))
        assert first[:length] == second[:length]
        with np.load(folder / 'normalization.npz') as a, np.load(old / 'normalization.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        matches.append(dict(directory=str(folder), shared_steps=length))
    receipt = dict(status='COMPLETE', fits=len(training['outputs']), component_models=len(components['models']),
        procedure_rows=len(pd.read_csv(output / 'procedures.csv')), component_rows=len(frame),
        component_sum_max_error=max(errors), matched_sampling_and_normalization=matches)
    atomic_write_json(output / 'inspection.json', receipt)
    print(changes[changes.metric == 'recall'].to_string(index=False))
    print(representation.to_string(index=False))
    print(component_summary[component_summary.boundary == 'original_scope'].to_string(index=False))
    print(selected.to_string())
    print(per_video[per_video.partition == 'validation'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    inspect(parser.parse_args().run)
