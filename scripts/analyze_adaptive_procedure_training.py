import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def analyze(run):
    config = read_json(run / 'config.json')
    baseline = Path(read_json(Path(config['baseline_run']) / 'config.json')['storage_root'])
    summary = read_json(Path(config['storage_root']) / 'training_summary.json')
    assert summary['status'] == 'COMPLETE' and len(summary['outputs']) == 6
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    rows, receipts = [], []
    for item in summary['outputs']:
        folder = Path(item['directory'])
        old = baseline / 'adaptive' / item['method'] / f"seed{item['seed']}" / 'foldfull'
        assert read_json(folder / 'sequence.json') == read_json(old / 'sequence.json')
        with np.load(folder / 'normalization.npz') as actual, np.load(old / 'normalization.npz') as expected:
            assert actual.files == expected.files
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key])
        history = pd.DataFrame(read_json(folder / 'history.json'))
        previous = pd.DataFrame(read_json(old / 'history.json'))
        assert len(history) == len(previous) == config['steps'] == 400
        assert np.isfinite(history.select_dtypes('number')).all().all()
        assert (history.backbone_gradient_norm > 0).all()
        np.testing.assert_allclose(history.identity, .5 * history.global_identity + .5 * history.procedure_identity)
        assert item['summary']['backbone_parameter_squared_change'] > 0
        column = ['token_sparse', 'token_dense'].index(item['method'])
        color = ['#0072B2', '#D55E00', '#009E73'][config['seeds'].index(item['seed'])]
        for row, metric in enumerate(['global_identity', 'procedure_identity', 'reconstruction']):
            axes[row, column].plot(history.step, history[metric], color=color, linewidth=.8,
                                  label=f"Conditioned, seed {item['seed']}")
            if metric != 'procedure_identity':
                baseline_metric = 'identity' if metric == 'global_identity' else metric
                axes[row, column].plot(previous.step, previous[baseline_metric], '--', color=color,
                                      alpha=.45, linewidth=.8, label=f"Global, seed {item['seed']}")
        rows.append(dict(method=item['method'], seed=item['seed'], **{
            metric + '_last50': float(history[metric].tail(50).mean()) for metric in
            ['global_identity', 'procedure_identity', 'identity', 'reconstruction', 'within_negative_fraction']},
            seconds=item['summary']['seconds'], parameter_change=item['summary']['backbone_parameter_squared_change']))
        receipts.append(dict(directory=str(folder), history_sha256=digest(folder / 'history.json'),
                             baseline_history_sha256=digest(old / 'history.json')))
    for column, title in enumerate(['Adapted SAE', 'Adapted Dense']):
        axes[0, column].set_title(title)
        axes[0, column].legend(fontsize=7, ncol=2)
        for row, label in enumerate(['Global identity loss', 'Within-procedure loss', 'Reconstruction loss']):
            axes[row, column].set_ylabel(label)
            axes[row, column].spines[['top', 'right']].set_visible(False)
        axes[2, column].set_xlabel('Optimizer update')
    figure.suptitle('Procedure-conditioned supervision with visual adaptation')
    figure.text(.5, .01, 'Training diagnostics; application benefit requires the saved causal-event evaluation.',
                ha='center', fontsize=10)
    figure.tight_layout(rect=(0, .03, 1, .96))
    output = run / 'training_analysis'
    output.mkdir(exist_ok=True)
    figure.savefig(output / 'training.png', dpi=140)
    figure.savefig(output / 'training.pdf')
    plt.close(figure)
    pd.DataFrame(rows).to_csv(output / 'per_model.csv', index=False)
    atomic_write_json(output / 'verification.json', dict(status='PASS', models=6, updates_per_model=400,
        baseline_sequence_equal=True, baseline_normalization_equal=True, objective_formula_equal=True,
        visual_gradients_positive=True, sources=receipts, script_sha256=digest(__file__)))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
