import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def plot(run, smoke):
    config = read_json(run / 'config.json')
    baseline = Path(read_json(Path(config['baseline_run']) / 'config.json')['storage_root'])
    root = run / 'smoke' if smoke else Path(config['storage_root'])
    summary = read_json(root / 'training_summary.json')
    assert summary['status'] == 'COMPLETE'
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    receipts = []
    for item in summary['outputs']:
        column = ['token_sparse', 'token_dense'].index(item['method'])
        seed_index = config['seeds'].index(item['seed'])
        color = ['#0072B2', '#D55E00', '#009E73'][seed_index]
        folder = Path(item['directory'])
        old = baseline / 'adaptive' / item['method'] / f"seed{item['seed']}" / 'foldfull'
        current = read_json(folder / 'history.json')
        previous = read_json(old / 'history.json')[:len(current)]
        x = np.array([r['step'] for r in current])
        assert len(current) == item['summary']['steps']
        for row, metric in enumerate(['identity', 'reconstruction']):
            y = np.array([r[metric] for r in current])
            original = np.array([r[metric] for r in previous])
            assert np.isfinite(y).all() and np.isfinite(original).all()
            axes[row, column].plot(x, original, '--', color=color, alpha=.5,
                label=f"REAL only, seed {item['seed']}", linewidth=.8)
            axes[row, column].plot(x, y, color=color, label=f"Supplemental, seed {item['seed']}", linewidth=.9)
            if smoke:
                chosen = x % config['replace_every'] == 0
                axes[row, column].scatter(x[chosen], y[chosen], color=color, s=28, zorder=3)
        receipts.append(dict(directory=str(folder), current_history_sha256=digest(folder / 'history.json'),
            baseline_history_sha256=digest(old / 'history.json')))
    for column, title in enumerate(['Adapted SAE', 'Adapted Dense']):
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel('Identity training loss')
        axes[1, column].set_ylabel('Reconstruction training loss')
        axes[1, column].set_xlabel('Optimizer update')
        axes[0, column].legend(fontsize=7)
        for ax in axes[:, column]:
            ax.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Training-path verification' if smoke else 'Fixed-budget source replacement', fontsize=13)
    figure.text(.5, .015, 'Different training sources change loss distributions; these curves do not measure application benefit.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .045, 1, .95))
    output = run / ('smoke_plot' if smoke else 'training_plot')
    output.mkdir(exist_ok=True)
    figure.savefig(output / 'training.png', dpi=150)
    figure.savefig(output / 'training.pdf')
    plt.close(figure)
    atomic_write_json(output / 'manifest.json', dict(status='COMPLETE', sources=receipts,
        summary_sha256=digest(root / 'training_summary.json'), script_sha256=digest(__file__), smoke=smoke))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plot(args.run, args.smoke)
