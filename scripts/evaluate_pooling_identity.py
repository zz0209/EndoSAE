import argparse
from pathlib import Path

import train_full_cohort_identity as parent
from train_temporal_view_identity import job_inputs
from train_token_identity_sae import evaluate
from src.checkpoint_io import atomic_write_json, read_json
from src.token_identity_sae import TokenIdentitySAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main(run, smoke):
    root = run / 'smoke' if smoke else run
    config = read_json(run / 'config.json')
    _, cohort, raw, offsets, records, _ = parent.load_inputs(Path(config['input_run']))
    raw, offsets, records = job_inputs(raw, offsets, records, cohort['fit_video_ids']['train'], 'original_four')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = read_json(root / 'training_summary.json')
    rows, checks = [], []
    for group in summary['outputs']:
        for item in group['outputs']:
            folder = Path(item['directory'])
            model_config = read_json(folder / 'model_config.json')
            selection = next(row for row in group['selection']
                             if row['method'] == item['method'] and row['seed'] == item['seed'])
            step = selection['step']
            model = TokenIdentitySAE(model_config, item['method']).cuda()
            with np.load(folder / f'model_{step:04d}.npz') as archive:
                model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files})
            with np.load(folder / 'normalization.npz') as archive:
                values = torch.from_numpy(((raw - archive['mean']) / archive['scale']).astype(np.float32)).cuda()
            for pooling in config['poolings']:
                target = folder / 'crossed_pooling'
                target.mkdir(exist_ok=True)
                model.pooling = pooling
                result = evaluate(model, values, offsets, records, item['held_videos'], target, pooling, model_config)
                if pooling == group['pooling']:
                    with np.load(folder / f'held_{step:04d}.npz') as previous, np.load(target / (pooling + '.npz')) as current:
                        error = float(np.max(np.abs(previous['scores'] - current['scores'])))
                        if error != 0:
                            raise ValueError('Matched pooling predictions did not reproduce')
                    checks.append(dict(directory=str(folder), matched_error=error))
                for video, metrics in result['by_video'].items():
                    nmse = [row['reconstruction_nmse'] for row in result['by_clip'] if row['video_id'] == video]
                    rows.append(dict(train_pooling=group['pooling'], eval_pooling=pooling,
                        method=item['method'], seed=item['seed'], fold=item['fold'], video=video,
                        population='validation' if item['fold'] == 'full' else 'exclusion',
                        recall=metrics['recall'], auroc=metrics['auroc'], nmse=float(np.mean(nmse))))
            del values
            print('CROSSED_EVALUATED', group['pooling'], item['method'], item['seed'], item['fold'], flush=True)
    target = root / 'analysis'
    target.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / 'procedures.csv', index=False)
    keys = ['population', 'method', 'train_pooling', 'eval_pooling', 'seed']
    seeds = frame.groupby(keys)[['recall', 'auroc', 'nmse']].mean().reset_index()
    seeds.to_csv(target / 'seeds.csv', index=False)
    means = seeds.groupby(keys[:-1])[['recall', 'auroc', 'nmse']].mean().reset_index()
    means.to_csv(target / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, population in enumerate(['exclusion', 'validation']):
        for col, metric in enumerate(['recall', 'auroc']):
            ax = axes[row, col]
            for method, color in [('token_sparse', '#147d92'), ('token_dense', '#b25530')]:
                part = seeds[(seeds.population == population) & (seeds.method == method)]
                labels = ['unit_mean/unit_mean', 'unit_mean/gmp', 'gmp/unit_mean', 'gmp/gmp']
                for seed, local in part.groupby('seed'):
                    values = [local[(local.train_pooling == key.split('/')[0]) &
                                    (local.eval_pooling == key.split('/')[1])][metric].item() for key in labels]
                    ax.plot(range(4), values, color=color, alpha=.45, marker='o',
                            label=method if seed == part.seed.min() else None)
            ax.set_xticks(range(4), ['Unit→Unit', 'Unit→GMP', 'GMP→Unit', 'GMP→GMP'], rotation=15)
            ax.set_title(population + ' | ' + metric)
            ax.set_ylabel('Procedure macro ' + metric)
            ax.grid(axis='y', alpha=.2)
            ax.legend()
    fig.suptitle('Training aggregation and inference aggregation | examined data')
    fig.savefig(target / 'crossed_pooling.png', dpi=160)
    fig.savefig(target / 'crossed_pooling.pdf')
    plt.close(fig)
    atomic_write_json(root / 'crossed_verification.json', dict(status='PASS', checks=checks, procedure_rows=len(frame)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(args.run, args.smoke)
