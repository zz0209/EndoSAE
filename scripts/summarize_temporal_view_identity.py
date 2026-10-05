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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json


def summarize(run, training_root, output):
    config = read_json(run / 'config.json')
    training = read_json(training_root / 'training_summary.json')
    component_results = read_json(training_root / 'components' / 'summary.json')
    if training['status'] != 'COMPLETE' or component_results['status'] != 'COMPLETE':
        raise ValueError('Incomplete analysis inputs')
    output.mkdir(parents=True, exist_ok=True)
    selection = {(row['condition'], row['method'], row['seed']): row['step'] for row in training['selection']}
    procedures, clips = [], []
    for item in training['outputs']:
        step = selection[item['condition'], item['method'], item['seed']]
        evaluation = next(row for row in item['summary']['evaluations'] if row['step'] == step)
        common = {key: item[key] for key in ['condition', 'method', 'seed', 'fold']}
        common.update(step=step, partition='validation' if item['fold'] == 'full' else 'training_oof')
        for video, row in evaluation['by_video'].items():
            procedures.append(dict(common, video_id=video, threshold=evaluation['threshold'], **row))
        clips.extend(dict(common, **row) for row in evaluation['by_clip'])
    frame = pd.DataFrame(procedures)
    clip_frame = pd.DataFrame(clips)
    frame.to_csv(output / 'procedures.csv', index=False)
    clip_frame.to_csv(output / 'representations.csv', index=False)
    metrics = ['recall', 'negative_retention', 'auroc', 'cross_interval_recall']
    summary = frame.groupby(['partition', 'condition', 'method'], sort=False)[metrics].mean().reset_index()
    seed_frame = frame.groupby(['partition', 'condition', 'method', 'seed'], sort=False)[metrics].mean().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    seed_frame.to_csv(output / 'seeds.csv', index=False)
    keys = ['partition', 'method', 'seed', 'video_id']
    original = frame[frame.condition == 'original_four'].set_index(keys)
    expanded = frame[frame.condition == 'expanded_real_views'].set_index(keys)
    if set(original.index) != set(expanded.index):
        raise ValueError('Observation-condition procedure rosters differ')
    contrasts = (expanded[metrics] - original[metrics]).reset_index()
    contrasts.to_csv(output / 'observation_contrasts.csv', index=False)
    ordinary = []
    for condition in config['conditions']:
        sparse = frame[(frame.condition == condition) & (frame.method == 'token_sparse')].set_index(['partition', 'seed', 'video_id'])
        for method in ['token_dense', 'raw_supcon']:
            baseline = frame[(frame.condition == condition) & (frame.method == method)].set_index(['partition', 'seed', 'video_id'])
            delta = (sparse[metrics] - baseline[metrics]).reset_index()
            delta['condition'], delta['baseline'] = condition, method
            ordinary.append(delta)
    pd.concat(ordinary, ignore_index=True).to_csv(output / 'ordinary_contrasts.csv', index=False)
    component_rows, effects = [], []
    for item in component_results['models']:
        directory = Path(item['analysis_directory'])
        common = {key: item[key] for key in ['condition', 'method', 'seed', 'fold']}
        common['partition'] = 'validation' if item['fold'] == 'full' else 'training_oof'
        component_rows.extend(dict(common, **row) for row in read_json(directory / 'procedures.json'))
        with np.load(directory / 'effects.npz', allow_pickle=False) as archive:
            selected = archive['selected_features']
            fit_difference = archive['fit_negative'] - archive['fit_positive']
            held_difference = archive['held_negative'] - archive['held_positive']
            for feature in range(config['latent_dim']):
                effects.append(dict(common, feature=feature, selected=bool(feature in selected),
                    fit_positive=float(archive['fit_positive'][feature]), fit_negative=float(archive['fit_negative'][feature]),
                    fit_confusion_contribution=float(fit_difference[feature]),
                    held_positive=float(archive['held_positive'][feature]), held_negative=float(archive['held_negative'][feature]),
                    held_confusion_contribution=float(held_difference[feature])))
    component_frame = pd.DataFrame(component_rows)
    effect_frame = pd.DataFrame(effects)
    component_frame.to_csv(output / 'component_procedures.csv', index=False)
    effect_frame.to_csv(output / 'component_effects.csv', index=False)
    component_summary = component_frame.groupby(['partition', 'condition', 'method', 'scope', 'variant', 'boundary'], sort=False)[
        ['recall', 'negative_retention', 'auroc', 'false_positive_correction', 'true_positive_damage',
         'positive_score_change', 'negative_score_change']].mean().reset_index()
    component_summary.to_csv(output / 'component_summary.csv', index=False)
    selected_effects = effect_frame[effect_frame.selected].groupby(['partition', 'condition', 'method', 'seed', 'fold'], sort=False)[
        ['fit_confusion_contribution', 'held_confusion_contribution']].sum().reset_index()
    selected_effects.to_csv(output / 'selected_component_transfer.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout='constrained')
    names = ['Direct SAE', 'Dense code', 'SupCon']
    colors = {'original_four': '#b7c2cb', 'expanded_real_views': '#197698'}
    for row, partition in enumerate(['training_oof', 'validation']):
        for column, metric in enumerate(['recall', 'negative_retention', 'auroc']):
            axis = axes[row, column]
            for index, method in enumerate(config['methods']):
                for side, condition in enumerate(config['conditions']):
                    subset = frame[(frame.partition == partition) & (frame.method == method) & (frame.condition == condition)]
                    per_video = subset.groupby('video_id')[metric].mean().to_numpy()
                    x = index + (side - .5) * .36
                    axis.bar(x, np.mean(per_video), width=.32, color=colors[condition],
                        label='4 real observations' if index == 0 and side == 0 else '32 real observations' if index == 0 else None)
                    axis.scatter(x + np.linspace(-.08, .08, len(per_video)), per_video, color='#253645', s=12, zorder=3)
            axis.set(xticks=range(3), xticklabels=names, ylim=(0, 1.04),
                title=('Grouped training' if row == 0 else 'Examined validation') + ': ' + metric.replace('_', ' '))
            if row == 0 and column == 0:
                axis.legend(fontsize=8, loc='upper left')
    figure.suptitle('Fixed original evaluation observations; dots are procedure means across seeds\nRecall uses a label-informed 0.99 negative-quantile boundary')
    figure.savefig(output / 'temporal_identity.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for axis, partition in zip(axes, ['training_oof', 'validation'], strict=True):
        selected = selected_effects[selected_effects.partition == partition]
        labels = []
        for index, (condition, method) in enumerate((c, m) for c in config['conditions'] for m in config['methods'][:2]):
            subset = selected[(selected.condition == condition) & (selected.method == method)]
            labels.append(('4' if condition == 'original_four' else '32') + ' views\n' + ('SAE' if method == 'token_sparse' else 'Dense'))
            axis.bar(index - .18, subset.fit_confusion_contribution.mean(), width=.34, color='#b7c2cb', label='Fitting procedures' if index == 0 else None)
            axis.bar(index + .18, subset.held_confusion_contribution.mean(), width=.34, color='#197698', label='Excluded procedures' if index == 0 else None)
        axis.axhline(0, color='#333333', linewidth=.7)
        axis.set(xticks=range(4), xticklabels=labels, ylabel='Different minus same identity contribution',
            title='Grouped training' if partition == 'training_oof' else 'Examined validation')
        axis.legend(fontsize=8)
    figure.suptitle('16 components ranked on original fitting observations; unchanged rankings on held procedures')
    figure.savefig(output / 'component_transfer.png', dpi=160)
    plt.close(figure)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', procedures=len(frame),
        component_procedures=len(component_frame), component_effects=len(effect_frame),
        summary=summary.to_dict('records')))
    atomic_write_json(output / 'progress.json', dict(status='COMPLETE', completed=1, total=1))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    training = args.training_root or args.run
    summarize(args.run, training, args.output or training / 'analysis')
