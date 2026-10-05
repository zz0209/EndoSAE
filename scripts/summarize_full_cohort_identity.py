import argparse
from pathlib import Path

from summarize_temporal_view_identity import summarize
import matplotlib.pyplot as plt
import pandas as pd
from src.checkpoint_io import atomic_write_json, read_json


def compare(run, training_root):
    output = training_root / 'analysis'
    summarize(run, training_root, output)
    config = read_json(run / 'config.json')
    comparison = Path(config['comparison_run']) / ('smoke' if training_root != run else '') / 'analysis'
    current = pd.read_csv(output / 'procedures.csv')
    original = pd.read_csv(comparison / 'procedures.csv')
    keys = ['partition', 'condition', 'method', 'seed', 'video_id']
    metrics = ['recall', 'negative_retention', 'auroc', 'cross_interval_recall']
    first, second = original.set_index(keys), current.set_index(keys)
    if not first.index.is_unique or not second.index.is_unique or set(first.index) != set(second.index):
        raise ValueError('Cohort comparisons have different evaluation units')
    delta = (second[metrics] - first[metrics]).reset_index()
    delta.to_csv(output / 'cohort_procedure_contrasts.csv', index=False)
    combined = pd.concat([original.assign(training_procedures=19), current.assign(training_procedures=27)], ignore_index=True)
    combined.to_csv(output / 'cohort_procedures.csv', index=False)
    summary = combined.groupby(['partition', 'condition', 'method', 'training_procedures'], sort=False)[metrics].mean().reset_index()
    summary.to_csv(output / 'cohort_summary.csv', index=False)
    delta.groupby(['partition', 'condition', 'method', 'seed'], sort=False)[metrics].mean().reset_index().to_csv(
        output / 'cohort_seed_contrasts.csv', index=False)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    for row, partition in enumerate(['training_oof', 'validation']):
        for column, condition in enumerate(config['conditions']):
            axis = axes[row, column]
            for index, method in enumerate(config['methods']):
                for side, count in enumerate([19, 27]):
                    selected = combined[(combined.partition == partition) & (combined.condition == condition) &
                        (combined.method == method) & (combined.training_procedures == count)]
                    values = selected.groupby('video_id').recall.mean()
                    x = index + (side - .5) * .36
                    axis.bar(x, values.mean(), width=.32, color=['#b7c2cb', '#197698'][side],
                        label=f'{count} training procedures' if index == 0 else None)
                    axis.scatter([x] * len(values), values, s=14, color='#253645', alpha=.65, zorder=3)
            axis.set(xticks=range(3), xticklabels=['Direct SAE', 'Dense code', 'SupCon'], ylim=(0, 1.04),
                ylabel='Same-lesion recall', title=('Grouped training' if row == 0 else 'Examined validation') +
                (' / 4 observations per lesion' if column == 0 else ' / 32 observations per lesion'))
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.suptitle('Unchanged evaluation procedures and observations; dots are procedure means across seeds\nLabel-informed 0.99 negative-quantile boundary; cohort extension changes quantity and composition')
    figure.savefig(output / 'cohort_recall.png', dpi=160)
    plt.close(figure)
    atomic_write_json(output / 'cohort_comparison.json', dict(status='COMPLETE',
        paired_procedure_rows=len(delta), comparison_run=config['comparison_run'], summary=summary.to_dict('records')))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--training-root', type=Path)
    args = parser.parse_args()
    compare(args.run, args.training_root or args.run)
