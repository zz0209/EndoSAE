import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def analyze(run):
    output = run / 'analysis'
    assert read_json(output / 'summary.json')['status'] == 'COMPLETE'
    events = pd.read_csv(output / 'events.csv')
    procedures = pd.read_csv(output / 'procedures.csv')
    calibration = read_json(output / 'calibration.json')
    assert len(calibration) == 120 and len(events) == 5394 * 24
    videos = sorted(events.video.unique())
    assert len(videos) == 5
    for row in calibration:
        assert row['video'] not in row['fit_procedures']
        assert set(row['fit_procedures']) == set(videos) - {row['video']}
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    keys = ['video', 'method', 'seed', 'variant']
    original = procedures.set_index(keys)
    discriminations, sources = [], []
    for identity, group in events.groupby(keys, sort=True):
        same, other = group[group.same_identity], group[~group.same_identity]
        first = other[other.first_prompt]
        computed = [(~same.retained).sum() / len(same), other.retained.sum() / len(other),
                    first.retained.sum() / len(first)]
        np.testing.assert_allclose(original.loc[identity, metrics].to_numpy(dtype=float), computed, atol=1e-14)
        available = group[np.isfinite(group.matched_score)]
        discriminations.append(dict(zip(keys, identity),
            auroc=roc_auc_score(available.same_identity, available.matched_score), available=len(available),
            missing=len(group) - len(available)))
    for identity, group in events.groupby(keys + ['episode'], sort=True):
        same, other = group[group.same_identity], group[~group.same_identity]
        sources.append(dict(zip(keys + ['episode'], identity), repeat_removal=(~same.retained).mean(),
            other_retention=other.retained.mean(), repeat_events=len(same), other_events=len(other)))
    pd.DataFrame(sources).to_csv(output / 'sources.csv', index=False)
    discriminations = pd.DataFrame(discriminations)
    discriminations.to_csv(output / 'discrimination.csv', index=False)
    procedures = procedures.merge(discriminations, on=keys, validate='one_to_one')
    metrics.append('auroc')
    by_video = procedures.groupby(['video', 'method', 'variant'])[metrics].mean()
    rng = np.random.default_rng(20261007)
    bootstrap = rng.integers(0, 5, size=(10000, 5))
    comparisons = []
    for budget in ['budget1024', 'budget32']:
        for method in ['token_sparse', 'token_dense']:
            comparisons.append((f'adaptation_{method}_{budget}', ('adaptive_' + method, budget),
                                ('frozen_' + method, budget)))
        comparisons.append((f'adapted_sparse_minus_dense_{budget}', ('adaptive_token_sparse', budget),
                            ('adaptive_token_dense', budget)))
    for method in ['frozen_token_sparse', 'adaptive_token_sparse', 'frozen_token_dense', 'adaptive_token_dense']:
        comparisons.append((f'compression_{method}', (method, 'budget32'), (method, 'budget1024')))
    contrasts, differences = [], []
    for name, changed, reference in comparisons:
        a = by_video.xs(changed, level=['method', 'variant']).sort_index()
        b = by_video.xs(reference, level=['method', 'variant']).sort_index()
        assert a.index.equals(b.index) and len(a) == 5
        for metric in metrics:
            delta = (a[metric] - b[metric]).to_numpy()
            lower, upper = np.quantile(delta[bootstrap].mean(1), [.025, .975])
            contrasts.append(dict(contrast=name, metric=metric, mean=delta.mean(), lower=lower, upper=upper,
                procedures=5, positive=int((delta > 0).sum()), negative=int((delta < 0).sum())))
            differences.extend(dict(contrast=name, metric=metric, video=video, difference=value)
                               for video, value in zip(a.index, delta, strict=True))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(output / 'contrasts.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_procedures.csv', index=False)
    summary = procedures.groupby(['method', 'variant'])[metrics].mean().reset_index()
    summary.to_csv(output / 'summary_with_discrimination.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    order = ['frozen_token_sparse', 'adaptive_token_sparse', 'frozen_token_dense', 'adaptive_token_dense']
    labels = ['Frozen SAE', 'Adapted SAE', 'Frozen Dense', 'Adapted Dense']
    for ax, metric, title in zip(axes, metrics[:3], ['Repeat events removed', 'Other events retained', 'First prompts retained'], strict=True):
        for budget, color in [('budget1024', '#15678A'), ('budget32', '#CA653F')]:
            values = summary[summary.variant == budget].set_index('method').loc[order, metric].to_numpy() * 100
            ax.plot(range(4), values, marker='o', label='Full code' if budget == 'budget1024' else '32 components', color=color)
            for j, method in enumerate(order):
                dots = by_video.xs((method, budget), level=['method', 'variant'])[metric] * 100
                ax.scatter(np.full(len(dots), j), dots, color=color, alpha=.3, s=14)
        ax.set_xticks(range(4), labels, rotation=25, ha='right')
        ax.set_ylabel('Procedure mean (%)')
        ax.set_title(title)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Fixed development events | Three initializations | Dots: procedure means')
    fig.savefig(output / 'causal_events.png', dpi=180)
    plt.close(fig)
    report = '\n\n'.join([
        '# Adapted visual identity at causal prompt events',
        'Five previously examined development procedures, 5394 fixed annotation-selected events, '
        '13 acknowledgement sources, twelve trained models and two storage budgets. '
        'Calibration excludes the evaluated procedure and weights the remaining procedures equally. '
        'Every decision reruns detection-to-lesion box matching. This sample measures selected event outcomes; '
        'it does not estimate continuous prompt time or independent confirmation.',
        summary.to_markdown(index=False, floatfmt='.6f'),
        'Paired differences average the three initializations within each procedure before a 10000-draw '
        'procedure bootstrap. Intervals describe five development procedures and have limited precision. '
        'Positive removal differences and positive retention differences are favorable. AUROC uses finite '
        'scores for the originally matched detections and describes a different measurement from event rematching.',
        contrasts.to_markdown(index=False, floatfmt='.6f'),
        '![Causal prompt events](causal_events.png)',
        'Complete initialization results: seeds.csv. Procedure results: procedures.csv. Source results: '
        'sources.csv. All event decisions: events.csv. Exact calibration populations: calibration.json.'
    ])
    (output / 'report.md').write_text(report + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', event_rows=len(events),
        procedure_rows=len(procedures), calibration_exclusion_checks=len(calibration),
        independent_aggregate_checks=len(procedures) * 3, summary_sha256=digest(output / 'summary.json'),
        source_sha256=digest(__file__), bootstrap_seed=20261007, bootstrap_draws=10000))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
