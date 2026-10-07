import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def analyze(run):
    config = read_json(run / 'config.json')
    output = run / 'comparison'
    output.mkdir(exist_ok=True)
    keys = ['video', 'method', 'seed']
    metrics = ['repeat_removal', 'other_retention', 'first_retention', 'auroc', 'oracle_recall99']
    procedures, events = [], []
    checks = 0
    for support, folder in [('detector', Path(config['reference_run'])), ('annotated', run)]:
        assert read_json(folder / 'analysis/summary.json')['status'] == 'COMPLETE'
        table = pd.read_csv(folder / 'analysis/events.csv')
        table = table[table.variant == 'budget1024'].copy()
        assert len(table) == 5394 * 12
        saved = pd.read_csv(folder / 'analysis/procedures.csv')
        saved = saved[saved.variant == 'budget1024'].set_index(keys)
        videos = set(table.video.unique())
        calibration = read_json(folder / 'analysis/calibration.json')
        for item in [r for r in calibration if r['budget'] == 1024]:
            assert set(item['fit_procedures']) == videos - {item['video']}
            checks += 1
        for identity, group in table.groupby(keys):
            same, other = group[group.same_identity], group[~group.same_identity]
            first = other[other.first_prompt]
            values = [(~same.retained).mean(), other.retained.mean(), first.retained.mean()]
            np.testing.assert_allclose(values, saved.loc[identity, metrics[:3]].to_numpy(float), atol=1e-14)
            checks += 3
            available = group[np.isfinite(group.matched_score)]
            fpr, tpr, _ = roc_curve(available.same_identity, available.matched_score, drop_intermediate=False)
            procedures.append(dict(zip(keys, identity), support=support,
                **dict(zip(metrics[:3], values)),
                auroc=roc_auc_score(available.same_identity, available.matched_score),
                oracle_recall99=tpr[fpr <= .01].max(), unavailable=len(group) - len(available)))
        table['support'] = support
        events.append(table)
    procedures = pd.DataFrame(procedures)
    procedures.to_csv(output / 'procedures.csv', index=False)
    summary = procedures.groupby(['support', 'method'])[metrics].mean().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    procedures.groupby(['support', 'method', 'seed'])[metrics].mean().to_csv(output / 'seeds.csv')
    average = procedures.groupby(['video', 'support', 'method'])[metrics].mean()
    bootstrap = np.random.default_rng(20261007).integers(0, 5, size=(10000, 5))
    contrasts, differences = [], []
    for method in sorted(procedures.method.unique()):
        a = average.xs(('annotated', method), level=['support', 'method']).sort_index()
        b = average.xs(('detector', method), level=['support', 'method']).sort_index()
        assert a.index.equals(b.index) and len(a) == 5
        for metric in metrics:
            delta = (a[metric] - b[metric]).to_numpy()
            low, high = np.quantile(delta[bootstrap].mean(1), [.025, .975])
            contrasts.append(dict(method=method, metric=metric, mean=delta.mean(), lower=low, upper=high,
                                  positive=int((delta > 0).sum()), negative=int((delta < 0).sum())))
            differences.extend(dict(method=method, metric=metric, video=v, difference=d)
                               for v, d in zip(a.index, delta, strict=True))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(output / 'contrasts.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_procedures.csv', index=False)
    event_keys = keys + ['event_id']
    paired = events[0].merge(events[1], on=event_keys, suffixes=('_detector', '_annotated'), validate='one_to_one')
    for column in ['episode', 'frame', 'lesion', 'same_identity', 'first_prompt', 'matched_position']:
        assert paired[column + '_detector'].equals(paired[column + '_annotated'])
    first = paired[(~paired.same_identity_detector) & paired.first_prompt_detector].copy()
    first.to_csv(output / 'first_prompts.csv', index=False)
    changes = first[first.retained_detector != first.retained_annotated]
    changes.to_csv(output / 'first_prompt_changes.csv', index=False)
    lost = changes[(changes.method == 'adaptive_token_sparse') & changes.retained_detector]
    cases = lost.sort_values(['video', 'frame_detector', 'seed']).drop_duplicates('video')
    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 4 * len(cases)), layout='constrained', squeeze=False)
    prefix = read_json(Path(config['prefix_run']) / 'config.json')
    for i, row in enumerate(cases.itertuples()):
        manifest = read_json(Path(prefix['event_reference']) / 'inputs' / row.video / 'events.json')
        source = next(e for e in manifest['episodes'] if e['episode_id'] == row.episode_detector)
        query = manifest['frames'][str(row.output_detector)]
        inputs = [(source['click']['input_frame'], source['source_info']['click']['detection']['xyxy'], 'Acknowledged source'),
                  (row.frame_detector, query['detections'][query['positions'].index(row.matched_position_detector)]['xyxy'],
                   'Other-lesion first prompt')]
        for j, (frame, box, label) in enumerate(inputs):
            with Image.open(Path(manifest['definition']['frame_root']) / row.video / f'{frame:06d}.jpg') as picture:
                axes[i, j].imshow(picture)
            x0, y0, x1, y1 = box
            axes[i, j].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#00FFFF', linewidth=2))
            axes[i, j].set_title(f'{label} | {row.video} | frame {frame}')
            axes[i, j].axis('off')
        axes[i, 1].set_title(f'Other-lesion first prompt | {row.video}\nDetector {row.matched_score_detector:.3f} < '
                            f'{row.threshold_detector:.3f}; annotated {row.matched_score_annotated:.3f} >= '
                            f'{row.threshold_annotated:.3f}')
    fig.savefig(output / 'first_prompt_cases.png', dpi=140)
    plt.close(fig)
    source_rows = []
    for table in events:
        for identity, group in table.groupby(keys + ['support', 'episode']):
            same, other = group[group.same_identity], group[~group.same_identity]
            source_rows.append(dict(zip(keys + ['support', 'episode'], identity),
                repeat_removal=(~same.retained).mean(), other_retention=other.retained.mean(),
                first_retention=other[other.first_prompt].retained.mean(),
                repeat_events=len(same), other_events=len(other)))
    pd.DataFrame(source_rows).to_csv(output / 'sources.csv', index=False)
    regions = pd.read_csv(run / 'regions/diagnostics.csv')
    regions.groupby(['video', 'status']).agg(count=('position', 'size'),
        original_tokens=('original_tokens', 'mean'), annotated_tokens=('annotated_tokens', 'mean'),
        support_jaccard=('support_jaccard', 'mean')).to_csv(output / 'region_coverage.csv')
    order = ['frozen_token_sparse', 'adaptive_token_sparse', 'frozen_token_dense', 'adaptive_token_dense']
    labels = ['Frozen SAE', 'Adapted SAE', 'Frozen Dense', 'Adapted Dense']
    titles = ['Repeat events removed', 'Other events retained', 'First prompts retained',
              'Matched-score AUROC', 'Label-defined recall at 99% specificity']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for support, color in [('detector', '#15678A'), ('annotated', '#CA653F')]:
            values = summary[summary.support == support].set_index('method').loc[order, metric]
            ax.plot(range(4), values * 100, marker='o', color=color, label=support)
            for j, method in enumerate(order):
                values = average.xs((support, method), level=['support', 'method'])[metric]
                ax.scatter(np.full(len(values), j), values * 100, color=color, alpha=.35, s=15)
        ax.set_xticks(range(4), labels, rotation=25, ha='right')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Procedure mean (%)')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    axes[1, 2].axis('off')
    axes[1, 2].text(0, .9, '5 exposed development procedures\n3 initializations\nIdentical camera inputs and models\n\nAnnotated support uses labels.\nLabel-defined recall is diagnostic.\nDots show procedure means.', va='top')
    fig.savefig(output / 'region_effects.png', dpi=180)
    plt.close(fig)
    report = '\n\n'.join(['# Fixed-model annotation-region intervention',
        'All twelve models, five exposed development procedures and 5394 fixed events. '
        'Original detections, eight camera images and model parameters are unchanged. '
        'Annotated support uses additional labels. Unmatched detections and incomplete annotation '
        'windows retain original support. Thresholds exclude the evaluated procedure.',
        summary.to_markdown(index=False, floatfmt='.6f'),
        'Paired differences average initialization results within procedure before a 10000-draw '
        'procedure bootstrap. AUROC and label-defined 99% specificity recall use finite originally '
        'matched scores. Label-defined recall is an in-sample capacity diagnostic; actual events '
        'use excluded-procedure thresholds and detection rematching.',
        contrasts.to_markdown(index=False, floatfmt='.6f'),
        '![Region effects](region_effects.png)',
        'Complete source, procedure, initialization and first-prompt changes accompany this report.'])
    (output / 'report.md').write_text(report + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', checks=checks,
        paired_events=len(paired), first_prompt_changes=len(changes), source_sha256=digest(__file__),
        inputs={str(p / 'analysis/events.csv'): digest(p / 'analysis/events.csv')
                for p in [Path(config['reference_run']), run]}, bootstrap_draws=10000, bootstrap_seed=20261007))
    print(summary.to_string(index=False), flush=True)
    print(contrasts.to_string(index=False), flush=True)
    print(average.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
