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

from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def analyze(run, comparison='supplemental', output=None):
    config = read_json(run / 'config.json')
    output = output or run / 'comparison'
    output.mkdir(exist_ok=True)
    labels = ('global_objective', 'procedure_objective') if comparison == 'procedure' else ('real_only', 'supplemental')
    reference_label, candidate_label = labels
    title = 'Procedure-conditioned objective application comparison' if comparison == 'procedure' else 'Supplemental-source application comparison'
    description = ('The training objective changes under matched inputs, initialization and optimization budget.'
                   if comparison == 'procedure' else 'Source, view distribution and supervision change together.')
    keys = ['video', 'method', 'seed']
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    methods = ['adaptive_token_sparse', 'adaptive_token_dense']
    tables, rows = [], []
    checks = 0
    for source, folder in [(reference_label, Path(config['reference_run'])), (candidate_label, run)]:
        assert read_json(folder / 'analysis/summary.json')['status'] == 'COMPLETE'
        table = pd.read_csv(folder / 'analysis/events.csv')
        table = table[(table.variant == 'budget1024') & table.method.isin(methods)].copy()
        assert len(table) == 5394 * 6
        assert set(table.seed) == {20261007, 20261008, 20261009}
        saved = pd.read_csv(folder / 'analysis/procedures.csv')
        saved = saved[(saved.variant == 'budget1024') & saved.method.isin(methods)].set_index(keys)
        assert len(saved) == 30
        videos = set(table.video)
        calibration = [c for c in read_json(folder / 'analysis/calibration.json')
                       if c['budget'] == 1024 and any(c['key'].startswith(m + '_seed') for m in methods)]
        assert len(calibration) == 30
        for item in calibration:
            assert set(item['fit_procedures']) == videos - {item['video']}
            checks += 1
        for identity, group in table.groupby(keys):
            same, other = group[group.same_identity], group[~group.same_identity]
            first = other[other.first_prompt]
            values = [(~same.retained).mean(), other.retained.mean(), first.retained.mean()]
            np.testing.assert_allclose(values, saved.loc[identity, metrics].to_numpy(float), atol=1e-14)
            checks += 3
            rows.append(dict(zip(keys, identity), source=source, **dict(zip(metrics, values))))
        table['source'] = source
        tables.append(table)
    procedures = pd.DataFrame(rows)
    procedures.to_csv(output / 'procedures.csv', index=False)
    summary = procedures.groupby(['source', 'method'])[metrics].mean().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    seeds = procedures.groupby(['source', 'method', 'seed'])[metrics].mean()
    seeds.to_csv(output / 'seeds.csv')
    average = procedures.groupby(['video', 'source', 'method'])[metrics].mean()
    average.to_csv(output / 'procedure_means.csv')
    boot = np.random.default_rng(20261007).integers(0, 5, size=(10000, 5))
    contrasts, differences = [], []
    for method in methods:
        a = average.xs((candidate_label, method), level=['source', 'method']).sort_index()
        b = average.xs((reference_label, method), level=['source', 'method']).sort_index()
        assert a.index.equals(b.index) and len(a) == 5
        for metric in metrics:
            delta = (a[metric] - b[metric]).to_numpy()
            low, high = np.quantile(delta[boot].mean(1), [.025, .975])
            contrasts.append(dict(method=method, metric=metric, mean=delta.mean(), lower=low, upper=high))
            differences.extend(dict(method=method, metric=metric, video=v, difference=d)
                               for v, d in zip(a.index, delta, strict=True))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(output / 'contrasts.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_procedures.csv', index=False)
    paired = tables[0].merge(tables[1], on=keys + ['event_id'], suffixes=('_reference', '_candidate'), validate='one_to_one')
    assert len(paired) == 32364
    for column in ['episode', 'frame', 'lesion', 'same_identity', 'first_prompt', 'matched_position']:
        assert paired[column + '_reference'].equals(paired[column + '_candidate'])
    first = paired[(~paired.same_identity_reference) & paired.first_prompt_reference]
    assert len(first) == 26 * 6
    first.to_csv(output / 'first_prompts.csv', index=False)
    changes = first[first.retained_reference != first.retained_candidate]
    changes.to_csv(output / 'first_prompt_changes.csv', index=False)
    cases = changes.sort_values(['video', 'event_id', 'method', 'seed']).drop_duplicates(['video', 'event_id'])
    if len(cases):
        fig, axes = plt.subplots(len(cases), 2, figsize=(10, 3.5 * len(cases)), layout='constrained', squeeze=False)
        prefix = read_json(Path(config['prefix_run']) / 'config.json')
        for i, row in enumerate(cases.itertuples()):
            manifest = read_json(Path(prefix['event_reference']) / 'inputs' / row.video / 'events.json')
            source = next(e for e in manifest['episodes'] if e['episode_id'] == row.episode_reference)
            query = manifest['frames'][str(row.output_reference)]
            inputs = [(source['click']['input_frame'], source['source_info']['click']['detection']['xyxy'], 'Acknowledged source'),
                      (row.frame_reference, query['detections'][query['positions'].index(row.matched_position_reference)]['xyxy'], 'Other-lesion first event')]
            for j, (frame, box, label) in enumerate(inputs):
                with Image.open(Path(manifest['definition']['frame_root']) / row.video / f'{frame:06d}.jpg') as picture:
                    axes[i, j].imshow(picture)
                x0, y0, x1, y1 = box
                axes[i, j].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#00FFFF', linewidth=2))
                axes[i, j].set_title(f'{label} | {row.video} | frame {frame}', fontsize=9)
                axes[i, j].axis('off')
        fig.savefig(output / 'first_prompt_cases.png', dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    titles = ['Repeat events removed', 'Other events retained', 'First prompts retained']
    for ax, metric, title in zip(axes, metrics, titles, strict=True):
        for j, method in enumerate(methods):
            a = average.xs((reference_label, method), level=['source', 'method']).sort_index()
            b = average.xs((candidate_label, method), level=['source', 'method']).sort_index()
            for k in range(5):
                ax.plot([j - .15, j + .15], [a[metric].iloc[k] * 100, b[metric].iloc[k] * 100],
                        color='gray', alpha=.4, linewidth=1)
        for source, shift, color in [(reference_label, -.15, '#15678A'), (candidate_label, .15, '#CA653F')]:
            values = summary[summary.source == source].set_index('method').loc[methods, metric]
            ax.scatter(np.arange(2) + shift, values * 100, color=color, s=70, label=source, zorder=3)
        ax.set_xticks([0, 1], ['SAE', 'Dense'])
        ax.set_title(title)
        ax.set_ylabel('Procedure mean (%)')
        ax.set_ylim(max(0, ax.get_ylim()[0] - .2), min(100.5, ax.get_ylim()[1]))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed training budget | 5 exposed procedures | 3 initializations\nLines: paired procedures; markers: means')
    fig.savefig(output / 'source_effects.png', dpi=180)
    plt.close(fig)
    report = '\n\n'.join(['# ' + title,
        'Identical 5394 development events, six models per training condition and excluded-procedure calibration. '
        'All model results and first-prompt changes are retained. ' + description,
        summary.to_markdown(index=False, floatfmt='.6f'),
        'Paired differences average seeds within each procedure before 10000 bootstrap samples of five procedures. '
        'These development intervals do not establish independent generalization.',
        contrasts.to_markdown(index=False, floatfmt='.6f'),
        '![Source effects](source_effects.png)'])
    (output / 'report.md').write_text(report + '\n', encoding='utf-8')
    atomic_write_json(output / 'verification.json', dict(status='PASS', checks=checks, paired_events=len(paired),
        first_prompt_changes=len(changes), source_sha256=digest(__file__),
        inputs={str(p / 'analysis/events.csv'): digest(p / 'analysis/events.csv')
                for p in [Path(config['reference_run']), run]}, bootstrap_seed=20261007, bootstrap_draws=10000))
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(seeds.to_string())
    print(average.to_string())
    print(changes[['video', 'method', 'seed', 'event_id', 'retained_reference', 'retained_candidate']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--comparison', choices=['supplemental', 'procedure'], default='supplemental')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.run, args.comparison, args.output)
