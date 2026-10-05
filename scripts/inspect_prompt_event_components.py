import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inspect(run):
    output = run / 'analysis'
    rows = pd.read_csv(output / 'events.csv')
    procedures = pd.read_csv(output / 'procedures.csv')
    seeds = pd.read_csv(output / 'seeds.csv')
    summary = pd.read_csv(output / 'summary.csv')
    metrics = ['repeat_removal', 'other_retention', 'first_retention',
               'false_removal_correction', 'correct_repeat_damage']
    expected = procedures.groupby(['method', 'variant', 'seed'])[metrics].mean().reset_index()
    pd.testing.assert_frame_equal(expected, seeds, check_exact=False, atol=1e-14, rtol=0)
    expected = seeds.groupby(['method', 'variant'])[metrics].mean().reset_index()
    pd.testing.assert_frame_equal(expected, summary, check_exact=False, atol=1e-14, rtol=0)
    index = ['video', 'method', 'seed', 'event_id']
    before = rows[rows.variant == 'before'].set_index(index)
    effects = []
    for variant, group in rows.groupby('variant'):
        current = group.set_index(index)
        assert current.index.equals(before.index)
        np.testing.assert_array_equal(current.before_retained, before.retained)
        for (method, seed), part in group.groupby(['method', 'seed']):
            wrong = ~part.same_identity & ~part.before_retained
            correct = part.same_identity & ~part.before_retained
            effects.append(dict(method=method, seed=seed, variant=variant,
                wrong_removals=int(wrong.sum()), corrected=int((wrong & part.retained).sum()),
                correct_repeat_removals=int(correct.sum()), damaged=int((correct & part.retained).sum()),
                new_wrong_removals=int((~part.same_identity & part.before_retained & ~part.retained).sum()),
                changed_events=int((part.retained != part.before_retained).sum())))
    effects = pd.DataFrame(effects)
    effects.to_csv(output / 'count_inspection.csv', index=False)
    keys = ['video', 'method', 'seed']
    base = procedures[procedures.variant == 'before'].set_index(keys)[metrics]
    deltas = []
    for variant, group in procedures.groupby('variant'):
        values = group.set_index(keys)[metrics] - base
        values['variant'] = variant
        deltas.append(values.reset_index())
    deltas = pd.concat(deltas, ignore_index=True)
    deltas.to_csv(output / 'procedure_deltas.csv', index=False)
    contribution = pd.read_csv(output / 'source_contributions.csv')
    contribution_summary = []
    for (method, variant), part in contribution.groupby(['method', 'variant']):
        contribution_summary.append(dict(method=method, variant=variant,
            positive_sources=int((part.other_minus_same > 0).sum()),
            negative_sources=int((part.other_minus_same < 0).sum()),
            zero_sources=int((part.other_minus_same == 0).sum()),
            undefined_sources=int(part.other_minus_same.isna().sum()),
            other_mean=float(part.other_contribution.mean()), same_mean=float(part.same_contribution.mean())))
    pd.DataFrame(contribution_summary).to_csv(output / 'contribution_inspection.csv', index=False)
    mass_rows = []
    for path in sorted((run / 'evaluation').glob('*/*/scores.npz')):
        data = read(run / 'inputs' / path.parent.parent.name / 'events.json')
        with np.load(path) as saved:
            unit, positions = saved['unit_codes'], saved['positions']
            lookup = {int(p): i for i, p in enumerate(positions)}
            for variant in ['selected', 'random0', 'random1']:
                chosen = saved[variant + '_features']
                mass = (unit[:, chosen] ** 2).sum(1)
                for episode in data['episodes']:
                    source = lookup[episode['source_position']]
                    mass_rows.append(dict(video=data['video'], model=path.parent.name, variant=variant,
                        episode=episode['episode_id'], source_squared_mass=float(mass[source]),
                        query_mean_squared_mass=float(mass.mean()), query_zero_fraction=float((mass == 0).mean())))
    pd.DataFrame(mass_rows).to_csv(output / 'component_mass.csv', index=False)
    receipts = [read(path) for path in sorted((run / 'evaluation').glob('*/*/complete.json'))]
    assert len(receipts) == 30 and all(r['original_decisions_exact'] for r in receipts)
    result = dict(status='PASS', unique_events=len(before) // 6, model_evaluations=30,
        maximum_original_score_error=max(r['original_score_max_error'] for r in receipts),
        counts=effects.groupby(['method', 'variant']).sum(numeric_only=True).drop(columns='seed').reset_index().to_dict('records'),
        contribution=contribution_summary,
        procedure_effects=deltas.groupby(['video', 'method', 'variant'])[metrics[:3]].mean().reset_index().to_dict('records'))
    (output / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metric, title in zip(axes, metrics[:2], ['Repeat-removal change', 'Other-retention change']):
        for offset, (method, part) in enumerate(deltas[deltas.variant == 'selected'].groupby('method')):
            means = part.groupby('video')[metric].mean() * 100
            ax.scatter(np.arange(len(means)) + .12 * offset, means, label='SAE' if method.endswith('sparse') else 'Dense')
            ax.set_xticks(range(len(means)), means.index, rotation=25)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel('Percentage points, selected minus before')
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(output / 'procedure_changes.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'procedure_effects'}, indent=2))


def draw(ax, path, box, title):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        ax.imshow(rgb)
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='#ffe100', linewidth=1.6))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def images(run):
    output = run / 'analysis' / 'images'
    output.mkdir(exist_ok=True)
    rows = pd.read_csv(run / 'analysis/events.csv')
    evidence = []
    for directory in sorted((run / 'inputs').iterdir()):
        if not directory.is_dir():
            continue
        data = read(directory / 'events.json')
        first = min(data['episodes'], key=lambda e: (e['click']['time'], e['episode_id']))
        events = [(i, e) for i, e in enumerate(data['events']) if e['episode'] == first['episode_id']]
        cases = []
        for role, predicate in [('same_return', lambda e: e['same_identity'] and e['stratum'] == 'later_reappearance'),
                                ('other_first', lambda e: not e['same_identity'] and e['first_prompt'])]:
            candidates = [(i, e) for i, e in events if predicate(e)]
            if candidates:
                index, event = min(candidates, key=lambda p: (p[1]['frame'], p[1]['lesion']))
                cases.append((role, index, event, first, 20261005, 'earliest chronological eligible event for earliest source'))
        wrong = rows[(rows.video == data['video']) & (rows.method == 'p27v4_token_sparse') &
                     (rows.variant == 'before') & ~rows.same_identity & ~rows.retained]
        if len(wrong):
            chosen = wrong.sort_values(['seed', 'frame', 'episode', 'lesion']).iloc[0]
            event = data['events'][int(chosen.event_id)]
            episode = next(e for e in data['episodes'] if e['episode_id'] == event['episode'])
            cases.append(('false_removal', int(chosen.event_id), event, episode, int(chosen.seed),
                          'post-hoc failure illustration: first available seed, then earliest mistaken event'))
        for role, index, event, episode, seed, rule in cases:
            local = rows[(rows.video == data['video']) & (rows.method == 'p27v4_token_sparse') &
                         (rows.seed == seed) & (rows.event_id == index)].set_index('variant')
            frame = data['frames'][str(event['output'])]
            match = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
            image_root = Path(data['definition']['frame_root']) / data['video']
            source_path = image_root / f"{episode['click']['input_frame']:06d}.jpg"
            query_path = image_root / f"{event['frame']:06d}.jpg"
            before, selected = local.loc['before'], local.loc['selected']
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
            draw(axes[0], source_path, episode['click']['detection']['xyxy'],
                 f"Confirmed source | {episode['source_lesion_id']}\nFrame {episode['click']['input_frame']}")
            draw(axes[1], query_path, frame['detections'][match['prediction_index']]['xyxy'],
                 f"{'Same lesion' if event['same_identity'] else 'Other lesion'} | {event['lesion']} | {role}\n"
                 f"Score {before.matched_score:.5f} to {selected.matched_score:.5f}; threshold {before.threshold:.5f}\n"
                 f"Prompt retained {bool(before.retained)} to {bool(selected.retained)}")
            fig.suptitle(f"{data['video']} | SAE seed {seed} | Yellow: detector support", fontsize=11)
            name = data['video'] + '_' + role + '.png'
            fig.savefig(output / name, dpi=140)
            plt.close(fig)
            evidence.append(dict(video=data['video'], role=role, event_id=index, seed=seed, rule=rule,
                source=str(source_path), query=str(query_path), output=name,
                source_box=episode['click']['detection']['xyxy'], query_box=frame['detections'][match['prediction_index']]['xyxy'],
                selected_contribution=float(selected.component_contribution),
                transformation='Full original RGB frame, no color or contrast changes; original detector boxes.',
                interpretation='Visible image content and detection support; not an attribution or clinical concept label.'))
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('IMAGE_PAIRS', len(evidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--images', action='store_true')
    args = parser.parse_args()
    images(args.run) if args.images else inspect(args.run)
