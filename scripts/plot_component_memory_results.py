import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/environments/endomind-inference-overlay'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RUN = ROOT / 'results/runs/20260930T0445Z_component_memory_v1'
OUT = ROOT / 'figures/component_memory_20260930'
POPS = ['development', 'previously_examined_extension']
METRICS = ['removal', 'retention', 'first_prompt']
SEEDS = [20260929, 20260930, 20261001]
OWN = 'own_operating_point'
SAME = 'learned_method_threshold'
MARKERS = ['o', '^', 's']


def mean(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    return float(values.mean())


def save_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = RUN / 'summary.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'COMPLETE' and data['seeds'] == SEEDS
    rows = data['procedure_rows']
    assert len(rows) == 480
    index = {(r['population'], r['method'], r['mode'], r['seed'], r['video']): r for r in rows}
    assert len(index) == len(rows)
    videos = {pop: sorted({r['video'] for r in rows if r['population'] == pop}) for pop in POPS}
    assert all(len(value) == 5 for value in videos.values())
    aggregates = {}
    seed_rows = []
    aggregate_rows = []
    for aggregate in data['aggregates']:
        key = (aggregate['population'], aggregate['method'], aggregate['mode'])
        aggregates[key] = aggregate
        for seed in SEEDS:
            selected = [index[(*key, seed, video)] for video in videos[key[0]]]
            record = dict(population=key[0], method=key[1], operating_point=key[2], seed=seed,
                          procedures=len(selected))
            for metric in METRICS:
                result = mean([r[metric] for r in selected])
                assert np.isclose(result, aggregate['seed_results'][str(seed)][metric], atol=1e-12)
                record[metric + '_percent'] = 100 * result
            seed_rows.append(record)
        record = dict(population=key[0], method=key[1], operating_point=key[2], procedures=5, seeds=3)
        for metric in METRICS:
            result = mean([aggregate['seed_results'][str(seed)][metric] for seed in SEEDS])
            assert np.isclose(result, aggregate[metric], atol=1e-12)
            record[metric + '_percent'] = 100 * result
        aggregate_rows.append(record)

    contrasts = data['paired_procedure_contrasts']
    assert len(contrasts) == 510
    contrast_map = {'sparse_minus_dense': ('sparse_edit', 'dense_edit', OWN)}
    for method in ['sparse_edit', 'dense_edit']:
        for baseline in ['reference_supcon', 'spatial_weighted']:
            contrast_map[f'{method}_minus_{baseline}'] = (method, baseline, OWN)
        for random in range(3):
            for mode in [OWN, SAME]:
                contrast_map[f'{method}_minus_random{random}__{mode}'] = (
                    method, f'{method}__random{random}', mode)
    for contrast in contrasts:
        method, baseline, mode = contrast_map[contrast['contrast']]
        left = index[(contrast['population'], method, OWN, contrast['seed'], contrast['video'])]
        right = index[(contrast['population'], baseline, mode, contrast['seed'], contrast['video'])]
        for metric in METRICS:
            assert np.isclose(left[metric] - right[metric], contrast[metric], atol=1e-12)
    procedure_pairs = []
    seed_pairs = []
    contrast_names = sorted({r['contrast'] for r in contrasts})
    for pop in POPS:
        for name in contrast_names:
            for video in videos[pop]:
                selected = [r for r in contrasts if r['population'] == pop and r['contrast'] == name and r['video'] == video]
                assert len(selected) == 3 and {r['seed'] for r in selected} == set(SEEDS)
                procedure_pairs.append(dict(population=pop, contrast=name, video=video,
                    **{metric + '_change_pp': 100 * mean([r[metric] for r in selected]) for metric in METRICS}))
            for seed in SEEDS:
                selected = [r for r in contrasts if r['population'] == pop and r['contrast'] == name and r['seed'] == seed]
                assert len(selected) == 5
                seed_pairs.append(dict(population=pop, contrast=name, seed=seed,
                    **{metric + '_change_pp': 100 * mean([r[metric] for r in selected]) for metric in METRICS}))

    control_rows = []
    for row in data['source_edit_diagnostics']:
        for name, control in row['random_controls'].items():
            assert control['gain_value_multiset_preserved']
            assert np.isclose(control['applied_delta_norm'], control['learned_delta_norm'], atol=1e-10)
            control_rows.append(dict(population=row['population'], video=row['video'],
                episode_id=row['episode_id'], seed=row['seed'], method=name,
                status=control['status'], learned_delta_norm=control['learned_delta_norm'],
                applied_delta_norm=control['applied_delta_norm']))
        assert row['reencoded_original_max_error'] == 0
        assert row['spatial_equal_weight_raw_max_error'] == 0
        for check in row['zero_edit_checks'].values():
            assert check['raw_max_error'] == check['reference_memory_max_error'] == 0

    selections = []
    for method in ['dense_edit', 'sparse_edit']:
        for row in json.loads((RUN / f'component_selection_{method}.json').read_text(encoding='utf-8')):
            selections.append(dict(method=method, seed=row['seed'], **row['selected_candidate']))
    assert all(r['strength'] == 0 for r in selections if r['method'] == 'dense_edit')
    for pop in POPS:
        reference = aggregates[(pop, 'reference_supcon', OWN)]
        for method in ['dense_edit'] + [f'dense_edit__random{n}' for n in range(3)]:
            for metric in METRICS:
                assert aggregates[(pop, method, OWN)][metric] == reference[metric]

    OUT.mkdir(parents=True, exist_ok=True)
    save_csv('aggregate_metrics.csv', aggregate_rows)
    save_csv('seed_metrics.csv', seed_rows)
    save_csv('procedure_metrics.csv', rows)
    save_csv('paired_seed_procedure_effects.csv', contrasts)
    save_csv('paired_procedure_effects.csv', procedure_pairs)
    save_csv('paired_seed_effects.csv', seed_pairs)
    save_csv('random_control_norms.csv', control_rows)
    save_csv('selected_components.csv', selections)

    groups = [(['reference_supcon'], OWN, 'Reference / dense', '#666666'),
              (['sparse_edit'], OWN, 'SAE edit', '#0072B2'),
              (['spatial_weighted'], OWN, 'Spatial weighting', '#D55E00'),
              ([f'sparse_edit__random{n}' for n in range(3)], OWN, 'Random: own threshold', '#7C6034'),
              ([f'sparse_edit__random{n}' for n in range(3)], SAME, 'Random: SAE threshold', '#7C6034')]
    style = {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 9,
             'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'axes.titlesize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(4.8, 6.5), sharey=True)
        fig.subplots_adjust(left=0.30, right=0.98, bottom=0.20, top=0.90, wspace=0.25, hspace=0.55)
        titles = ['Development\n5 procedures', 'Examined extension\n5 procedures']
        axis_labels = ['Repeat removal (%)', 'Other-prompt retention (%)', 'First-prompt retention (%)']
        for col, pop in enumerate(POPS):
            for rownum, metric in enumerate(METRICS):
                ax = axes[rownum, col]
                for groupnum, (methods, mode, label, color) in enumerate(groups):
                    for methodnum, method in enumerate(methods):
                        for seednum, seed in enumerate(SEEDS):
                            aggregate = aggregates[(pop, method, mode)]
                            offset = (seednum - 1) * 0.10 + ((methodnum - 1) * 0.25 if len(methods) == 3 else 0)
                            value = aggregate['seed_results'][str(seed)][metric] * 100
                            ax.scatter(value, groupnum + offset, marker=MARKERS[seednum], s=15,
                                       facecolors='white' if mode == SAME else color,
                                       edgecolors=color, linewidths=0.6, zorder=3)
                    avg = mean([aggregates[(pop, method, mode)][metric] for method in methods]) * 100
                    ax.plot([avg, avg], [groupnum - 0.38, groupnum + 0.38], color=color, lw=0.8, zorder=2)
                ax.set_yticks(range(len(groups)), [group[2] for group in groups])
                ax.set_ylim(4.6, -0.6)
                ax.set_xlabel(axis_labels[rownum])
                ax.grid(axis='x', color='#DDDDDD', linewidth=0.5)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='y', length=0)
                if rownum == 0:
                    ax.set_title(titles[col], pad=10)
                    ax.set_xlim(-1, 50 if col == 0 else 17)
                    ax.set_xticks([0, 20, 40] if col == 0 else [0, 5, 10, 15])
                elif rownum == 1:
                    ax.set_xlim(96.7, 100.25)
                    ax.set_xticks([97, 98, 99, 100])
                else:
                    ax.set_xlim(91.5, 100.55)
                    ax.set_xticks([92, 96, 100])
                ax.text(-0.04, 1.03, chr(65 + rownum * 2 + col), transform=ax.transAxes,
                        fontweight='bold', va='bottom')
        handles = [Line2D([], [], marker=marker, linestyle='none', color='#444444', markersize=4,
                          label=str(seed)) for seed, marker in zip(SEEDS, MARKERS)]
        fig.legend(handles=handles, ncol=3, frameon=False, loc='lower center',
                   bbox_to_anchor=(0.61, 0.078), fontsize=8, columnspacing=1)
        fig.text(0.02, 0.031, 'Symbols: seed means over procedures. Vertical marks: mean over seeds.\n'
                 'Random rows include all 3 fixed permutations. All 10 procedures were examined.',
                 fontsize=8, va='bottom')
        fig.savefig(OUT / 'component_memory_main.pdf')
        fig.savefig(OUT / 'component_memory_main.png', dpi=300)
        plt.close(fig)

    lines = ['# Component memory analysis', '',
             'Source: `results/runs/20260930T0445Z_component_memory_v1/summary.json`.', '',
             'All ten procedures were previously exposed. These are development and examined-extension results. '
             'Each population has five procedures and three algorithm seeds. Procedure means retain the saved '
             'lesion/episode hierarchy; seeds are repeated fits. All reported differences are descriptive.', '',
             '| Population | Method / threshold | Removal (%) | Other retention (%) | First prompt (%) |',
             '|---|---|---:|---:|---:|']
    for pop in POPS:
        for aggregate in data['aggregates']:
            if aggregate['population'] != pop:
                continue
            lines.append(f"| {pop} | {aggregate['method']} / {aggregate['mode']} | " +
                         ' | '.join(f"{100 * aggregate[m]:.4f}" for m in METRICS) + ' |')
    lines += ['', '## Paired SAE minus reference', '',
              '| Population | Seed | Removal (pp) | Other retention (pp) | First prompt (pp) |',
              '|---|---:|---:|---:|---:|']
    for r in seed_pairs:
        if r['contrast'] == 'sparse_edit_minus_reference_supcon':
            lines.append(f"| {r['population']} | {r['seed']} | " +
                         ' | '.join(f"{r[m + '_change_pp']:+.4f}" for m in METRICS) + ' |')
    lines += ['', '| Population | Procedure | Removal (pp) | Other retention (pp) | First prompt (pp) |',
              '|---|---|---:|---:|---:|']
    for r in procedure_pairs:
        if r['contrast'] == 'sparse_edit_minus_reference_supcon':
            lines.append(f"| {r['population']} | {r['video']} | " +
                         ' | '.join(f"{r[m + '_change_pp']:+.4f}" for m in METRICS) + ' |')
    lines += ['', '## Interpretation', '']
    for pop in POPS:
        sparse = aggregates[(pop, 'sparse_edit', OWN)]
        reference = aggregates[(pop, 'reference_supcon', OWN)]
        chosen = [r for r in procedure_pairs if r['population'] == pop and r['contrast'] == 'sparse_edit_minus_reference_supcon']
        positive = sum(r['removal_change_pp'] > 1e-10 for r in chosen)
        negative = sum(r['removal_change_pp'] < -1e-10 for r in chosen)
        lines.append(f"- {pop}: SAE minus reference removal {100 * (sparse['removal'] - reference['removal']):+.4f} pp; "
                     f"other retention {100 * (sparse['retention'] - reference['retention']):+.4f} pp; "
                     f"first-prompt retention {100 * sparse['first_prompt']:.2f}%. "
                     f"Procedure removal effects: {positive} positive, {negative} negative, {5-positive-negative} zero.")
        for mode in [OWN, SAME]:
            randoms = [aggregates[(pop, f'sparse_edit__random{n}', mode)] for n in range(3)]
            lines.append(f"- {pop}, {mode}: SAE minus mean of all three random controls: " +
                ', '.join(f"{metric} {100 * (sparse[metric] - mean([r[metric] for r in randoms])):+.4f} pp" for metric in METRICS) + '.')
    lines += ['- Dense selection chose zero edit for all three seeds. Dense and its random controls equal the reference.',
              '- SAE removal exceeds each independently calibrated random control in both populations. At the SAE threshold, '
              'all three random controls have higher extension removal and lower other-prompt retention than SAE. '
              'These operating points show a removal/protection tradeoff; removal alone does not establish component specificity.',
              '- Zero-edit and descriptor re-encoding errors are zero in the saved diagnostics. All random edit norms '
              'match the corresponding learned norm within numerical tolerance. The complete norm table is saved.', '',
              '## Reproduction and figure', '',
              '`artifacts/environments/modern/Scripts/python.exe scripts/plot_component_memory_results.py`', '',
              'The figure displays every seed mean and all three random permutations. Vertical marks represent equal-seed '
              'means, with an additional equal-permutation average for random rows. Axes use distinct units and limits; '
              'no confidence intervals are inferred from seeds. Reference and dense share one row because their metrics agree.', '',
              'CSV assets preserve aggregate, seed, procedure, seed-by-procedure contrasts, selection, and norm diagnostics. '
              '`procedure_metrics.csv` and `paired_seed_procedure_effects.csv` retain the original fraction units; '
              'other metric tables explicitly label percent or percentage-point units.']
    (OUT / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    script=str(Path(__file__).relative_to(ROOT)),
                    procedure_rows=len(rows), paired_seed_procedure_rows=len(contrasts),
                    random_control_rows=len(control_rows), width_inches=4.8, height_inches=6.5, png_dpi=300,
                    independent_unit='procedure', seeds=SEEDS)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
