import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path

from plot_reusable_component_comparison import ROOT, POPS, METRICS, SEEDS, OWN, SAME, MARKERS
from plot_reusable_component_comparison import plt, np, Line2D

RUNS = {'0445': ROOT / 'results/runs/20260930T0445Z_component_memory_v1',
        '0535': ROOT / 'results/runs/20260930T0535Z_reusable_component_memory_v1',
        '0630': ROOT / 'results/runs/20260930T0630Z_crossfit_component_discovery_v1'}
OUT = ROOT / 'figures/component_memory_20260930/crossfit_comparison'


def write_csv(output, name, rows):
    with (output / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-shared-effect', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = dict(RUNS)
    if args.include_shared_effect:
        runs['0730'] = ROOT / 'results/runs/20260930T0730Z_shared_effect_components_v1'
    latest = list(runs)[-1]
    output = args.output or (OUT.parent / 'shared_effect_comparison' if args.include_shared_effect else OUT)
    figure_name = 'shared_effect_component_comparison' if args.include_shared_effect else 'crossfit_component_comparison'
    figure_height = 9.4 if args.include_shared_effect else 8.6
    aggregate_rows, seed_rows, procedure_rows, selections = [], [], [], []
    index, aggregates, identities = {}, {}, {}
    for stage, directory in runs.items():
        source = directory / 'summary.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        assert data['status'] == 'COMPLETE' and data['seeds'] == SEEDS
        assert data['independent_unit'] == 'procedure' and len(data['procedure_rows']) == 480
        identities[stage] = dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for row in data['procedure_rows']:
            key = (stage, row['population'], row['method'], row['mode'], row['seed'], row['video'])
            assert key not in index and all(np.isfinite(row[m]) for m in METRICS)
            index[key] = row
            procedure_rows.append(dict(stage=stage, population=key[1], method=key[2], threshold_mode=key[3],
                seed=key[4], procedure=key[5], threshold=row['threshold'],
                **{m + '_percent': 100 * row[m] for m in METRICS}))
        for row in data['aggregates']:
            key = (stage, row['population'], row['method'], row['mode'])
            aggregates[key] = row
            for seed in SEEDS:
                values = [r for k, r in index.items() if k[:4] == key and k[4] == seed]
                assert len(values) == 5 and len({r['video'] for r in values}) == 5
                for metric in METRICS:
                    assert np.isclose(np.mean([r[metric] for r in values]), row['seed_results'][str(seed)][metric], atol=1e-12)
                seed_rows.append(dict(stage=stage, population=key[1], method=key[2], threshold_mode=key[3], seed=seed,
                    **{m + '_percent': 100 * row['seed_results'][str(seed)][m] for m in METRICS}))
            for metric in METRICS:
                assert np.isclose(np.mean([row['seed_results'][str(s)][metric] for s in SEEDS]), row[metric], atol=1e-12)
            aggregate_rows.append(dict(stage=stage, population=key[1], method=key[2], threshold_mode=key[3],
                **{m + '_percent': 100 * row[m] for m in METRICS}))
        for record in data['source_edit_diagnostics']:
            for control in record['random_controls'].values():
                assert control['gain_value_multiset_preserved']
                assert np.isclose(control['applied_delta_norm'], control['learned_delta_norm'], atol=1e-10)
        for method in ['sparse_edit', 'dense_edit']:
            for row in json.loads((directory / f'component_selection_{method}.json').read_text(encoding='utf-8')):
                selections.append(dict(stage=stage, method=method, seed=row['seed'], **row['selected_candidate']))
    paired = []
    for key, current in index.items():
        if key[0] != latest:
            continue
        for previous in list(runs)[:-1]:
            earlier = index[(previous, *key[1:])]
            paired.append(dict(comparison=f'{latest}_minus_{previous}', population=key[1], method=key[2], threshold_mode=key[3],
                seed=key[4], procedure=key[5], **{m + '_change_pp': 100 * (current[m] - earlier[m]) for m in METRICS}))
    assert len(paired) == 480 * (len(runs) - 1)
    for pop in POPS:
        reference = aggregates[('0445', pop, 'reference_supcon', OWN)]
        for stage in runs:
            assert all(aggregates[(stage, pop, 'reference_supcon', OWN)][m] == reference[m] for m in METRICS)
        for stage in ['0445', '0535']:
            assert all(aggregates[(stage, pop, 'dense_edit', OWN)][m] == reference[m] for m in METRICS)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output, 'aggregate_metrics.csv', aggregate_rows)
    write_csv(output, 'seed_metrics.csv', seed_rows)
    write_csv(output, 'procedure_metrics.csv', procedure_rows)
    write_csv(output, 'paired_stage_changes.csv', paired)
    write_csv(output, 'selected_components.csv', selections)

    sparse_random = [f'sparse_edit__random{i}' for i in range(3)]
    dense_random = [f'dense_edit__random{i}' for i in range(3)]
    groups = [('0445', ['reference_supcon'], OWN, 'Reference', '#666666'),
              ('0445', ['sparse_edit'], OWN, '0445 SAE', '#0072B2'),
              ('0535', ['sparse_edit'], OWN, '0535 SAE', '#009E73'),
              ('0630', ['sparse_edit'], OWN, '0630 SAE', '#8E4A9E'),
              ('0630', ['dense_edit'], OWN, '0630 dense', '#D55E00')]
    sparse_color, dense_color = '#8E4A9E', '#D55E00'
    if args.include_shared_effect:
        sparse_color, dense_color = '#A34D70', '#8C6418'
        groups += [('0730', ['sparse_edit'], OWN, '0730 SAE', sparse_color),
                   ('0730', ['dense_edit'], OWN, '0730 dense', dense_color)]
    groups += [(latest, sparse_random, OWN, 'SAE random: own', sparse_color),
               (latest, sparse_random, SAME, 'SAE random: same', sparse_color),
               (latest, dense_random, OWN, 'Dense random: own', dense_color),
               (latest, dense_random, SAME, 'Dense random: same', dense_color)]
    style = {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 9,
             'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(4.8, figure_height), sharey=True)
        fig.subplots_adjust(left=0.28, right=0.98, bottom=0.21, top=0.92, wspace=0.25, hspace=0.40)
        for col, pop in enumerate(POPS):
            for metricnum, metric in enumerate(METRICS):
                ax = axes[metricnum, col]
                all_values = []
                for groupnum, (stage, methods, mode, label, color) in enumerate(groups):
                    for methodnum, method in enumerate(methods):
                        record = aggregates[(stage, pop, method, mode)]
                        for seednum, seed in enumerate(SEEDS):
                            offset = (seednum - 1) * 0.085 + ((methodnum - 1) * 0.25 if len(methods) == 3 else 0)
                            value = 100 * record['seed_results'][str(seed)][metric]
                            all_values.append(value)
                            ax.scatter(value, groupnum + offset, s=14, marker=MARKERS[seednum], linewidths=0.6,
                                edgecolors=color, facecolors='white' if mode == SAME else color, zorder=3)
                    avg = 100 * np.mean([aggregates[(stage, pop, method, mode)][metric] for method in methods])
                    ax.plot([avg, avg], [groupnum - 0.38, groupnum + 0.38], color=color, lw=0.8, zorder=2)
                ax.set_yticks(range(len(groups)), [group[3] for group in groups])
                ax.set_ylim(len(groups) - 0.45, -0.55)
                ax.tick_params(axis='y', length=0)
                ax.grid(axis='x', color='#DDDDDD', lw=0.5)
                ax.spines[['top', 'right']].set_visible(False)
                if metricnum == 0:
                    ax.set_title(('Development' if col == 0 else 'Examined extension') + '\n5 procedures', pad=10)
                    ax.set_xlim((35, 45) if col == 0 else (6.5, 12))
                    ax.set_xticks([35, 40, 45] if col == 0 else [7, 9, 11])
                elif metricnum == 1:
                    ax.set_xlim(98, 100.2)
                    ax.set_xticks([98, 99, 100])
                else:
                    ax.set_xlim(96, 100.35)
                    ax.set_xticks([96, 98, 100])
                assert min(all_values) >= ax.get_xlim()[0] and max(all_values) <= ax.get_xlim()[1]
                ax.set_xlabel(['Repeat removal (%)', 'Other-prompt retention (%)', 'First-prompt retention (%)'][metricnum])
                ax.text(-0.035, 1.03, chr(65 + metricnum * 2 + col), transform=ax.transAxes, fontweight='bold')
        handles = [Line2D([], [], marker=marker, linestyle='none', color='#444444', markersize=4,
                          label=str(seed)) for seed, marker in zip(SEEDS, MARKERS)]
        fig.legend(handles=handles, ncol=3, frameon=False, bbox_to_anchor=(0.61, 0.128),
                   loc='lower center', fontsize=8, columnspacing=0.9)
        fig.text(0.02, 0.034,
            'Symbols: seed means. Vertical marks: means over displayed points.\n'
            f'All random rows show {latest}, including all 3 permutations.\n'
            f'Own: separately calibrated; same: corresponding {latest} method threshold.\n'
            '0445 and 0535 dense results equal the reference.\n'
            'All 10 procedures were examined; seeds are repeated model fits.', fontsize=8, va='bottom')
        fig.savefig(output / f'{figure_name}.pdf')
        fig.savefig(output / f'{figure_name}.png', dpi=300)
        plt.close(fig)

    lines = ['# ' + ('Four-stage' if args.include_shared_effect else 'Three-stage') + ' component discovery comparison', '',
        '0445 uses fitting-procedure mean component benefit minus harm. 0535 uses the minimum remaining effect '
        'after each fitting-procedure omission. 0630 returns to the mean-effect criterion and measures each discovery '
        'procedure through an identity head trained without it. Discovery-head training scope changes in 0630; '
        'the deployment head and saved application evaluation remain fixed.' +
        (' 0730 requires positive benefit and positive benefit minus harm under both the original and crossfit '
         'discovery heads, then ranks components by the smaller benefit-minus-harm value.' if args.include_shared_effect else ''), '',
        'There are five procedures in each population and three paired model seeds. All ten procedures are development-exposed, '
        'including the previously examined extension. Procedures are independent units; seeds, source episodes and frames '
        'are repeated or nested observations. Values average lesions within episode, episodes within procedure, '
        'procedures equally, and then seeds equally. These saved operating points have different achieved protection.', '',
        'Cells report **removal / other-prompt retention / first-prompt retention (%)**. Random rows below average '
        'all three fixed norm-matched permutations; complete CSV tables preserve each permutation. Own thresholds '
        'are calibrated separately on development; same thresholds are the corresponding learned method\'s '
        'threshold within that stage. No threshold or model selection is recomputed.']
    compact = [('Reference', ['reference_supcon'], OWN), ('SAE', ['sparse_edit'], OWN), ('Dense', ['dense_edit'], OWN),
               ('SAE random, own', sparse_random, OWN), ('SAE random, same', sparse_random, SAME),
               ('Dense random, own', dense_random, OWN), ('Dense random, same', dense_random, SAME)]
    for pop in POPS:
        lines += ['', f'## {pop}', '', '| Method | ' + ' | '.join(runs) + ' |', '|---|' + '---:|' * len(runs)]
        for label, methods, mode in compact:
            cells = [' / '.join(f"{100 * np.mean([aggregates[(stage, pop, method, mode)][m] for method in methods]):.4f}"
                               for m in METRICS) for stage in runs]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines += ['', f'## {latest} individual seed results', '',
        '| Population | Method | Seed | Removal (%) | Other retention (%) | First prompt (%) |',
        '|---|---|---:|---:|---:|---:|']
    for row in seed_rows:
        if row['stage'] == latest and row['method'] in ['sparse_edit', 'dense_edit']:
            lines.append(f"| {row['population']} | {row['method']} | {row['seed']} | " +
                         ' | '.join(f"{row[m + '_percent']:.4f}" for m in METRICS) + ' |')
    lines += ['', f'## {latest} procedure results', '',
        'Each cell averages the same procedure over three seeds and reports the same three percentages.', '',
        '| Population | Procedure | Reference | SAE | Dense |', '|---|---|---:|---:|---:|']
    for pop in POPS:
        videos = sorted({r['procedure'] for r in procedure_rows if r['population'] == pop})
        assert len(videos) == 5
        for video in videos:
            cells = []
            for method in ['reference_supcon', 'sparse_edit', 'dense_edit']:
                values = [index[(latest, pop, method, OWN, seed, video)] for seed in SEEDS]
                cells.append(' / '.join(f"{100 * np.mean([r[m] for r in values]):.4f}" for m in METRICS))
            lines.append(f'| {pop} | {video} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Interpretation', '']
    for pop in POPS:
        for method in ['sparse_edit', 'dense_edit']:
            current = aggregates[(latest, pop, method, OWN)]
            reference = aggregates[(latest, pop, 'reference_supcon', OWN)]
            lines.append(f'- {pop}, {latest} {method} minus reference: ' +
                ', '.join(f'{m} {100 * (current[m] - reference[m]):+.4f} pp' for m in METRICS) + '.')
            for mode in [OWN, SAME]:
                random = [aggregates[(latest, pop, f'{method}__random{i}', mode)] for i in range(3)]
                lines.append(f'- {pop}, {latest} {method} minus mean random ({mode}): ' +
                    ', '.join(f'{m} {100 * (current[m] - np.mean([r[m] for r in random])):+.4f} pp' for m in METRICS) + '.')
    if args.include_shared_effect:
        lines += ['- The 0730 SAE selects component caps of 4, 16 and 16 with attenuation strength 1.0 for seeds '
                  '20260929, 20260930 and 20261001. Dense selects a cap of 4 at strength 0.5 for seed 20260929 '
                  'and zero edit for the other seeds. Selection records are exported unchanged.',
                  '- Shared-effect selection increases SAE removal relative to 0630, with lower other-prompt '
                  'retention on the examined extension. It does not exceed reference removal at the saved '
                  'operating points. These adaptive-development results do not supply independent confirmation '
                  'of a reusable component benefit.']
    else:
        lines += ['- The 0630 SAE selects 64 fully attenuated components in every seed. Dense selects 16 components at '
              'strength 0.25 for seed 20260929 and zero edit for the other seeds. Selection records are exported unchanged.',
              '- Crossfit discovery does not improve the SAE removal result at the saved operating points. Its extension '
              'other-prompt retention increases, so removal and protection must be read together. These adaptive-development '
              'results do not supply independent confirmation of a reusable component benefit.']
    reproduction = 'artifacts/environments/modern/Scripts/python.exe scripts/plot_crossfit_component_comparison.py'
    if args.include_shared_effect:
        reproduction += ' --include-shared-effect'
    if args.output:
        reproduction += f' --output "{args.output.as_posix()}"'
    lines += ['', '## Data and reproduction', '',
              f'All {len(procedure_rows)} procedure/seed operating-point rows, {len(seed_rows)} seed summaries and '
              f'{len(aggregate_rows)} aggregate rows are exported in percent. The {len(paired)} paired changes compare '
              f'{latest} with each preceding stage in percentage points. All methods, populations, permutations and '
              f'threshold modes remain in the CSV files; the figure emphasizes the {latest} random controls.', '',
              f'`{reproduction}`']
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = dict(sources=identities, script=str(Path(__file__).relative_to(ROOT)),
        procedure_rows=len(procedure_rows), seed_rows=len(seed_rows), aggregate_rows=len(aggregate_rows), paired_rows=len(paired),
        seeds=SEEDS, independent_unit='procedure', figure_inches=[4.8, figure_height], png_dpi=300)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
