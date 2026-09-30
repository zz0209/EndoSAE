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

RUNS = {'0445': ROOT / 'results/runs/20260930T0445Z_component_memory_v1',
        '0535': ROOT / 'results/runs/20260930T0535Z_reusable_component_memory_v1'}
OUT = ROOT / 'figures/component_memory_20260930/reusable_comparison'
POPS = ['development', 'previously_examined_extension']
METRICS = ['removal', 'retention', 'first_prompt']
SEEDS = [20260929, 20260930, 20261001]
OWN = 'own_operating_point'
SAME = 'learned_method_threshold'
MARKERS = ['o', '^', 's']


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index, aggregates, procedures, sources = {}, {}, [], {}
    aggregate_rows, seed_rows, selection_rows = [], [], []
    for stage, directory in RUNS.items():
        path = directory / 'summary.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['status'] == 'COMPLETE' and data['seeds'] == SEEDS
        assert data['independent_unit'] == 'procedure'
        sources[stage] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows = data['procedure_rows']
        assert len(rows) == 480
        for row in rows:
            key = (stage, row['population'], row['method'], row['mode'], row['seed'], row['video'])
            assert key not in index
            index[key] = row
            assert all(np.isfinite(row[metric]) for metric in METRICS)
            procedures.append(dict(stage=stage, population=row['population'], method=row['method'],
                threshold_mode=row['mode'], seed=row['seed'], procedure=row['video'], threshold=row['threshold'],
                **{metric + '_percent': 100 * row[metric] for metric in METRICS}))
        for row in data['aggregates']:
            key = (stage, row['population'], row['method'], row['mode'])
            aggregates[key] = row
            for seed in SEEDS:
                selected = [r for r in rows if (r['population'], r['method'], r['mode'], r['seed']) == (*key[1:], seed)]
                assert len(selected) == 5 and len({r['video'] for r in selected}) == 5
                values = {metric: np.mean([r[metric] for r in selected]) for metric in METRICS}
                assert all(np.isclose(values[m], row['seed_results'][str(seed)][m], atol=1e-12) for m in METRICS)
                seed_rows.append(dict(stage=stage, population=key[1], method=key[2], threshold_mode=key[3], seed=seed,
                    **{metric + '_percent': 100 * values[metric] for metric in METRICS}))
            for metric in METRICS:
                assert np.isclose(row[metric], np.mean([row['seed_results'][str(seed)][metric] for seed in SEEDS]), atol=1e-12)
            aggregate_rows.append(dict(stage=stage, population=key[1], method=key[2], threshold_mode=key[3],
                **{metric + '_percent': 100 * row[metric] for metric in METRICS}))
        for source in data['source_edit_diagnostics']:
            for control in source['random_controls'].values():
                assert control['gain_value_multiset_preserved']
                assert np.isclose(control['applied_delta_norm'], control['learned_delta_norm'], atol=1e-10)
        for method in ['sparse_edit', 'dense_edit']:
            selection = json.loads((directory / f'component_selection_{method}.json').read_text(encoding='utf-8'))
            for row in selection:
                selection_rows.append(dict(stage=stage, method=method, seed=row['seed'], **row['selected_candidate']))
    assert all(row['strength'] == 0 for row in selection_rows if row['method'] == 'dense_edit')
    for pop in POPS:
        reference = aggregates[('0445', pop, 'reference_supcon', OWN)]
        for stage in RUNS:
            for method in ['reference_supcon', 'dense_edit'] + [f'dense_edit__random{i}' for i in range(3)]:
                for metric in METRICS:
                    assert aggregates[(stage, pop, method, OWN)][metric] == reference[metric]
    paired = []
    for key, before in index.items():
        if key[0] != '0445':
            continue
        after = index[('0535', *key[1:])]
        paired.append(dict(population=key[1], method=key[2], threshold_mode=key[3], seed=key[4], procedure=key[5],
                           **{metric + '_change_pp': 100 * (after[metric] - before[metric]) for metric in METRICS}))
    assert len(paired) == 480
    write_csv('aggregate_metrics.csv', aggregate_rows)
    write_csv('seed_metrics.csv', seed_rows)
    write_csv('procedure_metrics.csv', procedures)
    write_csv('paired_0535_minus_0445.csv', paired)
    write_csv('selected_components.csv', selection_rows)

    randoms = [f'sparse_edit__random{i}' for i in range(3)]
    groups = [('0445', ['reference_supcon'], OWN, 'Reference / dense', '#666666'),
              ('0445', ['sparse_edit'], OWN, '0445 SAE', '#0072B2'),
              ('0535', ['sparse_edit'], OWN, '0535 SAE', '#009E73'),
              ('0445', randoms, OWN, '0445 random: own', '#0072B2'),
              ('0535', randoms, OWN, '0535 random: own', '#009E73'),
              ('0445', randoms, SAME, '0445 random: same', '#0072B2'),
              ('0535', randoms, SAME, '0535 random: same', '#009E73')]
    style = {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 9,
             'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(4.8, 7.5), sharey=True)
        fig.subplots_adjust(left=0.285, right=0.98, bottom=0.20, top=0.91, wspace=0.25, hspace=0.46)
        for col, pop in enumerate(POPS):
            for metricnum, metric in enumerate(METRICS):
                ax = axes[metricnum, col]
                for groupnum, (stage, methods, mode, label, color) in enumerate(groups):
                    for methodnum, method in enumerate(methods):
                        record = aggregates[(stage, pop, method, mode)]
                        for seednum, seed in enumerate(SEEDS):
                            offset = (seednum - 1) * 0.085 + ((methodnum - 1) * 0.25 if len(methods) == 3 else 0)
                            ax.scatter(100 * record['seed_results'][str(seed)][metric], groupnum + offset,
                                s=15, marker=MARKERS[seednum], linewidths=0.65, edgecolors=color,
                                facecolors='white' if mode == SAME else color, zorder=3)
                    avg = 100 * np.mean([aggregates[(stage, pop, method, mode)][metric] for method in methods])
                    ax.plot([avg, avg], [groupnum - 0.38, groupnum + 0.38], color=color, lw=0.8, zorder=2)
                ax.set_yticks(range(len(groups)), [group[3] for group in groups])
                ax.set_ylim(6.55, -0.55)
                ax.tick_params(axis='y', length=0)
                ax.grid(axis='x', color='#DDDDDD', lw=0.5)
                ax.spines[['top', 'right']].set_visible(False)
                if metricnum == 0:
                    ax.set_title(('Development' if col == 0 else 'Examined extension') + '\n5 procedures', pad=10)
                    ax.set_xlim((30, 45.7) if col == 0 else (6, 14))
                    ax.set_xticks([30, 35, 40, 45] if col == 0 else [6, 8, 10, 12, 14])
                elif metricnum == 1:
                    ax.set_xlim(96.8, 100.15)
                    ax.set_xticks([97, 98, 99, 100])
                else:
                    ax.set_xlim(92, 100.55)
                    ax.set_xticks([92, 96, 100])
                ax.set_xlabel(['Repeat removal (%)', 'Other-prompt retention (%)', 'First-prompt retention (%)'][metricnum])
                ax.text(-0.035, 1.03, chr(65 + metricnum * 2 + col), transform=ax.transAxes, fontweight='bold')
        handles = [Line2D([], [], marker=marker, linestyle='none', color='#444444', markersize=4,
                          label=str(seed)) for seed, marker in zip(SEEDS, MARKERS)]
        fig.legend(handles=handles, ncol=3, frameon=False, bbox_to_anchor=(0.61, 0.105),
                   loc='lower center', fontsize=8, columnspacing=0.9)
        fig.text(0.02, 0.026,
            'Symbols: seed means; random rows include all 3 permutations.\n'
            'Vertical marks: means over seeds and, for controls, permutations.\n'
            'Own: separately calibrated. Same: corresponding stage SAE threshold.\n'
            'All 10 procedures were examined; seeds are repeated model fits.', fontsize=8, va='bottom')
        fig.savefig(OUT / 'reusable_component_comparison.pdf')
        fig.savefig(OUT / 'reusable_component_comparison.png', dpi=300)
        plt.close(fig)

    lines = ['# Component discovery comparison', '',
        '0445 ranks by the mean fitting-procedure component effect. 0535 ranks by the minimum remaining net effect '
        'after omitting each fitting procedure and requires positive benefit and net effect under every omission. '
        'Both use the saved nonlinear group evaluation and the original development threshold rule.', '',
        'Each population contains five independent procedures; the three seeds are paired repeated model fits. '
        'All ten procedures are development-exposed, including the previously examined extension. '
        'Metrics average lesions within source episode, episodes within procedure, procedures equally and then seeds equally. '
        'The following values are descriptive saved operating points. They do not impose equal achieved retention across methods.', '',
        'Each table cell is **removal / other-prompt retention / first-prompt retention**, in percent. '
        '`own` uses each control\'s independently calibrated development threshold; `same` uses its corresponding '
        'stage\'s learned SAE threshold. All three norm-matched random permutations are retained.']
    compact = [('Reference / dense', 'reference_supcon', OWN), ('SAE', 'sparse_edit', OWN)]
    compact += [(f'Random {i}, {label}', f'sparse_edit__random{i}', mode) for mode, label in [(OWN, 'own'), (SAME, 'same')] for i in range(3)]
    for pop in POPS:
        lines += ['', f'## {pop}', '', '| Method | 0445 | 0535 |', '|---|---:|---:|']
        for label, method, mode in compact:
            cells = [' / '.join(f"{100 * aggregates[(stage, pop, method, mode)][m]:.4f}" for m in METRICS) for stage in RUNS]
            lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Paired SAE changes', '',
        '| Population | Seed | Removal change (pp) | Other retention change (pp) | First-prompt change (pp) |',
        '|---|---:|---:|---:|---:|']
    for pop in POPS:
        for seed in SEEDS:
            selected = [r for r in paired if r['population'] == pop and r['method'] == 'sparse_edit' and r['seed'] == seed]
            assert len(selected) == 5
            lines.append(f'| {pop} | {seed} | ' + ' | '.join(f"{np.mean([r[m + '_change_pp'] for r in selected]):+.4f}" for m in METRICS) + ' |')
    lines += ['', '| Population | Procedure | Removal change (pp) | Other retention change (pp) | First-prompt change (pp) |',
              '|---|---|---:|---:|---:|']
    for pop in POPS:
        videos = sorted({r['procedure'] for r in paired if r['population'] == pop})
        assert len(videos) == 5
        for video in videos:
            selected = [r for r in paired if r['population'] == pop and r['method'] == 'sparse_edit' and r['procedure'] == video]
            assert len(selected) == 3
            lines.append(f'| {pop} | {video} | ' + ' | '.join(f"{np.mean([r[m + '_change_pp'] for r in selected]):+.4f}" for m in METRICS) + ' |')
    lines += ['', '## Interpretation', '']
    for pop in POPS:
        old = aggregates[('0445', pop, 'sparse_edit', OWN)]
        new = aggregates[('0535', pop, 'sparse_edit', OWN)]
        reference = aggregates[('0445', pop, 'reference_supcon', OWN)]
        lines.append(f"- {pop}: 0535 minus 0445 removal {100 * (new['removal'] - old['removal']):+.4f} pp; "
                     f"other retention {100 * (new['retention'] - old['retention']):+.4f} pp; "
                     f"first-prompt retention remains {100 * new['first_prompt']:.2f}%. "
                     f"0535 minus reference removal is {100 * (new['removal'] - reference['removal']):+.4f} pp.")
        for mode in [OWN, SAME]:
            controls = [aggregates[('0535', pop, method, mode)] for method in randoms]
            lines.append(f"- {pop}, 0535 SAE minus mean random controls ({mode}): " +
                ', '.join(f"{m} {100 * (new[m] - np.mean([r[m] for r in controls])):+.4f} pp" for m in METRICS) + '.')
    lines += ['- All dense fits select zero edit in both stages. Their learned and random-control outcomes equal the reference; '
              'the figure and compact table share that row, while complete tables preserve every method.',
              '- The 0535 SAE selections contain four components at strengths 0.5, 0.25 and 0.25; '
              '0445 selections contain 64, 64 and 16 components at strengths 1, 1 and 0.5. '
              'The procedure-deletion criterion produces smaller selected interventions and application outcomes close to the reference.',
              '- These results characterize the tested ranking rule. They do not establish reusable component specificity or '
              'independent confirmation of an application benefit.', '',
              '## Data and reproduction', '',
              '`procedure_metrics.csv` contains all 960 stage-by-procedure-by-seed operating-point records in percent. '
              '`paired_0535_minus_0445.csv` contains all 480 paired changes in percentage points. '
              'Aggregate and seed tables preserve all methods, including the spatial baseline and each dense random control. '
              'No fitting, threshold selection or video inference occurs during figure generation.', '',
              '`artifacts/environments/modern/Scripts/python.exe scripts/plot_reusable_component_comparison.py`']
    (OUT / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = dict(sources=sources, script=str(Path(__file__).relative_to(ROOT)),
        procedure_rows=len(procedures), paired_rows=len(paired), aggregate_rows=len(aggregate_rows),
        seed_rows=len(seed_rows), independent_unit='procedure', seeds=SEEDS,
        figure_inches=[4.8, 7.5], png_dpi=300)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
