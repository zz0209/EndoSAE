import argparse
import hashlib
import json
from pathlib import Path

from plot_source_support_components import (
    ROOT, SUPPORTS, POPS, SEEDS, METRICS, VARIANTS, LABELS, COLORS, STYLES,
    MARKERS, STYLE, Line2D, plt, np, finite_mean, equal, group_rows, csv_write, convert, flatten,
)

RUN = ROOT / 'results/runs/20260930T0935Z_support_protected_tradeoff_v1'
OUT = ROOT / 'figures/support_protected_tradeoff_20260930'
OWN = 'own_protected_threshold'
TARGET = 'target_threshold'
PKEYS = ['seed', 'population', 'video', 'support', 'variant', 'policy']
AKEYS = ['population', 'support', 'variant', 'policy']


def validate_export(data, output):
    assert data['status'] == 'COMPLETE' and data['seeds'] == SEEDS
    assert data['independent_unit'] == 'procedure'
    metrics = list(data['source_rows'][0]['metrics'])
    assert len(metrics) == 21
    sources = [{**{k: row[k] for k in PKEYS + ['episode_id', 'threshold']}, **row['metrics']}
               for row in data['source_rows']]
    assert len(sources) == 5544 and len(data['procedure_rows']) == 1980
    assert len({tuple(row[k] for k in PKEYS + ['episode_id']) for row in sources}) == len(sources)
    for row in sources:
        for metric in metrics:
            value = row[metric]
            assert value is None or (np.isfinite(value) and -1e-12 <= value <= 1 + 1e-12)
    stored = {tuple(row[k] for k in PKEYS): row for row in data['procedure_rows']}
    assert len(stored) == len(data['procedure_rows'])
    procedures, pindex = [], {}
    for key, group in group_rows(sources, PKEYS).items():
        row = dict(zip(PKEYS, key), threshold=group[0]['threshold'], sources=len(group),
                   **{m: finite_mean([r[m] for r in group]) for m in metrics})
        original = stored[key]
        equal(original['source_suppression'], row['removal'])
        for metric in metrics:
            if metric in original:
                equal(original[metric], row[metric])
        assert all(r['threshold'] == row['threshold'] for r in group)
        procedures.append(row)
        pindex[key] = row
    aggregates, seeds = {}, []
    original = {tuple(r[k] for k in AKEYS): r for r in data['aggregates']}
    for key, group in group_rows(procedures, AKEYS).items():
        assert len(group) == 15
        seed_results = {}
        for seed in SEEDS:
            local = [r for r in group if r['seed'] == seed]
            assert len(local) == 5 and len({r['video'] for r in local}) == 5
            row = dict(zip(AKEYS, key), seed=seed, threshold=local[0]['threshold'],
                       **{m: finite_mean([r[m] for r in local]) for m in metrics})
            for metric in METRICS:
                equal(row[metric], original[key]['seed_results'][str(seed)][metric])
            seed_results[str(seed)] = row
            seeds.append(row)
        row = dict(zip(AKEYS, key), seed_results=seed_results,
                   **{m: finite_mean([r[m] for r in seed_results.values()]) for m in metrics})
        for metric in METRICS:
            equal(row[metric], original[key][metric])
        if key[0] == 'development' and key[-1] == OWN:
            assert all(r['retention'] >= data['retention_floor'] - 1e-12 and r['first_prompt'] == 1
                       for r in seed_results.values())
        aggregates[key] = row
    assert len(aggregates) == len(original) == 132
    contrasts = []
    for row in procedures:
        if row['variant'] not in VARIANTS[1:] or row['policy'] != OWN:
            continue
        key = tuple(row[k] for k in PKEYS)
        references = [('target_minus_reference', 'reference', OWN)]
        references += [(f'target_minus_random{i}', row['variant'] + f'__random{i}', policy)
                       for policy in [OWN, TARGET] for i in range(3)]
        for contrast, variant, policy in references:
            other = pindex[key[:4] + (variant, policy)]
            contrasts.append({**{k: row[k] for k in PKEYS}, 'control_policy': policy, 'contrast': contrast,
                              **{m: row[m] - other[m] if row[m] is not None and other[m] is not None else None
                                 for m in metrics}})
    ckeys = ['population', 'video', 'support', 'variant', 'control_policy', 'contrast']
    paired_means = [dict(zip(ckeys, key), **{m: finite_mean([r[m] for r in group]) for m in metrics})
                    for key, group in group_rows(contrasts, ckeys).items()]
    for name, rows, suffix in [('source_metrics', sources, 'percent'), ('procedure_metrics', procedures, 'percent'),
                               ('seed_metrics', seeds, 'percent'), ('aggregate_metrics', list(aggregates.values()), 'percent'),
                               ('paired_procedure_contrasts', contrasts, 'change_pp'),
                               ('paired_procedure_seed_means', paired_means, 'change_pp')]:
        csv_write(output, name + '.csv', [convert(row, metrics, suffix) for row in rows])
    csv_write(output, 'source_details.csv', [flatten(r) for r in data['source_rows']])
    return aggregates, procedures, contrasts, dict(source_rows=len(sources), procedure_rows=len(procedures),
        seed_rows=len(seeds), aggregate_rows=len(aggregates), paired_procedure_contrasts=len(contrasts),
        paired_procedure_seed_means=len(paired_means), metric_columns=len(metrics))


def finish(fig, axes, output, name, rows):
    for ax in axes.flat:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xticks([value for value in ax.get_xticks() if xmin <= value <= xmax])
        ax.set_yticks([value for value in ax.get_yticks() if ymin <= value <= ymax])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in axes.flat:
        artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    for artist in artists:
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= -1 and bounds.y0 >= -1, (artist.get_text(), bounds)
            assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1, (artist.get_text(), bounds)
    fig.savefig(output / (name + '.pdf'))
    fig.savefig(output / (name + '.png'), dpi=300)
    plt.close(fig)
    csv_write(output, name + '_plotted_values.csv', rows)


def legends(fig, random=False, y=0.20):
    indices = range(1, 4) if random else range(4)
    fig.legend(handles=[Line2D([], [], color=COLORS[i], linestyle=STYLES[i], label=LABELS[i]) for i in indices],
               ncol=len(indices), loc='lower center', bbox_to_anchor=(0.53, y), frameon=False, fontsize=8)
    fig.legend(handles=[Line2D([], [], color='#444444', marker=marker, linestyle='none', markersize=3.5,
                              label=str(seed)) for seed, marker in zip(SEEDS, MARKERS)],
               ncol=3, loc='lower center', bbox_to_anchor=(0.53, y - 0.032), frameon=False, fontsize=8)


def plot_outcomes(aggregates, output, random=False):
    columns = [(pop, policy) for pop in POPS for policy in ([OWN, TARGET] if random else [OWN])]
    fig, axes = plt.subplots(3, len(columns), figsize=(7.2, 8.6 if random else 7.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30 if random else 0.26, top=0.88,
                        wspace=0.28, hspace=0.41)
    plotted = []
    for col, (pop, policy) in enumerate(columns):
        for mi, metric in enumerate(METRICS):
            ax, values = axes[mi, col], []
            for vi in range(1 if random else 0, 4):
                variant, center_values = VARIANTS[vi], []
                offset = (vi - (2 if random else 1.5)) * 0.115
                for si, support in enumerate(SUPPORTS):
                    record = aggregates[(pop, support, variant, OWN)]
                    seed_values = []
                    for seednum, seed in enumerate(SEEDS):
                        target = record['seed_results'][str(seed)][metric]
                        if random:
                            controls = [aggregates[(pop, support, variant + f'__random{i}', policy)]
                                        ['seed_results'][str(seed)][metric] for i in range(3)]
                            individual = [100 * (target - control) for control in controls]
                            for ri, value in enumerate(individual):
                                ax.scatter(si + offset + (seednum - 1) * 0.018 + (ri - 1) * 0.048, value,
                                           marker=MARKERS[seednum], s=9, facecolors='none', edgecolors=COLORS[vi],
                                           linewidths=0.5, alpha=0.5)
                                plotted.append(dict(population=pop, support=support, variant=variant, control_policy=policy,
                                    metric=metric, seed=seed, random=ri, value=value, units='percentage_points'))
                            values.extend(individual)
                            value = float(np.mean(individual))
                        else:
                            value = 100 * target
                            plotted.append(dict(population=pop, support=support, variant=variant, policy=OWN,
                                                metric=metric, seed=seed, value=value, units='percent'))
                        seed_values.append(value)
                        values.append(value)
                        ax.scatter(si + offset + (seednum - 1) * 0.023, value, marker=MARKERS[seednum],
                                   color=COLORS[vi], s=14, linewidths=0.4, zorder=4)
                    center_values.append(np.mean(seed_values))
                ax.plot(np.arange(3) + offset, center_values, color=COLORS[vi], linestyle=STYLES[vi], lw=1)
            ax.set_xticks(range(3), ['Actual', 'GT\nsame', 'GT\nall'])
            ax.set_xlim(-0.35, 2.35)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.tick_params(length=2.5, pad=2)
            ax.grid(axis='y', color='#DDDDDD', lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.text(-0.075, 1.04, chr(65 + mi * len(columns) + col), transform=ax.transAxes, weight='bold')
            if random:
                ax.axhline(0, color='#777777', lw=0.7)
            if col == 0:
                ax.set_ylabel(['Removal', 'Other retention', 'First-prompt retention'][mi] +
                              (' change (pp)' if random else ' (%)'))
            if mi == 0:
                title = 'Development' if pop == 'development' else 'Examined extension'
                title += '\n' + ('Each control calibrated' if policy == OWN else 'Controls at target threshold') if random else ''
                bounds = ax.get_position()
                fig.text(bounds.x0 + bounds.width / 2, 0.935, title, ha='center', va='center', fontsize=9)
            ax._values = values
    for mi in range(3):
        for cols in ([0, 1], [2, 3]) if random else ([0], [1]):
            values = [v for c in cols for v in axes[mi, c]._values]
            low, high = min(values), max(values)
            span = max(high - low, 0.5)
            low, high = low - 0.09 * span, high + 0.09 * span
            if not random:
                low, high = max(0, low), min(100.1, high)
                if mi == 2:
                    low, high = 99, 100.15
            for col in cols:
                axes[mi, col].set_ylim(low, high)
                if not random and mi == 2:
                    axes[mi, col].set_yticks([99, 99.5, 100])
    legends(fig, random, 0.225 if random else 0.185)
    note = ('Filled symbols: all 3 seeds; lines: means across seeds. Reference seeds overlap.\n'
            'Each method/support selects its threshold on development under the same protection requirement.\n')
    if random:
        note = ('Filled symbols: target minus mean of 3 norm-matched controls, for every seed. Open: each control.\n'
                'Left policy: target and each control calibrate separately under the same protection requirement.\n'
                'Right policy: all controls use the targeted method threshold; control protection may differ.\n')
    note += ('Development requires other retention >= 99.5393% and every first prompt retained; extension keeps thresholds.\n'
             'GT same / all: source GT boxes on actual support frames / all 8 cached past frames (oracle conditions).\n'
             'Each population contains 5 previously examined procedures. Seeds are repeated observations.\n'
             'Removal and retention must be interpreted jointly; symbols do not denote independent procedures.')
    fig.text(0.025, 0.035, note, fontsize=8, va='bottom', linespacing=1.3)
    finish(fig, axes, output, 'random_control_comparisons' if random else 'protected_tradeoff_outcomes', plotted)


def plot_procedures(contrasts, output):
    rows = [r for r in contrasts if r['contrast'] == 'target_minus_reference']
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.27, top=0.91, wspace=0.32, hspace=0.52)
    plotted = []
    for si, support in enumerate(SUPPORTS):
        for pi, pop in enumerate(POPS):
            ax = axes[si, pi]
            local = [r for r in rows if r['population'] == pop and r['support'] == support]
            videos = sorted({r['video'] for r in local})
            for vi, variant in enumerate(VARIANTS[1:], 1):
                for video_index, video in enumerate(videos):
                    group = [r for r in local if r['variant'] == variant and r['video'] == video]
                    assert len(group) == 3
                    for row in group:
                        seed_index = SEEDS.index(row['seed'])
                        ax.scatter(100 * row['removal'], 100 * row['retention'], color=COLORS[vi],
                                   marker=MARKERS[seed_index], s=15, alpha=0.7, linewidths=0.4)
                        plotted.append({k: row[k] for k in PKEYS} | dict(removal_change_pp=100 * row['removal'],
                            retention_change_pp=100 * row['retention'], procedure_number=video_index + 1))
            ax.axhline(0, color='#AAAAAA', lw=0.7)
            ax.axvline(0, color='#AAAAAA', lw=0.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(color='#EEEEEE', lw=0.5)
            ax.set_xlabel('Removal change (pp)')
            if pi == 0:
                ax.set_ylabel('Other retention change (pp)')
            ax.set_title(['Actual support', 'GT same frames', 'GT all past frames'][si], pad=8)
            ax.text(-0.09, 1.06, chr(65 + si * 2 + pi), transform=ax.transAxes, weight='bold')
            ax.margins(x=0.20, y=0.27)
            if si == 0:
                bounds = ax.get_position()
                fig.text(bounds.x0 + bounds.width / 2, 0.975, 'Development' if pi == 0 else 'Examined extension',
                         ha='center', va='top')
    legends(fig, True, 0.195)
    note = ('Each symbol: one procedure and seed, targeted method minus reference at their own calibrated thresholds.\n'
            'Upper-right values improve removal and retention together; axes may differ by panel.\n'
            'All 5 procedures and 3 seeds appear in each panel. Coincident values overlap; coordinates are not jittered.\n'
            'The plotted-values CSV maps every symbol to its procedure, method and seed.\n'
            'Both populations were previously examined; GT support is an oracle. No inferential intervals are shown.')
    fig.text(0.025, 0.035, note, fontsize=8, va='bottom', linespacing=1.3)
    finish(fig, axes, output, 'paired_procedure_outcomes', plotted)


def report(data, aggregates, contrasts, output):
    lines = ['# Protection-matched support and component outcomes', '',
        'All 21 metrics are independently recomputed from saved source outcomes, averaging sources within procedure, '
        'procedures within seed, and seeds. The saved procedure metrics and three saved aggregate metrics are checked '
        'against the recomputation. Both populations were previously examined; GT supports are oracle interventions. '
        'Each support and method uses development-only calibration under the same other-retention floor and every '
        'first prompt retained. Extension applies those thresholds unchanged.', '',
        'Random controls are shown under two distinct policies: each control calibrated independently to the same '
        'protection requirement; or each control evaluated at the targeted method threshold. The latter can change '
        'control protection. Removal, other retention and first-prompt retention are therefore reported together.', '',
        '## Main methods', '', '| Population | Support | Method | Removal (%) | Other retention (%) | First prompt (%) |',
        '|---|---|---|---:|---:|---:|']
    for pop in POPS:
        for support in SUPPORTS:
            for variant in VARIANTS:
                row = aggregates[(pop, support, variant, OWN)]
                lines.append('| ' + ' | '.join([pop, support, variant] + [f'{100 * row[m]:.6f}' for m in METRICS]) + ' |')
    lines += ['', '## Target minus reference', '',
        'Changes are percentage points. Positive/zero/negative procedure counts use means across all three seeds. '
        'A positive removal count alone does not establish joint protection improvement.', '',
        '| Population | Support | Method | Removal | Retention | First prompt | Procedure removal + / 0 / - |',
        '|---|---|---|---:|---:|---:|---|']
    for pop in POPS:
        for support in SUPPORTS:
            for variant in VARIANTS[1:]:
                row, ref = [aggregates[(pop, support, v, OWN)] for v in [variant, 'reference']]
                selected = [r for r in contrasts if r['population'] == pop and r['support'] == support and
                            r['variant'] == variant and r['contrast'] == 'target_minus_reference']
                effects = [finite_mean([r['removal'] for r in group]) for group in group_rows(selected, ['video']).values()]
                counts = [sum(value > 1e-12 for value in effects), sum(abs(value) <= 1e-12 for value in effects),
                          sum(value < -1e-12 for value in effects)]
                lines.append('| ' + ' | '.join([pop, support, variant] + [f'{100 * (row[m] - ref[m]):+.6f}' for m in METRICS]
                    + [' / '.join(map(str, counts))]) + ' |')
    lines += ['', '## Target minus random controls', '',
        'Cells give target minus mean of the three controls, followed by the minimum and maximum among all nine '
        'seed-by-permutation contrasts. Changes are percentage points; seeds/permutations are repeated observations.', '',
        '| Population | Support | Method | Control policy | Removal mean [min, max] | Retention mean [min, max] | First prompt mean [min, max] |',
        '|---|---|---|---|---:|---:|---:|']
    for pop in POPS:
        for support in SUPPORTS:
            for variant in VARIANTS[1:]:
                for policy in [OWN, TARGET]:
                    values = []
                    for metric in METRICS:
                        deltas = [100 * (aggregates[(pop, support, variant, OWN)]['seed_results'][str(seed)][metric] -
                            aggregates[(pop, support, variant + f'__random{i}', policy)]['seed_results'][str(seed)][metric])
                            for seed in SEEDS for i in range(3)]
                        values.append(f'{np.mean(deltas):+.6f} [{min(deltas):+.6f}, {max(deltas):+.6f}]')
                    lines.append('| ' + ' | '.join([pop, support, variant, policy] + values) + ' |')
    lines += ['', '## Interpretation and data exports', '',
        '0445 improves development removal for every support under the common calibration requirement. On the examined '
        'extension, its actual and GT-same removal decreases while other retention increases; GT-all improves both aggregate '
        'removal and retention. These changes depend on support and population. The procedure figure and CSV preserve '
        'heterogeneity; no uniform procedure benefit or independent confirmation is established.', '',
        'Under separately calibrated random controls, 0445 has positive mean removal contrasts for every support '
        'and population. For actual support, all nine seed-by-permutation removal contrasts are positive in both '
        'development and extension. The development actual contrast is +3.883252 pp removal and -0.079281 pp '
        'other retention, with every method meeting the same calibration floor. The extension actual contrast '
        'is +0.986849 pp removal, +0.203910 pp other retention and +0.277778 pp first-prompt retention. '
        'These aggregate changes do not imply that all individual controls or procedures improve jointly.', '',
        'At the target threshold, 0445 extension actual removal changes by -0.404908 pp against random controls, '
        'while other retention changes by +0.627474 pp and first-prompt retention by +0.277778 pp. '
        'The control-threshold policy changes the comparison. The two policies answer distinct questions: '
        'performance with each control calibrated under the same development requirement, and component '
        'effects with the targeted threshold held constant. The 0730 sparse and dense results do not show '
        'a consistent removal advantage across supports and populations.', '',
        'The complete source, procedure, seed and aggregate CSVs include all 21 metrics with missing values retained as NA. '
        'Paired procedure CSVs contain every target-reference and target-random comparison under both control policies. '
        'The full source-details export preserves lesion-level and temporal-stratum evidence. Metric tables use percent, '
        'paired contrasts use percentage points, and source-details retain native units. ', '',
        'Figures use vector PDF and 300-dpi PNG at 7.2 inches wide, Times New Roman 9-point base text and 8-point '
        'ticks/captions. Every seed is shown; the procedure page preserves all 5 procedures in each population. '
        'No fitting or threshold selection occurs in this plotting script.', '',
        'Reproduce: `artifacts/environments/modern/Scripts/python.exe scripts/plot_support_protected_tradeoff.py`', '']
    (output / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    source = args.run / 'summary.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    aggregates, procedures, contrasts, counts = validate_export(data, args.output)
    with plt.rc_context(STYLE):
        plot_outcomes(aggregates, args.output)
        plot_outcomes(aggregates, args.output, True)
        plot_procedures(contrasts, args.output)
    report(data, aggregates, contrasts, args.output)
    manifest = dict(summary=str(source), summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script=str(Path(__file__).relative_to(ROOT)), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plotting_dependency_sha256=hashlib.sha256((ROOT / 'scripts/plot_source_support_components.py').read_bytes()).hexdigest(),
        **counts, seeds=SEEDS, independent_unit='procedure', png_dpi=300,
        figures_inches={'protected_tradeoff_outcomes': [7.2, 7.8], 'random_control_comparisons': [7.2, 8.6],
                        'paired_procedure_outcomes': [7.2, 8.6]})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
