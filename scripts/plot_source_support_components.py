import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/environments/endomind-inference-overlay'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_RUN = ROOT / 'results/runs/20260930T0830Z_source_support_components_v1'
DEFAULT_OUT = ROOT / 'figures/source_support_components_20260930'
SUPPORTS = ['actual', 'gt_actual_frames', 'gt_all_frames']
MODES = ['fixed_reference', 'fixed_actual_method']
POPS = ['development', 'extension']
SEEDS = [20260929, 20260930, 20261001]
METRICS = ['removal', 'retention', 'first_prompt']
VARIANTS = ['reference', 'p0445_sparse_edit', 'p0730_sparse_edit', 'p0730_dense_edit']
LABELS = ['Reference', '0445 SAE', '0730 SAE', '0730 dense']
COLORS = ['#555555', '#0072B2', '#8E4A9E', '#B45A00']
STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', '^', 's']
STYLE = {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 9,
         'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'pdf.fonttype': 42, 'ps.fonttype': 42}


def finite_mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def equal(actual, expected):
    assert (actual is None) == (expected is None), (actual, expected)
    if actual is not None:
        assert np.isclose(actual, expected, atol=1e-12, rtol=0), (actual, expected)


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def csv_write(output, name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def convert(row, metrics, suffix='percent'):
    return {key + '_' + suffix if key in metrics else key:
            ('NA' if value is None else 100 * value) if key in metrics else value
            for key, value in row.items() if key != 'seed_results'}


def flatten(record, prefix=''):
    result = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '__'))
        else:
            result[name] = json.dumps(value) if isinstance(value, list) else ('NA' if value is None else value)
    return result


def validate_and_export(data, output):
    assert data['status'] == 'COMPLETE' and data['seeds'] == SEEDS
    assert data['independent_unit'] == 'procedure'
    sources, procedures = data['source_rows'], data['procedure_rows']
    metadata = {'seed', 'population', 'video', 'episode_id', 'support', 'variant', 'threshold_mode', 'threshold'}
    metrics = [key for key in sources[0] if key not in metadata]
    assert set(METRICS) <= set(metrics)
    source_index = {}
    for row in sources:
        key = tuple(row[key] for key in ['seed', 'population', 'episode_id', 'support', 'variant', 'threshold_mode'])
        assert key not in source_index
        source_index[key] = row
        for metric in metrics:
            value = row[metric]
            assert value is None or (np.isfinite(value) and -1e-12 <= value <= 1 + 1e-12)
        assert all(row[m] is not None for m in METRICS)
    assert len(sources) == 6552 and len(procedures) == 2340
    pkeys = ['seed', 'population', 'video', 'support', 'variant', 'threshold_mode']
    source_groups = group_rows(sources, pkeys)
    pindex = {}
    for row in procedures:
        key = tuple(row[k] for k in pkeys)
        assert key not in pindex
        pindex[key] = row
        group = source_groups[key]
        assert len(group) == row['sources']
        for metric in metrics:
            equal(row[metric], finite_mean([r[metric] for r in group]))
    akeys = ['population', 'support', 'variant', 'threshold_mode']
    procedure_groups = group_rows(procedures, akeys)
    aggregates, seed_rows = {}, []
    for row in data['aggregates']:
        key = tuple(row[k] for k in akeys)
        assert key not in aggregates
        aggregates[key] = row
        group = procedure_groups[key]
        assert len(group) == 15
        for seed in SEEDS:
            local = [r for r in group if r['seed'] == seed]
            assert len(local) == 5 and len({r['video'] for r in local}) == 5
            for metric in metrics:
                equal(row['seed_results'][str(seed)][metric], finite_mean([r[metric] for r in local]))
            seed_rows.append(dict(zip(akeys, key), seed=seed, **row['seed_results'][str(seed)]))
        for metric in metrics:
            equal(row[metric], finite_mean([row['seed_results'][str(seed)][metric] for seed in SEEDS]))
    for row in data['paired_procedure_contrasts']:
        key = tuple(row[k] for k in pkeys)
        current = pindex[key]
        support, variant, mode = row['support'], row['variant'], row['threshold_mode']
        if row['contrast'] == 'spatial_support':
            support = 'actual'
        elif row['contrast'] == 'historical_support':
            support = 'gt_actual_frames'
        elif row['contrast'] == 'component_effect_at_common_threshold':
            variant = 'reference'
        else:
            assert row['contrast'] in [f'target_minus_random{i}' for i in range(3)]
            variant += '__' + row['contrast'].removeprefix('target_minus_')
        other = pindex[key[:3] + (support, variant, mode)]
        for metric in metrics:
            value = current[metric] - other[metric] if current[metric] is not None and other[metric] is not None else None
            equal(row[metric], value)
    threshold_pairs = []
    for key, row in pindex.items():
        if key[-1] == 'fixed_actual_method':
            reference_policy = pindex[key[:-1] + ('fixed_reference',)]
            threshold_pairs.append(dict(zip(pkeys, key), contrast='method_threshold_minus_reference_threshold',
                **{m: row[m] - reference_policy[m] if row[m] is not None and reference_policy[m] is not None else None
                   for m in metrics}))
    for pop in POPS:
        for support in SUPPORTS:
            for metric in metrics:
                equal(aggregates[(pop, support, 'reference', MODES[0])][metric],
                      aggregates[(pop, support, 'reference', MODES[1])][metric])
    contrasts = data['paired_procedure_contrasts'] + threshold_pairs
    contrast_groups = group_rows(contrasts, ['population', 'video', 'support', 'variant', 'threshold_mode', 'contrast'])
    paired_means = []
    for key, group in contrast_groups.items():
        assert len(group) == 3
        paired_means.append(dict(zip(['population', 'video', 'support', 'variant', 'threshold_mode', 'contrast'], key),
            **{m: finite_mean([r[m] for r in group]) for m in metrics}))
    for name, rows, suffix in [('source_metrics.csv', sources, 'percent'), ('procedure_metrics.csv', procedures, 'percent'),
                             ('seed_metrics.csv', seed_rows, 'percent'), ('aggregate_metrics.csv', data['aggregates'], 'percent'),
                             ('paired_procedure_contrasts.csv', contrasts, 'change_pp'),
                             ('paired_procedure_seed_means.csv', paired_means, 'change_pp')]:
        csv_write(output, name, [convert(r, metrics, suffix) for r in rows])
    supports, scores, frames = [], [], []
    for row in data['source_diagnostics']:
        identity = {k: row[k] for k in ['seed', 'population', 'video', 'episode_id']}
        supports.extend(dict(identity, support=support, **flatten(record)) for support, record in row['supports'].items())
        frames.extend(dict(identity, **frame) for frame in row['frames'])
        for condition, record in row['scores'].items():
            control = record['random_control']
            if control is not None:
                assert control['gain_value_multiset_preserved']
                equal(control['applied_delta_norm'], control['learned_delta_norm'])
            scores.append(dict(identity, condition=condition, **flatten(record)))
    csv_write(output, 'support_diagnostics.csv', supports)
    csv_write(output, 'score_diagnostics.csv', scores)
    csv_write(output, 'source_frame_states.csv', frames)
    return aggregates, dict(source_rows=len(sources), procedure_rows=len(procedures), seed_rows=len(seed_rows),
        aggregate_rows=len(aggregates), saved_paired_rows=len(data['paired_procedure_contrasts']),
        threshold_paired_rows=len(threshold_pairs), procedure_seed_mean_contrasts=len(paired_means),
        source_diagnostics=len(data['source_diagnostics']), metric_columns=len(metrics))


def plot(aggregates, output, random_contrast):
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 8.6 if random_contrast else 7.8))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.30 if random_contrast else 0.245,
                        top=0.88 if random_contrast else 0.91, wspace=0.24, hspace=0.44 if random_contrast else 0.36)
    plotted = []
    for col, (pop, mode) in enumerate((p, m) for p in POPS for m in MODES):
        for metricnum, metric in enumerate(METRICS):
            ax = axes[metricnum, col]
            values = []
            methods = VARIANTS[1:] if random_contrast else VARIANTS
            for variant in methods:
                variantnum = VARIANTS.index(variant)
                color = COLORS[variantnum]
                position = (variantnum - (2 if random_contrast else 1.5)) * 0.115
                centers = []
                for supportnum, support in enumerate(SUPPORTS):
                    record = aggregates[(pop, support, variant, mode)]
                    seed_values = []
                    for seednum, seed in enumerate(SEEDS):
                        target = record['seed_results'][str(seed)][metric]
                        if random_contrast:
                            controls = [aggregates[(pop, support, variant + f'__random{i}', mode)]
                                        ['seed_results'][str(seed)][metric] for i in range(3)]
                            individual = [100 * (target - value) for value in controls]
                            for randomnum, value in enumerate(individual):
                                x = supportnum + position + (seednum - 1) * 0.019 + (randomnum - 1) * 0.052
                                ax.scatter(x, value, s=8, marker=MARKERS[seednum], facecolors='none',
                                           edgecolors=color, linewidths=0.45, alpha=0.50, zorder=2)
                            value = float(np.mean(individual))
                            values.extend(individual)
                        else:
                            value = 100 * target
                        seed_values.append(value)
                        values.append(value)
                        ax.scatter(supportnum + position + (seednum - 1) * 0.023, value, s=13,
                                   marker=MARKERS[seednum], color=color, linewidths=0.4, zorder=4)
                        plotted.append(dict(population=pop, threshold_mode=mode, metric=metric, variant=variant,
                                            support=support, seed=seed, value=value,
                                            measure='target_minus_mean_random_pp' if random_contrast else 'percent'))
                    centers.append(np.mean(seed_values))
                ax.plot(np.arange(3) + position, centers, color=color, lw=1.0, linestyle=STYLES[variantnum], zorder=3)
            ax.set_xlim(-0.35, 2.35)
            ax.set_xticks(range(3), ['Actual', 'GT\nsame', 'GT\nall'])
            ax.tick_params(axis='both', length=2.5, pad=2)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#DDDDDD', linewidth=0.5)
            ax.text(-0.08, 1.05, chr(65 + metricnum * 4 + col), transform=ax.transAxes, weight='bold')
            if random_contrast:
                ax.axhline(0, color='#777777', lw=0.7, zorder=1)
            if col == 0:
                ax.set_ylabel((['Removal', 'Other retention', 'First-prompt retention'][metricnum]) +
                              (' change (pp)' if random_contrast else ' (%)'))
            if metricnum == 0:
                title = ('Development' if pop == 'development' else 'Examined extension') + '\n' + \
                        ('Reference threshold' if mode == MODES[0] else 'Method threshold')
                if random_contrast:
                    bounds = ax.get_position()
                    fig.text(bounds.x0 + bounds.width / 2, 0.935, title, ha='center', va='center', fontsize=9)
                else:
                    ax.set_title(title, pad=12)
            ax._source_values = values
    for metricnum in range(3):
        for columns in ([0, 1], [2, 3]):
            values = [v for c in columns for v in axes[metricnum, c]._source_values]
            if metricnum == 2 and not random_contrast:
                values = [v for c in range(4) for v in axes[metricnum, c]._source_values]
            minimum, maximum = min(values), max(values)
            span = max(maximum - minimum, 0.5)
            lower, upper = minimum - 0.09 * span, maximum + 0.09 * span
            if not random_contrast:
                lower = max(0, lower)
                upper = min(100.65, upper)
            for c in columns:
                axes[metricnum, c].set_ylim(lower, upper)
                if c in [1, 3]:
                    axes[metricnum, c].tick_params(labelleft=False)
    methods = range(1, 4) if random_contrast else range(4)
    handles = [Line2D([], [], color=COLORS[i], linestyle=STYLES[i], lw=1.0, label=LABELS[i]) for i in methods]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc='lower center',
               bbox_to_anchor=(0.54, 0.225 if random_contrast else 0.17), fontsize=8, columnspacing=1.3)
    seed_handles = [Line2D([], [], marker=marker, color='#444444', linestyle='none', markersize=3.5,
                          label=str(seed)) for seed, marker in zip(SEEDS, MARKERS)]
    fig.legend(handles=seed_handles, ncol=3, frameon=False, loc='lower center',
               bbox_to_anchor=(0.54, 0.195 if random_contrast else 0.14), fontsize=8, columnspacing=1.6)
    if random_contrast:
        note = ('Filled symbols: target minus the mean of its 3 norm-matched random controls, for each seed.\n'
                'Open symbols: target minus each random control. Lines: means across seeds.\n'
                'Each target and its controls use the same threshold in each panel. Higher values favor the target.\n')
        name = 'component_random_contrasts'
    else:
        note = ('Symbols: all 3 seed means; lines: means across seeds. Reference overlaps across seeds.\n'
                'Left policy in each population: one reference threshold for every method. Right: original method thresholds.\n'
                'Reference keeps its original threshold in all panels; each threshold is fixed across the 3 supports.\n')
        name = 'support_and_threshold_outcomes'
    note += ('GT same: source GT boxes on actual support frames. GT all: source GT boxes on all 8 cached past frames.\n'
             'GT conditions are oracle interventions. Each population has 5 previously examined procedures.\n'
             'Procedures are independent units; seeds and overlapping source episodes are repeated observations.')
    fig.text(0.03, 0.055 if random_contrast else 0.024, note, fontsize=8, va='bottom', linespacing=1.35)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = list(fig.texts)
    for ax in axes.flat:
        text_artists.extend([ax.title, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    for text in text_artists:
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= -1 and bounds.y0 >= -1
            assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
    fig.savefig(output / (name + '.pdf'))
    fig.savefig(output / (name + '.png'), dpi=300)
    plt.close(fig)
    csv_write(output, name + '_plotted_values.csv', plotted)


def write_report(data, aggregates, counts, output):
    lines = ['# Source support, thresholds and component interventions', '',
             'All displayed values were recomputed from saved source outcomes through procedure and seed means. '
             'No model, threshold, component or example selection is performed by this script. '
             'The GT conditions use source-identity boxes in the same eight cached past frames. '
             'Both populations were previously examined; these are oracle mechanism results.', '',
             '`support_and_threshold_outcomes` shows absolute outcomes for reference and three selected methods. '
             '`component_random_contrasts` shows each selected method minus its three norm-matched permutations '
             'under exactly the same support and threshold. Filled symbols retain every seed; open symbols on '
             'the second page retain each permutation contrast. The lines show means, with no inferential interval.', '',
             'The two threshold conditions are fixed_reference (the same original reference threshold for every '
             'variant) and fixed_actual_method (each original targeted-method threshold, also applied to that '
             'method\'s random controls). Reference retains the same reference threshold in both. All thresholds '
             'are unchanged across support conditions. A component-versus-reference contrast in the method-threshold '
             'panels therefore includes a threshold-policy difference. A comparison at one fixed threshold '
             'does not determine the attainable removal–protection tradeoff or rule out a component benefit '
             'under a common protection requirement. The companion saved-score analysis '
             '(`results/runs/20260930T0830Z_source_support_components_v1/analysis/analysis.md`) reports changed '
             'identity ordering and improved detection-level AUROC for 0445. These findings preserve evidence '
             'of a selected-component effect; the fixed-threshold figures do not attribute the complete original '
             'application gain to threshold choice.', '',
             '## Main outcomes', '', 'Cells contain removal / other-prompt retention / first-prompt retention (%).']
    for pop in POPS:
        for mode in MODES:
            lines += ['', f'### {pop}; {mode}', '', '| Method | Actual | GT same frames | GT all frames |',
                      '|---|---:|---:|---:|']
            for variant, label in zip(VARIANTS, LABELS):
                cells = [' / '.join(f'{100 * aggregates[(pop, support, variant, mode)][m]:.4f}' for m in METRICS)
                         for support in SUPPORTS]
                lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Support effects in the reference', '',
              'These reference effects are identical under both threshold labels. Changes are percentage points.', '',
              '| Population | Support contrast | Removal | Other retention | First prompt | Current visibility | Later reappearance |',
              '|---|---|---:|---:|---:|---:|---:|']
    for pop in POPS:
        for before, after, label in [('actual', 'gt_actual_frames', 'Spatial: GT same minus actual'),
                                     ('gt_actual_frames', 'gt_all_frames', 'History: GT all minus GT same')]:
            first, second = [aggregates[(pop, support, 'reference', MODES[0])] for support in [before, after]]
            values = [f'{100 * (second[m] - first[m]):+.4f}' for m in METRICS +
                      ['current_visibility__removal', 'later_reappearance__removal']]
            lines.append('| ' + pop + ' | ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## CSV scope and units', '',
              f"The complete tables contain {counts['source_rows']} source rows, {counts['procedure_rows']} procedure rows, "
              f"{counts['seed_rows']} seed rows and {counts['aggregate_rows']} overall rows. All {counts['metric_columns']} "
              'saved outcome metrics are included, with temporal groups and first-prompt/bout protection. '
              'Outcome columns use percent; contrast columns use percentage points. NA preserves undefined metrics.', '',
              f"The {counts['saved_paired_rows']} saved paired contrasts are recomputed and exported with "
              f"{counts['threshold_paired_rows']} additional within-procedure threshold-policy contrasts. "
              '`paired_procedure_seed_means.csv` provides their paired procedure-level means across seeds. '
              'Source support, original frame states and score/edit diagnostics are exported separately. '
              'Source rows do not represent independent procedures.', '',
              '## Figure format', '',
              'The outcome page is 7.2 by 7.8 inches and the random-control page is 7.2 by 8.6 inches, '
              'with vector PDF and 300-dpi PNG, Times New Roman 9-point base text '
              'and 8-point ticks/captions. These are general manuscript figures; final venue dimensions remain '
              'subject to the manuscript layout. Outcome scales match between threshold policies within each '
              'population; the two populations have separate outcome ranges. Horizontal point offsets only '
              'separate methods/seeds/permutations within each categorical support.', '',
              '## Reproduction', '',
              '`artifacts/environments/modern/Scripts/python.exe scripts/plot_source_support_components.py`']
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.run / 'summary.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    aggregates, counts = validate_and_export(data, args.output)
    with plt.rc_context(STYLE):
        plot(aggregates, args.output, False)
        plot(aggregates, args.output, True)
    write_report(data, aggregates, counts, args.output)
    manifest = dict(summary=str(source), summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script=str(Path(__file__).relative_to(ROOT)), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **counts, seeds=SEEDS, independent_unit='procedure',
        figures_inches={'support_and_threshold_outcomes': [7.2, 7.8], 'component_random_contrasts': [7.2, 8.6]}, png_dpi=300)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
