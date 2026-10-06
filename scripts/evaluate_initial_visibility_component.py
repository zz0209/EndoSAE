import argparse
import time
from pathlib import Path

import evaluate_source_component_transfer as transfer
import numpy as np
import pandas as pd

from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest


def selection_name(smoke):
    return 'smoke_selection.json' if smoke else 'selection.json'


def cosine_effects(source, positives):
    baseline = np.clip(positives @ source, -1., 1.)
    numerator = positives @ source[:, None] - positives * source[None]
    denominator = np.sqrt((np.sum(positives ** 2, axis=1)[:, None] - positives ** 2) * (source @ source - source ** 2)[None])
    assert (denominator > 0).all()
    return baseline, np.clip(numerator / denominator, -1., 1.)


def prepare(run, smoke, resume):
    config, capacity, event, _, original = transfer.inputs(run)
    if smoke:
        original['seeds'] = [config['smoke_seed']]
    videos = [config['smoke_video']] if smoke else original['development_videos']
    root = run / ('smoke_selection' if smoke else 'selection')
    root.mkdir(exist_ok=True)
    reference = Path(config['reference_run']) / 'selection'
    choices, begin = [], time.perf_counter()
    tolerance = config['positive_preservation_tolerance']
    for video in videos:
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for key, model_path in transfer.model_specs(original):
            target = root / video / key
            target.mkdir(parents=True, exist_ok=True)
            prior = reference / video / key
            old = read_json(prior / 'complete.json')
            assert digest(prior / 'vectors.npz') == old['vectors_sha256']
            identity = dict(config=digest(run / 'config.json'), protocol=digest(run / 'protocol.json'),
                source=digest(__file__), prior=digest(prior / 'complete.json'),
                model=digest(model_path / 'model.npz'), events=digest(event / 'inputs' / video / 'events.json'),
                capacity=digest(capacity / 'evaluation' / video / key / 'complete.json'), smoke=smoke)
            if (target / 'complete.json').exists():
                receipt = read_json(target / 'complete.json')
                assert resume and receipt['identity'] == identity
                assert receipt['vectors_sha256'] == digest(target / 'vectors.npz')
                choices.extend(receipt['choices'])
                continue
            rows, arrays = [], {}
            with np.load(prior / 'vectors.npz') as saved:
                previous = {name: saved[name] for name in saved.files}
            with np.load(capacity / 'evaluation' / video / key / 'effects.npz') as saved:
                retained, before = saved['retained'], saved['before']
                codes, positions = saved['unit_codes'], saved['positions']
            for episode in manifest['episodes']:
                identifier = episode['episode_id']
                row = next(r for r in old['rows'] if r['episode'] == identifier)
                with np.load(Path(original['development_base']) / video / 'sources' / identifier / 'frame_data.npz') as data:
                    boundary = np.flatnonzero(data['post_click'] & (data['acknowledged_stratum'] != 'current_visibility'))
                    assert len(boundary)
                    activation = int(data['frame'][boundary[0]])
                indices = np.array([i for i, e in enumerate(manifest['events']) if e['episode'] == identifier
                    and e['same_identity'] and e['stratum'] == 'current_visibility' and e['frame'] <= activation], dtype=int)
                assert len(indices) and activation >= episode['click']['input_frame']
                selected_positions = []
                for index in indices:
                    current = manifest['events'][index]
                    frame = manifest['frames'][str(current['output'])]
                    match = next(m for m in frame['matches'] if m['lesion_id'] == current['lesion'])
                    selected_positions.append(frame['positions'][match['prediction_index']])
                positives = codes[[np.flatnonzero(positions == p).item() for p in selected_positions]]
                source = previous[identifier + '__source']
                baseline, effects = cosine_effects(source, positives)
                window = previous[identifier + '__feasible']
                history_score = window & np.all(effects >= baseline[:, None] - tolerance, axis=0)
                correct = indices[~before[indices]]
                decision = np.all(~retained[1:, correct], axis=1)
                window_correct = previous[identifier + '__positive_baseline'] >= row['threshold']
                decision &= np.all(previous[identifier + '__positive_effects'][window_correct] >= row['threshold'], axis=0)
                features = {'unchanged': -1, 'negative_only': row['original_feature'], 'window_score': row['feature']}
                for policy, feasible in [('history_score', history_score), ('history_decision', decision)]:
                    feature = -1
                    if row['feedback'] is not None and feasible.any():
                        negative = previous[identifier + '__feedback_effects']
                        best = int(np.argmin(np.where(feasible, negative, np.inf)))
                        if negative[best] < row['feedback_score'] - tolerance:
                            feature = best
                    features[policy] = feature
                    arrays[identifier + '__' + policy + '_feasible'] = feasible
                arrays[identifier + '__prefix_indices'] = indices
                arrays[identifier + '__prefix_positions'] = np.array(selected_positions)
                arrays[identifier + '__positive_baseline'] = baseline
                arrays[identifier + '__positive_effects'] = effects
                for policy, feature in features.items():
                    rows.append(dict(model=key, video=video, episode=identifier, policy=policy, feature=feature,
                        activation_frame=activation, threshold=row['threshold'], feedback_available=row['feedback'] is not None,
                        prefix_events=len(indices), correct_prefix_events=len(correct),
                        feasible_coordinates=int(arrays[identifier + '__' + policy + '_feasible'].sum()) if policy.startswith('history') else None,
                        original_feedback_score=row['feedback_score'],
                        edited_feedback_score=float(previous[identifier + '__feedback_effects'][feature]) if feature >= 0 else row['feedback_score']))
            np.savez_compressed(target / 'vectors.npz', **arrays)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', choices=rows, identity=identity,
                vectors_sha256=digest(target / 'vectors.npz')))
            choices.extend(rows)
            print('INITIAL_VISIBILITY_SELECTION', video, key, len(rows), flush=True)
        atomic_write_json(root / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos)))
        pause_after_checkpoint(root / 'progress.json')
    if not (run / selection_name(smoke)).exists():
        atomic_write_json(run / selection_name(smoke), dict(status='COMPLETE', choices=choices,
            seconds=time.perf_counter() - begin, target_outcomes_used_for_selection=True,
            future_outcomes_used_for_selection=False, source_sha256=digest(__file__)))


def evaluate(run, smoke, resume):
    config, _, event, _, original = transfer.inputs(run)
    root = run / ('smoke_scores' if smoke else 'scores')
    output = run / ('smoke_evaluation' if smoke else 'evaluation')
    output.mkdir(exist_ok=True)
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    selection = read_json(run / selection_name(smoke))
    videos = sorted({r['video'] for r in selection['choices']})
    base = Path(original['development_base'])
    settings = read_json(base / 'config.json')
    all_rows, all_events, begin = [], [], time.perf_counter()
    for video in videos:
        receipt, frames, records, _, offsets, _, _, first = transfer.application.load_video(base, settings, video)
        manifest = read_json(event / 'inputs' / video / 'events.json')
        for row in (r for r in selection['choices'] if r['video'] == video and r['policy'] == 'unchanged'):
            key, identifier = row['model'], row['episode']
            method, seed = key.rsplit('_seed', 1)
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == identifier)
            target = output / video / key / identifier
            target.mkdir(parents=True, exist_ok=True)
            identity = dict(source=digest(__file__), selection=digest(run / selection_name(smoke)), scores=digest(root / video / 'complete.json'))
            if (target / 'complete.json').exists():
                saved = read_json(target / 'complete.json')
                assert resume and saved['identity'] == identity
                all_rows.extend(saved['rows'])
                all_events.extend(pd.read_csv(target / 'events.csv').to_dict('records'))
                continue
            with np.load(base / video / 'sources' / identifier / 'frame_data.npz') as saved:
                data = {name: saved[name] for name in saved.files}
            before = np.load(root / video / key / identifier / 'unchanged.npy')
            rows, events = [], []
            for policy in config['policies']:
                scores = np.load(root / video / key / identifier / (policy + '.npy'))
                for horizon in ['full', 'after_activation']:
                    local = data if horizon == 'full' else dict(data, known=data['known'] & (data['frame'] > row['activation_frame']))
                    curve, checks, _ = transfer.application.fixed_curve(scores, row['threshold'], offsets, records, frames, local,
                        episode['source_lesion_id'], first, receipt['fps'])
                    for group, columns in episode['groups'].items():
                        curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                    result = transfer.application.result_row(curve, episode)
                    others = [l for l in result['lesions'] if 'all_other' in l['groups']]
                    retentions = [l['retention'] for l in others if l['retention'] is not None]
                    firsts = [l for l in others if l['baseline_first_prompt_time'] is not None]
                    rows.append(dict(method=method, seed=int(seed), video=video, episode=identifier, policy=policy, horizon=horizon,
                        repeat_removal=result['source_removal_fraction'], other_retention=float(np.mean(retentions)) if retentions else None,
                        source_baseline_seconds=result['source_baseline_seconds'], source_removed_seconds=result['source_removed_seconds'],
                        other_baseline_seconds=sum(l['baseline_seconds'] for l in others),
                        other_retained_seconds=sum(l['retained_baseline_seconds'] for l in others),
                        first_retention=sum(l['first_prompt_time'] == l['baseline_first_prompt_time'] for l in firsts) / len(firsts) if firsts else None,
                        feedback_available=row['feedback_available'], prefix_events=row['prefix_events'], activation_frame=row['activation_frame']))
                    np.savez_compressed(target / (policy + '_' + horizon + '_curve.npz'), **curve)
                local_events = [e for e in manifest['events'] if e['episode'] == identifier]
                observed = transfer.event_rows(dict(manifest, events=local_events), {identifier: scores}, {identifier: before}, row['threshold'], key, policy)
                for observed_row in observed:
                    observed_row['after_activation'] = observed_row['frame'] > row['activation_frame']
                    observed_row['feedback_available'] = row['feedback_available']
                events.extend(observed)
            pd.DataFrame(events).to_csv(target / 'events.csv', index=False)
            atomic_write_json(target / 'complete.json', dict(status='COMPLETE', identity=identity, rows=rows, checks=checks))
            all_rows.extend(rows)
            all_events.extend(events)
            print('INITIAL_VISIBILITY_REPLAY', video, key, identifier, flush=True)
        atomic_write_json(output / 'progress.json', dict(completed=videos.index(video) + 1, total=len(videos)))
        pause_after_checkpoint(output / 'progress.json')
    if not (output / 'summary.json').exists():
        pd.DataFrame(all_rows).to_csv(output / 'sources.csv', index=False)
        pd.DataFrame(all_events).to_csv(output / 'events.csv', index=False)
        atomic_write_json(output / 'summary.json', dict(status='COMPLETE', seconds=time.perf_counter() - begin, source_rows=len(all_rows)))


def summarize(run, smoke):
    root = run / ('smoke_evaluation' if smoke else 'evaluation')
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
    sources = pd.read_csv(root / 'sources.csv')
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    groups = ['method', 'seed', 'video', 'policy', 'horizon']
    procedures = []
    for name, local in [('complete', sources), ('reference_available', sources[sources.feedback_available])]:
        table = local.groupby(groups)[metrics].mean().reset_index()
        table['population'] = name
        procedures.append(table)
    procedures = pd.concat(procedures, ignore_index=True)
    seeds = procedures.groupby(['method', 'seed', 'policy', 'horizon', 'population'])[metrics].mean().reset_index()
    summary = seeds.groupby(['method', 'policy', 'horizon', 'population'])[metrics].mean().reset_index()
    for name, table in [('procedures', procedures), ('seeds', seeds), ('summary', summary)]:
        table.to_csv(output / (name + '.csv'), index=False)
    pd.DataFrame(read_json(run / selection_name(smoke))['choices']).to_csv(output / 'selections.csv', index=False)
    events = pd.read_csv(root / 'events.csv')
    rows = []
    for name, local in [('complete', events), ('reference_available', events[events.feedback_available])]:
        for keys, group in local.groupby(['method', 'seed', 'video', 'variant', 'after_activation']):
            measures = transfer.counts(group.retained.to_numpy(bool)[None], group.before_retained.to_numpy(bool), group.same_identity.to_numpy(bool))
            rows.append(dict(zip(['method', 'seed', 'video', 'policy', 'after_activation'], keys), population=name,
                **{k: int(v[0]) for k, v in measures.items()}, events=len(group), original_wrong=int((~group.same_identity & ~group.before_retained).sum())))
    pd.DataFrame(rows).to_csv(output / 'event_effects.csv', index=False)
    settings = read_json(run / 'config.json')
    policies = settings['policies']
    labels = settings.get('policy_labels', ['Original', 'Negative', 'Window score', 'History score', 'History decision'])
    fig, axes = transfer.plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for row, population in enumerate(['complete', 'reference_available']):
        for ax, metric in zip(axes[row], ['repeat_removal', 'other_retention']):
            for method, group in summary[(summary.population == population) & (summary.horizon == 'after_activation')].groupby('method'):
                group = group.set_index('policy').loc[policies]
                ax.plot(range(len(policies)), group[metric] * 100, marker='o', label='SAE' if method.endswith('sparse') else 'Dense')
            ax.set_xticks(range(len(policies)), labels, rotation=20)
            ax.set_title(population.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title() + ' (%)')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('After initial visibility | Annotated examined-development diagnostic')
    fig.savefig(output / 'initial_visibility.png', dpi=170)
    transfer.plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', rows=len(summary),
        aggregation='Lesion means, then sources, procedures and seeds; population and horizon separate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'score', 'evaluate', 'summarize'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run, args.smoke, args.resume)
    elif args.phase == 'score':
        transfer.score(args.run, args.smoke, args.resume, selection_name(args.smoke))
    elif args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
