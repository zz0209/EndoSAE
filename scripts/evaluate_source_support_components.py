import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import encode_causal_detection_identity as reference_api
import evaluate_acknowledgement_sae as evaluation
import evaluate_token_memory_edit as token_evaluation
from evaluate_endomind_observation import full_frame_metadata
from evaluate_fixed_acknowledgement_policy import result_row
from evaluate_region_identity_memory import finite_mean
from src import component_memory_intervention, token_memory_edit
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest, project_boxes


SUPPORTS = ('actual', 'gt_actual_frames', 'gt_all_frames')
MODES = ('fixed_reference', 'fixed_actual_method')
STRATA = ('current_visibility', 'later_reappearance', 'unknown_continuity')


def output_path(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path))
    return path


def save_arrays(path, arrays):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def hashes(paths):
    return {str(path): digest(path) for path in dict.fromkeys(map(Path, paths))}


def variants(config):
    values = {'reference': dict(parent=None, method='reference_supcon', random=None)}
    for edit in config['edits']:
        name = 'p' + edit['parent'] + '_' + edit['method']
        values[name] = dict(**edit, random=None)
        for index in range(config['random_controls']):
            values[name + f'__random{index}'] = dict(**edit, random=index)
    assert len(values) == 13
    return values


def original_name(specification):
    method = specification['method']
    return method if specification['random'] is None else method + f"__random{specification['random']}"


def numeric(value):
    return float(value) if np.isfinite(value) else None


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=len(values), mean=numeric(values.mean()) if len(values) else None,
                quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else [])


def mask_support(actual, frame_indices, episode, all_frames, known):
    gt = np.zeros_like(actual)
    rows = []
    for index, frame_index in enumerate(frame_indices):
        frame_index = int(frame_index)
        frame = all_frames.get(frame_index)
        count, fallback = 0, 0
        if frame is None:
            status = 'METADATA_MISSING'
        elif frame_index not in known:
            status = 'ANNOTATION_UNKNOWN'
        else:
            boxes = [box for box in frame['original_boxes_xyxy']
                     if box['lesion_id'] == episode['source_lesion_id']]
            count = len(boxes)
            if not boxes:
                status = 'SOURCE_NOT_VISIBLE'
            else:
                gt[index], fallback = project_boxes(dict(width=frame['width'], height=frame['height'], boxes_xyxy=boxes))
                status = 'PROJECTED' if gt[index].any() else 'SOURCE_BOX_OUTSIDE_TOKEN_CROP'
        rows.append(dict(frame_index=frame_index, status=status, source_boxes=count,
                         actual_tokens=int(actual[index].sum()), gt_tokens=int(gt[index].sum()),
                         small_box_fallback=fallback))
    if any(row['status'] in ('METADATA_MISSING', 'ANNOTATION_UNKNOWN') for row in rows):
        raise ValueError('Cached source annotation availability differs from the fixed availability record')
    supports = dict(actual=actual, gt_actual_frames=gt & actual.any(1)[:, None], gt_all_frames=gt)
    assert all(mask.shape == (8, 196) and mask.any() for mask in supports.values())
    return supports, rows


def source_design(records, offsets, known, data, source):
    labels = np.full(int(offsets[-1]), '', dtype='<U80')
    eligible = data['post_click'] & data['known']
    for index in np.flatnonzero(eligible):
        frame = known[int(data['frame'][index])]
        matched = evaluation.shared.memory.acknowledgement.detection.overlap(
            records[index]['detections'], frame['original_boxes_xyxy'])
        for match in matched['matches']:
            labels[int(offsets[index]) + match['prediction_index']] = match['lesion_id']
    return dict(source=np.flatnonzero(labels == source), other=np.flatnonzero((labels != '') & (labels != source)))


def measures(curve, episode):
    result = result_row(curve, episode)
    metric = dict(removal=result['source_removal_fraction'])
    source = curve['lesion_ids'].tolist().index(episode['source_lesion_id'])
    for group, columns in episode['groups'].items():
        for name in evaluation.shared.METRICS:
            metric[group + '__' + name] = finite_mean(curve[name][0, columns].tolist())
    metric['retention'] = metric['all_other__baseline_qualified_retention']
    metric['first_prompt'] = metric['all_other__first_baseline_frame_retention']
    strata = {}
    for name in STRATA:
        baseline = float(curve[name + '_baseline_seconds'][source])
        retained = float(curve[name + '_retained_seconds'][0, source])
        strata[name] = dict(baseline_seconds=baseline, retained_seconds=retained,
                            removed_seconds=baseline - retained,
                            removal=1 - retained / baseline if baseline > 0 else None)
        metric[name + '__removal'] = strata[name]['removal']
    result['source_strata'] = strata
    result['metrics'] = metric
    return result


def setup(config, seed, smoke):
    assert config['supports'] == list(SUPPORTS) and config['threshold_modes'] == list(MODES)
    assert config['random_controls'] == 3 and config['device'] == 'cpu'
    assert seed in config['seeds'] and config['smoke_sources_per_population'] == 1
    settings = read_json(config['evaluation_settings_file'])
    parents = {name: Path(path) for name, path in config['parent_runs'].items()}
    points, models = {}, {}
    assets = [Path(__file__), Path(token_evaluation.__file__), Path(evaluation.__file__),
              Path(evaluation.shared.__file__), Path(evaluation.shared.memory.__file__),
              Path(evaluation.shared.memory.acknowledgement.__file__),
              Path(evaluation.shared.memory.acknowledgement.detection.__file__),
              Path(reference_api.__file__), Path(component_memory_intervention.__file__),
              Path(token_memory_edit.__file__), ROOT / 'src/checkpoint_io.py',
              ROOT / 'src/evaluation/realcolon_task.py', ROOT / 'scripts/evaluate_fixed_acknowledgement_policy.py',
              ROOT / 'scripts/evaluate_region_identity_memory.py', Path(config['evaluation_settings_file']),
              Path(config['availability_record'])]
    for parent, run in parents.items():
        assert read_json(run / 'summary.json')['status'] == 'COMPLETE'
        root = run / 'evaluation' / f'seed{seed}'
        assert read_json(root / 'summary.json')['status'] == 'COMPLETE'
        parent_identity = read_json(root / 'identity.json')
        for source_path in (Path(token_evaluation.__file__), Path(evaluation.__file__),
                            Path(evaluation.shared.__file__), Path(evaluation.shared.memory.__file__),
                            Path(reference_api.__file__), Path(component_memory_intervention.__file__)):
            matches = [value for path, value in parent_identity['files'].items()
                       if Path(path).resolve() == source_path.resolve()]
            assert len(matches) == 1 and matches[0] == digest(source_path), source_path
        points[parent] = read_json(root / 'operating_points.json')['methods']
        assert read_json(root / 'evaluation_settings.json')['development_base'] == settings['development_base']
        assert read_json(root / 'evaluation_settings.json')['extension_base'] == settings['extension_base']
        assets += [run / 'config.json', run / 'protocol.json', run / 'summary.json',
                   root / 'identity.json', root / 'summary.json', root / 'operating_points.json', root / 'evaluation_settings.json']
    assert points['0445']['reference_supcon']['threshold'] == points['0730']['reference_supcon']['threshold']
    for specification in config['edits']:
        fit = parents[specification['parent']] / 'fit' / specification['method'] / f'seed{seed}'
        model_config = read_json(fit / 'model_config.json')
        assert Path(model_config['reference_fit']).resolve() == Path(config['reference_fit']).resolve()
        key = 'p' + specification['parent'] + '_' + specification['method']
        models[key] = component_memory_intervention.load_predictor(fit, device='cpu')
        assets += [fit / name for name in ('model.npz', 'normalization.npz', 'gains.npz', 'model_config.json', 'summary.json')]
    assets += [Path(config['reference_fit']) / name for name in ('model.npz', 'normalization.npz')]
    records = token_evaluation.source_records(config, settings, smoke)
    assert len(records) == (2 if smoke else 28)
    assert all(record['available'] and record['episode']['click']['available'] for record in records)
    return settings, parents, points, models, records, assets


def video_inputs(base, consumer, video):
    descriptor = Path(consumer['descriptors']) / video
    paths = [base / 'config.json', base / video / 'summary.json', base / video / 'indices.npz',
             Path(consumer['annotations']), Path(consumer['episodes']) / video / 'episodes.json',
             Path(consumer['metadata']) / (video + '.json'), Path(consumer['metadata']) / (video + '.jsonl'),
             Path(consumer['predictions']) / video / 'detections.jsonl',
             Path(consumer['predictions']) / video / 'complete.json']
    paths += [descriptor / name for name in ('identity.json', 'complete.json', 'indices.npz',
              'raw_mean.npy', 'supcon_l2.npy', 'available.npy', 'status_code.npy')]
    return paths


def score_memory(memory, codes, available, reference=False):
    scores = np.full(len(available), np.nan, dtype=np.float64)
    scores[available] = codes.astype(np.float64) @ memory.astype(np.float64) if reference else (
        token_evaluation.FixedSourceMemory.score_encoded(memory, codes))
    assert np.isfinite(scores[available]).all()
    return np.clip(scores, -1., 1.)


def prepare_source(record, parents, seed, source_directory, source_receipt, reference,
                   models, definitions, queries, cached_queries, available, all_frames, known, expected):
    archive = load_arrays(record['source'] / 'tokens.npz')
    tokens, actual, original = archive['tokens'], archive['mask'], archive['original_raw'].reshape(-1)
    assert tokens.shape == (8, 196, 768) and actual.dtype == np.bool_
    original_saved = load_arrays(record['original'])
    assert bool(original_saved['available']) and np.array_equal(original, original_saved['raw_mean'].reshape(-1))
    frames = archive['frame_indices']
    assert frames.tolist() == source_receipt['input_frames']
    assert int(frames[-1]) == record['episode']['click']['input_frame'] and np.all(np.diff(frames) == 1)
    actual_pool = tokens[actual].astype(np.float64).mean(axis=0)
    assert np.array_equal(actual_pool, archive['pooled_raw'].reshape(-1))
    supports, frame_rows = mask_support(actual, frames, record['episode'], all_frames, known)
    assert [row['status'] for row in frame_rows] == [row['status'] for row in expected['frames']]
    assert [row['gt_tokens'] for row in frame_rows] == [row['gt_tokens'] for row in expected['frames']]
    assert [row['actual_tokens'] for row in frame_rows] == [row['actual_tokens'] for row in expected['frames']]
    parent_scores, parent_edits = {}, {}
    for parent, run in parents.items():
        directory = run / 'evaluation' / f'seed{seed}'
        parent_scores[parent] = load_arrays(directory / record['population'] / record['video'] /
                                           'sources' / record['episode']['episode_id'] / 'scores.npz')
        parent_edits[parent] = load_arrays(directory / 'source_edits' / record['population'] /
                                         record['video'] / record['episode']['episode_id'] / 'edits.npz')
        assert np.array_equal(parent_edits[parent]['source_mask'], actual)
        assert np.array_equal(parent_edits[parent]['original_raw'], original)
    assert np.array_equal(parent_scores['0445']['reference_supcon'],
                          parent_scores['0730']['reference_supcon'], equal_nan=True)
    arrays = dict(original_raw=original, actual_cached_pool=actual_pool, frame_indices=frames)
    scores, diagnostics, mask_rows, actual_checks = {}, {}, {}, {}
    for support, mask in supports.items():
        equal_actual = bool(np.array_equal(mask, actual))
        pool = tokens[mask].astype(np.float64).mean(axis=0)
        base_raw = original + (pool - actual_pool)
        if equal_actual:
            assert np.array_equal(base_raw, original)
        arrays[support + '__pool'] = pool
        arrays[support + '__base_raw'] = base_raw
        learned = {}
        for name, predictor in models.items():
            learned[name] = predictor.source_delta(tokens, mask)
            codes = predictor.pooled_codes(tokens, mask)
            arrays[support + '__' + name + '__mean_codes'] = codes
        mask_rows[support] = dict(frames=int(mask.any(1).sum()), tokens=int(mask.sum()),
            tokens_per_frame=mask.sum(1).tolist(), equal_actual=equal_actual,
            raw_support_change_norm=float(np.linalg.norm(base_raw - original)),
            added_tokens=int((mask & ~actual).sum()), removed_tokens=int((actual & ~mask).sum()))
        for name, spec in definitions.items():
            key = support + '__' + name
            if spec['parent'] is None:
                delta = np.zeros(768, dtype=np.float64)
                control = None
                gains = np.zeros(1024, dtype=np.float32)
            else:
                targeted = 'p' + spec['parent'] + '_' + spec['method']
                predictor = models[targeted]
                if spec['random'] is None:
                    delta, gains, control = learned[targeted], predictor.source_gains, None
                else:
                    delta, gains, control = token_evaluation.random_delta(predictor, tokens, mask,
                        learned[targeted], seed, record['episode']['episode_id'], spec['method'], spec['random'])
            raw = base_raw + delta
            memory = reference(raw[None])['supcon_l2'][0]
            local_queries = cached_queries if spec['parent'] is None else queries
            calculated = score_memory(memory, local_queries, available, spec['parent'] is None)
            parent_key = original_name(spec)
            parent = spec['parent'] or '0730'
            if equal_actual:
                if spec['parent'] is None:
                    cached_memory = original_saved['supcon_l2'].reshape(-1)
                    assert np.max(np.abs(memory - cached_memory)) < 1e-6
                    exact_replay = score_memory(cached_memory, cached_queries, available, True)
                    reference_error = float(np.max(np.abs(exact_replay[available] - parent_scores[parent][parent_key][available])))
                    assert reference_error < 1e-12
                else:
                    old = parent_edits[parent]
                    for suffix, value in [('raw', raw), ('memory', memory), ('delta', delta), ('gains', gains)]:
                        assert np.array_equal(value, old[parent_key + '__' + suffix]), (name, suffix)
                error = float(np.max(np.abs(calculated[available] - parent_scores[parent][parent_key][available])))
                assert error < 1e-6, (name, error)
                scores[key] = parent_scores[parent][parent_key].copy()
                actual_checks[key] = dict(parent_scores_reused=True, recomputed_max_error=error,
                    recomputed_scores_exact=bool(np.array_equal(calculated, scores[key], equal_nan=True)),
                    raw_and_memory_parent_exact=spec['parent'] is not None)
            else:
                scores[key] = calculated
            arrays[key + '__raw'], arrays[key + '__memory'] = raw, memory
            arrays[key + '__delta'], arrays[key + '__gains'] = delta, gains
            diagnostics[key] = dict(raw_norm=float(np.linalg.norm(raw)), memory_norm=float(np.linalg.norm(memory)),
                edit_raw_norm=float(np.linalg.norm(delta)), full_raw_change_norm=float(np.linalg.norm(raw - original)),
                nonzero_gains=int(np.count_nonzero(gains)), random_control=control,
                scoring='cached original query FP32 values promoted to FP64 dot' if spec['parent'] is None
                        else 'original raw queries encoded in batches of512, FP32 dot',
                parent_scores_reused=equal_actual)
    save_arrays(source_directory / 'support_masks.npz', dict(frame_indices=frames, **supports))
    save_arrays(source_directory / 'source_edits.npz', arrays)
    save_arrays(source_directory / 'scores.npz', scores)
    return scores, arrays, dict(frames=frame_rows, supports=mask_rows, scores=diagnostics, actual_checks=actual_checks)


def verify_parent_curve(parents, seed, record, spec, curve, threshold):
    parent = spec['parent'] or '0730'
    name = original_name(spec)
    phase = record['population'] + ('_controls_at_method_threshold' if spec['random'] is not None else '')
    path = parents[parent] / 'evaluation' / f'seed{seed}' / phase / record['video'] / 'sources' / record['episode']['episode_id'] / (name + '__score_curve.npz')
    with np.load(path, allow_pickle=False) as old:
        column = int(np.searchsorted(old['threshold'], threshold, side='right') - 1) if (
            record['population'] == 'development' and spec['random'] is None) else 0
        errors = []
        for key in ('acknowledged_removed_seconds', 'retained_baseline_seconds',
                    'first_postclick_correct_prompt_time', 'first_baseline_frame_retention'):
            before, after = np.asarray(old[key][column]), np.asarray(curve[key][0])
            assert np.allclose(before, after, atol=1e-7, rtol=0, equal_nan=True), (name, key)
            difference = np.abs(before - after)
            errors.append(float(np.max(difference[np.isfinite(difference)], initial=0)))
    return dict(path=str(path), max_error=max(errors), frozen_policy_reproduced=True)


def evaluate(config_path, seed, smoke, resume, stop_after):
    config = read_json(config_path)
    run = output_path(config['run_dir'])
    base_output = run / 'smoke' if smoke else run
    output = base_output / 'evaluation' / f'seed{seed}'
    settings, parents, points, models, sources, assets = setup(config, seed, smoke)
    assets += [config_path, run / 'protocol.json']
    identity = dict(files=hashes(assets), seed=seed, smoke=smoke, python=sys.version,
                    numpy=np.__version__, torch=str(torch.__version__), device='cpu', threads=torch.get_num_threads())
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / 'identity.json'
    if identity_file.exists():
        assert resume and read_json(identity_file) == identity
    else:
        atomic_write_json(identity_file, identity)
    definitions = variants(config)
    reference = reference_api.Representations(config['reference_fit'])
    availability = {(r['population'], r['video'], r['episode_id']): r
                    for r in read_json(config['availability_record'])['rows']}
    threshold_reference = points['0730']['reference_supcon']['threshold']
    started, completed, new_sources = time.perf_counter(), 0, 0
    summaries = []
    atomic_write_json(output / 'progress.json', dict(completed=0, total=len(sources), phase='VERIFY_INPUTS'))
    for population, video in dict.fromkeys((r['population'], r['video']) for r in sources):
        source_base = Path(settings[population + '_base'])
        consumer = read_json(source_base / 'config.json')
        video_hashes = hashes(video_inputs(source_base, consumer, video))
        receipt, known, records, directory, offsets, raw, available, first = evaluation.load_video(source_base, consumer, video)
        clips, metadata_receipt, _, _, _ = full_frame_metadata(Path(consumer['metadata']), video)
        all_frames = {frame['frame_index']: frame for clip in clips for frame in clip['frames']}
        all_frames.update({frame['frame_index']: frame for frame in metadata_receipt['terminal_frame_records']})
        queries = token_evaluation.FixedSourceMemory(reference, {}).encode(np.asarray(raw[available]), batch_size=512)
        cached_queries = np.load(directory / 'supcon_l2.npy', mmap_mode='r')[available].copy()
        query_error = float(np.max(np.abs(queries - cached_queries)))
        assert queries.dtype == cached_queries.dtype == np.float32 and query_error < 1e-5
        target_video = output / population / video
        target_video.mkdir(parents=True, exist_ok=True)
        query_identity = dict(files=video_hashes, shapes=list(queries.shape), cached_query_max_error=query_error,
            component_query_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
            reference_query_sha256=hashlib.sha256(cached_queries.tobytes()).hexdigest(),
            component_query_definition='Original raw descriptors, frozen Representations, original batch512.',
            reference_query_definition='Original saved supcon_l2.npy, FP64 dot with source memory.')
        if (target_video / 'query_identity.json').exists():
            assert resume and read_json(target_video / 'query_identity.json') == query_identity
        else:
            atomic_write_json(target_video / 'query_identity.json', query_identity)
        for record in [r for r in sources if (r['population'], r['video']) == (population, video)]:
            episode = record['episode']
            target = target_video / 'sources' / episode['episode_id']
            target.mkdir(parents=True, exist_ok=True)
            stored = source_base / video / 'sources' / episode['episode_id']
            paths = [record['source'] / name for name in ('tokens.npz', 'source.json', 'complete.json')]
            paths += [record['original'], stored / 'frame_data.npz', stored / 'cosines.npz']
            for parent in parents.values():
                evaluation_root = parent / 'evaluation' / f'seed{seed}'
                paths += [evaluation_root / population / video / 'sources' / episode['episode_id'] / name
                          for name in ('scores.npz', 'summary.json')]
                paths += [evaluation_root / 'source_edits' / population / video / episode['episode_id'] / name
                          for name in ('edits.npz', 'summary.json')]
            for spec in definitions.values():
                parent = spec['parent'] or '0730'
                parent_phase = population + ('_controls_at_method_threshold' if spec['random'] is not None else '')
                paths.append(parents[parent] / 'evaluation' / f'seed{seed}' / parent_phase / video /
                    'sources' / episode['episode_id'] / (original_name(spec) + '__score_curve.npz'))
            local_identity = dict(global_identity_sha256=digest(identity_file), video_inputs=video_hashes, files=hashes(paths))
            saved_path = target / 'summary.json'
            if saved_path.exists():
                saved = read_json(saved_path)
                assert resume and saved['status'] == 'COMPLETE' and saved['identity'] == local_identity
                assert saved['outputs'] == hashes([target / name for name in saved['output_names']])
                summaries.append(saved)
                completed += 1
                continue
            source_info = read_json(record['source'] / 'source.json')
            for key in ('input_frame', 'output_index', 'detection_index', 'time'):
                assert source_info['click'][key] == episode['click'][key]
            data = load_arrays(stored / 'frame_data.npz')
            assert np.array_equal(data['frame'], [r['frame_index'] for r in records])
            design = source_design(records, offsets, known, data, episode['source_lesion_id'])
            scores, edits, diagnostics = prepare_source(record, parents, seed, target, source_info,
                reference, models, definitions, queries, cached_queries, available, all_frames, known,
                availability[(population, video, episode['episode_id'])])
            curves, keeps, outcomes, seen_policies = {}, {}, {}, {}
            for support in SUPPORTS:
                for name, spec in definitions.items():
                    key = support + '__' + name
                    before = scores['actual__' + name]
                    local_diagnostic = diagnostics['scores'][key]
                    local_diagnostic['memory_change_from_actual_norm'] = float(np.linalg.norm(
                        edits[key + '__memory'] - edits['actual__' + name + '__memory']))
                    local_diagnostic['score_groups'] = {group: dict(score=describe(scores[key][positions]),
                        change_from_actual=describe(scores[key][positions] - before[positions]))
                        for group, positions in design.items()}
                    for mode in MODES:
                        threshold = threshold_reference if mode == 'fixed_reference' or spec['parent'] is None else (
                            points[spec['parent']][spec['method']]['threshold'])
                        policy_key = mode + '__' + key
                        keep = ~np.isfinite(scores[key]) | (scores[key] < threshold)
                        policy_hash = hashlib.sha256(keep.tobytes()).hexdigest()
                        if policy_hash in seen_policies:
                            curve, checks = seen_policies[policy_hash]
                        else:
                            curve, checks, direct_keep = evaluation.fixed_curve(scores[key], threshold, offsets,
                                records, known, data, episode['source_lesion_id'], first, receipt['fps'])
                            assert np.array_equal(keep, direct_keep)
                            for group, columns in episode['groups'].items():
                                curve[group + '_lesion_columns'] = np.asarray(columns, dtype=int)
                            seen_policies[policy_hash] = (curve, checks)
                        curves.update({policy_key + '::' + field: value for field, value in curve.items()})
                        keeps[policy_key] = keep
                        result = measures(curve, episode)
                        result.update(support=support, variant=name, threshold_mode=mode, threshold=threshold,
                                      direct_mask_verification=checks)
                        if support == 'actual' and mode == 'fixed_actual_method':
                            result['parent_policy_check'] = verify_parent_curve(parents, seed, record, spec, curve, threshold)
                        outcomes[policy_key] = result
            save_arrays(target / 'curves.npz', curves)
            save_arrays(target / 'keeps.npz', keeps)
            save_arrays(target / 'score_detection_groups.npz', design)
            output_names = ['support_masks.npz', 'source_edits.npz', 'scores.npz', 'curves.npz', 'keeps.npz', 'score_detection_groups.npz']
            result = dict(status='COMPLETE', seed=seed, population=population, video=video,
                episode=episode, identity=local_identity, diagnostics=diagnostics, outcomes=outcomes,
                output_names=output_names, outputs=hashes([target / name for name in output_names]))
            atomic_write_json(saved_path, result)
            summaries.append(result)
            completed += 1
            new_sources += 1
            elapsed = time.perf_counter() - started
            atomic_write_json(output / 'progress.json', dict(completed=completed, total=len(sources),
                phase='SOURCE_COMPLETE', source=episode['episode_id'], elapsed_seconds=elapsed,
                measured_seconds_per_new_source=elapsed / new_sources,
                estimated_remaining_seconds=(len(sources) - completed) * elapsed / new_sources))
            print('SOURCE_SUPPORT', completed, len(sources), population, episode['episode_id'], round(elapsed, 3), flush=True)
            pause_after_checkpoint(saved_path)
            if stop_after and new_sources >= stop_after:
                raise SystemExit(75)
    assert identity['files'] == hashes(assets)
    atomic_write_json(output / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE',
        seed=seed, sources=len(summaries), conditions_per_source=len(SUPPORTS) * len(definitions) * len(MODES),
        source_summaries=[dict(population=r['population'], video=r['video'], episode_id=r['episode']['episode_id'],
            path=str(output / r['population'] / r['video'] / 'sources' / r['episode']['episode_id'] / 'summary.json')) for r in summaries],
        identity_sha256=digest(identity_file), seconds=time.perf_counter() - started,
        interpretation='GT support is an oracle source intervention on the same past frames; no deployment or independent confirmation claim.'))
    atomic_write_json(output / 'progress.json', dict(completed=len(sources), total=len(sources), phase='COMPLETE'))
    print('SOURCE_SUPPORT_COMPLETE', seed, flush=True)


def summarize(config_path, smoke, resume):
    config = read_json(config_path)
    run = output_path(config['run_dir'])
    output = run / 'smoke' if smoke else run
    seeds = config['seeds'][:1] if smoke else config['seeds']
    rows, diagnostics, assets = [], [], [config_path, run / 'protocol.json', Path(__file__)]
    for seed in seeds:
        folder = output / 'evaluation' / f'seed{seed}'
        receipt = read_json(folder / 'summary.json')
        assert receipt['status'] == ('REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE')
        assert digest(folder / 'identity.json') == receipt['identity_sha256']
        assets += [folder / 'summary.json', folder / 'identity.json']
        for source in receipt['source_summaries']:
            path = Path(source['path'])
            source_result = read_json(path)
            assert source_result['status'] == 'COMPLETE'
            assets.append(path)
            diagnostics.append(dict(seed=seed, population=source['population'], video=source['video'],
                episode_id=source['episode_id'], **source_result['diagnostics']))
            for result in source_result['outcomes'].values():
                rows.append(dict(seed=seed, population=source['population'], video=source['video'],
                    episode_id=source['episode_id'], support=result['support'], variant=result['variant'],
                    threshold_mode=result['threshold_mode'], threshold=result['threshold'], **result['metrics']))
    inputs = hashes(assets)
    if (output / 'summary.json').exists():
        assert resume and read_json(output / 'summary.json')['inputs'] == inputs
        print('REUSE_SOURCE_SUPPORT_SUMMARY', flush=True)
        return
    metric_names = list(next(iter(read_json(Path(receipt['source_summaries'][0]['path']))['outcomes'].values()))['metrics'])
    keys = ('seed', 'population', 'video', 'support', 'variant', 'threshold_mode')
    procedures = []
    for values in dict.fromkeys(tuple(r[key] for key in keys) for r in rows):
        local = [r for r in rows if tuple(r[key] for key in keys) == values]
        procedures.append(dict(zip(keys, values), sources=len(local),
            **{name: finite_mean([r[name] for r in local]) for name in metric_names}))
    keys = ('population', 'support', 'variant', 'threshold_mode')
    aggregates = []
    for values in dict.fromkeys(tuple(r[key] for key in keys) for r in procedures):
        local = [r for r in procedures if tuple(r[key] for key in keys) == values]
        per_seed = {seed: {name: finite_mean([r[name] for r in local if r['seed'] == seed])
                           for name in metric_names} for seed in seeds}
        aggregates.append(dict(zip(keys, values), seed_results=per_seed,
            **{name: finite_mean([r[name] for r in per_seed.values()]) for name in metric_names}))
    contrasts = []
    lookup = {(r['seed'], r['population'], r['video'], r['support'], r['variant'], r['threshold_mode']): r for r in procedures}
    for row in procedures:
        prefix = (row['seed'], row['population'], row['video'])
        comparisons = []
        if row['support'] == 'gt_actual_frames':
            comparisons.append(('spatial_support', 'actual', row['variant'], row['threshold_mode']))
        if row['support'] == 'gt_all_frames':
            comparisons.append(('historical_support', 'gt_actual_frames', row['variant'], row['threshold_mode']))
        if row['variant'] != 'reference' and row['threshold_mode'] == 'fixed_reference':
            comparisons.append(('component_effect_at_common_threshold', row['support'], 'reference', row['threshold_mode']))
        if '__random' not in row['variant'] and row['variant'] != 'reference':
            for index in range(3):
                comparisons.append((f'target_minus_random{index}', row['support'], row['variant'] + f'__random{index}', row['threshold_mode']))
        for label, support, variant, mode in comparisons:
            other = lookup[prefix + (support, variant, mode)]
            contrasts.append(dict(seed=row['seed'], population=row['population'], video=row['video'],
                support=row['support'], variant=row['variant'], threshold_mode=row['threshold_mode'], contrast=label,
                **{name: row[name] - other[name] if row[name] is not None and other[name] is not None else None for name in metric_names}))
    atomic_write_json(output / 'summary.json', dict(status='REAL_SMOKE_COMPLETE' if smoke else 'COMPLETE',
        inputs=inputs, seeds=seeds, source_rows=rows, procedure_rows=procedures, aggregates=aggregates,
        paired_procedure_contrasts=contrasts, source_diagnostics=diagnostics,
        independent_unit='procedure', aggregation='Lesions within source; sources within procedure; equal procedures within seed; equal seeds.',
        interpretation='Oracle support uses GT source boxes in existing past frames. Development and extension were previously examined. No model or threshold selection.',
        threshold_note='Component-minus-zero at fixed_reference uses a common threshold. fixed_actual_method retains each original targeted method policy; zero retains the reference threshold.'))
    lines = ['# Source support and frozen component intervention', '',
        'GT support is an oracle intervention on the same eight past frames. Queries, identity head, component groups, doses and thresholds remain frozen. Procedures are the independent units. The extension was previously examined.', '',
        '| Population | Support | Variant | Threshold | Repeat removal | Other retention | First prompt |',
        '|---|---|---|---|---:|---:|---:|']
    for row in aggregates:
        if '__random' in row['variant']:
            continue
        values = ['NA' if row[name] is None else f'{100 * row[name]:.4f}%' for name in ('removal', 'retention', 'first_prompt')]
        lines.append(f"| {row['population']} | {row['support']} | {row['variant']} | {row['threshold_mode']} | " + ' | '.join(values) + ' |')
    lines += ['', 'All random controls, natural temporal strata, paired procedure contrasts, per-source masks, frame states and score diagnostics are saved in summary.json and source artifacts.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_write_json(output / 'summary_progress.json', dict(completed=1, total=1, phase='COMPLETE'))
    print('SOURCE_SUPPORT_SUMMARY_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('evaluate', 'summary'), required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-sources', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(config['threads'])
    if args.phase == 'evaluate':
        if args.seed is None:
            parser.error('--seed is required for evaluate')
        evaluate(args.config, args.seed, args.smoke, args.resume, args.stop_after_sources)
    else:
        summarize(args.config, args.smoke, args.resume)


if __name__ == '__main__':
    main()
