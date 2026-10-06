import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared
from train_token_identity_sae import fit, source_identity as parent_sources
from src.checkpoint_io import atomic_write_json, read_json


def source_identity():
    return dict(parent_sources(), **{'scripts/train_temporal_view_identity.py': shared.file_sha256(__file__)})


def read_tokens(directory):
    receipt = read_json(directory / 'complete.json')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('Incomplete token input')
    if shared.file_sha256(directory / 'tokens.npz') != receipt['tokens_sha256']:
        raise ValueError('Token input changed')
    if shared.file_sha256(directory / 'records.json') != receipt['records_sha256']:
        raise ValueError('Observation records changed')
    records = read_json(directory / 'records.json')
    with np.load(directory / 'tokens.npz', allow_pickle=False) as archive:
        tokens, offsets = archive['tokens'].copy(), archive['offsets'].copy()
    if tokens.dtype != np.float32 or tokens.shape[1] != 768 or not np.isfinite(tokens).all():
        raise ValueError('Invalid actual token values')
    if offsets.shape != (len(records) + 1,) or offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Invalid token offsets')
    return tokens, offsets, records, receipt


def load_inputs(run):
    config = read_json(run / 'config.json')
    cohort = read_json(config['cohort_config'])
    original = Path(config['original_prepared_run'])
    expanded = Path(config['expanded_storage'])
    if read_json(expanded / 'preparation_summary.json')['status'] != 'COMPLETE':
        raise ValueError('Temporal preparation is incomplete')
    records, chunks, offsets, receipts = [], [], [0], []
    preserved = 0
    for partition in ['train', 'val']:
        for video in cohort['fit_video_ids'][partition]:
            directory = (expanded if partition == 'train' else original) / 'tokens' / video
            value, boundaries, local, receipt = read_tokens(directory)
            if any(row['video_id'] != video or row['split'] != partition for row in local):
                raise ValueError('Unexpected procedure or partition')
            if partition == 'train':
                old_value, old_offsets, old_records, old_receipt = read_tokens(original / 'tokens' / video)
                lookup = {(row['clip_id'], row['lesion_id']): i for i, row in enumerate(local)}
                expected_keys = {(row['clip_id'], row['lesion_id']) for row in old_records}
                actual_keys = {(row['clip_id'], row['lesion_id']) for row in local if row['original_observation']}
                if expected_keys != actual_keys or set(Counter(row['lesion_id'] for row in local).values()) != {32}:
                    raise ValueError('Expanded or original observation membership differs')
                for i, old in enumerate(old_records):
                    j = lookup[(old['clip_id'], old['lesion_id'])]
                    for key in ['video_id', 'split', 'start_frame', 'end_frame', 'fps', 'roi_tokens_per_frame',
                                'annotation_observation', 'lesion_first_frame']:
                        if old[key] != local[j][key]:
                            raise ValueError(f'Original metadata changed: {key}')
                    np.testing.assert_array_equal(value[boundaries[j]:boundaries[j + 1]],
                                                  old_value[old_offsets[i]:old_offsets[i + 1]])
                    preserved += 1
                receipts.append(dict(video=video, expanded=receipt, original=old_receipt))
                del old_value
            else:
                local = [dict(row, original_observation=True) for row in local]
                receipts.append(dict(video=video, original=receipt))
            base = len(records)
            records.extend(dict(row, partition=partition, global_index=base + i) for i, row in enumerate(local))
            offsets.extend((boundaries[1:] + offsets[-1]).tolist())
            chunks.append(value)
    if preserved != 244 or len(records) != 2044 or sum(row['partition'] == 'val' for row in records) != 92:
        raise ValueError('Unexpected temporal cohort size')
    if set(cohort['fit_video_ids']['train']) & set(cohort['fit_video_ids']['val']):
        raise ValueError('Procedure overlap')
    return config, cohort, np.concatenate(chunks), np.asarray(offsets), records, receipts


def job_inputs(raw, offsets, records, fitting, condition):
    if condition not in ('original_four', 'expanded_real_views'):
        raise ValueError(condition)
    chosen = [i for i, row in enumerate(records) if row['original_observation'] or
              (condition == 'expanded_real_views' and row['video_id'] in fitting)]
    local = [dict(records[i], parent_index=i, global_index=j) for j, i in enumerate(chosen)]
    bounds = np.concatenate(([0], np.cumsum([offsets[i + 1] - offsets[i] for i in chosen])))
    values = np.concatenate([raw[offsets[i]:offsets[i + 1]] for i in chosen])
    expected = 32 if condition == 'expanded_real_views' else 4
    counts = Counter((row['video_id'], row['lesion_id']) for row in local if row['video_id'] in fitting)
    if set(counts.values()) != {expected}:
        raise ValueError('Equal-lesion normalization requires the prescribed equal observation counts')
    if any(not row['original_observation'] and row['video_id'] not in fitting for row in local):
        raise ValueError('Additional held observation entered a job')
    return values, bounds, local


def train(run, smoke, resume, output, stop_after, input_loader=load_inputs, source_reader=source_identity):
    config, cohort, raw, offsets, records, receipts = input_loader(run)
    if not config['methods'] or len(set(config['methods'])) != len(config['methods']) or not set(config['methods']) <= {'token_sparse', 'token_dense', 'raw_supcon'}:
        raise ValueError('Unexpected method roster')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = output or (run / 'smoke' if smoke else run)
    root.mkdir(parents=True, exist_ok=True)
    sources = source_reader()
    identity = dict(input_sha256=shared.json_digest(receipts), source_hashes=sources)
    originals = [row for row in records if row['original_observation']]
    fold_videos = config.get('fold_training_videos', cohort['fit_video_ids']['train'])
    folds = shared.make_folds(originals, fold_videos, config['inner_folds'], config['fold_seed'])
    reference = read_json(Path(config['original_prepared_run']) / 'folds.json')
    if folds != reference:
        raise ValueError('Fixed procedure folds changed')
    atomic_write_json(root / 'records.json', records)
    atomic_write_json(root / 'folds.json', folds)
    atomic_write_json(root / 'input_identity.json', dict(identity, original_arrays_exact=config.get('original_arrays_exact', 244),
        train_observations=sum(row['partition'] == 'train' for row in records),
        validation_observations=sum(row['partition'] == 'val' for row in records), tokens=len(raw)))
    seeds = config['seeds'][:1] if smoke else config['seeds']
    active_folds = folds[:1] if smoke else folds
    steps = config['smoke_steps'] if smoke else config['steps']
    checkpoints = [steps] if smoke else config['checkpoints']
    total = len(config['conditions']) * len(config['methods']) * len(seeds) * (len(active_folds) + 1)
    outputs, selections = [], []
    for condition in config['conditions']:
        for seed in seeds:
            per_method = {method: [] for method in config['methods']}
            for fold_index, held in enumerate(active_folds):
                fitting = sorted(set(cohort['fit_video_ids']['train']) - set(held))
                values, bounds, local = job_inputs(raw, offsets, records, fitting, condition)
                pairs = shared.chronological_pairs(local, held)
                if any(not local[row[key]]['original_observation'] for row in pairs for key in ['source_index', 'query_index']):
                    raise ValueError('Held evaluation changed observations')
                for method in config['methods']:
                    directory = root / condition / 'inner' / method / f'seed{seed}' / f'fold{fold_index}'
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(directory / 'records.json', local)
                    job_identity = dict(identity, condition=condition, records_sha256=shared.json_digest(local))
                    context = dict(completed_jobs=len(outputs), total_jobs=total, condition=condition,
                        progress_path=str(root / 'training_progress.json'))
                    result = fit(config, method, seed, values, bounds, local, fitting, held, directory,
                        steps, checkpoints, job_identity, context, resume, stop_after)
                    item = dict(condition=condition, method=method, seed=seed, fold=fold_index,
                        fit_videos=fitting, held_videos=held, directory=str(directory), summary=result)
                    per_method[method].append(item)
                    outputs.append(item)
                del values
            values, bounds, local = job_inputs(raw, offsets, records, cohort['fit_video_ids']['train'], condition)
            for method in config['methods']:
                candidates = []
                for step in checkpoints:
                    rows = [row for item in per_method[method] for ev in item['summary']['evaluations']
                            if ev['step'] == step for row in ev['by_video'].values()]
                    candidates.append(dict(step=step, recall=float(np.mean([row['recall'] for row in rows])),
                                           procedures=len(rows)))
                selected = min(candidates, key=lambda row: (-row['recall'], row['step']))['step']
                selections.append(dict(condition=condition, method=method, seed=seed, step=selected, candidates=candidates))
                atomic_write_json(root / 'checkpoint_selection.json', selections)
                directory = root / condition / 'fit' / method / f'seed{seed}'
                directory.mkdir(parents=True, exist_ok=True)
                atomic_write_json(directory / 'records.json', local)
                job_identity = dict(identity, condition=condition, records_sha256=shared.json_digest(local))
                context = dict(completed_jobs=len(outputs), total_jobs=total, condition=condition,
                    progress_path=str(root / 'training_progress.json'))
                result = fit(config, method, seed, values, bounds, local, cohort['fit_video_ids']['train'],
                    cohort['fit_video_ids']['val'], directory, selected, [selected], job_identity, context, resume, stop_after)
                outputs.append(dict(condition=condition, method=method, seed=seed, fold='full',
                    fit_videos=cohort['fit_video_ids']['train'], held_videos=cohort['fit_video_ids']['val'],
                    directory=str(directory), summary=result))
            del values
    for condition in config['conditions']:
        for seed in seeds:
            for fold in list(range(len(active_folds))) + ['full']:
                sequences = [read_json(Path(row['directory']) / 'sequence.json') for row in outputs
                    if row['condition'] == condition and row['seed'] == seed and row['fold'] == fold]
                length = min(map(len, sequences))
                if any(sequence[:length] != sequences[0][:length] for sequence in sequences):
                    raise ValueError('Matched methods received different observations')
    if sources != source_reader():
        raise ValueError('Training sources changed')
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', outputs=outputs,
        selection=selections, **identity))
    atomic_write_json(root / 'training_progress.json', dict(status='COMPLETE', completed_jobs=total, total_jobs=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(args.run, args.smoke, args.resume, args.output, args.stop_after)
