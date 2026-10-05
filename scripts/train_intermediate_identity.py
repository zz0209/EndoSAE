import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from train_temporal_view_identity import read_tokens, source_identity as parent_sources, train
from analyze_temporal_identity_components import analyze
import train_acknowledgement_sae as shared
from src.checkpoint_io import read_json


def source_identity():
    return dict(parent_sources(), **{'scripts/train_intermediate_identity.py': shared.file_sha256(__file__)})


def load_inputs(run):
    config = read_json(run / 'config.json')
    cohort = read_json(config['cohort_config'])
    root = Path(config['intermediate_storage'])
    preparation = read_json(root / 'preparation_summary.json')
    if preparation['status'] != 'COMPLETE' or preparation['observations'] != 432:
        raise ValueError('Intermediate preparation is incomplete')
    if not all(row['block10_reference_exact'] and row['direct_index_exact'] for row in preparation['receipts']):
        raise ValueError('Matched original input verification failed')
    original_train = cohort['fit_video_ids']['train']
    added = config['added_training_videos']
    if original_train != config['fold_training_videos'] or set(added) & set(sum(cohort['fit_video_ids'].values(), [])):
        raise ValueError('Fitting or exclusion population changed')
    roster = [(video, partition) for partition in ['train', 'val'] for video in cohort['fit_video_ids'][partition]]
    roster.extend((video, 'train') for video in added)
    records, values, offsets, receipts = [], [], [0], []
    for video, partition in roster:
        value, boundaries, local, receipt = read_tokens(root / 'tokens' / video)
        if any(row['video_id'] != video or row['split'] != partition or not row['original_observation']
               or row['representation_layer'] != 5 for row in local):
            raise ValueError('Unexpected layer or input identity')
        if set(Counter(row['lesion_id'] for row in local).values()) != {4}:
            raise ValueError('Four original observations are required per lesion')
        base = len(records)
        records.extend(dict(row, partition=partition, global_index=base + i) for i, row in enumerate(local))
        values.append(value)
        offsets.extend((boundaries[1:] + offsets[-1]).tolist())
        receipts.append(dict(video=video, intermediate=receipt))
    if len(records) != 432 or sum(row['partition'] == 'train' for row in records) != 340:
        raise ValueError('Training and validation observation counts changed')
    if len({(row['clip_id'], row['lesion_id']) for row in records}) != 432:
        raise ValueError('Duplicate observation')
    cohort = dict(cohort, fit_video_ids=dict(cohort['fit_video_ids'], train=original_train + added))
    return config, cohort, np.concatenate(values), np.asarray(offsets), records, receipts


def component_inputs(config):
    _, _, values, offsets, records, _ = load_inputs(Path(config['training_run']))
    return values, offsets, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['train', 'components'], default='train')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.phase == 'train':
        train(args.run, args.smoke, args.resume, args.output, args.stop_after,
              input_loader=load_inputs, source_reader=source_identity)
    else:
        root = args.output or (args.run / 'smoke' if args.smoke else args.run)
        analyze(args.run, root, root / 'components', args.resume, input_loader=component_inputs)
