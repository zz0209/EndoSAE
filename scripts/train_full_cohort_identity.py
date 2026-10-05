import argparse
from collections import Counter
from pathlib import Path

from train_temporal_view_identity import load_inputs as original_inputs
from train_temporal_view_identity import read_tokens, source_identity as parent_sources, train
import train_acknowledgement_sae as shared
import numpy as np
from src.checkpoint_io import read_json


def source_identity():
    return dict(parent_sources(), **{'scripts/train_full_cohort_identity.py': shared.file_sha256(__file__)})


def load_inputs(run):
    config, cohort, raw, offsets, records, receipts = original_inputs(run)
    storage = Path(config['added_training_storage'])
    preparation = read_json(storage / 'preparation_summary.json')
    if preparation['status'] != 'COMPLETE' or preparation['clips'] != 768:
        raise ValueError('Added input preparation is incomplete')
    added = config['added_training_videos']
    if len(added) != 8 or len(set(added)) != 8 or set(added) & {row['video_id'] for row in records}:
        raise ValueError('Added procedure membership differs')
    if config['fold_training_videos'] != cohort['fit_video_ids']['train']:
        raise ValueError('Original exclusion population changed')
    chunks, boundaries = [raw], offsets.tolist()
    for video in added:
        value, local_offsets, local, receipt = read_tokens(storage / 'tokens' / video)
        if any(row['video_id'] != video or row['split'] != 'train' for row in local):
            raise ValueError('Added input has an unexpected partition')
        counts = Counter(row['lesion_id'] for row in local)
        originals = Counter(row['lesion_id'] for row in local if row['original_observation'])
        if set(counts.values()) != {32} or set(originals) != set(counts) or set(originals.values()) != {4}:
            raise ValueError('Added lesion observation counts differ')
        base = len(records)
        records.extend(dict(row, partition='train', global_index=base + i) for i, row in enumerate(local))
        boundaries.extend((local_offsets[1:] + boundaries[-1]).tolist())
        chunks.append(value)
        receipts.append(dict(video=video, added=receipt))
    train_rows = [row for row in records if row['partition'] == 'train']
    if len(records) != 2812 or len(train_rows) != 2720:
        raise ValueError('Full training observation count differs')
    if len({(row['video_id'], row['lesion_id']) for row in train_rows}) != 85:
        raise ValueError('Full training lesion membership differs')
    if len({(row['clip_id'], row['lesion_id']) for row in records}) != len(records):
        raise ValueError('Duplicate observation identity')
    cohort = dict(cohort, fit_video_ids=dict(cohort['fit_video_ids'],
        train=cohort['fit_video_ids']['train'] + added))
    return config, cohort, np.concatenate(chunks), np.asarray(boundaries), records, receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(args.run, args.smoke, args.resume, args.output, args.stop_after,
          input_loader=load_inputs, source_reader=source_identity)
