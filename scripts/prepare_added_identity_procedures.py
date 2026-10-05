import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
from prepare_identity_view_expansion import prepare_metadata, extract, selected_clips, source_identity as parent_sources
from run_continuous_confirmation import prepare_input
from train_acknowledgement_sae import now, save_npz
from train_frozen_identity_supcon import observation_interval
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames


def source_identity():
    return dict(parent_sources(), **{'scripts/prepare_added_identity_procedures.py': digest(__file__)})


def encode(run, config, cohort, videos, root, smoke, resume, stop_after):
    encoding = read_json(config['source_encoding_config'])
    annotations = {row['video_id']: row for row in read_json(cohort['annotation_summary'])['videos']}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(encoding, torch.device(config['device']))
    sources = source_identity()
    total = sum(len(selected_clips(run, cohort, video, smoke)) for video in videos)
    completed, reused, receipts = 0, 0, []
    started = time.perf_counter()
    for video in videos:
        clips = selected_clips(run, cohort, video, smoke)
        original_metadata = Path(cohort['metadata_root']) / (video + '.json')
        original = read_json(original_metadata)
        original_keys = {(row['clip_id'], lesion) for row in original['clips'] for lesion in row['sampling_lesion_ids']}
        directory = root / 'tokens' / video
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(config_sha256=digest(run / 'config.json'), source_hashes=sources,
            metadata_sha256=digest(run / 'metadata' / (video + '.json')),
            original_metadata_sha256=digest(original_metadata),
            extraction_sha256=digest(root / 'extraction' / video / ('extracted_' + video + '.json')),
            backbone_sha256=encoder.state_sha256, selected=[row['clip_id'] for row in clips],
            python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=config['device'])
        reference, reference_indices = None, {}
        if video in config['native_reference_videos']:
            cache = Path(cohort['cache_root']) / video
            cache_identity = read_json(cache / 'identity.json')
            cache_receipt = read_json(cache / 'complete.json')
            if cache_receipt['status'] != 'COMPLETE' or digest(cache / 'identity.json') != cache_receipt['identity_sha256']:
                raise ValueError('Incomplete native reference cache')
            if cache_identity['backbone_sha256'] != encoder.state_sha256 or cache_identity['metadata_sha256'] != digest(original_metadata):
                raise ValueError('Reference input or backbone differs')
            reference = np.load(cache / 'block10.npy', mmap_mode='r', allow_pickle=False)
            if reference.shape != (len(cache_identity['clip_ids']), 1569, 768) or reference.dtype != np.float32:
                raise ValueError('Invalid native reference dimensions')
            reference_indices = {key: i for i, key in enumerate(cache_identity['clip_ids'])}
            identity['native_reference_identity_sha256'] = cache_receipt['identity_sha256']
        identity_path = directory / 'identity.json'
        if identity_path.exists():
            if not resume or read_json(identity_path) != identity:
                raise ValueError('Added-procedure encoding identity changed')
        else:
            atomic_write_json(identity_path, identity)
        records, values, positions, offsets, checks = [], [], [], [0], []
        for clip_index, clip in enumerate(clips):
            asset = directory / (clip['clip_id'] + '.npz')
            receipt_path = asset.with_suffix('.json')
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if not resume or receipt['identity_sha256'] != digest(identity_path) or digest(asset) != receipt['sha256']:
                    raise ValueError('Saved clip identity changed')
                reused += 1
            else:
                inputs, _ = prepare_input(clip, root / 'frames')
                native, _ = encoder(inputs)
                frames = native[1:].reshape(196, 8, 768).transpose(1, 0, 2)
                np.testing.assert_array_equal(frames, tokens_to_frames(native[None]))
                verified = []
                if clip['clip_id'] in reference_indices:
                    np.testing.assert_array_equal(native, reference[reference_indices[clip['clip_id']]])
                    verified.append(dict(clip_id=clip['clip_id'], native_reference_exact=True))
                arrays, rows = {}, []
                for lesion_index, lesion in enumerate(clip['sampling_lesion_ids']):
                    masks = np.stack([project_boxes(dict(frame, boxes_xyxy=[box for box in frame['boxes_xyxy']
                        if box['lesion_id'] == lesion]))[0] for frame in clip['frames']])
                    if not masks.any(axis=1).all():
                        raise ValueError('Lesion lacks projected support')
                    value = frames[masks].astype(np.float32)
                    if value.shape[1] != config['input_dim'] or not np.isfinite(value).all():
                        raise ValueError('Invalid region tokens')
                    interval = observation_interval(annotations[video], lesion, clip)
                    rows.append(dict(index=lesion_index, video_id=video, split='train', lesion_id=lesion,
                        clip_id=clip['clip_id'], cache_clip_index=clip_index, start_frame=clip['start_frame'],
                        end_frame=clip['end_frame'], fps=clip['fps'], roi_tokens_per_frame=masks.sum(axis=1).tolist(),
                        original_mean_norm=float(np.linalg.norm(value.mean(axis=0, dtype=np.float64))),
                        annotation_observation=interval, lesion_first_frame=interval['lesion_first_frame'],
                        original_observation=(clip['clip_id'], lesion) in original_keys))
                    arrays[f'tokens_{lesion_index}'] = value
                    arrays[f'positions_{lesion_index}'] = np.column_stack(np.where(masks)).astype(np.int16)
                save_npz(asset, **arrays)
                receipt = dict(identity_sha256=digest(identity_path), sha256=digest(asset),
                    records=rows, checks=verified, completed_at=now())
                atomic_write_json(receipt_path, receipt)
            with np.load(asset, allow_pickle=False) as archive:
                for i, row in enumerate(receipt['records']):
                    value = archive[f'tokens_{i}'].copy()
                    records.append(dict(row, index=len(records)))
                    values.append(value)
                    positions.append(archive[f'positions_{i}'].copy())
                    offsets.append(offsets[-1] + len(value))
            checks.extend(receipt['checks'])
            completed += 1
            progress = dict(status='RUNNING', completed=completed, total=total, reused=reused,
                video=video, seconds=time.perf_counter() - started, clip_id=clip['clip_id'])
            atomic_write_json(root / 'encoding_progress.json', progress)
            print('ADDED_PROCEDURE_CLIP', progress, flush=True)
            pause_after_checkpoint(receipt_path)
            if completed == stop_after:
                raise SystemExit(75)
        if not smoke and {(row['clip_id'], row['lesion_id']) for row in records if row['original_observation']} != original_keys:
            raise ValueError('Original four-observation membership changed')
        save_npz(directory / 'tokens.npz', tokens=np.concatenate(values), offsets=np.array(offsets),
            positions=np.concatenate(positions))
        atomic_write_json(directory / 'records.json', records)
        receipt = dict(status='COMPLETE', identity=identity, clips=len(records), tokens=offsets[-1],
            original_native_checks=checks, tokens_sha256=digest(directory / 'tokens.npz'),
            records_sha256=digest(directory / 'records.json'), completed_at=now())
        atomic_write_json(directory / 'complete.json', receipt)
        receipts.append(receipt)
        del reference
    if sources != source_identity():
        raise ValueError('Preparation sources changed')
    summary = dict(status='COMPLETE', receipts=receipts, clips=sum(row['clips'] for row in receipts),
        tokens=sum(row['tokens'] for row in receipts), seconds=time.perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), completed_at=now())
    if not smoke and set(videos) == set(config['added_training_videos']) and summary['clips'] != config['expected_added_observations']:
        raise ValueError('Incomplete added cohort')
    atomic_write_json(root / 'preparation_summary.json', summary)
    atomic_write_json(root / 'encoding_progress.json', dict(status='COMPLETE', completed=total, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['metadata', 'extract', 'encode'], required=True)
    parser.add_argument('--video')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    config = read_json(args.run / 'config.json')
    cohort = read_json(config['cohort_config'])
    added = config['added_training_videos']
    if len(added) != 8 or len(set(added)) != 8 or set(added) & set(sum(cohort['fit_video_ids'].values(), [])):
        raise ValueError('Added training roster overlaps existing cohort')
    videos = [args.video or config['smoke_video']] if args.smoke or args.video else added
    if not set(videos) <= set(added):
        raise ValueError('Unspecified procedure')
    root = Path(config['storage_root']) / 'smoke' if args.smoke else Path(config['storage_root'])
    if args.phase == 'metadata':
        prepare_metadata(args.run, config, cohort, videos, args.resume)
    elif args.phase == 'extract':
        extract(args.run, config, cohort, videos, root, args.smoke)
    else:
        encode(args.run, config, cohort, videos, root, args.smoke, args.resume, args.stop_after)
