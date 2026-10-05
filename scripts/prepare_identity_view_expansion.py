import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder, extraction_request
from prepare_rc27_task_cohort import sample_video
from run_continuous_confirmation import prepare_input
from scripts.build_realcolon_visibility_pack import extract_task_manifest
from train_acknowledgement_sae import now, save_npz
from train_frozen_identity_supcon import observation_interval
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames


def source_identity():
    return {name: digest(ROOT / name) for name in [
        'scripts/prepare_identity_view_expansion.py', 'scripts/prepare_rc27_task_cohort.py',
        'scripts/encode_rc27_cohort.py', 'scripts/build_realcolon_visibility_pack.py',
        'scripts/run_continuous_confirmation.py', 'scripts/train_frozen_identity_supcon.py',
        'src/evaluation/realcolon_task.py']}


def prepare_metadata(run, config, cohort, videos, resume):
    output = run / 'metadata'
    output.mkdir(parents=True, exist_ok=True)
    sampling = dict(read_json(config['source_sampling_config']),
                    positive_clips_per_lesion=config['positive_clips_per_lesion'],
                    negative_clips_per_video=config['negative_clips_per_video'])
    with (Path(sampling['annotation_directory']) / 'video_info.csv').open(encoding='utf-8-sig', newline='') as stream:
        information = {row['unique_video_name']: row for row in csv.DictReader(stream)}
    summaries = []
    for index, video in enumerate(videos):
        old_path = Path(cohort['metadata_root']) / (video + '.json')
        old = read_json(old_path)
        required = {row['lesion_id']: row['selected_starts'] for row in old['summary']['lesions']}
        spec = {key: old['summary'][key] for key in ['video_id', 'split', 'official_lesion_ids']}
        assert spec['split'] == 'train'
        identity = dict(config=digest(run / 'config.json'), original_metadata=digest(old_path),
                        sampling_config=digest(config['source_sampling_config']), sources=source_identity())
        receipt = output / (video + '_complete.json')
        if receipt.exists():
            saved = read_json(receipt)
            assert resume and saved['identity'] == identity
            assert digest(output / (video + '.json')) == saved['metadata_sha256']
        else:
            summary, clips = sample_video(spec, sampling, information[video], output, required)
            original = {row['clip_id']: row for row in old['clips'] if row['sampling_lesion_ids']}
            selected = {row['clip_id']: row for row in clips}
            assert set(original) <= set(selected)
            for key, row in original.items():
                assert row['frames'] == selected[key]['frames']
                assert set(row['sampling_lesion_ids']) <= set(selected[key]['sampling_lesion_ids'])
            saved = dict(status='COMPLETE', identity=identity,
                metadata_sha256=digest(output / (video + '.json')), clips=len(clips),
                observations=sum(len(row['sampling_lesion_ids']) for row in clips),
                original_observations=sum(map(len, required.values())), seconds=summary['seconds'],
                completed_at=now(), video_id=video)
            atomic_write_json(receipt, saved)
        summaries.append(saved)
        atomic_write_json(run / 'metadata_progress.json', dict(status='RUNNING', video=video,
            completed=index + 1, total=len(videos)))
        pause_after_checkpoint(receipt)
    atomic_write_json(run / 'metadata_summary.json', dict(status='COMPLETE', receipts=summaries,
        clips=sum(row['clips'] for row in summaries), observations=sum(row['observations'] for row in summaries)))


def selected_clips(run, cohort, video, smoke):
    clips = read_json(run / 'metadata' / (video + '.json'))['clips']
    if smoke:
        old_ids = {row['clip_id'] for row in read_json(Path(cohort['input_descriptor_root']) / video / 'records.json')}
        clips = [next(row for row in clips if row['clip_id'] in old_ids),
                 next(row for row in clips if row['clip_id'] not in old_ids)]
    return clips


def extract(run, config, cohort, videos, root, smoke):
    encoding = read_json(config['source_encoding_config'])
    started = time.perf_counter()
    receipts = []
    for index, video in enumerate(videos):
        clips = selected_clips(run, cohort, video, smoke)
        output = root / 'extraction' / video
        output.mkdir(parents=True, exist_ok=True)
        request = output / 'requested_frames.jsonl'
        count = extraction_request(clips, request)
        extract_task_manifest(request, Path(encoding['frame_archive_directory']), root / 'frames', output)
        receipt = read_json(output / 'extraction_summary.json')
        assert receipt['status'] == 'EXTRACTED_DIMENSIONS_VERIFIED' and receipt['frames'] == count
        receipts.append(dict(video_id=video, frames=count, summary=receipt))
        atomic_write_json(root / 'extraction_progress.json', dict(status='RUNNING', completed=index + 1,
            total=len(videos), video=video, seconds=time.perf_counter() - started))
        pause_after_checkpoint(output / 'extraction_summary.json')
    atomic_write_json(root / 'extraction_complete.json', dict(status='COMPLETE', receipts=receipts,
        seconds=time.perf_counter() - started))


def encode(run, config, cohort, videos, root, smoke, resume, stop_after):
    encoding = read_json(config['source_encoding_config'])
    annotations = {row['video_id']: row for row in read_json(cohort['annotation_summary'])['videos']}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(encoding, torch.device(config['device']))
    sources = source_identity()
    completed, reused, receipts = 0, 0, []
    total = sum(len(selected_clips(run, cohort, video, smoke)) for video in videos)
    started = time.perf_counter()
    for video in videos:
        clips = selected_clips(run, cohort, video, smoke)
        directory = root / 'tokens' / video
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(config_sha256=digest(run / 'config.json'), source_hashes=sources,
            metadata_sha256=digest(run / 'metadata' / (video + '.json')),
            extraction_sha256=digest(root / 'extraction' / video / ('extracted_' + video + '.json')),
            backbone_sha256=encoder.state_sha256, selected=[row['clip_id'] for row in clips],
            python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=config['device'])
        if (directory / 'identity.json').exists():
            assert resume and read_json(directory / 'identity.json') == identity
        else:
            atomic_write_json(directory / 'identity.json', identity)
        original = Path(config['original_prepared_run']) / 'tokens' / video
        original_records = read_json(original / 'records.json')
        lookup = {(row['clip_id'], row['lesion_id']): i for i, row in enumerate(original_records)}
        original_receipt = read_json(original / 'complete.json')
        assert digest(original / 'tokens.npz') == original_receipt['tokens_sha256']
        with np.load(original / 'tokens.npz', allow_pickle=False) as archive:
            original_tokens, original_offsets = archive['tokens'].copy(), archive['offsets'].copy()
        records, values, positions, offsets, checks = [], [], [], [0], []
        for clip_index, clip in enumerate(clips):
            asset = directory / (clip['clip_id'] + '.npz')
            receipt_path = asset.with_suffix('.json')
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert resume and receipt['identity_sha256'] == digest(directory / 'identity.json')
                assert digest(asset) == receipt['sha256']
                reused += 1
            else:
                inputs, _ = prepare_input(clip, root / 'frames')
                native, _ = encoder(inputs)
                manual = native[1:].reshape(196, 8, 768).transpose(1, 0, 2)
                np.testing.assert_array_equal(manual, tokens_to_frames(native[None]))
                arrays, rows, verified = {}, [], []
                for lesion_index, lesion in enumerate(clip['sampling_lesion_ids']):
                    masks = np.stack([project_boxes(dict(frame, boxes_xyxy=[box for box in frame['boxes_xyxy']
                        if box['lesion_id'] == lesion]))[0] for frame in clip['frames']])
                    assert masks.any(axis=1).all()
                    value = manual[masks].astype(np.float32)
                    assert value.shape[1] == config['input_dim'] and np.isfinite(value).all()
                    key = (clip['clip_id'], lesion)
                    if key in lookup:
                        i = lookup[key]
                        np.testing.assert_array_equal(value, original_tokens[original_offsets[i]:original_offsets[i + 1]])
                        verified.append(dict(clip_id=clip['clip_id'], lesion_id=lesion, original_tokens_exact=True))
                    interval = observation_interval(annotations[video], lesion, clip)
                    row = dict(index=lesion_index, video_id=video, split='train', lesion_id=lesion,
                        clip_id=clip['clip_id'], cache_clip_index=clip_index, start_frame=clip['start_frame'],
                        end_frame=clip['end_frame'], fps=clip['fps'], roi_tokens_per_frame=masks.sum(axis=1).tolist(),
                        original_mean_norm=float(np.linalg.norm(value.mean(axis=0, dtype=np.float64))),
                        annotation_observation=interval, lesion_first_frame=interval['lesion_first_frame'],
                        original_observation=key in lookup)
                    rows.append(row)
                    arrays[f'tokens_{lesion_index}'] = value
                    arrays[f'positions_{lesion_index}'] = np.column_stack(np.where(masks)).astype(np.int16)
                save_npz(asset, **arrays)
                receipt = dict(identity_sha256=digest(directory / 'identity.json'), sha256=digest(asset),
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
            print('TEMPORAL_CLIP', json.dumps(progress), flush=True)
            pause_after_checkpoint(receipt_path)
            if completed == stop_after:
                raise SystemExit(75)
        save_npz(directory / 'tokens.npz', tokens=np.concatenate(values), offsets=np.array(offsets),
                 positions=np.concatenate(positions))
        atomic_write_json(directory / 'records.json', records)
        receipt = dict(status='COMPLETE', identity=identity, clips=len(records), tokens=offsets[-1],
            original_exact_checks=checks, tokens_sha256=digest(directory / 'tokens.npz'),
            records_sha256=digest(directory / 'records.json'), completed_at=now())
        atomic_write_json(directory / 'complete.json', receipt)
        receipts.append(receipt)
    assert sources == source_identity()
    summary = dict(status='COMPLETE', receipts=receipts, clips=sum(row['clips'] for row in receipts),
        tokens=sum(row['tokens'] for row in receipts), seconds=time.perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), completed_at=now())
    atomic_write_json(root / 'preparation_summary.json', summary)
    atomic_write_json(root / 'encoding_progress.json', dict(status='COMPLETE', completed=total, total=total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['metadata', 'extract', 'encode'])
    parser.add_argument('--video')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    config = read_json(args.run / 'config.json')
    cohort = read_json(config['cohort_config'])
    roster = cohort['fit_video_ids']['train']
    videos = [args.video or config['smoke_video']] if args.smoke or args.video else roster
    assert set(videos) <= set(roster) and len(set(roster)) == 19
    root = Path(config['storage_root']) / 'smoke' if args.smoke else Path(config['storage_root'])
    if args.phase == 'metadata':
        prepare_metadata(args.run, config, cohort, videos, args.resume)
    elif args.phase == 'extract':
        extract(args.run, config, cohort, videos, root, args.smoke)
    else:
        encode(args.run, config, cohort, videos, root, args.smoke, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
