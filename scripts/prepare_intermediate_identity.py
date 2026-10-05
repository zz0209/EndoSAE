import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_temporal_identity_components import original_inputs
from encode_rc27_cohort import Encoder
from run_continuous_confirmation import prepare_input
from train_acknowledgement_sae import now, save_npz, json_digest
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames


def inputs(run):
    config = read_json(run / 'config.json')
    reference = read_json(config['reference_training_config'])
    cohort = read_json(reference['cohort_config'])
    encoding = read_json(config['encoding_config'])
    raw, offsets, records = original_inputs(reference)
    if len(records) != config['expected_observations'] or config['block_index'] != 5:
        raise ValueError('Prescribed cohort or layer changed')
    metadata = {video: read_json(Path(cohort['metadata_root']) / (video + '.json'))
                for video in sorted({row['video_id'] for row in records})}
    clips = {video: {row['clip_id']: row for row in value['clips']} for video, value in metadata.items()}
    frame_roots = {video: Path(config['added_frame_root'] if video in reference['added_training_videos']
                              else encoding['frame_root']) for video in metadata}
    frames = set()
    for row in records:
        clip = clips[row['video_id']][row['clip_id']]
        if clip['split'] != row['split'] or row['split'] not in ['train', 'val']:
            raise ValueError('Observation split differs')
        for frame in clip['frames']:
            frames.add(frame_roots[row['video_id']] / row['video_id'] / ('%06d.jpg' % frame['frame_index']))
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    identity = dict(config_sha256=digest(run / 'config.json'), reference_sha256=digest(config['reference_training_config']),
        records_sha256=json_digest(records), metadata_sha256=json_digest(metadata),
        sources={name: digest(ROOT / name) for name in ['scripts/prepare_intermediate_identity.py',
            'scripts/encode_rc27_cohort.py', 'scripts/run_model_port_parity.py',
            'scripts/run_continuous_confirmation.py', 'src/evaluation/realcolon_task.py',
            'scripts/analyze_temporal_identity_components.py', 'scripts/train_temporal_view_identity.py']})
    return config, encoding, raw, offsets, records, clips, frame_roots, identity, len(frames)


@torch.no_grad()
def encode(run, smoke, resume, stop_after, preflight):
    config, encoding, raw, bounds, records, clips, frame_roots, identity, frame_count = inputs(run)
    atomic_write_json(run / 'input_verification.json', dict(status='PASS', observations=len(records),
        procedures=len({row['video_id'] for row in records}), frame_count=frame_count,
        token_bytes=int(raw.nbytes), identity=identity))
    if preflight:
        print('INPUTS_PASS', len(records), 'observations', frame_count, 'frames', flush=True)
        return
    root = Path(config['storage_root']) / 'smoke' if smoke else Path(config['storage_root'])
    selected = list(range(len(records)))
    if smoke:
        selected = [index for video in config['smoke_videos']
                    for index in [i for i, row in enumerate(records) if row['video_id'] == video][:2]]
        if len(selected) != 6:
            raise ValueError('Smoke requires two real observations from each prescribed source')
    identity['selected_indices'] = selected
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / 'identity.json'
    if identity_path.exists():
        if not resume or read_json(identity_path) != identity:
            raise ValueError('Prepared input identity changed')
    else:
        atomic_write_json(identity_path, identity)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(encoding, torch.device(config['device']))
    captured = {}
    handle = encoder.model.blocks[config['block_index']].register_forward_hook(
        lambda _module, _inputs, output: captured.update(value=output.detach()))
    receipts, grouped, reused = [], {}, 0
    started = time.perf_counter()
    for completed, index in enumerate(selected, 1):
        row = records[index]
        directory = root / 'observations'
        directory.mkdir(exist_ok=True)
        asset = directory / f'{index:04d}.npz'
        receipt_path = asset.with_suffix('.json')
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if not resume or receipt['identity_sha256'] != digest(identity_path) or digest(asset) != receipt['sha256']:
                raise ValueError('Saved observation changed')
            reused += 1
        else:
            clip = clips[row['video_id']][row['clip_id']]
            image, _ = prepare_input(clip, frame_roots[row['video_id']])
            native, _ = encoder(image)
            middle = captured['value'][0].cpu().numpy().copy()
            if middle.shape != (1569, 768) or not np.isfinite(middle).all():
                raise ValueError('Invalid middle-layer activations')
            masks = np.stack([project_boxes(dict(frame, boxes_xyxy=[box for box in frame['boxes_xyxy']
                if box['lesion_id'] == row['lesion_id']]))[0] for frame in clip['frames']])
            np.testing.assert_array_equal(masks.sum(axis=1), row['roi_tokens_per_frame'])
            positions = np.column_stack(np.where(masks)).astype(np.int16)
            reference = tokens_to_frames(native[None])[masks]
            np.testing.assert_array_equal(reference, raw[bounds[index]:bounds[index + 1]])
            values = tokens_to_frames(middle[None])[masks].copy()
            direct = middle[1 + positions[:, 1].astype(np.int64) * 8 + positions[:, 0]]
            np.testing.assert_array_equal(values, direct)
            suffix_error = None
            if smoke:
                replay = captured['value'].clone()
                for block in encoder.model.blocks[config['block_index'] + 1:11]:
                    replay = block(replay, 1, 8, 14)
                np.testing.assert_array_equal(replay[0].cpu().numpy(), native)
                suffix_error = float(np.max(np.abs(replay[0].cpu().numpy() - native)))
            save_npz(asset, tokens=values, positions=positions)
            receipt = dict(identity_sha256=digest(identity_path), sha256=digest(asset), index=index,
                block10_reference_exact=True, direct_index_exact=True, suffix_max_error=suffix_error,
                backbone_sha256=encoder.state_sha256, token_count=len(values), completed_at=now())
            atomic_write_json(receipt_path, receipt)
        receipts.append(receipt)
        grouped.setdefault(row['video_id'], []).append((index, asset))
        progress = dict(status='RUNNING', completed=completed, total=len(selected), reused=reused,
            seconds=time.perf_counter() - started, video=row['video_id'])
        atomic_write_json(root / 'encoding_progress.json', progress)
        print('MIDDLE_LAYER_OBSERVATION', progress, flush=True)
        pause_after_checkpoint(receipt_path)
        if completed == stop_after:
            raise SystemExit(75)
    for video, assets in grouped.items():
        values, positions, local, offsets = [], [], [], [0]
        for index, asset in assets:
            with np.load(asset, allow_pickle=False) as saved:
                values.append(saved['tokens'].copy())
                positions.append(saved['positions'].copy())
            local.append(dict(records[index], index=len(local), original_observation=True,
                representation_layer=config['block_index']))
            offsets.append(offsets[-1] + len(values[-1]))
        directory = root / 'tokens' / video
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / 'tokens.npz', tokens=np.concatenate(values), offsets=np.asarray(offsets),
            positions=np.concatenate(positions))
        atomic_write_json(directory / 'records.json', local)
        atomic_write_json(directory / 'complete.json', dict(status='COMPLETE', observations=len(local),
            tokens_sha256=digest(directory / 'tokens.npz'), records_sha256=digest(directory / 'records.json'),
            identity_sha256=digest(identity_path), completed_at=now()))
    handle.remove()
    if identity['sources'] != {name: digest(ROOT / name) for name in identity['sources']}:
        raise ValueError('Encoding sources changed')
    summary = dict(status='COMPLETE', observations=len(selected), reused=reused, receipts=receipts,
        seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                     device=config['device'], tf32=False), completed_at=now())
    atomic_write_json(root / 'preparation_summary.json', summary)
    atomic_write_json(root / 'encoding_progress.json', dict(status='COMPLETE', completed=len(selected), total=len(selected)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    encode(args.run, args.smoke, args.resume, args.stop_after, args.preflight)
