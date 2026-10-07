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
from run_continuous_confirmation import prepare_input
from train_acknowledgement_sae import now, save_npz, json_digest
from train_temporal_view_identity import read_tokens
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames


def inputs(run):
    config = read_json(run / 'config.json')
    reference = read_json(Path(config['reference_run']) / 'config.json')
    encoding = read_json(config['encoding_config'])
    sources = {}
    records = []
    for kind, videos, storage, preparation in [
        ('expanded', reference['fold_training_videos'], reference['expanded_storage'], config['expanded_preparation']),
        ('added', reference['added_training_videos'], reference['added_training_storage'], reference['preparation_run'])]:
        for video in videos:
            directory = Path(storage) / 'tokens' / video
            local = read_json(directory / 'records.json')
            receipt = read_json(directory / 'complete.json')
            if receipt['status'] != 'COMPLETE' or digest(directory / 'records.json') != receipt['records_sha256']:
                raise ValueError('Incomplete or changed input records')
            metadata_path = Path(preparation) / 'metadata' / (video + '.json')
            metadata = read_json(metadata_path)
            clips = {row['clip_id']: row for row in metadata['clips']}
            frame_root = Path(storage) / 'frames'
            sources[video] = dict(directory=directory, clips=clips, frame_root=frame_root,
                                  metadata_sha256=digest(metadata_path), receipt=receipt)
            for local_index, row in enumerate(local):
                assert row['split'] == 'train' and row['video_id'] == video
                clip = clips[row['clip_id']]
                for frame in clip['frames']:
                    path = frame_root / video / ('%06d.jpg' % frame['frame_index'])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                records.append(dict(row, index=len(records), local_index=local_index, input_kind=kind))
    assert len(records) == 2720 and len(sources) == 27
    assert len({(row['video_id'], row['lesion_id']) for row in records}) == 85
    identity = dict(config_sha256=digest(run / 'config.json'), records_sha256=json_digest(records),
        metadata={video: value['metadata_sha256'] for video, value in sources.items()},
        inputs={video: value['receipt'] for video, value in sources.items()},
        source_hashes={name: digest(ROOT / name) for name in ['scripts/prepare_adaptive_identity.py',
            'scripts/encode_rc27_cohort.py', 'scripts/run_model_port_parity.py',
            'scripts/run_continuous_confirmation.py', 'src/evaluation/realcolon_task.py']})
    return config, encoding, records, sources, identity


@torch.no_grad()
def prepare(run, smoke, resume, stop_after, preflight):
    config, encoding, records, sources, identity = inputs(run)
    if preflight:
        atomic_write_json(run / 'input_verification.json', dict(status='PASS', observations=len(records),
            procedures=len(sources), identity=identity))
        print('INPUTS_PASS', len(records), len(sources), flush=True)
        return
    root = Path(config['storage_root']) / ('smoke' if smoke else 'prepared')
    root.mkdir(parents=True, exist_ok=True)
    selected = list(range(len(records)))
    if smoke:
        selected = [index for video in config['smoke_videos']
                    for index in [i for i, row in enumerate(records) if row['video_id'] == video][:2]]
        assert len(selected) == 6
    identity['selected_indices'] = selected
    signature = json_digest(identity)
    identity_path = root / 'identity.json'
    if identity_path.exists():
        if not resume or read_json(identity_path) != identity:
            raise ValueError('Preparation identity differs')
    else:
        atomic_write_json(identity_path, identity)
    atomic_write_json(root / 'records.json', records)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(encoding, torch.device('cuda'))
    captured = {}
    handle = encoder.model.blocks[8].register_forward_hook(
        lambda _module, _inputs, output: captured.update(value=output.detach()))
    started = time.perf_counter()
    previous_video, reused = None, 0
    receipts = []
    for completed, index in enumerate(selected, 1):
        row = records[index]
        folder = root / 'observations' / f'{index:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        receipt_path = folder / 'complete.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            assert resume and receipt['identity_sha256'] == signature
            assert digest(folder / 'prefix.npy') == receipt['prefix_sha256']
            assert digest(folder / 'roi.npz') == receipt['roi_sha256']
            reused += 1
        else:
            source = sources[row['video_id']]
            if previous_video != row['video_id']:
                raw, offsets, local, _ = read_tokens(source['directory'])
                previous_video = row['video_id']
            clip = source['clips'][row['clip_id']]
            image, _ = prepare_input(clip, source['frame_root'])
            native, _ = encoder(image)
            prefix = captured['value'][0].cpu().numpy().copy()
            assert prefix.shape == (1569, 768) and np.isfinite(prefix).all()
            masks = np.stack([project_boxes(dict(frame, boxes_xyxy=[box for box in frame['boxes_xyxy']
                if box['lesion_id'] == row['lesion_id']]))[0] for frame in clip['frames']])
            np.testing.assert_array_equal(masks.sum(1), row['roi_tokens_per_frame'])
            positions = np.column_stack(np.where(masks)).astype(np.int16)
            value = tokens_to_frames(native[None])[masks]
            j = row['local_index']
            np.testing.assert_array_equal(value, raw[offsets[j]:offsets[j + 1]])
            suffix_error = None
            if smoke:
                replay = captured['value'].clone()
                for block in encoder.model.blocks[9:11]:
                    replay = block(replay, 1, 8, 14)
                np.testing.assert_array_equal(replay[0].cpu().numpy(), native)
                suffix_error = float(np.abs(replay[0].cpu().numpy() - native).max())
            temporary = folder / 'prefix.pending.npy'
            np.save(temporary, prefix, allow_pickle=False)
            temporary.replace(folder / 'prefix.npy')
            save_npz(folder / 'roi.npz', native=value, positions=positions)
            receipt = dict(identity_sha256=signature, index=index, prefix_sha256=digest(folder / 'prefix.npy'),
                roi_sha256=digest(folder / 'roi.npz'), native_roi_exact=True, suffix_max_error=suffix_error,
                backbone_sha256=encoder.state_sha256, completed_at=now())
            atomic_write_json(receipt_path, receipt)
        receipts.append(receipt)
        progress = dict(status='RUNNING', completed=completed, total=len(selected), reused=reused,
            seconds=time.perf_counter() - started, video=row['video_id'], updated_at=now())
        atomic_write_json(root / 'progress.json', progress)
        if completed % 10 == 0 or smoke:
            print('PREFIX_PREPARED', progress, flush=True)
        pause_after_checkpoint(receipt_path)
        if completed == stop_after:
            raise SystemExit(75)
    handle.remove()
    summary = dict(status='COMPLETE', observations=len(selected), reused=reused, identity_sha256=signature,
        seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version, tf32=False),
        receipts=receipts, completed_at=now())
    atomic_write_json(root / 'summary.json', summary)
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=len(selected), total=len(selected)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    prepare(args.run, args.smoke, args.resume, args.stop_after, args.preflight)
