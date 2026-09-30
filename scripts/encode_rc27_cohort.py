import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from run_continuous_confirmation import prepare_input, read, rows, extract_task_manifest
from train_nested_dictionary import put, sha
from verify_tail_runtime import FrozenTail
import run_model_port_parity as parity


ROOT = Path(__file__).resolve().parents[1]
ARRAY_SHAPES = {'block10': (1569, 768), 'final_features': (8, 196, 768), 'masks': (8, 196)}
ARRAY_DTYPES = {'block10': np.float32, 'final_features': np.float32, 'masks': np.bool_}


def load_video(run, video):
    metadata = run / 'metadata' / (video + '.json')
    selected = read(metadata)['clips']
    assert selected and all(r['video_id'] == video and r['split'] in ['train', 'val'] for r in selected)
    assert len({r['clip_id'] for r in selected}) == len(selected)
    assert all(len(r['frames']) == 8 for r in selected)
    return selected, metadata


def video_list(run, selected):
    if selected is not None:
        return [selected]
    manifest = rows(run / 'metadata/clip_manifest.jsonl')
    assert all(r['split'] in ['train', 'val'] for r in manifest)
    return sorted({r['video_id'] for r in manifest})


def extraction_request(selected, path):
    unique = {}
    for clip in selected:
        for frame in clip['frames']:
            index = frame['frame_index']
            if index in unique:
                assert unique[index] == frame
            unique[index] = frame
    request = dict(video_id=selected[0]['video_id'], kind='unique_frame_extraction_request',
                   frames=[unique[index] for index in sorted(unique)])
    content = json.dumps(request) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == content
    else:
        path.write_text(content, encoding='utf-8')
    return len(unique)


def extract_video(run, config, video):
    selected, metadata = load_video(run, video)
    destination = Path(config['cache_root']) / 'extraction' / video
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'requested_frames.jsonl'
    count = extraction_request(selected, manifest)
    print('EXTRACT_VIDEO', video, len(selected), 'clips', count, 'unique frames', flush=True)
    extract_task_manifest(manifest, Path(config['frame_archive_directory']), Path(config['frame_root']), destination)
    return dict(video_id=video, unique_frames=count, clips=len(selected), metadata_sha256=sha(metadata),
                extraction_receipt=str(destination / ('extracted_' + video + '.json')))


class Encoder:
    def __init__(self, config, device):
        self.device = device
        source = read(config['source_config'])
        assert parity.sha256_file(source['state_exchange']) == source['state_exchange_sha256']
        module = parity.load_timesformer(str(ROOT / 'third_party/Endo-FM/models'))
        self.model = parity.build_model(torch, module)
        self.model.load_state_dict(parity.load_state_exchange(source['state_exchange'], torch, np), strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.captured = {}
        self.handle = self.model.blocks[10].register_forward_hook(
            lambda _module, _inputs, value: self.captured.update(block10=value.detach()))
        self.state_sha256 = source['state_exchange_sha256']
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

    def __call__(self, inputs):
        with torch.no_grad():
            final = self.model.forward_features(torch.from_numpy(inputs[None]).to(self.device), get_all=True)
        native = self.captured['block10'][0].cpu().numpy().copy()
        features = final[0, 1:].reshape(196, 8, 768).permute(1, 0, 2).cpu().numpy().copy()
        assert native.shape == ARRAY_SHAPES['block10'] and features.shape == ARRAY_SHAPES['final_features']
        assert np.isfinite(native).all() and np.isfinite(features).all()
        return native, features


def encode_video(run, config, video, encoder):
    selected, metadata = load_video(run, video)
    destination = Path(config['cache_root']) / 'videos' / video
    destination.mkdir(parents=True, exist_ok=True)
    identity = dict(video_id=video, split=selected[0]['split'], clips=len(selected), metadata_sha256=sha(metadata),
                    backbone_sha256=encoder.state_sha256, clip_ids=[r['clip_id'] for r in selected],
                    source_sha256=sha(__file__), python_version=platform.python_version(),
                    dtype='float32', execution_device=str(encoder.device), torch_version=str(torch.__version__), tf32=False,
                    block10_layout='clip,CLS_plus_spatial_major_temporal_minor,768',
                    final_layout='clip,time,spatial,768', final_normalization='Original backbone final LayerNorm; no dataset standardization',
                    preprocessing='Existing prepare_input: RGB BICUBIC280x224 center224, ImageNet mean/std')
    identity_path = destination / 'identity.json'
    if identity_path.exists():
        assert read(identity_path) == identity
    else:
        put(identity_path, identity)
        shutil.copyfile(metadata, destination / 'metadata.json')
    complete = destination / 'complete.json'
    if complete.exists():
        record = read(complete)
        assert record['identity_sha256'] == sha(identity_path)
        assert all((destination / (name + '.npy')).stat().st_size == size for name, size in record['array_file_bytes'].items())
        print('ENCODE_REUSE', video, len(selected), 'clips', flush=True)
        return record
    progress_path = destination / 'progress.json'
    start = read(progress_path)['completed'] if progress_path.exists() else 0
    arrays = {}
    for name, shape in ARRAY_SHAPES.items():
        path = destination / (name + '.npy')
        if path.exists():
            arrays[name] = np.lib.format.open_memmap(path, mode='r+')
            assert arrays[name].shape == (len(selected), *shape)
            assert arrays[name].dtype == np.dtype(ARRAY_DTYPES[name])
        else:
            assert start == 0
            arrays[name] = np.lib.format.open_memmap(path, mode='w+', dtype=ARRAY_DTYPES[name], shape=(len(selected), *shape))
    started = time.perf_counter()
    for index in range(start, len(selected)):
        inputs, masks = prepare_input(selected[index], Path(config['frame_root']))
        native, final = encoder(inputs)
        arrays['block10'][index], arrays['final_features'][index], arrays['masks'][index] = native, final, masks
        if (index + 1) % config['checkpoint_clips'] == 0 or index + 1 == len(selected):
            for value in arrays.values():
                value.flush()
            progress = dict(video_id=video, completed=index + 1, total=len(selected),
                            seconds_this_invocation=time.perf_counter() - started)
            put(progress_path, progress)
            print('ENCODE_PROGRESS', json.dumps(progress), flush=True)
    frame_indices = np.array([[f['frame_index'] for f in row['frames']] for row in selected], dtype=np.int64)
    np.save(destination / 'frame_indices.npy', frame_indices)
    record = dict(status='COMPLETE', video_id=video, split=selected[0]['split'], clips=len(selected),
                  identity_sha256=sha(identity_path), cache_directory=str(destination),
                  source_sha256=identity['source_sha256'],
                  runtime=dict(python=platform.python_version(), torch=str(torch.__version__), device=str(encoder.device)),
                  array_file_bytes={name: (destination / (name + '.npy')).stat().st_size for name in ARRAY_SHAPES},
                  seconds_this_invocation=time.perf_counter() - started)
    put(complete, record)
    return record


def estimate(run, config, videos):
    counts = {'train': 0, 'val': 0}
    video_records = []
    for video in videos:
        selected, _ = load_video(run, video)
        counts[selected[0]['split']] += len(selected)
        unique = {f['frame_index'] for row in selected for f in row['frames']}
        archive = Path(config['frame_archive_directory']) / (video + '_frames.tar.gz')
        video_records.append(dict(video_id=video, split=selected[0]['split'], clips=len(selected),
             unique_frames=len(unique), archive_available=archive.is_file(), archive_bytes=archive.stat().st_size if archive.is_file() else None))
    bytes_per_clip = {name: int(np.prod(shape) * np.dtype(ARRAY_DTYPES[name]).itemsize) for name, shape in ARRAY_SHAPES.items()}
    total = sum(counts.values())
    result = dict(clips=counts, total_clips=total, bytes_per_clip=bytes_per_clip,
        total_array_payload_bytes=total * sum(bytes_per_clip.values()),
        largest_video_array_payload_bytes=max(r['clips'] for r in video_records) * sum(bytes_per_clip.values()),
        unique_jpeg_count=sum(r['unique_frames'] for r in video_records), videos=video_records,
        working_memory='Single clip input4.82MB plus block10/final9.64MB; arrays are memory mapped per video. Backbone parameters and transient inference activations are additional and measured in smoke.',
        storage_free_bytes=shutil.disk_usage(Path(config['cache_root']).anchor).free)
    return result


def smoke(run, config, video, device, output):
    output.mkdir(parents=True, exist_ok=False)
    selected, metadata = load_video(run, video)
    assert selected[0]['split'] == 'train'
    source_run = Path(config['smoke_source_run'])
    previous = rows(source_run / 'clip_manifest.jsonl')
    previous_lookup = {r['clip_id']: i for i, r in enumerate(previous)}
    row = next(r for r in selected if r['clip_id'] in previous_lookup)
    index = previous_lookup[row['clip_id']]
    source_cache = Path(read(config['source_config'])['training_cache'])
    frame_root = Path(read(source_run / 'extraction_summary.json')['output_dir'])
    inputs, masks = prepare_input(row, frame_root)
    assert np.array_equal(inputs, np.load(source_cache / 'inputs.npy', mmap_mode='r')[index])
    assert np.array_equal(masks, np.load(source_cache / 'masks.npy', mmap_mode='r')[index] > 0)
    request = output / 'requested_frames.jsonl'
    unique = extraction_request([row, row], request)
    assert unique == 8
    started = time.perf_counter()
    encoder = Encoder(config, device)
    native, final = encoder(inputs)
    source = read(config['source_config'])
    tail = FrozenTail(source['state_exchange'], device)
    with torch.no_grad():
        replay = tail(torch.from_numpy(native[None]).to(device))[0, 1:].reshape(196, 8, 768).permute(1, 0, 2).cpu().numpy()
    maximum = float(np.max(np.abs(replay - final)))
    assert maximum < 1e-5
    np.savez(output / 'actual_clip.npz', block10=native, final_features=final, masks=masks,
             frame_indices=np.array([f['frame_index'] for f in row['frames']]))
    process_peak = int(subprocess.check_output(['pwsh', '-NoProfile', '-Command', f'(Get-Process -Id {os.getpid()}).PeakWorkingSet64'], text=True).strip())
    result = dict(status='PASS_SMOKE', video_id=video, clip_id=row['clip_id'],
        input_exact=True, mask_exact=True, duplicate_frame_request_unique=unique,
        tail_replay_maximum_error=maximum, parameter_bytes=encoder.parameter_bytes,
        process_peak_working_set_bytes=process_peak,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        seconds=time.perf_counter() - started, metadata_sha256=sha(metadata), source_sha256=sha(__file__),
        environment=dict(python=platform.python_version(), torch=torch.__version__, device=str(device)))
    put(output / 'summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['estimate', 'extract', 'encode', 'all', 'smoke'])
    parser.add_argument('--video')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    config = read(args.run / 'encoding_config.json')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.phase == 'smoke':
        assert args.video is not None and args.output is not None
        smoke(args.run, config, args.video, device, args.output)
        return
    videos = video_list(args.run, args.video)
    if args.phase == 'estimate':
        result = estimate(args.run, config, videos)
        put(args.run / 'encoding_estimate.json', result)
        print(json.dumps(result), flush=True)
        return
    encoder = Encoder(config, device) if args.phase in ['encode', 'all'] else None
    completed, pending = [], []
    for video in videos:
        archive = Path(config['frame_archive_directory']) / (video + '_frames.tar.gz')
        extraction = Path(config['cache_root']) / 'extraction' / video / ('extracted_' + video + '.json')
        if args.available_only and ((args.phase in ['extract', 'all'] and not archive.is_file()) or
                                    (args.phase == 'encode' and not extraction.exists())):
            pending.append(video)
            print('PENDING', video, args.phase, flush=True)
            continue
        if args.phase in ['extract', 'all']:
            result = extract_video(args.run, config, video)
        if args.phase in ['encode', 'all']:
            assert extraction.exists()
            result = encode_video(args.run, config, video, encoder)
        completed.append(result)
        put(args.run / ('encoding_' + args.phase + '_progress.json'), dict(completed=completed, pending=pending, phase=args.phase))
    put(args.run / ('encoding_' + args.phase + '_summary.json'), dict(
        status='PARTIAL_WAITING_FOR_INPUTS' if pending else 'COMPLETE', completed=completed, pending=pending,
        source_sha256=sha(__file__), peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None))


if __name__ == '__main__':
    main()
