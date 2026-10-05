import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from encode_token_causal_identity import ROOT, native_api, shared, targets_for, video_inputs
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest, tokens_to_frames


def cache(run, smoke, resume):
    config = read_json(run / 'config.json')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    root = run / 'smoke_raw' if smoke else Path(config['raw_root'])
    root.mkdir(parents=True, exist_ok=True)
    videos = [config['development_videos'][0], config['extension_videos'][0]] if smoke else config['development_videos'] + config['extension_videos']
    total = sum(len(targets_for(video_inputs(config, v)[5], video_inputs(config, v)[6], smoke)) for v in videos)
    source_paths = [Path(__file__), Path(native_api.__file__), ROOT / 'scripts/encode_rc27_cohort.py',
                    ROOT / 'scripts/encode_token_causal_identity.py', run / 'config.json', run / 'protocol.json']
    source_hashes = {str(p): digest(p) for p in source_paths}
    backbone, completed, results = None, 0, []
    started = time.perf_counter()
    for video in videos:
        definition, reference, records, tracks, available, targets, sources, paths = video_inputs(config, video)
        targets = targets_for(targets, sources, smoke)
        destination = root / video
        destination.mkdir(exist_ok=True)
        identity = dict(files=source_hashes, inputs={str(p): digest(p) for p in paths},
            targets=targets.tolist(), smoke=smoke, video=video, torch=str(torch.__version__), numpy=np.__version__)
        identity_path = destination / 'identity.json'
        if identity_path.exists():
            assert resume and read_json(identity_path) == identity
        else:
            atomic_write_json(identity_path, identity)
        if (destination / 'complete.json').exists():
            results.append(read_json(destination / 'complete.json'))
            completed += len(targets)
            continue
        if backbone is None:
            backbone = native_api.Encoder(read_json(definition['encoder_config']), device)
        prepare = native_api.Inputs(definition['frame_root'], video)
        original = np.load(reference / 'raw_mean.npy', mmap_mode='r')
        progress_path = destination / 'progress.json'
        previous = read_json(progress_path) if progress_path.exists() else dict(processed=0, shards=[], seconds=0., max_mean_error=0., source_checks=[])
        assert previous['processed'] == 0 or resume
        shards, source_checks = previous['shards'], previous['source_checks']
        max_error = previous['max_mean_error']
        status_path = destination / 'encoded.npy'
        encoded = np.lib.format.open_memmap(status_path, mode='r+' if status_path.exists() else 'w+', dtype=np.bool_, shape=available.shape)
        if previous['processed'] == 0:
            encoded[:] = False
        begin = time.perf_counter()
        for begin_index in range(previous['processed'], len(targets), config['checkpoint_outputs']):
            stop = min(begin_index + config['checkpoint_outputs'], len(targets))
            values, positions, detection_positions, offsets = [], [], [], [0]
            for output in targets[begin_index:stop]:
                output = int(output)
                native, _ = backbone(prepare(records, output))
                tokens = tokens_to_frames(native[None])
                first, last = tracks['offsets'][output:output + 2]
                for index in range(int(last - first)):
                    position = int(first) + index
                    mask, support, reason = native_api.support(records, tracks, output, index)
                    assert (reason == 'available') == bool(available[position])
                    assert all(s['output_index'] <= output for s in support)
                    if not available[position]:
                        continue
                    raw = tokens[mask].astype(np.float32)
                    error = float(np.max(np.abs(raw.mean(0, dtype=float) - original[position])))
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(raw.mean(0, dtype=float), original[position], atol=1e-5, rtol=0)
                    token_positions = np.column_stack(np.where(mask)).astype(np.int16)
                    for episode in sources.get(output, []):
                        if episode['click']['detection_index'] == index:
                            old = Path(config['source_token_root']) / video / 'sources' / episode['episode_id']
                            with np.load(old / 'observed_tokens.npz') as saved:
                                np.testing.assert_array_equal(raw, saved['tokens'])
                                np.testing.assert_array_equal(token_positions, saved['positions'])
                            source_checks.append(episode['episode_id'])
                    values.append(raw)
                    positions.append(token_positions)
                    detection_positions.append(position)
                    offsets.append(offsets[-1] + len(raw))
                    encoded[position] = True
            name = f'chunk_{begin_index:06d}_{stop:06d}.npz'
            shared.save_npz(destination / name, tokens=np.concatenate(values), token_positions=np.concatenate(positions),
                detection_positions=np.array(detection_positions), offsets=np.array(offsets), outputs=targets[begin_index:stop])
            encoded.flush()
            shards.append(dict(file=name, first=begin_index, stop=stop, detections=len(values), tokens=offsets[-1],
                               bytes=(destination / name).stat().st_size, sha256=digest(destination / name)))
            elapsed = previous['seconds'] + time.perf_counter() - begin
            atomic_write_json(progress_path, dict(processed=stop, total=len(targets), shards=shards,
                seconds=elapsed, max_mean_error=max_error, source_checks=source_checks))
            atomic_write_json(run / ('smoke_cache_progress.json' if smoke else 'cache_progress.json'),
                dict(status='RUNNING', completed=completed + stop, total=total, video=video,
                    tokens=sum(s['tokens'] for s in shards), seconds=time.perf_counter() - started, updated_at=shared.now()))
            print('CAUSAL_ROI_CACHE', video, stop, '/', len(targets), 'total', completed + stop, '/', total, flush=True)
            pause_after_checkpoint(progress_path)
        if not smoke:
            np.testing.assert_array_equal(encoded, available)
        expected_sources = [e['episode_id'] for group in sources.values() for e in group
                            if available[int(tracks['offsets'][e['click']['output_index']]) + e['click']['detection_index']]]
        assert sorted(source_checks) == sorted(expected_sources)
        result = dict(status='COMPLETE', video=video, smoke=smoke, native_outputs=len(targets),
            detections=int(encoded.sum()), max_mean_error=max_error, source_checks=source_checks, shards=shards,
            seconds=previous['seconds'] + time.perf_counter() - begin, completed_at=shared.now(),
            bytes=sum(s['bytes'] for s in shards), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            identity_sha256=digest(identity_path))
        atomic_write_json(destination / 'complete.json', result)
        results.append(result)
        completed += len(targets)
        del encoded, original
    assert source_hashes == {str(p): digest(p) for p in source_paths}
    atomic_write_json(run / ('smoke_cache_summary.json' if smoke else 'cache_summary.json'),
        dict(status='COMPLETE', smoke=smoke, videos=results, total_native_outputs=total,
            bytes=sum(r['bytes'] for r in results), seconds=sum(r['seconds'] for r in results)))
    atomic_write_json(run / ('smoke_cache_progress.json' if smoke else 'cache_progress.json'),
                      dict(status='COMPLETE', completed=total, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cache(args.run, args.smoke, args.resume)
