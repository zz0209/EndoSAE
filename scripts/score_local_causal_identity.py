import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from encode_token_causal_identity import ROOT, load_models, model_specs, shared, video_inputs
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import digest
from src.token_identity_sae import TokenIdentitySAE, symmetric_maxsim
from verify_token_identity_sae import numpy_forward


@torch.no_grad()
def vectors(model, mean, scale, raw, device):
    values = ((raw.astype(float) - mean) / scale).astype(np.float32)
    code = model.encode(torch.from_numpy(values).to(device))
    projected = model.readout(code)
    if not torch.isfinite(projected).all() or torch.any(projected.norm(dim=-1) <= 0):
        raise ValueError('Invalid local identity vectors')
    pooled = F.normalize(model.readout(code.mean(0)), dim=0)
    return F.normalize(projected, dim=-1), pooled.cpu().numpy().astype(float)


def score(run, smoke, resume, output):
    config = read_json(run / 'config.json')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    root = output or (run / 'smoke_encoding' if smoke else Path(config['embedding_root']))
    raw_root = run / 'smoke_raw' if smoke else Path(config['raw_root'])
    root.mkdir(parents=True, exist_ok=True)
    models = load_models(config, device)
    videos = [config['development_videos'][0], config['extension_videos'][0]] if smoke else config['development_videos'] + config['extension_videos']
    paths = [run / 'config.json', run / 'protocol.json', Path(__file__), ROOT / 'src/token_identity_sae.py',
             ROOT / 'scripts/encode_token_causal_identity.py', ROOT / 'scripts/verify_token_identity_sae.py']
    paths += [folder / name for _, folder in model_specs(config)
              for name in ['model.npz', 'model_config.json', 'normalization.npz']]
    code_identity = {str(p): digest(p) for p in paths}
    total = sum(read_json(raw_root / v / 'complete.json')['detections'] for v in videos)
    completed, results = 0, []
    for video in videos:
        _, reference, _, tracks, available, _, by_output, _ = video_inputs(config, video)
        raw_receipt = read_json(raw_root / video / 'complete.json')
        assert raw_receipt['status'] == 'COMPLETE'
        encoded = np.load(raw_root / video / 'encoded.npy')
        episodes = [e for group in by_output.values() for e in group]
        folder = root / video
        folder.mkdir(exist_ok=True)
        sources, source_files = {}, []
        for episode in episodes:
            identifier = episode['episode_id']
            source = Path(config['source_token_root']) / video / 'sources' / identifier
            position = int(tracks['offsets'][episode['click']['output_index']]) + episode['click']['detection_index']
            if not available[position]:
                continue
            source_files.extend([source / 'source.json', source / 'observed_tokens.npz'])
            info = read_json(source / 'source.json')
            assert info['position'] == position and not info['future_frames_used'] and not info['ground_truth_regions_used']
            assert (info['status'] == 'available') == bool(available[position])
            if available[position]:
                assert encoded[position]
                with np.load(source / 'observed_tokens.npz') as saved:
                    raw = saved['tokens'].copy()
                sources[identifier] = dict(position=position, raw=raw,
                    vectors={key: vectors(model, mean, scale, raw, device) for key, (model, mean, scale) in models.items()})
        identity = dict(files=code_identity, smoke=smoke, video=video,
            raw_receipt_sha256=digest(raw_root / video / 'complete.json'),
            sources={str(p): digest(p) for p in source_files})
        identity_path = folder / 'identity.json'
        if identity_path.exists():
            assert resume and read_json(identity_path) == identity
        else:
            atomic_write_json(identity_path, identity)
        if (folder / 'complete.json').exists():
            results.append(read_json(folder / 'complete.json'))
            completed += int(encoded.sum())
            continue
        np.save(folder / 'encoded.npy', encoded)
        progress_path = folder / 'progress.json'
        previous = read_json(progress_path) if progress_path.exists() else dict(shards=0, processed=0, seconds=0., checks=[], model_seconds={key: 0. for key in models})
        assert previous['shards'] == 0 or resume
        arrays = {}
        for episode in episodes:
            identifier = episode['episode_id']
            target = folder / 'sources' / identifier
            target.mkdir(parents=True, exist_ok=True)
            for key in models:
                method, seed = key.rsplit('_seed', 1)
                for mode in ['pooled_cosine', 'symmetric_maxsim']:
                    path = target / f'{method}_{mode}_seed{seed}.npy'
                    value = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float64, shape=available.shape)
                    if previous['shards'] == 0:
                        value[:] = np.nan
                    arrays[identifier, key, mode] = value
        checks, timings = previous['checks'], previous['model_seconds']
        original_vectors = {key: np.load(Path(config['source_token_root']) / video / (key.replace('pooled_', 'projected_', 1) + '.npy'), mmap_mode='r')
                            for key in models if key.startswith('pooled_')}
        begin, processed = time.perf_counter(), previous['processed']
        for shard_index in range(previous['shards'], len(raw_receipt['shards'])):
            shard = raw_receipt['shards'][shard_index]
            path = raw_root / video / shard['file']
            assert digest(path) == shard['sha256']
            with np.load(path) as saved:
                raw_values, offsets, indices = saved['tokens'].copy(), saved['offsets'].copy(), saved['detection_positions'].copy()
            assert np.all(encoded[indices]) and np.all(np.diff(indices) > 0)
            for i, position in enumerate(indices):
                raw = raw_values[offsets[i]:offsets[i + 1]]
                for key, (model, mean, scale) in models.items():
                    start = time.perf_counter()
                    local, pooled = vectors(model, mean, scale, raw, device)
                    if key in original_vectors:
                        np.testing.assert_allclose(pooled, original_vectors[key][position], atol=2e-6, rtol=0)
                    for identifier, source in sources.items():
                        source_local, source_pooled = source['vectors'][key]
                        arrays[identifier, key, 'pooled_cosine'][position] = np.clip(pooled @ source_pooled, -1., 1.)
                        arrays[identifier, key, 'symmetric_maxsim'][position] = np.clip(float(symmetric_maxsim(local, source_local)), -1., 1.)
                    timings[key] += time.perf_counter() - start
                    if smoke and not any(c['model'] == key for c in checks):
                        definition = read_json(dict(model_specs(config))[key] / 'model_config.json')
                        independent = TokenIdentitySAE(definition, model.method).double().eval()
                        independent.load_state_dict({k: v.detach().cpu().double() for k, v in model.state_dict().items()})
                        expected, _ = numpy_forward(independent, ((raw.astype(float) - mean) / scale)[None], 'symmetric_maxsim')
                        np.testing.assert_allclose(local.cpu().numpy(), expected[0], atol=2e-5, rtol=1e-4)
                        identifier, source = next(iter(sources.items()))
                        expected_source, _ = numpy_forward(independent, ((source['raw'].astype(float) - mean) / scale)[None], 'symmetric_maxsim')
                        matrix = expected[0] @ expected_source[0].T
                        target = .5 * (matrix.max(0).mean() + matrix.max(1).mean())
                        observed = arrays[identifier, key, 'symmetric_maxsim'][position]
                        np.testing.assert_allclose(observed, target, atol=2e-6, rtol=0)
                        checks.append(dict(model=key, actual_position=int(position), source=identifier,
                                           numpy_score_error=abs(float(observed) - target)))
                processed += 1
            for value in arrays.values():
                value.flush()
            atomic_write_json(progress_path, dict(shards=shard_index + 1, processed=processed,
                seconds=previous['seconds'] + time.perf_counter() - begin, checks=checks, model_seconds=timings))
            atomic_write_json(run / ('smoke_encoding_progress.json' if smoke else 'encoding_progress.json'),
                dict(status='RUNNING', completed=completed + processed, total=total, video=video, updated_at=shared.now()))
            print('LOCAL_CAUSAL_SCORES', video, processed, '/', int(encoded.sum()), 'total', completed + processed, '/', total, flush=True)
            pause_after_checkpoint(progress_path)
        assert processed == int(encoded.sum())
        for (identifier, key, mode), value in arrays.items():
            if identifier in sources:
                assert np.isfinite(value[encoded]).all() and np.isnan(value[~encoded]).all()
                assert np.isclose(value[sources[identifier]['position']], 1., atol=2e-5)
            else:
                assert np.isnan(value).all()
        result = dict(status='COMPLETE', video=video, smoke=smoke, detections=processed, checks=checks,
            model_seconds=timings, seconds=previous['seconds'] + time.perf_counter() - begin,
            completed_at=shared.now(), identity_sha256=digest(identity_path), peak_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_write_json(folder / 'complete.json', result)
        results.append(result)
        completed += processed
        del arrays, original_vectors
    assert code_identity == {str(p): digest(p) for p in paths}
    if output is None:
        atomic_write_json(run / ('smoke_encoding_summary.json' if smoke else 'encoding_summary.json'),
            dict(status='COMPLETE', smoke=smoke, videos=results, total_detections=total, models=list(models)))
    atomic_write_json(root / 'complete.json', dict(status='COMPLETE', videos=results, smoke=smoke))
    atomic_write_json(run / ('smoke_encoding_progress.json' if smoke else 'encoding_progress.json'),
                      dict(status='COMPLETE', completed=total, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    score(args.run, args.smoke, args.resume, args.output)
