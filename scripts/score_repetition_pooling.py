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


@torch.no_grad()
def pool(model, mean, scale, raw, device, verify=False):
    values = ((raw.astype(float) - mean) / scale).astype(np.float32)
    code = model.encode(torch.from_numpy(values).to(device))
    assert model.identity_space == 'code'
    assert torch.isfinite(code).all() and torch.all(code.norm(dim=1) > 0)
    unit = F.normalize(code.double(), dim=1)
    count, dimension = unit.shape
    # GMP ridge is fixed to one for a unit-diagonal Gram matrix.
    if count <= dimension:
        gram = unit @ unit.T
        weights = torch.linalg.solve(gram + torch.eye(count, device=device, dtype=unit.dtype),
                                     torch.ones(count, device=device, dtype=unit.dtype))
        aggregate = unit.T @ weights
    else:
        aggregate = torch.linalg.solve(unit.T @ unit + torch.eye(dimension, device=device, dtype=unit.dtype),
                                       unit.sum(0))
        weights = 1. - unit @ aggregate
    unit_mean = unit.mean(0)
    response_mean, response_gmp = unit @ F.normalize(unit_mean, dim=0), unit @ F.normalize(aggregate, dim=0)
    result = dict(mean=F.normalize(code.mean(0), dim=0), unit_mean=F.normalize(unit_mean, dim=0),
                  gmp=F.normalize(aggregate, dim=0))
    result = {key: value.cpu().numpy().astype(float) for key, value in result.items()}
    assert all(np.isfinite(v).all() and np.isclose(np.linalg.norm(v), 1., atol=2e-6) for v in result.values())
    residual = float((aggregate - unit.T @ (1. - unit @ aggregate)).abs().max())
    assert residual < 1e-7
    stats = dict(tokens=count, negative_weight_fraction=float((weights < 0).double().mean()),
                 mean_response_cv=float(response_mean.std(correction=0) / response_mean.mean().abs()),
                 gmp_response_cv=float(response_gmp.std(correction=0) / response_gmp.mean().abs()),
                 mean_gmp_cosine=float(result['mean'] @ result['gmp']), equation_residual=residual)
    if verify:
        local = code.cpu().numpy().astype(float)
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        # Independent augmented least squares verifies both normal-equation branches.
        design = np.concatenate([local, np.eye(dimension)], axis=0)
        response = np.concatenate([np.ones(count), np.zeros(dimension)])
        expected = np.linalg.lstsq(design, response, rcond=None)[0]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result['gmp'], expected, atol=2e-8, rtol=2e-6)
        reversed_unit = unit.flip(0)
        reversed_gmp = torch.linalg.solve(reversed_unit.T @ reversed_unit + torch.eye(dimension, device=device, dtype=unit.dtype),
                                          reversed_unit.sum(0))
        np.testing.assert_allclose(F.normalize(reversed_gmp, dim=0).cpu().numpy(), result['gmp'], atol=2e-8, rtol=2e-6)
        stats['independent_max_error'] = float(np.abs(expected - result['gmp']).max())
    return result, stats


def score(run, smoke, resume):
    config = read_json(run / 'config.json')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    root = run / 'smoke_encoding' if smoke else Path(config['embedding_root'])
    raw_root = Path(config['smoke_raw_root'] if smoke else config['raw_root'])
    videos = [config['development_videos'][0], config['extension_videos'][0]] if smoke else config['development_videos'] + config['extension_videos']
    models = load_models(config, device)
    paths = [run / 'config.json', run / 'protocol.json', Path(__file__), ROOT / 'scripts/encode_token_causal_identity.py',
             ROOT / 'src/token_identity_sae.py']
    paths += [folder / name for _, folder in model_specs(config) for name in ('model.npz', 'model_config.json', 'normalization.npz')]
    identities = {str(p): digest(p) for p in paths}
    root.mkdir(parents=True, exist_ok=True)
    total = sum(read_json(raw_root / v / 'complete.json')['detections'] for v in videos)
    completed, summaries = 0, []
    for video in videos:
        _, _, _, tracks, available, _, by_output, input_files = video_inputs(config, video)
        receipt = read_json(raw_root / video / 'complete.json')
        assert receipt['status'] == 'COMPLETE'
        encoded = np.load(raw_root / video / 'encoded.npy', allow_pickle=False)
        episodes = [e for group in by_output.values() for e in group]
        folder = root / video
        folder.mkdir(exist_ok=True)
        source_files, sources, source_stats = [], {}, []
        for episode in episodes:
            identifier = episode['episode_id']
            source = Path(config['source_token_root']) / video / 'sources' / identifier
            info = read_json(source / 'source.json')
            source_files += [source / 'source.json', source / 'observed_tokens.npz']
            position = int(tracks['offsets'][episode['click']['output_index']] + episode['click']['detection_index'])
            assert position == info['position'] and not info['future_frames_used'] and not info['ground_truth_regions_used']
            assert available[position] and encoded[position]
            with np.load(source / 'observed_tokens.npz', allow_pickle=False) as data:
                raw = data['tokens'].copy()
            representations = {}
            for key, (model, mean, scale) in models.items():
                representations[key], stats = pool(model, mean, scale, raw, device, verify=smoke)
                source_stats.append(dict(source=identifier, model=key, **stats))
            sources[identifier] = dict(position=position, vectors=representations)
        identity = dict(files=identities, smoke=smoke, video=video,
                        inputs={str(p): digest(p) for p in input_files + source_files + [raw_root / video / 'complete.json']},
                        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, device=str(device)))
        if (folder / 'identity.json').exists():
            assert resume and read_json(folder / 'identity.json') == identity
        else:
            atomic_write_json(folder / 'identity.json', identity)
        if (folder / 'complete.json').exists():
            summaries.append(read_json(folder / 'complete.json'))
            completed += int(encoded.sum())
            continue
        np.save(folder / 'encoded.npy', encoded)
        atomic_write_json(folder / 'source_diagnostics.json', dict(rows=source_stats))
        progress_path = folder / 'progress.json'
        previous = read_json(progress_path) if progress_path.exists() else dict(shards=0, processed=0, seconds=0., checks=[])
        assert previous['shards'] == 0 or resume
        arrays, original = {}, {}
        for identifier in sources:
            target = folder / 'sources' / identifier
            target.mkdir(parents=True, exist_ok=True)
            for key in models:
                method, seed = key.rsplit('_seed', 1)
                original[identifier, key] = np.load(Path(config['reference_scores']) / video / 'sources' / identifier /
                                                    f'{method}_pooled_cosine_seed{seed}.npy', mmap_mode='r')
                for mode in ('mean', 'unit_mean', 'gmp'):
                    path = target / f'{method}_{mode}_seed{seed}.npy'
                    value = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float64, shape=available.shape)
                    if previous['shards'] == 0:
                        value[:] = np.nan
                    arrays[identifier, key, mode] = value
        start, processed, checks = time.perf_counter(), previous['processed'], previous['checks']
        for shard_index in range(previous['shards'], len(receipt['shards'])):
            shard = receipt['shards'][shard_index]
            path = raw_root / video / shard['file']
            assert digest(path) == shard['sha256']
            with np.load(path, allow_pickle=False) as data:
                raw_values, offsets, positions = data['tokens'].copy(), data['offsets'].copy(), data['detection_positions'].copy()
            assert np.all(encoded[positions])
            statistics = []
            for number, position in enumerate(positions):
                raw = raw_values[offsets[number]:offsets[number + 1]]
                for key, (model, mean, scale) in models.items():
                    need_check = smoke and not any(c['model'] == key for c in checks)
                    vectors, stats = pool(model, mean, scale, raw, device, verify=need_check)
                    statistics.append(dict(position=int(position), model=key, **stats))
                    for identifier, source in sources.items():
                        for mode, vector in vectors.items():
                            arrays[identifier, key, mode][position] = np.clip(vector @ source['vectors'][key][mode], -1., 1.)
                        np.testing.assert_allclose(arrays[identifier, key, 'mean'][position], original[identifier, key][position], atol=2e-6, rtol=0)
                    if need_check:
                        checks.append(dict(model=key, position=int(position), **stats))
                processed += 1
            for array in arrays.values():
                array.flush()
            atomic_write_json(folder / f'diagnostics_{shard_index:05d}.json', dict(rows=statistics))
            atomic_write_json(progress_path, dict(shards=shard_index + 1, processed=processed, checks=checks,
                                                  seconds=previous['seconds'] + time.perf_counter() - start))
            atomic_write_json(run / ('smoke_encoding_progress.json' if smoke else 'encoding_progress.json'),
                              dict(status='RUNNING', completed=completed + processed, total=total, video=video, updated_at=shared.now()))
            print('REPETITION_POOLING', video, processed, '/', int(encoded.sum()), 'total', completed + processed, '/', total, flush=True)
            pause_after_checkpoint(progress_path)
        assert processed == int(encoded.sum())
        for (identifier, key, mode), value in arrays.items():
            assert np.isfinite(value[encoded]).all() and np.isnan(value[~encoded]).all()
            assert np.isclose(value[sources[identifier]['position']], 1., atol=2e-5)
        result = dict(status='COMPLETE', video=video, smoke=smoke, detections=processed, checks=checks,
                      seconds=previous['seconds'] + time.perf_counter() - start, completed_at=shared.now(),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(), identity_sha256=digest(folder / 'identity.json'))
        atomic_write_json(folder / 'complete.json', result)
        summaries.append(result)
        completed += processed
        del arrays, original
    assert identities == {str(p): digest(p) for p in paths}
    atomic_write_json(root / 'complete.json', dict(status='COMPLETE', videos=summaries, smoke=smoke))
    atomic_write_json(run / ('smoke_encoding_progress.json' if smoke else 'encoding_progress.json'),
                      dict(status='COMPLETE', completed=total, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    score(args.run, args.smoke, args.resume)
