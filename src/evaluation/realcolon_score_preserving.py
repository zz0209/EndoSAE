"""A fixed task direction plus residual PCA: a simple compression control."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_fixed_confirmation import arrays, tokens_cuda, training_inputs, validate_assets
from src.evaluation.realcolon_task import digest, summarize, write_json


def constrained_basis(covariance, direction, width, torch):
    """Keep the task direction and maximize residual retained variance."""
    u = direction.double() / direction.double().norm()
    projector = torch.eye(len(u), device=u.device, dtype=u.dtype) - u[:, None] * u[None, :]
    residual = projector @ covariance.double() @ projector
    _, vectors = torch.linalg.eigh((residual + residual.T) / 2)
    rest = vectors[:, -(width - 1):].flip(1)
    rest = rest - u[:, None] * (u @ rest)[None, :]
    rest, _ = torch.linalg.qr(rest, mode='reduced')
    basis = torch.cat([u[:, None], rest], dim=1).float()
    error = (basis.T @ basis - torch.eye(width, device=u.device)).abs().max()
    if error > 1e-5:
        raise ValueError('Compression basis is not orthonormal')
    return basis


def probabilities(x, state, torch):
    result = []
    with torch.no_grad():
        for batch in x.split(4096):
            codes = batch @ state['basis']
            logits = codes[:, 0] * state['score_scale'] + state['score_intercept']
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def execute(run, config, torch):
    reference = Path(config['reference_run'])
    confirmation = Path(config['confirmation_run'])
    base = json.loads((confirmation / 'model_config.json').read_text())
    validate_assets(base)
    raw, masks, rows, source_indices = training_inputs(base, torch)
    norm = arrays(Path(base['normalization']), torch, 'cuda')
    x = (raw - norm['mean']) / norm['rms']
    del raw
    covariance = torch.zeros((768, 768), device='cuda', dtype=torch.float64)
    with torch.no_grad():
        for batch in x.split(4096):
            covariance += batch.double().T @ batch.double()
    covariance /= len(x)
    output = run / 'comparison'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(output / 'training_scope.json', {'all_rows': rows, 'source_clip_indices': source_indices.tolist()})
    models, train_predictions, checks = {}, {}, {}
    for seed in config['seeds']:
        seed_config = json.loads((reference / 'replication' / str(seed) / 'model_config.json').read_text())
        record = seed_config['models']['raw_balanced']
        head = arrays(Path(record['head']), torch, 'cuda')
        direction = head['weight'].squeeze(0).double() / head['feature_scale'].double()
        basis = constrained_basis(covariance, direction, config['width'], torch)
        state = {'basis': basis, 'score_scale': direction.norm().float(),
                 'score_intercept': (head['bias'].squeeze().double() - direction @ head['feature_mean'].double()).float()}
        destination = output / str(seed)
        destination.mkdir()
        np.savez(destination / 'projection.npz', **{k: v.cpu().numpy() for k, v in state.items()})
        predictions = probabilities(x, state, torch).reshape(masks.shape)
        original = np.load(record['training_predictions'], allow_pickle=False)
        error = float(np.max(np.abs(predictions - original)))
        if error > config['score_tolerance']:
            raise ValueError('Training raw-score preservation failed')
        np.save(destination / 'training_predictions.npy', predictions, allow_pickle=False)
        checks[str(seed)] = {'train_max_probability_error': error}
        train_predictions[seed], models[seed] = predictions, state
    write_json(output / 'models_frozen.json', {'status': 'ALL_PROJECTIONS_FITTED_TRAIN_ONLY',
        'sha256': {str(output / str(s) / 'projection.npz'): digest(output / str(s) / 'projection.npz') for s in config['seeds']}})
    del x, covariance
    new_rows = [json.loads(line) for line in (confirmation / 'clip_manifest.jsonl').read_text().splitlines()]
    if {r['video_id'] for r in new_rows} & {r['video_id'] for r in rows}:
        raise ValueError('Train/evaluation overlap')
    cache = Path(base['confirmation_cache'])
    for name, identity in config['confirmation_cache_identity'].items():
        stat = (cache / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity['bytes'], identity['mtime_ns']):
            raise ValueError('Confirmation cache changed')
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(new_rows))), torch)
    x = (raw - norm['mean']) / norm['rms']
    del raw
    new_masks = np.load(cache / 'masks.npy', allow_pickle=False)
    combined_rows, combined_masks = rows + new_rows, np.concatenate([masks, new_masks])
    write_json(output / 'fit_scope.json', {'all_rows': combined_rows, 'source_clip_indices': source_indices.tolist()})
    records, fidelity = [], {}
    for seed, state in models.items():
        path = output / str(seed)
        predictions = np.concatenate([train_predictions[seed], probabilities(x, state, torch).reshape(new_masks.shape)])
        original = np.load(reference / 'replication' / str(seed) / 'raw_balanced/predictions.npy', allow_pickle=False)
        error = float(np.max(np.abs(predictions - original)))
        if error > config['score_tolerance']:
            raise ValueError('Evaluation raw-score preservation failed')
        checks[str(seed)]['all_max_probability_error'] = error
        np.save(path / 'predictions.npy', predictions, allow_pickle=False)
        write_json(path / 'metrics.json', summarize(predictions, combined_masks, combined_rows))
        totals = {v: {'sse': 0., 'denominator': 0.} for v in base['confirmation_videos']}
        with torch.no_grad():
            for row, token in zip(new_rows, x.reshape(len(new_rows), 8 * 196, 768)):
                difference = (token @ state['basis']) @ state['basis'].T - token
                totals[row['video_id']]['sse'] += float(difference.double().square().sum())
                totals[row['video_id']]['denominator'] += float(token.double().square().sum())
        fidelity[str(seed)] = {'raw_sums': totals, 'per_video_nmse': {v: n['sse'] / n['denominator'] for v, n in totals.items()}}
        records.append({'seed': seed, 'method': 'score_pca48', 'path': path.as_posix()})
        print(json.dumps({'seed': seed, 'score_error': error}), flush=True)
    write_json(output / 'summary.json', {'records': records, 'score_preservation': checks})
    write_json(output / 'fidelity.json', fidelity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen control input changed: ' + path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    write_json(run / 'status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        execute(run, config, torch)
    except Exception as error:
        write_json(run / 'status.json', {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(run / 'status.json', {'status': 'COMPLETED', 'elapsed_seconds': time.time() - start,
               'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
