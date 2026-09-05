"""Training-only PCA controls for the frozen REAL-Colon representation study."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, summarize, write_json
from src.evaluation.realcolon_fixed_confirmation import tokens_cuda
from src.evaluation.realcolon_task_regularization import validate_scope
from src.evaluation.realcolon_task_supervised import SupportSampler, feature_stats, fit_one, fold_masks


def execute(run, config):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for path, expected in config['asset_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen source/fold asset changed: ' + path)
    reference = Path(config['reference_run'])
    task = json.loads((reference / 'config.json').read_text())
    source = Path(task['source_run'])
    all_rows = [json.loads(line) for line in (source / 'clip_manifest.jsonl').read_text().splitlines()]
    selected = [i for i, row in enumerate(all_rows) if row['split'] == 'train']
    rows = [all_rows[i] for i in selected]
    cache = Path(task['cache_dir'])
    for name, expected in task['cache_file_identity'].items():
        stat = (cache / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise ValueError('Previously verified training cache changed')
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    if tokens.shape != tuple(task['source_token_shape']):
        raise ValueError('Training cache shape mismatch')
    raw = tokens_cuda(tokens, selected, torch)
    masks = np.load(cache / 'masks.npy', mmap_mode='r', allow_pickle=False)[selected]
    output = run / 'cv'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(output / 'environment.json', {'torch': torch.__version__, 'numpy': np.__version__})
    records = []
    for index, fit_clips in enumerate(fold_masks(rows, task['heldout_folds'])):
        previous = reference / 'cv' / ('fold' + str(index))
        scope = json.loads((previous / 'fit_scope.json').read_text())
        validate_scope(scope, rows, fit_clips)
        with np.load(previous / 'input_normalization_pca.npz', allow_pickle=False) as state:
            mean, rms, basis = [torch.tensor(state[key], device='cuda') for key in ('mean', 'rms', 'basis')]
        x = (raw - mean) / rms
        fit_indices = torch.tensor(np.flatnonzero(np.repeat(fit_clips, 8 * 196)), device='cuda')
        sampler = SupportSampler(rows, masks, fit_clips, raw.device, torch)
        fold = output / previous.name
        fold.mkdir()
        write_json(fold / 'fit_scope.json', scope)
        for dimension in config['pca_dimensions']:
            projection = basis[:, :dimension]
            stats = {'pca': feature_stats(x, fit_indices, lambda v: v @ projection, torch)}
            method = 'pca' + str(dimension) + '_balanced'
            for candidate, weight_decay in enumerate(config['weight_decay_candidates']):
                destination = fold / (method + '_candidate' + str(candidate))
                destination.mkdir()
                hyper = {'head_weight_decay': weight_decay}
                predictions = fit_one('pca64_balanced', hyper, x, fit_indices, sampler,
                                      {'pca': projection}, stats, task, destination, torch).reshape(masks.shape)
                training = json.loads((destination / 'training.json').read_text())
                training.update(method=method, projection_dimension=dimension,
                                backend='existing generic PCA head; explicit basis width')
                write_json(destination / 'training.json', training)
                np.save(destination / 'predictions.npy', predictions, allow_pickle=False)
                metrics = summarize(predictions, masks, scope['all_rows'])
                write_json(destination / 'metrics.json', metrics)
                held = {v: metrics['per_video'][v] for v in scope['validation_videos']}
                records.append({'method': method, 'hyperparameters': hyper,
                                'path': destination.as_posix(), 'validation': held})
                print(json.dumps({'fold': index, 'method': method, 'wd': weight_decay,
                                  'auroc': {v: m['clip_detection']['auroc'] for v, m in held.items()}}), flush=True)
        del x, sampler, stats, basis, mean, rms
    aggregate = {}
    for dimension in config['pca_dimensions']:
        method = 'pca' + str(dimension) + '_balanced'
        aggregate[method] = []
        for wd in config['weight_decay_candidates']:
            hyper = {'head_weight_decay': wd}
            videos = [v for r in records if r['method'] == method and r['hyperparameters'] == hyper
                      for v in r['validation'].values()]
            if len(videos) != len({r['video_id'] for r in rows}):
                raise ValueError('Incomplete held-out training coverage')
            aggregate[method].append({'hyperparameters': hyper,
                'macro_video_auroc': float(np.mean([v['clip_detection']['auroc'] for v in videos])),
                'macro_positive_frame_patch_ap': float(np.mean([v['positive_frame_mean_patch_ap'] for v in videos]))})
    write_json(output / 'summary.json', {'status': 'COMPLETED_CV', 'records': records,
        'candidates': aggregate,
        'selected': {m: max(items, key=lambda x: x['macro_video_auroc']) for m, items in aggregate.items()},
        'elapsed_seconds': time.time() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'Training-video-only selection; no development/confirmation input materialized'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    write_json(run / 'status.json', {'status': 'RUNNING', 'phase': 'pca_cv', 'pid': os.getpid()})
    try:
        execute(run, config)
    except Exception as error:
        write_json(run / 'status.json', {'status': 'FAILED', 'phase': 'pca_cv', 'error': repr(error)})
        raise
    write_json(run / 'status.json', {'status': 'COMPLETED', 'phase': 'pca_cv'})


if __name__ == '__main__':
    main()
