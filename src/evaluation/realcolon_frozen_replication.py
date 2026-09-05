"""Fit the predeclared seed replications, then evaluate all frozen models."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_fixed_confirmation import arrays, predict, tokens_cuda, training_inputs, validate_assets
from src.evaluation.realcolon_task import digest, summarize, write_json
from src.evaluation.realcolon_task_cv import train_dictionary
from src.evaluation.realcolon_task_supervised import SupportSampler, feature_stats, fit_one, save_state


def fit_models(run, plan, base, torch):
    output = run / 'replication'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    raw, masks, rows, original_indices = training_inputs(base, torch)
    norm = arrays(Path(base['normalization']), torch, 'cuda')
    x = (raw - norm['mean']) / norm['rms']
    indices = torch.arange(len(x), device='cuda')
    sampler = SupportSampler(rows, masks, np.ones(len(rows), dtype=bool), 'cuda', torch)
    scope = {'fit_videos': sampler.videos, 'all_rows': rows, 'source_clip_indices': original_indices.tolist()}
    write_json(output / 'training_scope.json', scope)
    task = json.loads(Path(base['task_config']).read_text())
    if task['seed'] != plan['seeds'][0]:
        raise ValueError('Reused seed no longer matches the reference model')
    common_stats = {'raw': feature_stats(x, indices, lambda batch: batch, torch)}
    for dimension in (32, 48):
        basis = norm['basis'][:, :dimension]
        common_stats[dimension] = feature_stats(x, indices, lambda batch: batch @ basis, torch)
    seed_configs = []
    for seed in plan['seeds']:
        directory = output / str(seed)
        directory.mkdir()
        config = copy.deepcopy(base)
        config['models'] = {}
        active = dict(task, seed=seed)
        original = seed == plan['seeds'][0]
        model = None
        if original:
            dictionary_path = Path(base['models']['topk_frozen_balanced']['dictionary'])
        else:
            model, history = train_dictionary(x, active, torch)
            dictionary_path = directory / 'topk_initial.npz'
            save_state(dictionary_path, model)
            write_json(directory / 'pretraining.json', {'seed': seed, 'history': history})
            topk_stats = feature_stats(x, indices, model.encode_inference, torch)
        for method in plan['methods']:
            destination = directory / method
            destination.mkdir()
            record = copy.deepcopy(base['models'][method])
            if original and method in ('raw_balanced', 'topk_frozen_balanced'):
                predictions = np.load(record['training_predictions'], mmap_mode='r', allow_pickle=False)[original_indices]
                write_json(destination / 'reuse.json', {'source_head': record['head'], 'source_predictions': record['training_predictions'],
                           'reason': 'Same fixed seed and WD; preserve the validated model without refitting'})
            else:
                backend = method
                if method.startswith('pca'):
                    dimension = record['dimension']
                    references = {'pca': norm['basis'][:, :dimension]}
                    stats = {'pca': common_stats[dimension]}
                    backend = 'pca64_balanced'
                elif method == 'raw_balanced':
                    references, stats = {}, {'raw': common_stats['raw']}
                else:
                    references, stats = {'topk': model}, {'topk': topk_stats}
                predictions = fit_one(backend, {'head_weight_decay': plan['head_weight_decay'][method]},
                                      x, indices, sampler, references, stats, active, destination, torch).reshape(masks.shape)
                training = json.loads((destination / 'training.json').read_text())
                training.update(method=method, seed=seed, projection_dimension=record.get('dimension'))
                write_json(destination / 'training.json', training)
                record['head'] = (destination / 'head.npz').as_posix()
            record['training_predictions'] = (destination / 'training_predictions.npy').as_posix()
            np.save(record['training_predictions'], predictions, allow_pickle=False)
            if method == 'topk_frozen_balanced':
                record['dictionary'] = dictionary_path.as_posix()
            config['models'][method] = record
            reproduced = predict(raw, method, config, torch).reshape(masks.shape)
            error = float(np.max(np.abs(reproduced - predictions)))
            if error > 1e-6:
                raise ValueError('Training/inference consumer mismatch: ' + method)
            write_json(destination / 'training_replay.json', {'maximum_absolute_error': error,
                       'n_values': predictions.size, 'exact': bool(np.array_equal(reproduced, predictions))})
            print(json.dumps({'phase': 'fit', 'seed': seed, 'method': method,
                              'reused': original and method in ('raw_balanced', 'topk_frozen_balanced')}), flush=True)
        write_json(directory / 'model_config.json', config)
        seed_configs.append({'seed': seed, 'config': (directory / 'model_config.json').as_posix()})
        del model
    frozen = {}
    for record in seed_configs:
        path = Path(record['config'])
        frozen[path.as_posix()] = digest(path)
        for model in json.loads(path.read_text())['models'].values():
            for key in ('head', 'dictionary', 'training_predictions'):
                if key in model:
                    frozen[model[key]] = digest(Path(model[key]))
    write_json(output / 'models_frozen.json', {'status': 'ALL_MODELS_FITTED_BEFORE_EVALUATION',
               'seeds': seed_configs, 'sha256': frozen, 'plan_sha256': digest(run / 'replication_plan.json')})
    return scope


def evaluate(run, plan, base, scope, execution, torch):
    output = run / 'replication'
    frozen = json.loads((output / 'models_frozen.json').read_text())
    for path, expected in frozen['sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Fitted model changed before evaluation')
    confirmation = Path(plan['confirmation_run'])
    rows = [json.loads(line) for line in (confirmation / 'clip_manifest.jsonl').read_text().splitlines()]
    if {r['video_id'] for r in rows} != set(base['confirmation_videos']):
        raise ValueError('Confirmation videos changed')
    if set(scope['fit_videos']) & {r['video_id'] for r in rows}:
        raise ValueError('Training/confirmation overlap')
    cache = Path(base['confirmation_cache'])
    for name, expected in execution['confirmation_cache_identity'].items():
        stat = (cache / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise ValueError('Previously verified confirmation cache changed')
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(rows))), torch)
    masks = np.load(cache / 'masks.npy', allow_pickle=False)
    training_masks = np.load(Path(base['training_cache']) / 'masks.npy', mmap_mode='r', allow_pickle=False)[scope['source_clip_indices']]
    combined_masks = np.concatenate([training_masks, masks])
    combined_rows = scope['all_rows'] + rows
    write_json(output / 'fit_scope.json', {**scope, 'all_rows': combined_rows, 'validation_videos': base['confirmation_videos']})
    records, replay, fidelity = [], {}, {}
    norm = arrays(Path(base['normalization']), torch, 'cuda')
    for seed_record in frozen['seeds']:
        seed = seed_record['seed']
        config = json.loads(Path(seed_record['config']).read_text())
        for method in plan['methods']:
            path = output / str(seed) / method
            new = predict(raw, method, config, torch).reshape(masks.shape)
            train = np.load(config['models'][method]['training_predictions'], allow_pickle=False)
            predictions = np.concatenate([train, new])
            np.save(path / 'predictions.npy', predictions, allow_pickle=False)
            if seed == plan['seeds'][0] and method in ('raw_balanced', 'topk_frozen_balanced'):
                previous = np.load(confirmation / 'confirmation' / method / 'predictions.npy', allow_pickle=False)
                if not np.array_equal(previous, predictions):
                    raise ValueError('Unchanged seed-zero predictor did not replay exactly: ' + method)
                replay[method] = {'n_values': predictions.size, 'max_error': 0.}
            metrics = summarize(predictions, combined_masks, combined_rows)
            write_json(path / 'metrics.json', metrics)
            records.append({'seed': seed, 'method': method, 'path': path.as_posix()})
        from src.sae.baselines import TopKAutoencoder
        model = TopKAutoencoder(768, 1536, 32).cuda()
        model.load_state_dict(arrays(Path(config['models']['topk_frozen_balanced']['dictionary']), torch, 'cuda'), strict=True)
        totals = {v: {'denominator': 0., 'topk32': 0., 'pca32': 0., 'pca48': 0., 'nonzero': 0, 'tokens': 0}
                  for v in base['confirmation_videos']}
        with torch.no_grad():
            for row, token in zip(rows, raw.reshape(len(rows), 8 * 196, 768)):
                x = (token - norm['mean']) / norm['rms']
                value = totals[row['video_id']]
                value['denominator'] += float(x.double().square().sum())
                codes = model.encode_inference(x)
                value['topk32'] += float((model.decode(codes) - x).double().square().sum())
                value['nonzero'] += int(torch.count_nonzero(codes))
                value['tokens'] += len(x)
                for d in (32, 48):
                    basis = norm['basis'][:, :d]
                    value['pca' + str(d)] += float(((x @ basis) @ basis.T - x).double().square().sum())
        fidelity[str(seed)] = {'raw_sums': totals, 'per_video': {v: {
            **{m: sums[m] / sums['denominator'] for m in ('topk32', 'pca32', 'pca48')},
            'topk_mean_nonzero': sums['nonzero'] / sums['tokens']} for v, sums in totals.items()}}
        print(json.dumps({'phase': 'evaluated', 'seed': seed}), flush=True)
    write_json(output / 'original_predictor_replay.json', replay)
    write_json(output / 'summary.json', {'status': 'ALL_SEEDS_EVALUATED_WITHOUT_SELECTION', 'records': records})
    write_json(output / 'representation_fidelity.json', fidelity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    execution = json.loads((run / 'replication_execution.json').read_text())
    for path, expected in execution['asset_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Replication input changed: ' + path)
    plan = json.loads((run / 'replication_plan.json').read_text())
    base = json.loads((Path(plan['confirmation_run']) / 'model_config.json').read_text())
    validate_assets(base)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    write_json(run / 'replication_status.json', {'status': 'RUNNING', 'phase': 'fit', 'pid': os.getpid()})
    try:
        scope = fit_models(run, plan, base, torch)
        write_json(run / 'replication_status.json', {'status': 'RUNNING', 'phase': 'evaluate', 'pid': os.getpid()})
        evaluate(run, plan, base, scope, execution, torch)
    except Exception as error:
        write_json(run / 'replication_status.json', {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(run / 'replication_status.json', {'status': 'COMPLETED', 'elapsed_seconds': time.time() - start,
               'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
