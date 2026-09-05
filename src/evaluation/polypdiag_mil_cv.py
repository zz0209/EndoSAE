"""Grouped target-video supervision with frozen representations and MIL pooling."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_fixed_confirmation import arrays, tokens_cuda
from src.evaluation.realcolon_task import binary_metrics, digest, write_json
from src.evaluation.realcolon_task_supervised import feature_stats


def bag_probabilities(features, head, torch):
    """One loss target per video; never assign the video label to its patches."""
    probabilities = torch.sigmoid(head(features).squeeze(-1))
    return probabilities.topk(4, dim=-1).values.mean(-1).mean(-1).max(-1).values


def evaluate(features, head, mean, scale, torch):
    scores = []
    with torch.no_grad():
        for batch in features.split(4):
            scores.append(bag_probabilities((batch - mean) / scale, head, torch).cpu().numpy())
    return np.concatenate(scores)


def target_head(features, old_head, train, labels, config, seed, output, reference_scores, torch):
    width = features.shape[-1]
    normalization = config.get('head_normalization', 'target_fold')
    if normalization == 'target_fold':
        flat = features.reshape(-1, width)
        token_indices = torch.cat([torch.arange(i * 4704, (i + 1) * 4704, device='cuda') for i in np.flatnonzero(train)])
        mean, scale = feature_stats(flat, token_indices, lambda value: value, torch)
    elif normalization == 'frozen_source':
        mean, scale = old_head['feature_mean'], old_head['feature_scale']
    else:
        raise ValueError('Unknown head normalization policy')
    head = torch.nn.Linear(width, 1, device='cuda')
    with torch.no_grad():
        head.weight.copy_(old_head['weight'] * (scale / old_head['feature_scale']))
        head.bias.copy_(old_head['bias'] + (old_head['weight'] * ((mean - old_head['feature_mean']) / old_head['feature_scale'])).sum())
    warm = evaluate(features, head, mean, scale, torch)
    error = float(np.max(np.abs(warm - reference_scores)))
    if error > 1e-6:
        raise ValueError('Warm initialization does not reproduce frozen video scores')
    write_json(output / 'warm_replay.json', {'max_absolute_error': error, 'n_video_scores': len(warm)})
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(seed)
    positive = torch.tensor(np.flatnonzero(train & (labels == 1)), device='cuda')
    negative = torch.tensor(np.flatnonzero(train & (labels == 0)), device='cuda')
    targets = torch.tensor([1., 1., 0., 0.], device='cuda')
    losses = []
    for step in range(config['steps']):
        indices = torch.cat([positive[torch.randint(len(positive), (2,), generator=generator, device='cuda')],
                             negative[torch.randint(len(negative), (2,), generator=generator, device='cuda')]])
        scores = bag_probabilities((features[indices] - mean) / scale, head, torch)
        loss = torch.nn.functional.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            losses.append({'step': step + 1, 'batch_bce': float(loss.detach())})
    np.savez(output / 'head.npz', weight=head.weight.detach().cpu().numpy(), bias=head.bias.detach().cpu().numpy(),
             feature_mean=mean.cpu().numpy(), feature_scale=scale.cpu().numpy())
    write_json(output / 'training.json', {'steps': config['steps'], 'seed': seed, 'loss_history': losses,
        'optimizer': 'AdamW', 'learning_rate': config['learning_rate'], 'weight_decay': config['weight_decay'],
        'fit_video_indices': np.flatnonzero(train).tolist(), 'held_video_indices': np.flatnonzero(~train).tolist(),
        'normalization_fit_video_indices': np.flatnonzero(train).tolist() if normalization == 'target_fold' else [],
        'head_normalization': normalization,
        'loss_unit': 'video bag only', 'frozen_head_sha256_before_evaluation': digest(output / 'head.npz')})
    return evaluate(features, head, mean, scale, torch)


def execute(run, config, torch):
    from src.sae.baselines import TopKAutoencoder
    source = Path(config['source_run'])
    source_config = json.loads((source / 'config.json').read_text())
    encoded = json.loads((source / 'encoded.json').read_text())
    cache = Path(source_config['cache_dir'])
    if digest(cache / 'tokens.npy') != encoded['tokens_sha256']:
        raise ValueError('Target cache changed')
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(tokens))), torch)
    labels = np.array([v['label'] for v in source_config['videos']])
    fold_ids = np.asarray(config['fold_ids'])
    if len(fold_ids) != len(labels) or set(fold_ids) != set(range(4)):
        raise ValueError('Invalid fold assignment')
    for fold in range(4):
        if sum((fold_ids == fold) & (labels == 1)) != 4 or sum((fold_ids == fold) & (labels == 0)) != 4:
            raise ValueError('Each held fold must have four files per class')
    output = run / 'cv'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for seed in config['seeds']:
        model_config = json.loads(Path(source_config['model_configs'][str(seed)]).read_text())
        norm = arrays(Path(model_config['normalization']), torch, 'cuda')
        x = (raw - norm['mean']) / norm['rms']
        for method in config['methods']:
            if method == 'raw_balanced':
                features = x
            elif method == 'pca48_balanced':
                features = x @ norm['basis'][:, :48]
            else:
                dictionary = TopKAutoencoder(768, 1536, 32).cuda()
                dictionary.load_state_dict(arrays(Path(model_config['models'][method]['dictionary']), torch, 'cuda'), strict=True)
                dictionary.eval()
                with torch.no_grad():
                    features = torch.empty((len(x), 1536), device='cuda')
                    for start in range(0, len(x), 4096):
                        features[start:start + 4096] = dictionary.encode_inference(x[start:start + 4096])
                del dictionary
            features = features.reshape(32, 3, 8, 196, -1)
            old_head = arrays(Path(model_config['models'][method]['head']), torch, 'cuda')
            original = np.load(source / 'predictions' / str(seed) / method / 'video_scores.npy', allow_pickle=False)
            oof_scores, oof_threshold, calibration_threshold = [np.full(32, np.nan) for _ in range(3)]
            fold_metrics = []
            for fold in range(4):
                train = fold_ids != fold
                destination = output / str(seed) / method / ('fold' + str(fold))
                destination.mkdir(parents=True)
                scores = target_head(features, old_head, train, labels, config, seed + fold,
                                     destination, original, torch)
                threshold = float(np.quantile(scores[train & (labels == 0)], .95, method='higher'))
                baseline_threshold = float(np.quantile(original[train & (labels == 0)], .95, method='higher'))
                oof_scores[~train] = scores[~train]
                oof_threshold[~train] = threshold
                calibration_threshold[~train] = baseline_threshold
                np.save(destination / 'all_video_scores.npy', scores, allow_pickle=False)
                record = {'fold': fold, 'threshold': threshold, 'frozen_score_threshold': baseline_threshold,
                          'head_held': binary_metrics(labels[~train], scores[~train]),
                          'frozen_held': binary_metrics(labels[~train], original[~train])}
                write_json(destination / 'metrics.json', record)
                fold_metrics.append(record)
                print(json.dumps({'seed': seed, 'method': method, **record}), flush=True)
            dest = output / str(seed) / method
            np.savez(dest / 'oof.npz', labels=labels, fold_ids=fold_ids, scores=oof_scores, thresholds=oof_threshold,
                     frozen_scores=original, frozen_thresholds=calibration_threshold)
            scores_record = {'seed': seed, 'method': method, 'folds': fold_metrics,
                'adapted_oof': binary_metrics(labels, oof_scores), 'frozen_oof': binary_metrics(labels, original)}
            for prefix, scores, thresholds in [('adapted', oof_scores, oof_threshold), ('calibrated_frozen', original, calibration_threshold)]:
                decisions = scores >= thresholds
                scores_record[prefix + '_sensitivity'] = float(decisions[labels == 1].mean())
                scores_record[prefix + '_fpr'] = float(decisions[labels == 0].mean())
            write_json(dest / 'summary.json', scores_record)
            records.append(scores_record)
            del features
        del x
    write_json(output / 'summary.json', {'records': records, 'status': 'COMPLETED', 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    run = parser.parse_args().run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen asset changed: ' + path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    write_json(run / 'status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        execute(run, config, torch)
    except Exception as error:
        write_json(run / 'status.json', {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(run / 'status.json', {'status': 'COMPLETED', 'elapsed_seconds': time.time() - started,
                                    'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
