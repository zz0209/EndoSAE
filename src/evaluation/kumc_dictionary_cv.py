"""Bounded task/reconstruction representation adaptation on grouped box support."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.evaluation.realcolon_task import digest, write_json
from src.evaluation.realcolon_fixed_confirmation import arrays, tokens_cuda
from src.evaluation.kumc_head_cv import FrameSupportSampler
from src.evaluation.kumc_localization import localization_metrics
from src.sae.baselines import TopKAutoencoder


class LinearBottleneck(torch.nn.Module):
    """Trainable linear encoder/decoder initialized to the existing PCA basis."""
    def __init__(self, basis):
        super().__init__()
        self.encoder = torch.nn.Linear(basis.shape[0], basis.shape[1], bias=False, device=basis.device)
        self.decoder = torch.nn.Linear(basis.shape[1], basis.shape[0], bias=False, device=basis.device)
        with torch.no_grad():
            self.encoder.weight.copy_(basis.T)
            self.decoder.weight.copy_(basis)

    def encode_inference(self, values):
        return self.encoder(values)

    def decode(self, codes):
        return self.decoder(codes)


def task_loss(model, head, batch, labels, mean, scale, task_updates_representation):
    code = model.encode_inference(batch)
    if not task_updates_representation:
        code = code.detach()
    logits = head((code - mean) / scale).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def evaluate(model, head, mean, scale, x, shape):
    predictions, squared_errors, denominators, active = [], [], [], []
    with torch.no_grad():
        for batch in x.split(4096):
            code = model.encode_inference(batch)
            reconstruction = model.decode(code)
            predictions.append(torch.sigmoid(head((code - mean) / scale)).squeeze(-1).cpu().numpy())
            squared_errors.append((reconstruction - batch).double().square().sum(-1).cpu().numpy())
            denominators.append(batch.double().square().sum(-1).cpu().numpy())
            active.append((code != 0).sum(-1).cpu().numpy())
    scores = np.concatenate(predictions).reshape(shape)
    diagnostics = {key: np.concatenate(parts).reshape(shape) for key, parts in
                   [('sse', squared_errors), ('denominator', denominators), ('l0', active)]}
    if not np.isfinite(scores).all() or not all(np.isfinite(v).all() for v in diagnostics.values()):
        raise ValueError('Nonfinite predictions or reconstruction')
    return scores, diagnostics


def execute(run, config):
    source = Path(config['source_run'])
    previous = json.loads((source / 'config.json').read_text())
    rows = [json.loads(line) for line in (source / 'clip_manifest.jsonl').read_text().splitlines()]
    families = np.array([r['video_id'].split('/')[-1] for r in rows])
    if families.tolist() != config['clip_families']:
        raise ValueError('Family map mismatch')
    cache = Path(previous['cache_dir'])
    encoded = json.loads((source / 'encoded.json').read_text())
    prepared = json.loads((source / 'prepared.json').read_text())
    for name, expected in [('tokens.npy', encoded['tokens_sha256']), ('masks.npy', prepared['assets']['masks.npy'])]:
        if digest(cache / name) != expected:
            raise ValueError('Input cache changed')
    masks = np.load(cache / 'masks.npy', allow_pickle=False)
    raw = tokens_cuda(np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False), list(range(len(rows))), torch)
    seed = config['seed']
    model_config = json.loads(Path(previous['model_configs'][str(seed)]).read_text())
    norm = arrays(Path(model_config['normalization']), torch, 'cuda')
    x = (raw - norm['mean']) / norm['rms']
    del raw
    output = run / 'cv'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for method in config['methods']:
        sparse = method.startswith('topk')
        parent = 'topk_frozen_balanced' if sparse else 'pca48_balanced'
        old_head = arrays(Path(model_config['models'][parent]['head']), torch, 'cuda')
        mean, scale = old_head['feature_mean'], old_head['feature_scale']
        original = np.load(source / 'predictions' / str(seed) / parent / 'patch_predictions.npy', allow_pickle=False)
        oof = np.full(masks.shape, np.nan, dtype=np.float32)
        oof_stats = {key: np.full(masks.shape, np.nan) for key in ['sse', 'denominator', 'l0']}
        coverage = np.zeros(len(rows), dtype=int)
        for fold, held in enumerate(config['held_families']):
            fit = families != held
            destination = output / method / ('family_' + held)
            destination.mkdir(parents=True)
            if sparse:
                model = TopKAutoencoder(768, 1536, 32).cuda()
                model.load_state_dict(arrays(Path(model_config['models'][parent]['dictionary']), torch, 'cuda'), strict=True)
                if config.get('representation_update') == 'low_rank':
                    from src.sae.low_rank_topk import LowRankTopK
                    torch.manual_seed(seed + fold + 200000)
                    model = LowRankTopK(model, config['update_rank'])
            else:
                model = LinearBottleneck(norm['basis'][:, :48])
            head = torch.nn.Linear(len(mean), 1, device='cuda')
            with torch.no_grad():
                head.weight.copy_(old_head['weight']); head.bias.copy_(old_head['bias'])
            warm, _ = evaluate(model, head, mean, scale, x, masks.shape)
            warm_error = float(np.max(np.abs(warm - original)))
            if warm_error > 1e-6:
                raise ValueError('Initial representation/head does not match frozen source')
            sampler = FrameSupportSampler(rows, masks, fit, torch, 'cuda')
            fit_tokens = torch.tensor(np.flatnonzero(np.repeat(fit, 8 * 196)), device='cuda')
            task_generator = torch.Generator(device='cuda').manual_seed(seed + fold)
            mse_generator = torch.Generator(device='cuda').manual_seed(seed + fold + 100000)
            optimizer = torch.optim.AdamW([
                {'params': head.parameters(), 'lr': config['head_learning_rate'], 'weight_decay': config['head_weight_decay']},
                {'params': model.parameters(), 'lr': config['representation_learning_rate'], 'weight_decay': 0.}])
            history = []
            joint = method != 'topk_reconstruction_only'
            for step in range(config['steps']):
                indices, labels = sampler.sample(config['batch_size'], task_generator)
                bce = task_loss(model, head, x[indices], labels, mean, scale, joint)
                chosen = fit_tokens[torch.randint(len(fit_tokens), (config['reconstruction_batch_size'],),
                                                 generator=mse_generator, device='cuda')]
                batch = x[chosen]
                reconstruction = model.decode(model.encode_inference(batch))
                mse = (reconstruction - batch).square().mean()
                loss = bce + config['mse_weight'] * mse
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if sparse and config.get('representation_update') != 'low_rank':
                    model.project_decoder_gradient_()
                # Separate clipping prevents head BCE from rescaling MSE-only dictionary gradients.
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if sparse and config.get('representation_update') != 'low_rank':
                    model.normalize_decoder_()
                if (step + 1) % 50 == 0:
                    history.append({'step': step + 1, 'bce': float(bce.detach()), 'mse': float(mse.detach())})
            np.savez(destination / 'representation.npz', **{k: v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            if config.get('representation_update') == 'low_rank':
                if not model.base_unchanged():
                    raise ValueError('Frozen low-rank base changed')
                np.savez(destination / 'materialized_representation.npz',
                         **{k: v.cpu().numpy() for k,v in model.materialized_state().items()})
            np.savez(destination / 'head.npz', weight=head.weight.detach().cpu().numpy(), bias=head.bias.detach().cpu().numpy(),
                feature_mean=mean.cpu().numpy(), feature_scale=scale.cpu().numpy())
            write_json(destination / 'training.json', {'held_family': held, 'fit_clip_indices': np.flatnonzero(fit).tolist(),
                'held_clip_indices': np.flatnonzero(~fit).tolist(), 'fit_frame_indices': sampler.fit_frame_indices,
                'reconstruction_fit_clip_indices': np.flatnonzero(fit).tolist(), 'fit_sources': sampler.sources,
                'normalization_fit_target_indices': [], 'task_updates_representation': joint,
                'representation_parameters': sum(p.numel() for p in model.parameters()),
                'representation_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'representation_update': config.get('representation_update', 'full'),
                'low_rank_base_unchanged': model.base_unchanged() if config.get('representation_update') == 'low_rank' else None,
                'head_parameters': sum(p.numel() for p in head.parameters()), 'feature_payload_bytes': 192,
                'warm_max_absolute_difference': warm_error, 'history': history,
                'frozen_before_evaluation': {name: digest(destination / name) for name in
                    (['representation.npz', 'head.npz', 'materialized_representation.npz'] if config.get('representation_update') == 'low_rank'
                     else ['representation.npz', 'head.npz'])}})
            scores, stats = evaluate(model, head, mean, scale, x, masks.shape)
            np.save(destination / 'all_patch_predictions.npy', scores, allow_pickle=False)
            np.savez(destination / 'all_reconstruction_statistics.npz', **stats)
            oof[~fit] = scores[~fit]
            for key in oof_stats:
                oof_stats[key][~fit] = stats[key][~fit]
            coverage[~fit] += 1
            write_json(destination / 'metrics.json', {'held': localization_metrics(scores[~fit], masks[~fit],
                [r for r,use in zip(rows,~fit) if use]), 'train': localization_metrics(scores[fit], masks[fit],
                [r for r,use in zip(rows,fit) if use])})
            print('FIT {} held={}'.format(method, held), flush=True)
        if not (coverage == 1).all() or not np.isfinite(oof).all():
            raise ValueError('OOF coverage failure')
        np.save(output / method / 'oof_patch_predictions.npy', oof, allow_pickle=False)
        np.savez(output / method / 'oof_reconstruction_statistics.npz', **oof_stats)
        metric = localization_metrics(oof, masks, rows)
        reconstruction = {}
        for group in sorted(set(r['video_id'] for r in rows)):
            select = np.array([r['video_id'] == group for r in rows])
            reconstruction[group] = {'nmse': float(oof_stats['sse'][select].sum() / oof_stats['denominator'][select].sum()),
                                     'mean_l0': float(oof_stats['l0'][select].mean())}
        record = {'method': method, 'seed': seed, 'localization': metric, 'reconstruction': reconstruction}
        records.append(record)
        print(json.dumps({'method': method, 'mean_group_ap': float(np.mean([v['frame_mean_patch_ap'] for v in metric.values()])),
                          'mean_group_nmse': float(np.mean([v['nmse'] for v in reconstruction.values()]))}), flush=True)
    write_json(output / 'summary.json', {'records': records, 'status': 'COMPLETED'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    run = parser.parse_args().run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen asset changed: ' + path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    write_json(run / 'status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        execute(run, config)
    except Exception as error:
        write_json(run / 'status.json', {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(run / 'status.json', {'status': 'COMPLETED', 'elapsed_seconds': time.time() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
