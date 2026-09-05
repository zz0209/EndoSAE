"""Target box-support head adaptation with frozen representations and family CV."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_fixed_confirmation import arrays, tokens_cuda
from src.evaluation.realcolon_task import digest, write_json
from src.evaluation.kumc_localization import localization_metrics


class FrameSupportSampler:
    """Uniform fitting sources, then valid frames, then balanced inside/outside.

    Frames without both classes never enter either sampling table. A pair shares
    the source frame but samples one inside and one outside patch independently.
    """
    def __init__(self, rows, masks, fit_clips, torch, device):
        self.torch, self.device = torch, device
        if masks.shape != (len(rows), 8, 196):
            raise ValueError('Expected clip,time,patch support')
        self.sources = sorted({row['video_id'] for row, use in zip(rows, fit_clips) if use})
        groups, eligible_frames = [], []
        valid = masks.any(-1) & ~masks.all(-1)
        for source in self.sources:
            frames = [i * 8 + t for i, row in enumerate(rows) if fit_clips[i] and row['video_id'] == source
                      for t in range(8) if valid[i, t]]
            if not frames:
                raise ValueError('Fitting source lacks valid frames')
            groups.append(frames)
            eligible_frames.extend(frames)
        if not groups:
            raise ValueError('No fitting sources')
        frames = np.zeros((len(groups), max(map(len, groups))), dtype=np.int64)
        for i, group in enumerate(groups):
            frames[i, :len(group)] = group
        support = np.zeros((len(rows) * 8, 2, 196), dtype=np.int64)
        counts = np.zeros((len(rows) * 8, 2), dtype=np.int64)
        for frame in eligible_frames:
            target = masks.reshape(-1, 196)[frame]
            for category, selection in enumerate([target, ~target]):
                indices = np.flatnonzero(selection) + frame * 196
                support[frame, category, :len(indices)] = indices
                counts[frame, category] = len(indices)
        self.fit_frame_indices = sorted(eligible_frames)
        self.frames = torch.tensor(frames, device=device)
        self.frame_counts = torch.tensor([len(g) for g in groups], device=device)
        self.support = torch.tensor(support, device=device)
        self.counts = torch.tensor(counts, device=device)

    def sample(self, batch_size, generator):
        torch, device = self.torch, self.device
        if batch_size % 2:
            raise ValueError('An even batch size is required')
        source = torch.randint(len(self.sources), (batch_size // 2,), device=device, generator=generator)
        offset = (torch.rand(batch_size // 2, device=device, generator=generator) * self.frame_counts[source]).long()
        frame = self.frames[source, offset].repeat_interleave(2)
        category = torch.arange(batch_size, device=device) % 2
        patch = (torch.rand(batch_size, device=device, generator=generator) * self.counts[frame, category]).long()
        return self.support[frame, category, patch], (category == 0).float()


def head_predictions(features, head, shape, torch):
    with torch.no_grad():
        values = [torch.sigmoid(head(batch)).squeeze(-1).cpu().numpy() for batch in features.split(4096)]
    return np.concatenate(values).reshape(shape)


def execute(run, config, torch):
    from src.sae.baselines import TopKAutoencoder
    source = Path(config['source_run'])
    previous = json.loads((source / 'config.json').read_text())
    rows = [json.loads(line) for line in (source / 'clip_manifest.jsonl').read_text().splitlines()]
    cache = Path(previous['cache_dir'])
    encoded = json.loads((source / 'encoded.json').read_text())
    prepared = json.loads((source / 'prepared.json').read_text())
    if digest(cache / 'tokens.npy') != encoded['tokens_sha256'] or digest(cache / 'masks.npy') != prepared['assets']['masks.npy']:
        raise ValueError('Frozen input cache changed')
    masks = np.load(cache / 'masks.npy', allow_pickle=False)
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(rows))), torch)
    families = np.array([row['video_id'].split('/')[-1] for row in rows])
    if families.tolist() != config['clip_families'] or sorted(set(families)) != config['held_families']:
        raise ValueError('Family assignment mismatch')
    output = run / 'cv'
    output.mkdir(exist_ok=False)
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for seed in config['seeds']:
        model_config = json.loads(Path(previous['model_configs'][str(seed)]).read_text())
        norm = arrays(Path(model_config['normalization']), torch, 'cuda')
        x = (raw - norm['mean']) / norm['rms']
        for method in config['methods']:
            if method == 'raw_balanced':
                features = x.clone()
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
            old_head = arrays(Path(model_config['models'][method]['head']), torch, 'cuda')
            features.sub_(old_head['feature_mean']).div_(old_head['feature_scale'])
            if not bool(torch.isfinite(features).all()):
                raise ValueError('Nonfinite source-normalized features')
            original = np.load(source / 'predictions' / str(seed) / method / 'patch_predictions.npy', allow_pickle=False)
            oof = np.full(masks.shape, np.nan, dtype=np.float32)
            assigned = np.zeros(len(rows), dtype=int)
            for fold, held in enumerate(config['held_families']):
                fit = families != held
                destination = output / str(seed) / method / ('family_' + held)
                destination.mkdir(parents=True)
                sampler = FrameSupportSampler(rows, masks, fit, torch, 'cuda')
                head = torch.nn.Linear(features.shape[1], 1, device='cuda')
                with torch.no_grad():
                    head.weight.copy_(old_head['weight'])
                    head.bias.copy_(old_head['bias'])
                warm = head_predictions(features, head, masks.shape, torch)
                error = float(np.max(np.abs(warm - original)))
                if error > 1e-6:
                    raise ValueError('Warm head differs from frozen predictions')
                optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
                generator = torch.Generator(device='cuda').manual_seed(seed + fold)
                losses = []
                for step in range(config['steps']):
                    indices, labels = sampler.sample(config['batch_size'], generator)
                    logits = head(features[indices]).squeeze(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    optimizer.step()
                    if (step + 1) % 50 == 0:
                        losses.append({'step': step + 1, 'sampled_bce': float(loss.detach())})
                np.savez(destination / 'head.npz', weight=head.weight.detach().cpu().numpy(), bias=head.bias.detach().cpu().numpy(),
                    feature_mean=old_head['feature_mean'].cpu().numpy(), feature_scale=old_head['feature_scale'].cpu().numpy())
                write_json(destination / 'training.json', {'held_family': held, 'fit_clip_indices': np.flatnonzero(fit).tolist(),
                    'held_clip_indices': np.flatnonzero(~fit).tolist(), 'fit_sources': sampler.sources,
                    'fit_frame_indices': sampler.fit_frame_indices, 'normalization_fit_target_indices': [],
                    'sampling_seed': seed + fold, 'losses': losses, 'warm_max_absolute_difference': error,
                    'frozen_head_sha256_before_evaluation': digest(destination / 'head.npz')})
                scores = head_predictions(features, head, masks.shape, torch)
                np.save(destination / 'all_patch_predictions.npy', scores, allow_pickle=False)
                oof[~fit] = scores[~fit]
                assigned[~fit] += 1
                held_rows = [r for r, use in zip(rows, ~fit) if use]
                train_rows = [r for r, use in zip(rows, fit) if use]
                write_json(destination / 'metrics.json', {'held': localization_metrics(scores[~fit], masks[~fit], held_rows),
                    'train': localization_metrics(scores[fit], masks[fit], train_rows)})
            if not (assigned == 1).all() or not np.isfinite(oof).all():
                raise ValueError('OOF prediction coverage failed')
            np.save(output / str(seed) / method / 'oof_patch_predictions.npy', oof, allow_pickle=False)
            record = {'seed': seed, 'method': method, 'adapted': localization_metrics(oof, masks, rows),
                      'frozen': localization_metrics(original, masks, rows)}
            records.append(record)
            print(json.dumps({'seed': seed, 'method': method, 'mean_group_ap': float(np.mean([
                v['frame_mean_patch_ap'] for v in record['adapted'].values()]))}), flush=True)
            del features
        del x
    write_json(output / 'summary.json', {'records': records, 'status': 'COMPLETED', 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    run = parser.parse_args().run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen asset changed: ' + path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
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
