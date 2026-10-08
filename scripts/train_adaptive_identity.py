import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
import train_acknowledgement_sae as shared
from train_frozen_identity_supcon import sample_batch, supcon
from src.token_identity_sae import TokenIdentitySAE, procedure_identity_loss
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint


class Observations:
    def __init__(self, prepared, indices):
        self.prepared = prepared
        all_records = read_json(prepared / 'records.json')
        self.records = [dict(all_records[i], cache_index=i) for i in indices]
        self.native, self.positions = [], []
        for i in indices:
            with np.load(prepared / 'observations' / f'{i:04d}' / 'roi.npz', allow_pickle=False) as saved:
                self.native.append(saved['native'].copy())
                position = saved['positions'].astype(np.int64)
                self.positions.append(1 + position[:, 1] * 8 + position[:, 0])

    def prefix(self, indices):
        values = [np.load(self.prepared / 'observations' / f"{self.records[i]['cache_index']:04d}" /
                          'prefix.npy', allow_pickle=False) for i in indices]
        return torch.from_numpy(np.stack(values)).to('cuda')

    def normalization(self, videos):
        chosen = [i for i, row in enumerate(self.records) if row['video_id'] in videos]
        means, seconds = [], []
        for i in chosen:
            value = self.native[i].astype(np.float64)
            means.append(value.mean(0))
            seconds.append(np.square(value).mean(0))
        mean = np.mean(means, 0)
        variance = np.mean(seconds, 0) - mean ** 2
        assert np.min(variance) > -1e-10
        scale = np.sqrt(np.maximum(variance, 0))
        scale[scale == 0] = 1
        return mean.astype(np.float32), scale.astype(np.float32)


def replay(blocks, value, gradients):
    for block in blocks:
        if gradients:
            value = checkpoint(block, value, value.shape[0], 8, 14, use_reentrant=False)
        else:
            value = block(value, value.shape[0], 8, 14)
    return value


def token_batch(data, chosen, token_indices, blocks, mean, scale, adaptive, microbatch):
    if not adaptive:
        raw = np.stack([data.native[i][j] for i, j in zip(chosen, token_indices, strict=True)])
        return (torch.from_numpy(raw).to('cuda') - mean) / scale
    samples = []
    for start in range(0, len(chosen), microbatch):
        local = chosen[start:start + microbatch]
        value = replay(blocks, data.prefix(local), True)
        for offset, i in enumerate(local):
            selected = data.positions[i][token_indices[start + offset]]
            samples.append(value[offset, torch.as_tensor(selected, device='cuda')])
    return (torch.stack(samples) - mean) / scale


@torch.no_grad()
def evaluate(data, blocks, model, mean, scale, adaptive, videos, folder):
    indices = [i for i, row in enumerate(data.records)
               if row['video_id'] in videos and row['original_observation']]
    records = [data.records[i] for i in indices]
    embeddings, codes, metrics = [], [], []
    for i in indices:
        frozen = (torch.from_numpy(data.native[i]).to('cuda') - mean) / scale
        if adaptive:
            value = replay(blocks, data.prefix([i]), False)
            native = value[0, torch.as_tensor(data.positions[i], device='cuda')]
            samples = (native - mean) / scale
        else:
            samples = frozen
        projected, decoded, local = model(samples[None])
        assert torch.isfinite(projected).all() and projected.norm() > 0
        embeddings.append(projected[0].cpu().numpy())
        codes.append(local.mean(1)[0].cpu().numpy())
        metrics.append(dict(video_id=data.records[i]['video_id'], cache_index=data.records[i]['cache_index'],
            reconstruction_nmse=float((decoded[0] - samples).square().mean() / samples.square().mean()),
            native_anchor_nmse=float((decoded[0] - frozen).square().mean() / frozen.square().mean()),
            representation_drift=float((samples - frozen).square().mean() / frozen.square().mean()),
            local_active=float((local > 0).sum(-1).float().mean())))
    pairs = shared.chronological_pairs(records, videos)
    unit = np.stack(embeddings)
    source = np.array([r['source_index'] for r in pairs], dtype=np.int64)
    query = np.array([r['query_index'] for r in pairs], dtype=np.int64)
    scores = np.sum(unit[source] * unit[query], 1)
    shared.save_npz(folder / 'held.npz', embeddings=unit, pooled_codes=np.stack(codes),
        source=source, query=query, scores=scores, same_identity=np.array([r['same_identity'] for r in pairs]))
    atomic_write_json(folder / 'held_records.json', records)
    atomic_write_json(folder / 'held_metrics.json', metrics)


def fit(config, data, initial_blocks, method, adaptive, seed, fitting, held, folder,
        steps, identity, progress_path, stop_after=None, batch_schedule=None):
    folder.mkdir(parents=True, exist_ok=True)
    job = dict(identity=identity, config=config, method=method, adaptive=adaptive, seed=seed,
               fit_videos=fitting, held_videos=held, steps=steps)
    signature = shared.json_digest(job)
    if (folder / 'summary.json').exists():
        result = read_json(folder / 'summary.json')
        assert result['signature'] == signature
        return result
    atomic_write_json(folder / 'job.json', job)
    torch.manual_seed(seed)
    model = TokenIdentitySAE(config, method).cuda()
    blocks = copy.deepcopy(initial_blocks).cuda().eval().requires_grad_(adaptive)
    mean, scale = data.normalization(fitting)
    shared.save_npz(folder / 'normalization.npz', mean=mean, scale=scale)
    mean, scale = torch.from_numpy(mean).cuda(), torch.from_numpy(scale).cuda()
    groups = [dict(params=list(model.parameters()), lr=config['learning_rate'])]
    if adaptive:
        groups.append(dict(params=list(blocks.parameters()), lr=config['backbone_learning_rate']))
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    parameters = [p for group in groups for p in group['params']]
    train_records = [dict(row, split='train' if row['video_id'] in fitting else 'excluded') for row in data.records]
    label_map = {key: i for i, key in enumerate(sorted({(r['video_id'], r['lesion_id']) for r in data.records}))}
    labels = torch.tensor([label_map[(r['video_id'], r['lesion_id'])] for r in data.records], device='cuda')
    procedure_map = {key: i for i, key in enumerate(sorted({r['video_id'] for r in data.records}))}
    procedures = torch.tensor([procedure_map[r['video_id']] for r in data.records], device='cuda')
    procedure_weight = float(config.get('procedure_loss_weight', 0.))
    assert 0 <= procedure_weight <= 1
    rng = np.random.default_rng(seed)
    token_rng = np.random.default_rng(np.random.SeedSequence([seed, 71005]))
    history, sequence, initial, previous = [], [], 1, 0.
    checkpoint_path = folder / 'checkpoint.pt'
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
        assert saved['signature'] == signature
        model.load_state_dict(saved['model'])
        blocks.load_state_dict(saved['blocks'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = json.loads(saved['rng_json'])
        token_rng.bit_generator.state = json.loads(saved['token_rng_json'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        history, sequence = saved['history'], saved['sequence']
        initial, previous = saved['step'] + 1, saved['seconds']
    started = time.perf_counter()
    for step in range(initial, steps + 1):
        if batch_schedule is None:
            chosen = sample_batch(train_records, rng, config)
            token_indices = [token_rng.integers(0, len(data.native[i]), size=config['tokens_per_clip']) for i in chosen]
        else:
            scheduled = batch_schedule[step - 1]
            chosen = scheduled['indices']
            token_indices = [np.asarray(row, dtype=np.int64) for row in scheduled['tokens']]
        assert all(data.records[i]['video_id'] in fitting for i in chosen)
        samples = token_batch(data, chosen, token_indices, blocks, mean, scale, adaptive, config['microbatch'])
        projected, decoded, local = model(samples)
        task_loss = supcon(projected, labels[chosen], config['temperature'])
        global_loss = task_loss
        if procedure_weight:
            conditioned_loss = procedure_identity_loss(projected, labels[chosen], procedures[chosen], config['temperature'])
            task_loss = (1 - procedure_weight) * global_loss + procedure_weight * conditioned_loss
        reconstruction = (decoded - samples.detach()).square().mean()
        loss = task_loss + config['reconstruction_weight'] * reconstruction
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        assert norm > 0 and all(p.grad is not None for p in parameters)
        backbone_norm = float(torch.stack([p.grad.square().sum() for p in blocks.parameters()]).sum().sqrt()) if adaptive else 0.
        assert not adaptive or backbone_norm > 0
        optimizer.step()
        model.normalize_decoder()
        sequence.append(dict(indices=chosen, tokens_sha256=hashlib.sha256(np.stack(token_indices).tobytes()).hexdigest()))
        history.append(dict(step=step, loss=float(loss.detach()), identity=float(task_loss.detach()),
            reconstruction=float(reconstruction.detach()), gradient_norm=float(norm), backbone_gradient_norm=backbone_norm))
        if procedure_weight:
            negative = labels[chosen, None] != labels[chosen][None, :]
            within = procedures[chosen, None] == procedures[chosen][None, :]
            history[-1].update(global_identity=float(global_loss.detach()),
                procedure_identity=float(conditioned_loss.detach()),
                within_negative_fraction=float((negative & within).sum() / negative.sum()))
        if step % config['checkpoint_every'] == 0 or step == steps or step == stop_after:
            elapsed = previous + time.perf_counter() - started
            shared.save_torch(checkpoint_path, dict(signature=signature, model=model.state_dict(), blocks=blocks.state_dict(),
                optimizer=optimizer.state_dict(), rng_json=json.dumps(rng.bit_generator.state),
                token_rng_json=json.dumps(token_rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), step=step, history=history, sequence=sequence, seconds=elapsed))
            progress = dict(status='RUNNING', method=method, adaptive=adaptive, seed=seed, folder=str(folder),
                step=step, steps=steps, seconds=elapsed, losses=history[-1], updated_at=shared.now())
            atomic_write_json(progress_path, progress)
            print('ADAPTIVE_TRAINING', json.dumps(progress), flush=True)
            pause_after_checkpoint(checkpoint_path)
            if step == stop_after:
                raise SystemExit(75)
    model.eval()
    if held:
        evaluate(data, blocks, model, mean, scale, adaptive, held, folder)
    shared.save_npz(folder / 'model.npz', **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    shared.save_npz(folder / 'blocks.npz', **{k: v.detach().cpu().numpy() for k, v in blocks.state_dict().items()})
    atomic_write_json(folder / 'history.json', history)
    atomic_write_json(folder / 'sequence.json', sequence)
    drift = sum(float((a.detach() - b.to('cuda')).square().sum()) for a, b in
                zip(blocks.parameters(), initial_blocks.parameters(), strict=True))
    assert (drift > 0) if adaptive else (drift == 0)
    result = dict(status='COMPLETE', signature=signature, method=method, adaptive=adaptive, seed=seed,
        steps=steps, seconds=previous + time.perf_counter() - started, backbone_parameter_squared_change=drift,
        trainable_parameters=sum(p.numel() for p in parameters), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        sequence_sha256=shared.file_sha256(folder / 'sequence.json'),
        model_sha256=shared.file_sha256(folder / 'model.npz'), blocks_sha256=shared.file_sha256(folder / 'blocks.npz'),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version, tf32=False),
        completed_at=shared.now())
    atomic_write_json(folder / 'summary.json', result)
    return result


def train(run, smoke, output, stop_after):
    config = read_json(run / 'config.json')
    preparation = read_json(Path(config['preparation_run']) / 'config.json')
    prepared = Path(preparation['storage_root']) / ('smoke' if smoke and not config.get('smoke_full_inputs') else 'prepared')
    receipt = read_json(prepared / 'summary.json')
    assert receipt['status'] == 'COMPLETE'
    data = Observations(prepared, [r['index'] for r in receipt['receipts']])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(read_json(preparation['encoding_config']), torch.device('cuda'))
    initial_blocks = copy.deepcopy(encoder.model.blocks[9:11]).cpu()
    del encoder
    torch.cuda.empty_cache()
    root = output or (run / 'smoke' if smoke else Path(config['storage_root']))
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(prepared_sha256=shared.file_sha256(prepared / 'summary.json'),
        sources={name: shared.file_sha256(ROOT / name) for name in ['scripts/train_adaptive_identity.py',
            'src/token_identity_sae.py', 'scripts/train_frozen_identity_supcon.py', 'scripts/train_acknowledgement_sae.py']})
    folds = read_json(Path(config['reference_run']) / 'folds.json')
    videos = sorted({row['video_id'] for row in data.records})
    full_cohort = config.get('fit_full_cohort', False)
    if full_cohort:
        folds = [[]]
    elif smoke:
        folds = [videos]
    local = dict(config, checkpoint_every=2 if smoke else config['checkpoint_every'])
    steps = config['smoke_steps'] if smoke else config['steps']
    seeds = config['seeds'][:1] if smoke else config['seeds']
    outputs = []
    adaptive_states = config.get('adaptive_states', [False, True])
    total = len(seeds) * len(folds) * 2 * len(adaptive_states)
    for seed in seeds:
        for fold, held in enumerate(folds):
            fitting = videos if smoke else sorted(set(videos) - set(held))
            for method in ['token_sparse', 'token_dense']:
                for adaptive in adaptive_states:
                    fold_name = 'full' if full_cohort else str(fold)
                    folder = root / ('adaptive' if adaptive else 'frozen') / method / f'seed{seed}' / f'fold{fold_name}'
                    result = fit(local, data, initial_blocks, method, adaptive, seed, fitting, held, folder,
                                 steps, identity, root / 'training_progress.json', stop_after)
                    outputs.append(dict(method=method, adaptive=adaptive, seed=seed, fold=fold,
                        fit_videos=fitting, held_videos=held, directory=str(folder), summary=result))
                    atomic_write_json(root / 'progress.json', dict(status='RUNNING', completed=len(outputs), total=total))
    for seed in seeds:
        for fold in range(len(folds)):
            assert len({r['summary']['sequence_sha256'] for r in outputs if r['seed'] == seed and r['fold'] == fold}) == 1
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', outputs=outputs,
        paired_sampling_exact=True, smoke=smoke, identity=identity, completed_at=shared.now()))
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=total, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(args.run, args.smoke, args.output, args.stop_after)
