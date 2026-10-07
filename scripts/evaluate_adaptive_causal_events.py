import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
from train_adaptive_identity import replay
from train_acknowledgement_sae import now, save_npz, json_digest
from evaluate_prompt_event_components import event_rows
from evaluate_query_conditioned_components import boundary
from src.token_identity_sae import TokenIdentitySAE
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


def specification(run):
    config = read_json(run / 'config.json')
    preparation = read_json(Path(config['prefix_run']) / 'config.json')
    training = read_json(Path(preparation['training_run']) / 'config.json')
    fitted = read_json(Path(training['storage_root']) / 'training_summary.json')
    assert fitted['status'] == 'COMPLETE' and len(fitted['outputs']) == 12
    return config, preparation, training, fitted


@torch.no_grad()
def encode(run, smoke):
    config, preparation, training, fitted = specification(run)
    prefix = Path(preparation['storage_root']) / ('smoke' if smoke else 'prepared')
    ready = read_json(prefix / 'summary.json')
    assert ready['status'] == 'COMPLETE'
    region_root = run / ('smoke_regions' if smoke else 'regions') if config.get('annotated_regions') else None
    if region_root:
        assert read_json(region_root / 'summary.json')['status'] == 'COMPLETE'
    evaluation_videos = {r['video'] for r in ready['receipts']}
    assert all(not evaluation_videos & set(item['fit_videos']) for item in fitted['outputs'])
    output = Path(config['storage_root']) / ('smoke' if smoke else 'encoded')
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(config=digest(run / 'config.json'), prefixes=digest(prefix / 'summary.json'),
        training=digest(Path(training['storage_root']) / 'training_summary.json'),
        sources={name: digest(ROOT / name) for name in ['scripts/evaluate_adaptive_causal_events.py',
            'scripts/train_adaptive_identity.py', 'src/token_identity_sae.py']})
    signature = json_digest(identity)
    if region_root:
        identity['regions'] = digest(region_root / 'summary.json')
        signature = json_digest(identity)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = read_json(Path(training['preparation_run']) / 'config.json')
    encoder = Encoder(read_json(original['encoding_config']), torch.device('cuda'))
    initial = copy.deepcopy(encoder.model.blocks[9:11]).cpu()
    del encoder
    models = {}
    for item in fitted['outputs']:
        folder = Path(item['directory'])
        condition = 'adaptive' if item['adaptive'] else 'frozen'
        key = f"{condition}_{item['method']}_seed{item['seed']}"
        assert digest(folder / 'model.npz') == item['summary']['model_sha256']
        assert digest(folder / 'blocks.npz') == item['summary']['blocks_sha256']
        model = TokenIdentitySAE(training, item['method']).cuda().eval()
        with np.load(folder / 'model.npz', allow_pickle=False) as saved:
            model.load_state_dict({k: torch.from_numpy(saved[k].copy()) for k in saved.files})
        blocks = None
        if item['adaptive']:
            blocks = copy.deepcopy(initial).cuda().eval()
            with np.load(folder / 'blocks.npz', allow_pickle=False) as saved:
                blocks.load_state_dict({k: torch.from_numpy(saved[k].copy()) for k in saved.files})
        with np.load(folder / 'normalization.npz', allow_pickle=False) as saved:
            mean, scale = [torch.from_numpy(saved[k].copy()).cuda() for k in ['mean', 'scale']]
        models[key] = (model, blocks, mean, scale)
    if region_root:
        initial = initial.cuda().eval()
    started = time.perf_counter()
    receipts = []
    for index, source in enumerate(ready['receipts']):
        folder = prefix / source['video'] / f"{source['output']:06d}"
        target = output / source['video'] / f"{source['output']:06d}"
        target.mkdir(parents=True, exist_ok=True)
        receipt_path = target / 'complete.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            assert receipt['signature'] == signature
            assert digest(target / 'codes.npz') == receipt['codes_sha256']
        else:
            assert digest(folder / 'prefix.npy') == source['prefix_sha256']
            assert digest(folder / 'roi.npz') == source['roi_sha256']
            value = torch.from_numpy(np.load(folder / 'prefix.npy', allow_pickle=False)).cuda()[None]
            with np.load(folder / 'roi.npz', allow_pickle=False) as saved:
                native = torch.from_numpy(saved['native'].copy()).cuda()
                positions, offsets = saved['positions'].copy(), saved['offsets'].copy()
                selected = torch.from_numpy(saved['token_positions'].copy()).cuda()
            if region_root:
                region = region_root / source['video'] / f"{source['output']:06d}"
                assert digest(region / 'roi.npz') == read_json(region / 'complete.json')['roi_sha256']
                frozen_map = replay(initial, value.clone(), False)[0]
                if smoke:
                    np.testing.assert_array_equal(frozen_map[selected].cpu().numpy(), native.cpu().numpy())
                with np.load(region / 'roi.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(positions, saved['positions'])
                    offsets = saved['offsets'].copy()
                    selected = torch.from_numpy(saved['token_positions'].copy()).cuda()
                native = frozen_map[selected]
            arrays = dict(positions=positions)
            largest = 0.
            for key, (model, blocks, mean, scale) in models.items():
                raw = native if blocks is None else replay(blocks, value.clone(), False)[0, selected]
                samples = (raw - mean) / scale
                codes = model.encode(samples)
                pooled = torch.stack([codes[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:], strict=True)])
                assert torch.isfinite(pooled).all() and torch.all(pooled.norm(dim=1) > 0)
                arrays[key] = pooled.cpu().numpy()
                if smoke:
                    for j, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                        projected, _, local = model(samples[None, a:b])
                        np.testing.assert_allclose(local.mean(1)[0].cpu(), arrays[key][j], atol=2e-6, rtol=2e-5)
                        normalized = torch.nn.functional.normalize(pooled[j], dim=0)
                        error = float((projected[0] - normalized).abs().max())
                        assert error < 2e-6
                        largest = max(largest, error)
                    if blocks is None:
                        direct = np.maximum(samples.cpu().numpy() @ model.encoder.weight.detach().cpu().numpy().T +
                                            model.encoder.bias.detach().cpu().numpy(), 0)
                        if model.method.endswith('sparse'):
                            keep = np.argsort(-direct, axis=1, kind='stable')[:, :model.top_k]
                            sparse = np.zeros_like(direct)
                            np.put_along_axis(sparse, keep, np.take_along_axis(direct, keep, 1), 1)
                            direct = sparse
                        independent = np.stack([direct[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:], strict=True)])
                        np.testing.assert_allclose(independent, arrays[key], atol=2e-5, rtol=2e-4)
            save_npz(target / 'codes.npz', **arrays)
            receipt = dict(signature=signature, video=source['video'], output=source['output'],
                detections=len(positions), codes_sha256=digest(target / 'codes.npz'),
                smoke_forward_error=largest if smoke else None, completed_at=now())
            atomic_write_json(receipt_path, receipt)
        receipts.append(receipt)
        progress = dict(status='RUNNING', completed=index + 1, total=len(ready['receipts']),
            seconds=time.perf_counter() - started, updated_at=now())
        atomic_write_json(output / 'progress.json', progress)
        if smoke or (index + 1) % 10 == 0:
            print('ADAPTIVE_EVENT_CODES', progress, flush=True)
        pause_after_checkpoint(receipt_path)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', receipts=receipts,
        keys=list(models), identity=identity, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.perf_counter() - started,
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version, tf32=False),
        completed_at=now()))
    atomic_write_json(output / 'progress.json', dict(status='COMPLETE', completed=len(receipts), total=len(receipts)))


def evaluate(run):
    config, preparation, training, fitted = specification(run)
    root = Path(config['storage_root']) / 'encoded'
    ready = read_json(root / 'summary.json')
    assert ready['status'] == 'COMPLETE'
    output = run / 'analysis'
    output.mkdir(exist_ok=True)
    videos = sorted({r['video'] for r in ready['receipts']})
    manifests, codes, positions = {}, {}, {}
    for video in videos:
        manifests[video] = read_json(Path(preparation['event_reference']) / 'inputs' / video / 'events.json')
        chunks = {key: [] for key in ready['keys']}
        local_positions = []
        for item in [r for r in ready['receipts'] if r['video'] == video]:
            path = root / video / f"{item['output']:06d}" / 'codes.npz'
            assert digest(path) == item['codes_sha256']
            with np.load(path, allow_pickle=False) as saved:
                local_positions.append(saved['positions'].copy())
                for key in chunks:
                    chunks[key].append(saved[key].copy())
        positions[video] = np.concatenate(local_positions)
        assert len(np.unique(positions[video])) == len(positions[video])
        codes[video] = {key: np.concatenate(value).astype(np.float64) for key, value in chunks.items()}
    all_rows, calibration = [], []
    for key in ready['keys']:
        for budget in config['budgets']:
            scores, matches = {}, []
            for video in videos:
                unit = codes[video][key].copy()
                if budget < unit.shape[1]:
                    keep = np.argsort(-unit, axis=1, kind='stable')[:, :budget]
                    compact = np.zeros_like(unit)
                    np.put_along_axis(compact, keep, np.take_along_axis(unit, keep, 1), 1)
                    unit = compact
                unit /= np.linalg.norm(unit, axis=1, keepdims=True)
                manifest = manifests[video]
                lookup = {int(p): i for i, p in enumerate(positions[video])}
                length = max(max(f['positions']) for f in manifest['frames'].values()) + 1
                local = {}
                for episode in manifest['episodes']:
                    source = unit[lookup[episode['source_position']]]
                    value = np.full(max(length, int(positions[video].max()) + 1), np.nan)
                    value[positions[video]] = unit @ source
                    local[episode['episode_id']] = value
                scores[video] = local
                for event in manifest['events']:
                    frame = manifest['frames'][str(event['output'])]
                    matched = next(m for m in frame['matches'] if m['lesion_id'] == event['lesion'])
                    position = frame['positions'][matched['prediction_index']]
                    matches.append(dict(video=video, label=event['same_identity'], score=local[event['episode']][position]))
            table = pd.DataFrame(matches)
            for video in videos:
                used = table[(table.video != video) & np.isfinite(table.score)]
                threshold = boundary(used.score.to_numpy(), used.video.to_numpy(), used.label.to_numpy(), .99)
                calibration.append(dict(key=key, budget=budget, video=video, threshold=threshold,
                    fit_procedures=sorted(used.video.unique().tolist()), fit_events=len(used),
                    unavailable_matches=int((~np.isfinite(table[table.video == video].score)).sum())))
                all_rows.extend(event_rows(manifests[video], scores[video], scores[video], threshold,
                                           key, f'budget{budget}'))
            print('ADAPTIVE_EVENT_EVALUATION', key, budget, 'complete', flush=True)
    events = pd.DataFrame(all_rows)
    events.to_csv(output / 'events.csv', index=False)
    rows = []
    for keys, group in events.groupby(['video', 'method', 'seed', 'variant'], sort=True):
        same, other = group[group.same_identity], group[~group.same_identity]
        first = other[other.first_prompt]
        rows.append(dict(zip(['video', 'method', 'seed', 'variant'], keys),
            repeat_removal=(~same.retained).mean(), other_retention=other.retained.mean(),
            first_retention=first.retained.mean(), repeat_events=len(same), other_events=len(other), first_events=len(first)))
    procedures = pd.DataFrame(rows)
    procedures.to_csv(output / 'procedures.csv', index=False)
    metrics = ['repeat_removal', 'other_retention', 'first_retention']
    seeds = procedures.groupby(['method', 'variant', 'seed'])[metrics].mean().reset_index()
    seeds.to_csv(output / 'seeds.csv', index=False)
    seeds.groupby(['method', 'variant'])[metrics].mean().reset_index().to_csv(output / 'summary.csv', index=False)
    atomic_write_json(output / 'calibration.json', calibration)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', events=len(events),
        procedures=len(videos), models=len(ready['keys']), encoded_summary_sha256=digest(root / 'summary.json'),
        calibration='Other-procedure 99th negative-score percentile; equal procedure weighting',
        scope='Previously selected development events; full box rematching; no continuous-time estimand', completed_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['encode', 'evaluate'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.phase == 'encode':
        encode(args.run, args.smoke)
    else:
        assert not args.smoke
        evaluate(args.run)
