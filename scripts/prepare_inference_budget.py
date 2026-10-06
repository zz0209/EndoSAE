import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import evaluate_source_component_transfer as transfer
from train_temporal_view_identity import read_tokens
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest


@torch.no_grad()
def observations(raw, offsets, indices, model, mean, scale, top_k, bank, alternative_model=None):
    original, alternative, diagnostics = [], [], []
    for index in indices:
        normalized = ((raw[offsets[index]:offsets[index + 1]].astype(float) - mean) / scale).astype(np.float32)
        values = torch.from_numpy(normalized[None]).cuda()
        native_vector, _, native_codes = model(values)
        alternate, alternate_values = model, values
        if alternative_model is not None:
            alternate, other_mean, other_scale = alternative_model
            alternate_values = torch.from_numpy(((raw[offsets[index]:offsets[index + 1]].astype(float) - other_mean) / other_scale).astype(np.float32)[None]).cuda()
        changed = transfer.budget_codes(alternate_values, alternate, top_k)
        vector = F.normalize(changed.mean(1)[0], dim=0)
        assert torch.isfinite(vector).all() and vector.norm() > 0
        full = F.relu(alternate.encoder(alternate_values))
        assert torch.all(changed <= full)
        energy = values.square().sum().item()
        diagnostics.append(dict(tokens=len(normalized), native_l0=float((native_codes > 0).sum().item() / len(normalized)),
            changed_l0=float((changed > 0).sum().item() / len(normalized)),
            native_nmse=float((model.decoder(native_codes) - values).square().sum().item() / energy),
            changed_nmse=float((alternate.decoder(changed) - alternate_values).square().sum().item() / alternate_values.square().sum().item())))
        if bank:
            native = native_codes.mean(1)[0].cpu().numpy().astype(float)
            changed_vector = changed.mean(1)[0].cpu().numpy().astype(float)
            original.append(native / np.linalg.norm(native))
            alternative.append(changed_vector / np.linalg.norm(changed_vector))
        else:
            original.append(native_vector[0].cpu().numpy().astype(float))
            alternative.append(vector.cpu().numpy().astype(float))
    return np.asarray(original), np.asarray(alternative), diagnostics


def prepare(run, resume):
    config, capacity, event, _, original = transfer.inputs(run)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = transfer.load_models(original, torch.device(config['device']))
    alternatives = transfer.load_models(dict(original, training_runs=config['alternative_training_runs']), torch.device(config['device'])) if 'alternative_training_runs' in config else {}
    old_bank = Path(config['native_bank_run']) / 'bank'
    bank_receipt = read_json(old_bank / 'summary.json')
    records = bank_receipt['records']
    assert len(records) == 340 and all(r['split'] == 'train' for r in records)
    training = read_json(Path(config['training_run']) / 'config.json')
    root = run / 'preparation'
    root.mkdir(exist_ok=True)
    (run / 'bank').mkdir(exist_ok=True)
    identity = {str(p): digest(p) for p in [run / 'config.json', run / 'protocol.json', Path(__file__),
                Path(transfer.__file__), old_bank / 'summary.json']}
    for key, folder in transfer.model_specs(original):
        for name in ['model.npz', 'normalization.npz', 'model_config.json']:
            identity[str(folder / name)] = digest(folder / name)
    if alternatives:
        for key, folder in transfer.model_specs(dict(original, training_runs=config['alternative_training_runs'])):
            for name in ['model.npz', 'normalization.npz', 'model_config.json']:
                identity[str(folder / name)] = digest(folder / name)
    begin, completed, receipts = time.perf_counter(), 0, []
    training_videos = sorted({r['video_id'] for r in records})
    total = (len(training_videos) + len(original['development_videos'])) * len(models)
    for stage, videos in [('bank', training_videos), ('events', original['development_videos'])]:
        for video in videos:
            if stage == 'bank':
                token_root = training['added_training_storage'] if video in training['added_training_videos'] else training['original_prepared_run']
                directory = Path(token_root) / 'tokens' / video
                raw, offsets, local, _ = read_tokens(directory)
                selected = [i for i, r in enumerate(records) if r['video_id'] == video]
                indices = [next(i for i, candidate in enumerate(local) if
                    (candidate['clip_id'], candidate['lesion_id']) == (records[j]['clip_id'], records[j]['lesion_id'])) for j in selected]
                for i, j in zip(indices, selected):
                    assert all(local[i][field] == records[j][field] for field in records[j])
                positions = np.array(selected)
            else:
                directory = event / 'inputs' / video
                receipt = read_json(directory / 'complete.json')
                assert digest(directory / 'tokens.npz') == receipt['assets']['tokens.npz']
                with np.load(directory / 'tokens.npz') as saved:
                    raw, offsets, positions = saved['tokens'], saved['offsets'], saved['positions']
                indices = range(len(positions))
            for key, (model, mean, scale) in models.items():
                folder = root / stage / video / key
                folder.mkdir(parents=True, exist_ok=True)
                local_identity = dict(identity, token_receipt=digest(directory / 'complete.json'))
                if (folder / 'complete.json').exists():
                    receipt = read_json(folder / 'complete.json')
                    assert resume and receipt['identity'] == local_identity
                    assert digest(folder / 'vectors.npz') == receipt['vectors_sha256']
                else:
                    native, changed, diagnostics = observations(raw, offsets, indices, model, mean, scale,
                                                               config['inference_top_k'][model.method], stage == 'bank', alternatives.get(key))
                    reference = old_bank / (key + '.npz') if stage == 'bank' else capacity / 'evaluation' / video / key / 'effects.npz'
                    with np.load(reference) as saved:
                        if stage == 'bank':
                            np.testing.assert_array_equal(native, saved['vectors'][selected])
                        else:
                            np.testing.assert_array_equal(positions, saved['positions'])
                            np.testing.assert_array_equal(native, saved['unit_codes'])
                    np.savez_compressed(folder / 'vectors.npz', positions=positions, unit_codes=changed)
                    receipt = dict(status='COMPLETE', identity=local_identity, observations=len(indices),
                        native_exact=True, diagnostics=diagnostics, vectors_sha256=digest(folder / 'vectors.npz'))
                    atomic_write_json(folder / 'complete.json', receipt)
                receipts.append(dict(stage=stage, video=video, model=key, **receipt))
                completed += 1
                atomic_write_json(root / 'progress.json', dict(completed=completed, total=total))
                print('BUDGET_PREPARE', completed, '/', total, stage, video, key, flush=True)
                pause_after_checkpoint(root / 'progress.json')
    for key in models:
        vectors = np.empty((len(records), models[key][0].latent_dim))
        for video in training_videos:
            with np.load(root / 'bank' / video / key / 'vectors.npz') as saved:
                vectors[saved['positions']] = saved['unit_codes']
        with np.load(old_bank / (key + '.npz')) as saved:
            weights = saved['weights']
        target = run / 'bank' / (key + '.npz')
        if target.exists():
            assert resume
            with np.load(target) as saved:
                np.testing.assert_array_equal(saved['vectors'], vectors)
        else:
            np.savez_compressed(target, vectors=vectors, weights=weights)
        for video in original['development_videos']:
            target = Path(config['fitting_vectors_root']) / video / (key + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            source = root / 'events' / video / key / 'vectors.npz'
            if target.exists():
                assert resume and digest(target) == digest(source)
            else:
                import shutil
                shutil.copyfile(source, target)
    atomic_write_json(run / 'bank/summary.json', dict(status='COMPLETE', records=records, observations=340,
        files={p.name: digest(p) for p in (run / 'bank').glob('*.npz')}, identity=identity))
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', receipts=receipts, seconds=time.perf_counter() - begin,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare(args.run, args.resume)
