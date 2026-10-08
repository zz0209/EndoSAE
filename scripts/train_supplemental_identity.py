import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import collections
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
from train_adaptive_identity import Observations, fit
from train_frozen_identity_supcon import sample_batch
from train_acknowledgement_sae import json_digest, now
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


class CombinedObservations:
    def __init__(self, real, supplemental):
        self.real, self.supplemental = real, supplemental
        self.records = list(real.records) + [dict(r, lesion_id=r['video_id'] + ':source_lesion', split='train')
                                             for r in supplemental.records]
        self.native = real.native + supplemental.native
        self.positions = real.positions + supplemental.positions
        self.addresses = [(real.prepared, r['cache_index']) for r in real.records]
        self.addresses += [(supplemental.prepared, r['cache_index']) for r in supplemental.records]
        self.statistics = real.normalization({r['video_id'] for r in real.records})

    def prefix(self, indices):
        values = [np.load(root / 'observations' / f'{index:04d}' / 'prefix.npy', allow_pickle=False)
                  for root, index in [self.addresses[i] for i in indices]]
        return torch.from_numpy(np.stack(values)).to('cuda')

    def normalization(self, videos):
        assert {r['video_id'] for r in self.real.records} <= set(videos)
        return self.statistics


def load_data(config):
    paths, receipts, parts = [], [], []
    for key in ['preparation_run', 'supplemental_preparation_run']:
        prepared = Path(read_json(Path(config[key]) / 'config.json')['storage_root']) / 'prepared'
        receipt = read_json(prepared / 'summary.json')
        assert receipt['status'] == 'COMPLETE'
        indices = [r['index'] for r in receipt['receipts']]
        assert indices == list(range(len(indices)))
        for r in receipt['receipts']:
            folder = prepared / 'observations' / f"{r['index']:04d}"
            assert digest(folder / 'roi.npz') == r['roi_sha256']
            assert (folder / 'prefix.npy').stat().st_size == 1569 * 768 * 4 + 128
        paths.append(prepared)
        receipts.append(digest(prepared / 'summary.json'))
        parts.append(Observations(prepared, indices))
    assert len(parts[0].records) == 2720 and len(parts[1].records) == 1088
    combined = CombinedObservations(*parts)
    assert len({(r['video_id'], r['lesion_id']) for r in combined.records}) == 119
    assert len({r['video_id'] for r in parts[1].records}) == 34
    assert not {r['video_id'] for r in parts[0].records} & {r['video_id'] for r in parts[1].records}
    return combined, dict(prepared=[dict(path=str(p), summary_sha256=s) for p, s in zip(paths, receipts)])


def schedule(config, data, seed, baseline):
    records = [dict(r, split='train') for r in data.real.records]
    rng = np.random.default_rng(seed)
    token_rng = np.random.default_rng(np.random.SeedSequence([seed, 71005]))
    supplemental_rng = np.random.default_rng(np.random.SeedSequence([seed, 71007]))
    supplemental_token_rng = np.random.default_rng(np.random.SeedSequence([seed, 71008]))
    available = collections.defaultdict(list)
    for index, r in enumerate(data.records[len(records):], len(records)):
        available[r['lesion_id']].append(index)
    identities = sorted(available)
    original_sequence = read_json(baseline / 'sequence.json')
    assert len(original_sequence) == config['steps']
    output, counts = [], []
    for step in range(1, config['steps'] + 1):
        original = sample_batch(records, rng, config)
        tokens = [token_rng.integers(0, len(data.native[i]), size=config['tokens_per_clip']) for i in original]
        assert original_sequence[step - 1] == dict(indices=original,
            tokens_sha256=hashlib.sha256(np.stack(tokens).tobytes()).hexdigest())
        original_counts = collections.Counter(data.records[i]['lesion_id'] for i in original)
        assert set(original_counts.values()) == {config['views_per_lesion']}
        chosen = original
        replace = step % config['replace_every'] == 0
        if replace:
            selected_ids = supplemental_rng.choice(identities, len(original_counts), replace=False)
            chosen = [int(i) for identity in selected_ids for i in supplemental_rng.choice(
                available[identity], config['views_per_lesion'], replace=False)]
            tokens = [supplemental_token_rng.integers(0, len(data.native[i]), size=config['tokens_per_clip']) for i in chosen]
            assert len(chosen) == len(original)
        assert len(chosen) == len(set(chosen))
        count = collections.Counter(data.records[i]['lesion_id'] for i in chosen)
        assert set(count.values()) == {config['views_per_lesion']} and len(count) == len(original_counts)
        output.append(dict(indices=chosen, tokens=[t.tolist() for t in tokens], supplemental=replace))
        counts.append(dict(step=step, observations=len(chosen), identities=len(count), supplemental=replace))
    assert sum(row['supplemental'] for row in output) == 100
    return output, counts


def initial_blocks(config):
    original = read_json(Path(config['preparation_run']) / 'config.json')
    encoder = Encoder(read_json(original['encoding_config']), torch.device('cuda'))
    blocks = copy.deepcopy(encoder.model.blocks[9:11]).cpu()
    del encoder
    torch.cuda.empty_cache()
    return blocks


def assert_arrays_equal(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def run_training(run, phase):
    config = read_json(run / 'config.json')
    baseline_config = read_json(Path(config['baseline_run']) / 'config.json')
    for key in ['steps', 'input_dim', 'latent_dim', 'top_k', 'readout_dim', 'identity_space', 'pooling',
                'seeds', 'microbatch', 'batch_procedures', 'lesions_per_procedure', 'views_per_lesion',
                'tokens_per_clip', 'learning_rate', 'backbone_learning_rate', 'weight_decay',
                'reconstruction_weight', 'temperature', 'gradient_clip_norm']:
        assert config[key] == baseline_config[key]
    data, identity = load_data(config)
    identity['sources'] = {name: digest(ROOT / name) for name in ['scripts/train_supplemental_identity.py',
        'scripts/train_adaptive_identity.py', 'scripts/train_frozen_identity_supcon.py',
        'scripts/train_acknowledgement_sae.py', 'src/token_identity_sae.py']}
    identity['config_sha256'] = digest(run / 'config.json')
    base = Path(baseline_config['storage_root']) / 'adaptive'
    (run / 'sampling').mkdir(exist_ok=True)
    schedules = {}
    for seed in config['seeds']:
        original = base / 'token_sparse' / f'seed{seed}' / 'foldfull'
        planned, counts = schedule(config, data, seed, original)
        schedules[seed] = planned
        with np.load(original / 'normalization.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['mean'], data.statistics[0])
            np.testing.assert_array_equal(saved['scale'], data.statistics[1])
        reference_hash = digest(original / 'sequence.json')
        assert reference_hash == digest(base / 'token_dense' / f'seed{seed}' / 'foldfull' / 'sequence.json')
        receipt = dict(seed=seed, baseline_sequence_sha256=reference_hash, schedule_sha256=json_digest(planned),
            original_batch_checks=config['steps'], unchanged_real_updates=300, supplemental_updates=100, counts=counts)
        atomic_write_json(run / 'sampling' / f'{seed}.json', receipt)
    identity['schedules'] = {str(seed): json_digest(value) for seed, value in schedules.items()}
    atomic_write_json(run / 'input_verification.json', dict(status='PASS', identity=identity,
        observations=len(data.records), real_identities=85, supplemental_source_identities=34,
        original_sequence_checks=1200, native_bytes=sum(v.nbytes for v in data.native)))
    print('INPUTS_PASS', len(data.records), 'observations; 1200 original batch/token checks', flush=True)
    if phase == 'preflight':
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    blocks = initial_blocks(config)
    fitting = sorted({r['video_id'] for r in data.records})
    outputs = []
    smoke = phase == 'smoke'
    root = run / 'smoke' if smoke else Path(config['storage_root'])
    root.mkdir(parents=True, exist_ok=True)
    local = dict(config, checkpoint_every=4) if smoke else config
    seeds = config['seeds'][:1] if smoke else config['seeds']
    steps = config['smoke_steps'] if smoke else config['steps']
    for seed in seeds:
        for method in ['token_sparse', 'token_dense']:
            folder = root / 'adaptive' / method / f'seed{seed}' / 'foldfull'
            result = fit(local, data, blocks, method, True, seed, fitting, [], folder, steps, identity,
                         root / 'training_progress.json', batch_schedule=schedules[seed])
            if smoke:
                old_history = read_json(base / method / f'seed{seed}' / 'foldfull' / 'history.json')
                new_history = read_json(folder / 'history.json')
                for actual, expected in zip(new_history[:3], old_history[:3], strict=True):
                    for key in ['loss', 'identity', 'reconstruction', 'gradient_norm', 'backbone_gradient_norm']:
                        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-7)
                resumed = root / 'resumed' / method
                if not (resumed / 'checkpoint.pt').exists():
                    try:
                        fit(local, data, blocks, method, True, seed, fitting, [], resumed, steps, identity,
                            root / 'resume_progress.json', stop_after=4, batch_schedule=schedules[seed])
                    except SystemExit as stopped:
                        assert stopped.code == 75
                fit(local, data, blocks, method, True, seed, fitting, [], resumed, steps, identity,
                    root / 'resume_progress.json', batch_schedule=schedules[seed])
                for name in ['model.npz', 'blocks.npz', 'normalization.npz']:
                    assert_arrays_equal(folder / name, resumed / name)
                assert read_json(folder / 'history.json') == read_json(resumed / 'history.json')
                assert read_json(folder / 'sequence.json') == read_json(resumed / 'sequence.json')
            outputs.append(dict(method=method, adaptive=True, seed=seed, fold=0, fit_videos=fitting,
                                held_videos=[], directory=str(folder), summary=result))
            atomic_write_json(root / 'progress.json', dict(status='RUNNING', completed=len(outputs), total=len(seeds) * 2))
    for seed in seeds:
        assert len({item['summary']['sequence_sha256'] for item in outputs if item['seed'] == seed}) == 1
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', outputs=outputs, identity=identity,
        paired_sampling_exact=True, smoke=smoke, baseline_initial_three_steps_verified=smoke,
        resume_exact=smoke, completed_at=now()))
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=len(outputs), total=len(outputs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase', choices=['preflight', 'smoke', 'train'], required=True)
    args = parser.parse_args()
    run_training(args.run, args.phase)
