import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp

from train_adaptive_identity import Observations, fit, token_batch
from train_supplemental_identity import initial_blocks, assert_arrays_equal
from train_frozen_identity_supcon import sample_batch
from src.token_identity_sae import TokenIdentitySAE
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def verify(run):
    config = read_json(run / 'config.json')
    base_config = read_json(Path(config['baseline_run']) / 'config.json')
    for key in ['steps', 'input_dim', 'latent_dim', 'top_k', 'identity_space', 'pooling', 'seeds',
                'microbatch', 'batch_procedures', 'lesions_per_procedure', 'views_per_lesion',
                'tokens_per_clip', 'learning_rate', 'backbone_learning_rate', 'weight_decay',
                'reconstruction_weight', 'temperature', 'gradient_clip_norm']:
        assert config[key] == base_config[key]
    prepared = Path(read_json(Path(config['preparation_run']) / 'config.json')['storage_root']) / 'prepared'
    receipt = read_json(prepared / 'summary.json')
    assert receipt['status'] == 'COMPLETE'
    data = Observations(prepared, [r['index'] for r in receipt['receipts']])
    assert len(data.records) == 2720
    fitting = sorted({r['video_id'] for r in data.records})
    records = [dict(r, split='train') for r in data.records]
    baseline = Path(base_config['storage_root']) / 'adaptive'
    sequences, first_batches = {}, {}
    for seed in config['seeds']:
        rng = np.random.default_rng(seed)
        trng = np.random.default_rng(np.random.SeedSequence([seed, 71005]))
        sequence = []
        for step in range(config['steps']):
            chosen = sample_batch(records, rng, config)
            tokens = [trng.integers(0, len(data.native[i]), size=config['tokens_per_clip']) for i in chosen]
            sequence.append(dict(indices=chosen, tokens_sha256=hashlib.sha256(np.stack(tokens).tobytes()).hexdigest()))
            if step == 0:
                first_batches[seed] = (chosen, tokens)
        for method in ['token_sparse', 'token_dense']:
            assert sequence == read_json(baseline / method / f'seed{seed}/foldfull/sequence.json')
        sequences[seed] = sequence
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    blocks = initial_blocks(config)
    root = run / 'smoke'
    root.mkdir(exist_ok=True)
    identity = dict(prepared_sha256=digest(prepared / 'summary.json'), config_sha256=digest(run / 'config.json'),
        sources={name: digest(name) for name in ['scripts/verify_adaptive_procedure_training.py',
            'scripts/train_adaptive_identity.py', 'src/token_identity_sae.py']})
    local = dict(config, checkpoint_every=2)
    seed = config['seeds'][0]
    reports, curves = [], []
    for method in ['token_sparse', 'token_dense']:
        folder = root / method
        result = fit(local, data, blocks, method, True, seed, fitting, [], folder, 4, identity, root / 'progress.json')
        original = root / (method + '_global')
        fit(dict(local, procedure_loss_weight=0.), data, blocks, method, True, seed, fitting, [],
            original, 4, identity, root / 'progress.json')
        actual = read_json(original / 'history.json')
        expected = read_json(baseline / method / f'seed{seed}/foldfull/history.json')[:4]
        for a, b in zip(actual, expected, strict=True):
            for key in ['identity', 'reconstruction', 'loss', 'gradient_norm', 'backbone_gradient_norm']:
                np.testing.assert_allclose(a[key], b[key], rtol=2e-6, atol=2e-7)
        assert read_json(folder / 'sequence.json') == sequences[seed][:4]
        resumed = root / (method + '_resumed')
        if not (resumed / 'checkpoint.pt').exists():
            try:
                fit(local, data, blocks, method, True, seed, fitting, [], resumed, 4, identity,
                    root / 'progress.json', stop_after=2)
            except SystemExit as stopped:
                assert stopped.code == 75
        fit(local, data, blocks, method, True, seed, fitting, [], resumed, 4, identity, root / 'progress.json')
        for name in ['model.npz', 'blocks.npz', 'normalization.npz']:
            assert_arrays_equal(folder / name, resumed / name)
        history = read_json(folder / 'history.json')
        assert history == read_json(resumed / 'history.json')
        torch.manual_seed(seed)
        model = TokenIdentitySAE(config, method).cuda()
        chosen, tokens = first_batches[seed]
        mean, scale = data.normalization(fitting)
        with torch.no_grad():
            samples = token_batch(data, chosen, tokens, blocks.cuda(), torch.from_numpy(mean).cuda(),
                                  torch.from_numpy(scale).cuda(), True, config['microbatch'])
            unit, decoded, _ = model(samples)
        vectors = unit.cpu().numpy().astype(float)
        logits = vectors @ vectors.T / config['temperature']
        labels = np.array([records[i]['lesion_id'] for i in chosen])
        videos = np.array([records[i]['video_id'] for i in chosen])
        positive = (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)
        losses = []
        for allowed in [~np.eye(len(labels), dtype=bool),
                        (videos[:, None] == videos[None, :]) & ~np.eye(len(labels), dtype=bool)]:
            logp = logits - logsumexp(np.where(allowed, logits, -np.inf), axis=1)[:, None]
            losses.append(float(-(np.where(positive, logp, 0).sum(1) / positive.sum(1)).mean()))
        np.testing.assert_allclose(losses, [history[0]['global_identity'], history[0]['procedure_identity']], atol=2e-6)
        np.testing.assert_allclose(.5 * sum(losses), history[0]['identity'], atol=2e-6)
        assert all(r['backbone_gradient_norm'] > 0 and r['within_negative_fraction'] > 0 for r in history)
        reports.append(dict(method=method, global_replay=True, resume_exact=True, numpy_losses=losses,
            actual_first=history[0], training=result))
        curves.append(history)
        del model
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for axis, method, history in zip(axes, ['SAE', 'Dense'], curves, strict=True):
        for metric in ['global_identity', 'procedure_identity', 'identity']:
            axis.plot([r['step'] for r in history], [r[metric] for r in history], marker='o', label=metric)
        axis.set_title(method)
        axis.set_xlabel('Actual optimizer update')
        axis.set_ylabel('Training objective')
        axis.legend(fontsize=8)
    figure.suptitle('Real-input objective and gradient smoke; application effect remains unmeasured')
    figure.savefig(root / 'training.png', dpi=140)
    plt.close(figure)
    atomic_write_json(root / 'verification.json', dict(status='PASS', reports=reports,
        original_sampling_checks=1200, original_observations=2720, native_bytes=sum(x.nbytes for x in data.native),
        identity=identity))
    print('PASS: two methods; NumPy objectives; original replay; exact resumed arrays; 1200 baseline batches', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
