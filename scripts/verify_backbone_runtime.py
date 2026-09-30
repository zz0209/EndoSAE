import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run_model_port_parity as parity


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def slice_array(path, index):
    return np.array(np.load(str(path), mmap_mode='r', allow_pickle=False)[index], copy=True)


def digest_array(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def head_arrays(config):
    with np.load(config['final_normalization'], allow_pickle=False) as archive:
        normalization = {key: archive[key].copy() for key in ('mean', 'rms')}
    heads = []
    for fold in range(4):
        path = Path(config['heads']) / ('models/raw/%d/head.npz' % fold)
        with np.load(str(path), allow_pickle=False) as archive:
            heads.append({key: archive[key].copy() for key in archive.files})
    weight = np.stack([h['weight'] / h['feature_scale'] for h in heads]).mean(0)
    bias = np.stack([h['bias'] - (h['feature_mean'] * h['weight'] / h['feature_scale']).sum()
                     for h in heads]).mean(0)
    return normalization, weight, bias


def prepare(config_path, output):
    config = read(config_path)
    normalization, weight, bias = head_arrays(config)
    cases = [('cvc_0', config['old_cache'], 0, config['cache'], 0)]
    for label, value in [('real_positive', 1), ('real_negative', 0)]:
        index = config['train_classes'].index(value)
        cases.append((label, config['training_cache'], config['train_indices'][index],
                      config['new_cache'], index))
    payload = dict(normalization_mean=normalization['mean'], normalization_rms=normalization['rms'],
                   head_weight=weight, head_bias=bias)
    records = []
    for label, cache, index, block_cache, block_index in cases:
        cache = Path(cache)
        inputs = slice_array(cache / 'inputs.npy', index)
        block10 = slice_array(Path(block_cache) / 'block10.npy', block_index)
        tokens = slice_array(cache / 'tokens.npy', index)
        cls = slice_array(cache / 'cls.npy', index).reshape(1, 768)
        final = np.concatenate((cls, tokens.transpose(1, 0, 2).reshape(1568, 768)), axis=0)
        normalized = (tokens - normalization['mean']) / normalization['rms']
        logits = normalized.astype(np.float64) @ weight + bias
        assert inputs.shape == (3, 8, 224, 224)
        assert block10.shape == final.shape == (1569, 768)
        assert tokens.shape == (8, 196, 768) and logits.shape == (8, 196)
        arrays = dict(input=inputs, block10=block10, final=final, logits=logits)
        for name, value in arrays.items():
            assert np.isfinite(value).all(), (label, name)
            payload[label + '_' + name] = value
        records.append(dict(case=label, input_cache=str(cache), input_index=index,
                            block10_cache=str(block_cache), block10_index=block_index,
                            array_sha256={name: digest_array(value) for name, value in arrays.items()}))
        print('PREPARED', label, 'input_index', index, 'block10_index', block_index, flush=True)
    np.savez(str(output / 'fixtures.npz'), **payload)
    identity_paths = [Path(config_path), Path(config['final_normalization']), Path(__file__),
                      ROOT / 'scripts/run_model_port_parity.py',
                      ROOT / 'results/runs/20260906T1025Z_cvc_block10_nonlinear_sae_v1/capture_receipt.json',
                      ROOT / 'results/runs/20260906T2305Z_mixed_presence_roles_v1/capture_receipt.json']
    identity_paths.extend(Path(config['heads']) / ('models/raw/%d/head.npz' % fold) for fold in range(4))
    save(output / 'fixtures.json', dict(
        status='PREPARED', cases=records, config=str(Path(config_path).resolve()),
        state_exchange=config['state_exchange'], state_exchange_sha256=config['state_exchange_sha256'],
        fixture_sha256=parity.sha256_file(str(output / 'fixtures.npz')),
        identity={str(path): parity.sha256_file(str(path)) for path in identity_paths},
        reference='Saved LR1 CPU block10 and final tokens; fixed four-head mean evaluated on saved final tokens',
        criteria=parity.CRITERIA))


def verify(fixtures, output, device_name, threads):
    manifest = read(fixtures / 'fixtures.json')
    assert parity.sha256_file(str(fixtures / 'fixtures.npz')) == manifest['fixture_sha256']
    assert parity.sha256_file(manifest['state_exchange']) == manifest['state_exchange_sha256']
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    module = parity.load_timesformer(str(ROOT / 'third_party/Endo-FM/models'))
    model = parity.build_model(torch, module)
    model.load_state_dict(parity.load_state_exchange(manifest['state_exchange'], torch, np), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = {}
    handle = model.blocks[10].register_forward_hook(
        lambda _module, _inputs, value: captured.update(block10=value.detach()))
    arrays = {}
    records = []
    with np.load(str(fixtures / 'fixtures.npz'), allow_pickle=False) as archive, torch.no_grad():
        mean = torch.from_numpy(archive['normalization_mean'].copy()).to(device)
        rms = torch.from_numpy(archive['normalization_rms'].copy()).to(device)
        weight = torch.from_numpy(archive['head_weight'].copy()).to(device)
        bias = torch.from_numpy(archive['head_bias'].copy()).to(device)
        for case in manifest['cases']:
            label = case['case']
            value = torch.from_numpy(archive[label + '_input'].copy()[None]).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            final = model.forward_features(value, get_all=True)
            tokens = final[0, 1:].reshape(196, 8, 768).permute(1, 0, 2)
            logits = ((tokens - mean) / rms).double() @ weight + bias
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            current = dict(block10=captured['block10'][0], final=final[0], logits=logits)
            comparisons = {}
            for name, tensor in current.items():
                actual = tensor.cpu().numpy().copy()
                expected = archive[label + '_' + name]
                assert actual.shape == expected.shape
                assert np.isfinite(actual).all(), (label, name)
                arrays[label + '_' + name] = actual
                metrics = parity.array_error_metrics(expected, actual, np)
                metrics['passed'] = parity.metrics_pass(metrics)
                comparisons[name] = metrics
            actual_query = arrays[label + '_logits'].argmax(axis=1)
            expected_query = archive[label + '_logits'].argmax(axis=1)
            record = dict(case=label, comparisons=comparisons,
                          top1_query_equal=bool(np.array_equal(actual_query, expected_query)),
                          forward_seconds=elapsed)
            records.append(record)
            print('CASE', json.dumps(record), flush=True)
    handle.remove()
    passed = all(row['passed'] for record in records for row in record['comparisons'].values())
    np.savez(str(output / 'outputs.npz'), **arrays)
    report = dict(status='PASS' if passed else 'FAIL', cases=records, criteria=parity.CRITERIA,
                  device=str(device), torch=torch.__version__, python=platform.python_version(),
                  numpy=np.__version__, threads=threads, backbone_dtype='float32', head_dtype='float64',
                  tf32=False if device.type == 'cuda' else None,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  fixtures=str(fixtures.resolve()), fixture_sha256=manifest['fixture_sha256'],
                  state_exchange_sha256=manifest['state_exchange_sha256'],
                  script_sha256=parity.sha256_file(__file__), seconds=time.perf_counter() - started)
    save(output / 'report.json', report)
    print('COMPLETE', report['status'], report['seconds'], flush=True)
    if not passed:
        raise RuntimeError('Backbone parity failed; inspect report.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='results/runs/20260906T2305Z_mixed_presence_roles_v1/config.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the project root')
    if not args.prepare_only and args.fixtures is None:
        parser.error('--fixtures is required for runtime comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepare_only:
        prepare(args.config, args.output)
    else:
        verify(args.fixtures, args.output, args.device, args.threads)


if __name__ == '__main__':
    main()
