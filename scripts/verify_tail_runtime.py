import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_model_port_parity as parity
import run_block10_task_dictionary as core


def arrays(path, device):
    with np.load(str(path), allow_pickle=False) as archive:
        return {key: torch.from_numpy(np.array(archive[key], copy=True)).to(device)
                for key in archive.files}


class FrozenTail(torch.nn.Module):
    def __init__(self, state_path, device):
        super(FrozenTail, self).__init__()
        module = parity.load_timesformer(str(ROOT / 'third_party/Endo-FM/models'))
        block = module.Block(768, 12, mlp_ratio=4., qkv_bias=True,
                             drop_path=.1,
                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                             attention_type='divided_space_time')
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(11)] + [block])
        self.norm = torch.nn.LayerNorm(768, eps=1e-6)
        with np.load(str(state_path), allow_pickle=False) as archive:
            state = {key: torch.from_numpy(np.array(archive[key], copy=True))
                     for key in archive.files
                     if key.startswith('blocks.11.') or key.startswith('norm.')}
        self.load_state_dict(state, strict=True)
        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, y):
        return self.norm(self.blocks[11](y, y.shape[0], 8, 14))


def load_role_sae(config, checkpoint, device):
    path = ROOT / 'results/runs/20260906T2040Z_cvc_role_sae_v1/runner.py'
    spec = importlib.util.spec_from_file_location('tail_parity_role', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ae = module.RoleSAE(config).to(device)
    ae.load_state_dict(arrays(checkpoint, device), strict=True)
    return ae.eval()


def metrics(actual, expected):
    difference = actual.astype('float64') - expected.astype('float64')
    return dict(max_abs=float(np.abs(difference).max()),
                mean_abs=float(np.abs(difference).mean()),
                relative_l2=float(np.linalg.norm(difference.ravel()) /
                                  max(np.linalg.norm(expected.ravel()), 1e-12)))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='results/runs/20260906T2305Z_mixed_presence_roles_v1/config.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the project root')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    config = json.loads(Path(args.config).read_text())
    source = Path(args.config).parent
    checkpoint = source / 'folds' / str(args.fold) / 'presence_role/model.npz'
    statistics = Path(config['source_folds'][args.fold]['source_fold']) / 'input_statistics.npz'
    stats = arrays(statistics, device)
    head = arrays(Path(config['heads']) / ('models/raw/%d/head.npz' % args.fold), device)
    normalization = arrays(config['final_normalization'], device)
    started = time.perf_counter()
    tail = FrozenTail(config['state_exchange'], device)
    ae = load_role_sae(config, checkpoint, device)
    print('LOADED tail and RoleSAE', flush=True)
    fitting = [i for i in range(28) if i not in config['folds'][args.fold]]
    cases = [('cvc', fitting[0]), ('rc', config['train_classes'].index(1)),
             ('rc', config['train_classes'].index(0))]
    outputs = {}
    records = []
    for source_name, index in cases:
        before = time.perf_counter()
        cache = config['cache'] if source_name == 'cvc' else config['new_cache']
        native = np.load(str(Path(cache) / 'block10.npy'), mmap_mode='r', allow_pickle=False)
        y = torch.from_numpy(np.array(native[index:index + 1], copy=True)).to(device)
        x = (y[0, 1:] - stats['mean']) / stats['rms']
        with np.load(str(checkpoint.parent / (source_name + '_predictions.npz')),
                     allow_pickle=False) as archive:
            saved = np.array(archive['logits'][index], copy=True)
            saved_removed = np.array(archive['ablated_logits'][index, 0], copy=True)
        query = np.argsort(saved.reshape(-1))[-4:]
        mask = torch.zeros(8 * 196, device=device)
        mask[torch.from_numpy(query.copy()).to(device)] = 1
        native_mask = mask.reshape(8, 196).transpose(0, 1).reshape(1568, 1)

        def predict(rec):
            edited = torch.cat((y[:, :1], (rec * stats['rms'] + stats['mean'])[None]), 1)
            return core.logits(config, tail, edited, head, normalization)

        ae.zero_grad()
        code = ae.encode_inference(x)
        reconstruction = ae.decode(code)
        positive = code[:, :32] @ ae.decoder.weight[:, :32].T
        factual = predict(reconstruction)
        local_removed = predict(reconstruction - positive * native_mask)
        difference = factual - local_removed
        # This scalar tests the intended local-edit backward path without fitting.
        loss = (difference.reshape(-1)[torch.from_numpy(query.copy()).to(device)] - 1).square().mean()
        loss = loss + (difference * (1 - mask.reshape(8, 196))).square().mean()
        loss.backward()
        synchronize(device)
        forward_backward_seconds = time.perf_counter() - before
        with torch.no_grad():
            global_removed = predict(reconstruction - positive)
        synchronize(device)
        label = '%s_%d' % (source_name, index)
        current = dict(code=code, reconstruction=reconstruction, factual=factual,
                       local_removed=local_removed, global_removed=global_removed,
                       encoder_gradient=ae.encoder.weight.grad,
                       decoder_gradient=ae.decoder.weight.grad,
                       decoder_bias_gradient=ae.b_dec.grad)
        for name, value in current.items():
            array = value.detach().cpu().numpy().copy()
            if not np.isfinite(array).all():
                raise RuntimeError('Nonfinite parity output: ' + label + '_' + name)
            outputs[label + '_' + name] = array
        outputs[label + '_query'] = query
        record = dict(case=label, query=query.tolist(),
                      factual_vs_saved=metrics(outputs[label + '_factual'], saved),
                      global_removed_vs_saved=metrics(outputs[label + '_global_removed'], saved_removed),
                      forward_backward_seconds=forward_backward_seconds,
                      total_seconds=time.perf_counter() - before,
                      encoder_gradient_norm=float(ae.encoder.weight.grad.norm()),
                      decoder_gradient_norm=float(ae.decoder.weight.grad.norm()))
        records.append(record)
        print('CASE', json.dumps(record), flush=True)
    np.savez(str(output / 'outputs.npz'), **outputs)
    comparisons = {}
    if args.reference:
        with np.load(str(Path(args.reference) / 'outputs.npz'), allow_pickle=False) as reference:
            if set(reference.files) != set(outputs):
                raise RuntimeError('Reference cases differ')
            for key, actual in outputs.items():
                expected = reference[key]
                item = metrics(actual, expected)
                if key.endswith('_query'):
                    item['passed'] = bool(np.array_equal(actual, expected))
                elif key.endswith('_code'):
                    item['support_equal'] = bool(np.array_equal(actual != 0, expected != 0))
                    item['passed'] = item['support_equal'] and bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4))
                elif 'gradient' in key:
                    item['passed'] = item['relative_l2'] <= 1e-3
                else:
                    item['passed'] = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4))
                comparisons[key] = item
    saved_pass = all(record[key]['max_abs'] <= 1e-4
                     for record in records for key in ['factual_vs_saved', 'global_removed_vs_saved'])
    passed = saved_pass and all(row['passed'] for row in comparisons.values())
    report = dict(status='PASS' if passed else 'FAIL', cases=records,
                  cross_runtime_comparisons=comparisons, reference=args.reference,
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  device=str(device), threads=args.threads,
                  tail_parameters=sum(p.numel() for p in tail.parameters()),
                  role_parameters=sum(p.numel() for p in ae.parameters()),
                  tail_dtype='float32', fixed_head_dtype='float64',
                  config_sha256=parity.sha256_file(args.config),
                  checkpoint_sha256=parity.sha256_file(str(checkpoint)),
                  script_sha256=parity.sha256_file(__file__),
                  seconds=time.perf_counter() - started)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COMPLETE', report['status'], report['seconds'], flush=True)
    if not passed:
        raise RuntimeError('Runtime parity failed; inspect report.json')


if __name__ == '__main__':
    main()
