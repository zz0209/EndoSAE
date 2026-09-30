import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from verify_tail_runtime import FrozenTail, arrays, load_role_sae


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_task(logits, counts):
    positive = counts.sum(-1, keepdim=True)
    negative = (256 - counts).sum(-1, keepdim=True)
    wp = counts / positive.clamp_min(1)
    wn = (256 - counts) / negative.clamp_min(1)
    scale = ((positive > 0).double() + (negative > 0).double()).reciprocal()
    return (scale * (wp * torch.nn.functional.softplus(-logits) +
                     wn * torch.nn.functional.softplus(logits))).sum(-1).mean()


class TrainingData:
    def __init__(self, config, device):
        self.c = config
        self.source = read(config['source_config'])
        self.device = device
        fold = config['fold']
        self.stats = arrays(Path(self.source['source_folds'][fold]['source_fold']) / 'input_statistics.npz', device)
        self.norm = arrays(self.source['final_normalization'], device)
        self.head = arrays(Path(self.source['heads']) / f'models/raw/{fold}/head.npz', device)
        self.tail = FrozenTail(self.source['state_exchange'], device)
        self.y = {name: np.load(Path(self.source[key]) / 'block10.npy', mmap_mode='r')
                  for name, key in [('cvc', 'cache'), ('rc', 'new_cache')]}
        self.counts = {
            'cvc': np.load(Path(self.source['old_cache']) / 'fine_counts.npy', mmap_mode='r'),
            'rc': np.load(Path(self.source['training_cache']) / 'masks.npy', mmap_mode='r')[self.source['train_indices']] * 256,
        }
        self.sequence = self.source['mixed_sequences'][fold]
        assert len(self.sequence) == config['steps'] == 600
        assert self.y['rc'].shape == (256, 1569, 768)
        for name, index in self.sequence:
            assert name in self.y and 0 <= index < len(self.y[name])
            assert name != 'cvc' or index not in self.source['folds'][fold]
        assert all(bool(self.counts['rc'][i].any()) == bool(label)
                   for i, label in enumerate(self.source['train_classes']))
        initial = read(Path(config['source_config']).parent / f'folds/{fold}/initial_losses.json')
        self.alpha = initial['alpha']

    def sample(self, name, index):
        y = torch.from_numpy(np.array(self.y[name][index:index + 1], copy=True)).to(self.device)
        x = (y[0, 1:] - self.stats['mean']) / self.stats['rms']
        counts = torch.from_numpy(np.array(self.counts[name][index], dtype='float64', copy=True)).to(self.device)
        return x, y[:, :1], counts

    def logits(self, cls, reconstruction):
        patches = reconstruction * self.stats['rms'] + self.stats['mean']
        y = torch.cat((cls, patches[None]), dim=1)
        feature = self.tail(y)[0, 1:].reshape(196, 8, 768).permute(1, 0, 2)
        normalized = ((feature - self.norm['mean']) / self.norm['rms']).double()
        return (normalized - self.head['feature_mean']) / self.head['feature_scale'] @ self.head['weight'] + self.head['bias']

    def objective(self, ae, name, index, method):
        x, cls, counts = self.sample(name, index)
        code = ae.encode_inference(x)
        reconstructed = ae.decode(code)
        prefix = code[:, :self.c['prefix_size']] @ ae.decoder.weight[:, :self.c['prefix_size']].T + ae.b_dec
        logits = self.logits(cls, reconstructed)
        task = balanced_task(logits, counts)
        full = (reconstructed - x).square().mean()
        prefix_error = (prefix - x).square().mean()
        reconstruction = full if method == 'control' else .5 * (full + prefix_error)
        loss = reconstruction + self.alpha * task
        return loss, dict(full_mse=full, prefix_mse=prefix_error, task=task), logits


def evaluate_reconstruction(data, ae, destination, smoke):
    held = data.source['folds'][data.c['fold']]
    if smoke:
        held = held[:1]
    records = []
    predictions = []
    with torch.no_grad():
        for index in held:
            x, cls, counts = data.sample('cvc', index)
            z = ae.encode_inference(x)
            full = ae.decode(z)
            size = data.c['prefix_size']
            prefix = z[:, :size] @ ae.decoder.weight[:, :size].T + ae.b_dec
            outputs = [data.logits(cls, representation) for representation in [x, full, prefix]]
            records.append(dict(group=index, full_mse=float((full - x).square().mean()),
                                prefix_mse=float((prefix - x).square().mean()),
                                suffix_l0=float((z[:, size:] > 0).sum(-1).float().mean()),
                                suffix_rms=float((full - prefix).square().mean().sqrt()),
                                suffix_output_rms=float((outputs[1] - outputs[2]).square().mean().sqrt()),
                                **{name + '_task': float(balanced_task(value, counts))
                                   for name, value in zip(['native', 'full', 'prefix'], outputs)}))
            predictions.append(torch.stack(outputs).cpu().numpy())
    np.savez(destination / 'cvc_reconstruction.npz', groups=np.array(held),
             logits=np.array(predictions), consumers=np.array(['native', 'full', 'prefix']))
    summary = dict(scope='Previously exposed held CVC groups; read-only reconstruction characterization',
                   per_group=records, mean={key: float(np.mean([row[key] for row in records]))
                                            for key in records[0] if key != 'group'})
    put(destination / 'cvc_reconstruction.json', summary)
    return summary['mean']


def training_code_diagnostic(data, ae, destination, stage):
    with np.load(data.c['training_queries'], allow_pickle=False) as saved:
        x = torch.from_numpy(np.array(saved['raw'], copy=True).reshape(-1, 768)).to(data.device)
        scores = saved['factual'].reshape(-1)
        labels = saved['labels'].reshape(-1).astype(bool)
    alarms = scores >= data.c['diagnostic_alarm_threshold']
    output = {}
    numeric = {}
    with torch.no_grad():
        z = ae.encode_inference(x)
        start = data.c['prefix_size']
        suffix = z[:, start:] @ ae.decoder.weight[:, start:].T
        for name, selected in [('all_alerts', alarms), ('correct_alerts', alarms & labels),
                               ('incorrect_alerts', alarms & ~labels)]:
            selected = torch.from_numpy(selected).to(data.device)
            code = z[selected, start:]
            contribution = suffix[selected]
            frequency = (code > 0).float().mean(0)
            mean = contribution.mean(0)
            energy = contribution.square().sum(-1).mean()
            frequent = frequency >= .95
            output[name] = dict(count=int(selected.sum()), suffix_l0=float((code > 0).sum(-1).float().mean()),
                suffix_mean_norm=float(mean.norm()), suffix_rms=float(energy.sqrt() / math.sqrt(768)),
                suffix_mean_energy_fraction=float(mean.square().sum() / energy.clamp_min(1e-12)),
                suffix_channels_active_at_least95pct=int(frequent.sum()),
                fraction_nonzeros_from_frequent_channels=float((code[:, frequent] > 0).sum() /
                                                               (code > 0).sum().clamp_min(1)))
            numeric[name + '_frequency'] = frequency.cpu().numpy()
            numeric[name + '_mean_activation'] = code.mean(0).cpu().numpy()
    np.savez(destination / ('training_codes_' + stage + '.npz'), **numeric)
    put(destination / ('training_codes_' + stage + '.json'), output)
    return output


def train_method(data, output, method, smoke, resume):
    c = data.c
    destination = output / method
    destination.mkdir(exist_ok=resume)
    if resume and (destination / 'receipt.json').exists():
        return read(destination / 'receipt.json')
    torch.manual_seed(c['seed'])
    ae = load_role_sae(data.source, c['initial_checkpoint'], data.device)
    if not (destination / 'training_codes_before.json').exists():
        training_code_diagnostic(data, ae, destination, 'before')
    optimizer = torch.optim.AdamW(ae.parameters(), lr=c['learning_rate'], weight_decay=0.)
    sequence = data.sequence
    if smoke:
        sequence = [sequence[0], ['rc', data.source['train_classes'].index(0)],
                    ['rc', data.source['train_classes'].index(1)]]
    start_step = 0
    history = []
    checkpoint = destination / 'checkpoint.pt'
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=data.device, weights_only=True)
        assert saved['config_sha256'] == sha(output / 'config.json')
        ae.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step']
        history = saved['history']
    started = time.perf_counter()
    initial_parity = []
    if smoke:
        with torch.no_grad():
            for source, index in sequence:
                logits = data.objective(ae, source, index, method)[2]
                archived = Path(c['initial_checkpoint']).parent / (source + '_predictions.npz')
                with np.load(archived, allow_pickle=False) as saved:
                    expected = saved['logits'][index]
                error = float(np.max(np.abs(logits.cpu().numpy() - expected)))
                assert error < c['smoke_logit_tolerance'], (source, index, error)
                initial_parity.append(dict(source=source, index=index, max_logit_error=error))
    for step in range(start_step, len(sequence)):
        name, index = sequence[step]
        loss, terms, _ = data.objective(ae, name, index, method)
        optimizer.zero_grad()
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in ae.parameters())
        ae.project_decoder_gradient_()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.)
        rate = c['learning_rate'] * .5 * (1 + math.cos(math.pi * step / (c['steps'] - 1)))
        optimizer.param_groups[0]['lr'] = rate
        optimizer.step()
        ae.normalize_decoder_()
        history.append(dict(step=step + 1, source=name, index=index, learning_rate=rate,
                            loss=float(loss.detach()), **{key: float(value.detach()) for key, value in terms.items()}))
        if smoke or (step + 1) % c['checkpoint_interval'] == 0 or step + 1 == len(sequence):
            torch.save(dict(model=ae.state_dict(), optimizer=optimizer.state_dict(), step=step + 1,
                            history=history, config_sha256=sha(output / 'config.json')), checkpoint)
            put(destination / 'training.json', history)
            put(output / 'progress.json', dict(method=method, step=step + 1, total=len(sequence),
                                              seconds=time.perf_counter() - started))
            print('TRAIN', method, step + 1, '/', len(sequence), history[-1], flush=True)
    np.savez(destination / 'model.npz', **{key: value.detach().cpu().numpy() for key, value in ae.state_dict().items()})
    restored = load_role_sae(data.source, destination / 'model.npz', data.device)
    with torch.no_grad():
        reference = data.objective(ae, *sequence[-1], method)[2]
        replay = data.objective(restored, *sequence[-1], method)[2]
    replay_error = float((reference - replay).abs().max())
    assert replay_error == 0
    reconstruction = evaluate_reconstruction(data, restored, destination, smoke)
    code_diagnostic = training_code_diagnostic(data, restored, destination, 'after')
    receipt = dict(status='SMOKE_PASSED' if smoke else 'TRAINED', method=method,
                   steps=len(sequence), alpha=data.alpha, prefix_size=c['prefix_size'],
                   seconds=time.perf_counter() - started, initial_parity=initial_parity,
                   replay_error=replay_error, model_sha256=sha(destination / 'model.npz'),
                   held_cvc_reconstruction=reconstruction,
                   training_code_diagnostic=code_diagnostic,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    put(destination / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = read(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        assert sha(args.config) == sha(output / 'config.json')
    else:
        shutil.copyfile(args.config, output / 'config.json')
        shutil.copyfile(__file__, output / 'source.py')
    data = TrainingData(c, torch.device(args.device))
    put(output / 'execution.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
         python=sys.version, torch=torch.__version__, numpy=np.__version__, device=args.device,
         config_sha256=sha(args.config), script_sha256=sha(__file__),
         source_config_sha256=sha(c['source_config']), initial_checkpoint_sha256=sha(c['initial_checkpoint']),
         sequence=data.sequence, alpha=data.alpha, smoke=args.smoke,
         feedback_eligible_feature_start=c['prefix_size']))
    receipts = [train_method(data, output, method, args.smoke, args.resume) for method in c['methods']]
    put(output / 'summary.json', dict(status='SMOKE_PASSED' if args.smoke else 'TRAINED', methods=receipts,
                                    max_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0))


if __name__ == '__main__':
    main()
