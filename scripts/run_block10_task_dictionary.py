"""Train and replay configured block-10 dictionary/task controls in LR1.

The configuration supplies local, authorized caches, frozen model identity,
fitting groups, initialization assets and the fixed optimizer schedule.
No data, checkpoint or experiment configuration is bundled with this source.
This packages the executed AY local-MSE/output-fidelity/true-mask objectives;
generic TopK, end-to-end and discriminative dictionary training are prior art.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import run_model_port_parity as parity

def arr(path):
    with np.load(str(path), allow_pickle=False) as f:
        return {k: torch.from_numpy(np.array(f[k], copy=True)) for k in f.files}

def model(c):
    assert sys.version_info[:2] == (3, 7) and torch.__version__.startswith('1.8.0')
    torch.set_num_threads(c['cpu_threads'])
    torch.set_num_interop_threads(1)
    assert parity.sha256_file(c['state_exchange']) == c['state_exchange_sha256']
    m = parity.build_model(torch, parity.load_timesformer(str(ROOT / 'third_party/Endo-FM/models')))
    m.load_state_dict(parity.load_state_exchange(c['state_exchange'], torch, np), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def frames(y):
    return y[0, 1:].reshape(196, 8, 768).permute(1, 0, 2)

def tail(m, y):
    return m.norm(m.blocks[11](y, 1, 8, 14))

def logits(c, m, y, h, norm):
    f = frames(tail(m, y))
    x = (f - norm['mean']) / norm['rms']
    x = x.double()
    return (x - h['feature_mean']) / h['feature_scale'] @ h['weight'] + h['bias']

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
import torch
from src.sae.baselines import TopKAutoencoder

class Rank48(torch.nn.Module):

    def __init__(self, basis):
        super(Rank48, self).__init__()
        self.encoder = torch.nn.Linear(768, 48)
        self.decoder = torch.nn.Linear(48, 768)
        with torch.no_grad():
            self.encoder.weight.copy_(basis.T)
            self.decoder.weight.copy_(basis)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    def encode_inference(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def fresh(name, basis=None):
    return Rank48(basis) if name == 'rank48_task' else TopKAutoencoder(768, 1536, 32)

def residual_mlp():
    m = torch.nn.Sequential(torch.nn.Linear(768, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    torch.nn.init.zeros_(m[-1].weight)
    torch.nn.init.zeros_(m[-1].bias)
    return m
endofm = model

def run(run_directory):
    import os, json, time, math, hashlib
    from pathlib import Path
    import numpy as np
    import torch
    r = Path(run_directory).resolve()
    c = json.loads((r / 'fit_config.json').read_text())
    start = time.time()
    source = Path(c['source_fold'])
    for p, hsh in c['asset_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == hsh, p
    m = endofm(c)
    y = torch.from_numpy(np.load(str(Path(c['cache']) / 'block10.npy'), allow_pickle=False))
    norm = arr(c['final_normalization'])
    stats = arr(source / 'input_statistics.npz')
    x = (y[:, 1:] - stats['mean']) / stats['rms']
    h = arr(Path(c['heads']) / ('models/raw/%d/head.npz' % c['fold_index']))
    held = c['folds'][c['fold_index']]
    fit = [i for i in range(28) if i not in held]
    original = torch.from_numpy(np.load(str(source / 'base_logits.npy'), allow_pickle=False))
    basis = arr(source / 'models/pca48.npz')['basis']
    warm = arr(source / 'warmup_dictionary.npz')
    counts = torch.from_numpy(np.load(str(Path(c['old_cache']) / 'fine_counts.npy'), allow_pickle=False).astype(float))
    wp = counts / counts.sum(-1, keepdim=True)
    wn = (256 - counts) / (256 - counts).sum(-1, keepdim=True)
    out = r / 'models'
    out.mkdir(exist_ok=False)
    np.savez(str(r / 'input_statistics.npz'), mean=stats['mean'].numpy(), rms=stats['rms'].numpy())

    def bce(i, p):
        return 0.5 * (wp[i] * torch.nn.functional.softplus(-p) + wn[i] * torch.nn.functional.softplus(p)).sum(-1).mean()

    def predict(i, rec):
        return logits(c, m, torch.cat((y[i:i + 1, :1], (rec * stats['rms'] + stats['mean'])[None]), dim=1), h, norm)
    gen = np.random.RandomState(c['sampling_seed'])
    warmseq = [fit[j] for j in gen.randint(len(fit), size=80)]
    sequence = [fit[j] for j in gen.randint(len(fit), size=c['steps'])]
    oldseq = json.loads((source / 'training_sequences.json').read_text())
    assert oldseq['warmup'] == warmseq and oldseq['shared_adaptation'] == sequence[:80]
    save(r / 'training_sequences.json', {'fit_groups': fit, 'held_groups': held, 'warmup': warmseq, 'adaptation': sequence})
    receipts = []
    assert 'residual_mlp' in c['methods'] and set(c['methods']) <= {'sae_functional', 'sae_task', 'rank48_task', 'residual_mlp'}
    for name in [name for name in c['methods'] if name != 'residual_mlp']:
        t = time.time()
        dest = out / name
        dest.mkdir()
        ae = fresh(name, basis)
        if name != 'rank48_task':
            ae.load_state_dict(warm, strict=True)
        with torch.no_grad():
            init = {'local': [], 'functional': [], 'bce': []}
            for i in fit:
                rec = ae.decode(ae.encode_inference(x[i]))
                p = predict(i, rec)
                init['local'].append(float((rec - x[i]).square().mean()))
                init['functional'].append(float((p - original[i]).square().mean()))
                init['bce'].append(float(bce(i, p)))
        initial = {k: float(np.mean(v)) for k, v in init.items()}
        if name == 'sae_functional':
            alpha = json.loads((source / 'functional_weight.json').read_text())['alpha']
            assert abs(alpha - initial['local'] / initial['functional']) < 1e-06 * alpha
        else:
            alpha = initial['local'] / initial['bce']
        assert np.isfinite(alpha) and alpha > 0
        save(dest / 'initial_losses.json', {'fit_groups': fit, 'means': initial, 'alpha': alpha, 'target': 'functional' if name == 'sae_functional' else 'true_balanced_bce'})
        optimizer = torch.optim.AdamW(ae.parameters(), lr=c['learning_rate'], weight_decay=0.0)
        history = []
        for step, i in enumerate(sequence):
            rec = ae.decode(ae.encode_inference(x[i]))
            local = (rec - x[i]).square().mean()
            p = predict(i, rec)
            target = (p - original[i]).square().mean() if name == 'sae_functional' else bce(i, p)
            loss = local + alpha * target
            optimizer.zero_grad()
            loss.backward()
            if name != 'rank48_task':
                ae.project_decoder_gradient_()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.param_groups[0]['lr'] = c['learning_rate'] * 0.5 * (1 + math.cos(math.pi * step / (c['steps'] - 1)))
            optimizer.step()
            if name != 'rank48_task':
                ae.normalize_decoder_()
            assert bool(torch.isfinite(loss))
            if step == 0 or (step + 1) % 50 == 0:
                history.append({'step': step + 1, 'fit_group': i, 'local_mse': float(local.detach()), 'target_loss': float(target.detach()), 'loss': float(loss.detach()), 'seconds': time.time() - t})
                save(r / 'fit_status.json', {'status': 'RUNNING', 'method': name, 'step': step + 1, 'steps': c['steps'], 'pid': os.getpid(), 'seconds': time.time() - start})
                print(name, step + 1, 'loss', float(loss.detach()), 'seconds', time.time() - t, flush=True)
        np.savez(str(dest / 'model.npz'), **{k: v.detach().numpy() for k, v in ae.state_dict().items()})
        save(dest / 'training.json', {'alpha': alpha, 'steps': c['steps'], 'history': history, 'training_and_initialization_seconds': time.time() - t})
        with torch.no_grad():
            codes = torch.stack([ae.encode_inference(x[i]) for i in range(28)])
            width = codes.shape[-1]
            framecodes = codes.reshape(28, 196, 8, width).permute(0, 2, 1, 3)
            v = framecodes[fit]
            if name == 'rank48_task':
                center = v.reshape(-1, width).mean(0)
                framecodes = framecodes - center
                v = framecodes[fit]
                contrast = (v * (wp[fit] - wn[fit])[:, :, :, None]).sum(2).mean((0, 1))
                orientation = torch.sign(contrast)
                orientation[orientation == 0] = 1.0
                framecodes = framecodes * orientation
                v = framecodes[fit]
            else:
                center = torch.zeros(width)
                orientation = torch.ones(width)
            scale = v.reshape(-1, width).std(0, unbiased=False).clamp_min(1e-06)
            contrast = (v * (wp[fit] - wn[fit])[:, :, :, None]).sum(2).mean((0, 1)) / scale
            support = (v > 0).reshape(21, -1, width).any(1).sum(0)
            contrast[(support < 6) | (contrast <= 0)] = -float('inf')
            ids = torch.argsort(contrast, descending=True)[:3]
            assert bool(torch.isfinite(contrast[ids]).all())
            feature = (framecodes[held][:, :, :, ids] / scale[ids]).mean(-1)
            l0 = (codes[held] != 0).sum(-1).reshape(7, 196, 8).permute(0, 2, 1)
            pred = []
            local_error = []
            for i in held:
                rec = ae.decode(ae.encode_inference(x[i]))
                pred.append(predict(i, rec).numpy())
                local_error.append((rec - x[i]).square().sum(-1).reshape(196, 8).T.numpy())
            fit_pred = torch.stack([predict(i, ae.decode(ae.encode_inference(x[i]))) for i in fit])
            restored = fresh(name, basis)
            restored.load_state_dict(arr(dest / 'model.npz'), strict=True)
            rr = restored.decode(restored.encode_inference(x[held[0]]))
            replayerr = float((predict(held[0], rr) - torch.from_numpy(pred[0])).abs().max())
            assert replayerr < 1e-06
            np.savez(str(dest / 'predictions.npz'), fit_groups=np.array(fit), held_groups=np.array(held), fit_logits=fit_pred.numpy(), held_logits=np.array(pred), held_reconstruction_error=np.array(local_error), l0=l0.numpy(), feature_scores=feature.numpy())
            np.savez(str(dest / 'replay_sample.npz'), group=held[0], input=x[held[0]].numpy(), reconstruction=rr.numpy(), logits=pred[0], original_cls=y[held[0], 0].numpy())
            save(dest / 'feature_selection.json', {'ids': ids.tolist(), 'fit_groups': fit, 'fit_contrast': contrast[ids].tolist(), 'fit_scale': scale[ids].tolist(), 'center': center[ids].tolist(), 'orientation': orientation[ids].tolist()})
        receipts.append({'method': name, 'parameters': sum((p.numel() for p in ae.parameters())), 'alpha': alpha, 'seconds': time.time() - t, 'full_clip_reload_max_logit_error': replayerr})
        print('EVALUATED', name, 'seconds', time.time() - start, flush=True)
        del codes, framecodes, v, ae
    tokens = torch.from_numpy(np.load(str(Path(c['old_cache']) / 'tokens.npy'), allow_pickle=False))
    raw = ((tokens - norm['mean']) / norm['rms']).double()
    u = (raw - h['feature_mean']) / h['feature_scale']
    cacheerr = float((u @ h['weight'] + h['bias'] - original).abs().max())
    assert cacheerr < 1e-06
    u = u.float()
    del tokens, raw
    t = time.time()
    torch.manual_seed(c['seed'])
    mlp = residual_mlp()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=c['mlp_learning_rate'], weight_decay=0.0)
    history = []
    dest = out / 'residual_mlp'
    dest.mkdir()
    with torch.no_grad():
        assert float(mlp(u[fit[0]]).abs().max()) == 0
    for step, i in enumerate(sequence):
        p = original[i] + mlp(u[i]).squeeze(-1)
        loss = bce(i, p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert bool(torch.isfinite(loss))
        if step == 0 or (step + 1) % 50 == 0:
            history.append({'step': step + 1, 'fit_group': i, 'bce': float(loss.detach())})
    with torch.no_grad():
        fit_pred = original[fit] + mlp(u[fit]).squeeze(-1)
        pred = original[held] + mlp(u[held]).squeeze(-1)
        np.savez(str(dest / 'model.npz'), **{k: v.numpy() for k, v in mlp.state_dict().items()})
        restore = residual_mlp()
        restore.load_state_dict(arr(dest / 'model.npz'), strict=True)
        err = float((original[held[0]] + restore(u[held[0]]).squeeze(-1) - pred[0]).abs().max())
        assert err < 1e-05
        np.savez(str(dest / 'predictions.npz'), fit_groups=np.array(fit), held_groups=np.array(held), fit_logits=fit_pred.numpy(), held_logits=pred.numpy())
    save(dest / 'training.json', {'steps': c['steps'], 'history': history, 'seconds': time.time() - t})
    receipts.append({'method': 'residual_mlp', 'parameters': sum((p.numel() for p in mlp.parameters())), 'seconds': time.time() - t, 'reload_logit_error': err, 'cached_final_head_error': cacheerr})
    np.save(str(r / 'base_logits.npy'), original.numpy(), allow_pickle=False)
    np.save(str(r / 'held_input_energy.npy'), x[held].square().sum(-1).reshape(7, 196, 8).permute(0, 2, 1).numpy(), allow_pickle=False)
    assert all((p.grad is None for p in m.parameters()))
    save(r / 'fit_receipt.json', {'status': 'FITTED', 'fold': c['fold_index'], 'seed': c['seed'], 'methods': receipts, 'seconds': time.time() - start, 'fit_config_sha256': hashlib.sha256((r / 'fit_config.json').read_bytes()).hexdigest()})
    save(r / 'fit_status.json', {'status': 'FITTED', 'seconds': time.time() - start})
    print('ALL FITTED', time.time() - start, flush=True)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Directory containing fit_config.json; outputs must not already exist.')
    run(parser.parse_args().run_dir)
