import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_token_identity_sae as training
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest
from src.token_identity_sae import TokenIdentitySAE


def jobs_for(config):
    jobs = []
    for space, path in config['training_runs'].items():
        source = Path(path)
        receipt = read_json(source / 'training_summary.json')
        assert receipt['status'] == 'COMPLETE'
        for item in receipt['outputs']:
            if item['method'] in config['methods'] and item['seed'] in config['seeds']:
                jobs.append(dict(item, space=space, name=space + '_' + item['method']))
    return jobs


def load_vectors(job, raw, offsets, indices, device):
    folder = Path(job['directory'])
    definition = read_json(folder / 'model_config.json')
    model = TokenIdentitySAE(definition, definition['method'])
    with np.load(folder / 'model.npz') as saved:
        model.load_state_dict({k: torch.from_numpy(saved[k].copy()) for k in saved.files})
    model.to(device).eval().requires_grad_(False)
    with np.load(folder / 'normalization.npz') as saved:
        mean, scale = saved['mean'].copy(), saved['scale'].copy()
    local, pooled = {}, {}
    with torch.no_grad():
        for index in indices:
            tokens = raw[offsets[index]:offsets[index + 1]]
            values = torch.from_numpy(((tokens.astype(float) - mean) / scale).astype(np.float32)).to(device)
            code = model.encode(values)
            projected = model.readout(code)
            assert torch.isfinite(projected).all() and torch.all(projected.norm(dim=1) > 0)
            local[int(index)] = F.normalize(projected, dim=1)
            pooled[int(index)] = F.normalize(model.readout(code.mean(0)), dim=0)
    return local, pooled


def pair_scores(local, pooled, pairs, output, resume, every):
    saved_path = output / 'scores.npz'
    arrays = {name: np.full(len(pairs), np.nan) for name in ('pooled_cosine', 'symmetric_maxsim')}
    completed, previous_checks = 0, []
    if (output / 'pair_progress.json').exists():
        assert resume
        receipt = read_json(output / 'pair_progress.json')
        completed = receipt['completed']
        previous_checks = receipt['checks']
        with np.load(saved_path) as saved:
            arrays = {key: saved[key].copy() for key in arrays}
        assert all(np.isfinite(a[:completed]).all() for a in arrays.values())
    checks = list(previous_checks)
    for number in range(completed, len(pairs)):
        pair = pairs[number]
        left, right = pair['source_index'], pair['query_index']
        a, b = local[left], local[right]
        with torch.no_grad():
            matrix = a @ b.T
            score = .5 * (matrix.amax(1).mean() + matrix.amax(0).mean())
            arrays['pooled_cosine'][number] = float(pooled[left] @ pooled[right])
            arrays['symmetric_maxsim'][number] = float(score)
        assert np.isfinite(arrays['symmetric_maxsim'][number]) and abs(float(score)) <= 1.00001
        category = 'other' if not pair['same_identity'] else 'cross_interval' if not pair['same_annotation_interval'] else 'same_interval'
        if category not in {c['category'] for c in checks}:
            av, bv = a.cpu().numpy().astype(float), b.cpu().numpy().astype(float)
            matrix_np = av @ bv.T
            expected = .5 * (matrix_np.max(1).mean() + matrix_np.max(0).mean())
            distance = 1 - cdist(av, bv, metric='cosine')
            independent = .5 * (distance.max(1).mean() + distance.max(0).mean())
            np.testing.assert_allclose(float(score), [expected, independent], atol=2e-6, rtol=0)
            reversed_matrix = bv[::-1] @ av[::-1].T
            permuted = .5 * (reversed_matrix.max(1).mean() + reversed_matrix.max(0).mean())
            np.testing.assert_allclose(expected, permuted, atol=1e-12, rtol=0)
            checks.append(dict(category=category, pair=number, source_tokens=len(a), query_tokens=len(b),
                               numpy_error=abs(float(score) - expected), scipy_error=abs(float(score) - independent),
                               symmetry_and_permutation=True))
        if (number + 1) % every == 0 or number + 1 == len(pairs):
            training.shared.save_npz(saved_path, **arrays,
                source=np.array([p['source_index'] for p in pairs]), query=np.array([p['query_index'] for p in pairs]))
            atomic_write_json(output / 'pair_progress.json', dict(completed=number + 1, total=len(pairs), checks=checks))
            print('LOCAL_PAIRS', output.name, number + 1, '/', len(pairs), flush=True)
    return arrays, checks


def metrics(pairs, arrays, quantile, metadata):
    rows = []
    labels = np.array([p['same_identity'] for p in pairs])
    cross = labels & ~np.array([p['same_annotation_interval'] for p in pairs])
    for name, scores in arrays.items():
        report = training.protected_statistics(pairs, scores, quantile)
        for video, row in report['by_video'].items():
            selected = np.array([p['video_id'] == video for p in pairs])
            rows.append(dict(**metadata, readout=name, video=video, threshold=report['threshold'], **row,
                auroc=float(roc_auc_score(labels[selected], scores[selected])),
                cross_interval_recall=float(np.mean(scores[selected & cross] > report['threshold'])) if np.any(selected & cross) else None,
                positive_pairs=int(np.sum(selected & labels)), negative_pairs=int(np.sum(selected & ~labels)),
                cross_interval_pairs=int(np.sum(selected & cross))))
    return rows


def evaluate(run, smoke, resume):
    config = read_json(run / 'config.json')
    prepared = Path(config['training_runs']['projected'])
    _, cohort, raw, offsets, records, receipts = training.load_inputs(prepared)
    jobs = jobs_for(config)
    if smoke:
        jobs = [next(j for j in jobs if j['name'] == name and j['fold'] == 'full')
                for name in ('projected_token_sparse', 'projected_token_dense', 'direct_token_sparse', 'direct_token_dense')]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    root = run / ('smoke' if smoke else 'evaluation')
    root.mkdir(exist_ok=True)
    paths = [run / 'config.json', run / 'protocol.json', Path(__file__), Path(training.__file__), ROOT / 'src/token_identity_sae.py']
    paths += [Path(path) / 'training_summary.json' for path in config['training_runs'].values()]
    identity = dict(inputs={str(p): digest(p) for p in paths}, tokens=[r['tokens_sha256'] for r in receipts], smoke=smoke)
    if (root / 'identity.json').exists():
        assert resume and read_json(root / 'identity.json') == identity
    atomic_write_json(root / 'identity.json', identity)
    rows, raw_done = [], set()
    started = time.perf_counter()
    for number, job in enumerate(jobs):
        target = root / f"{job['name']}_seed{job['seed']}_fold{job['fold']}"
        target.mkdir(exist_ok=True)
        folder = Path(job['directory'])
        summary = read_json(folder / 'summary.json')
        assert summary['status'] == 'COMPLETE' and summary['model_sha256'] == digest(folder / 'model.npz')
        step = summary['steps']
        path = folder / f'held_{step:04d}.npz'
        local_identity = {str(p): digest(p) for p in [path, folder / 'model.npz', folder / 'model_config.json', folder / 'normalization.npz']}
        with np.load(path) as saved:
            indices, reference = saved['indices'].copy(), saved['scores'].copy()
            expected_a, expected_b = saved['source'].copy(), saved['query'].copy()
        videos = sorted({records[int(i)]['video_id'] for i in indices})
        pairs = training.shared.chronological_pairs(records, videos)
        np.testing.assert_array_equal(expected_a, [p['source_index'] for p in pairs])
        np.testing.assert_array_equal(expected_b, [p['query_index'] for p in pairs])
        metadata = dict(method=job['name'], seed=job['seed'], fold=job['fold'], step=step,
                        partition='validation' if job['fold'] == 'full' else 'training_oof_final_step')
        if (target / 'summary.json').exists():
            saved = read_json(target / 'summary.json')
            assert resume and saved['identity'] == local_identity
            rows.extend(saved['rows'])
        else:
            if (target / 'identity.json').exists():
                assert resume and read_json(target / 'identity.json') == local_identity
            atomic_write_json(target / 'identity.json', local_identity)
            local, pooled = load_vectors(job, raw, offsets, indices, device)
            arrays, checks = pair_scores(local, pooled, pairs, target, resume, config['checkpoint_pairs'])
            np.testing.assert_allclose(arrays['pooled_cosine'], reference, atol=2e-6, rtol=0)
            local_rows = metrics(pairs, arrays, config['negative_quantile'], metadata)
            atomic_write_json(target / 'summary.json', dict(status='COMPLETE', rows=local_rows, identity=local_identity,
                checks=checks, pooled_max_error=float(np.max(np.abs(arrays['pooled_cosine'] - reference))), pairs=len(pairs)))
            rows.extend(local_rows)
            del local, pooled
        if job['fold'] not in raw_done:
            target_raw = root / f"raw_fold{job['fold']}"
            target_raw.mkdir(exist_ok=True)
            if (target_raw / 'summary.json').exists():
                assert resume
                rows.extend(read_json(target_raw / 'summary.json')['rows'])
            else:
                local, pooled = {}, {}
                for index in indices:
                    value = torch.from_numpy(raw[offsets[index]:offsets[index + 1]]).to(device)
                    assert torch.all(value.norm(dim=1) > 0)
                    local[int(index)] = F.normalize(value, dim=1)
                    pooled[int(index)] = F.normalize(value.mean(0), dim=0)
                arrays, checks = pair_scores(local, pooled, pairs, target_raw, resume, config['checkpoint_pairs'])
                raw_metadata = dict(metadata, method='raw_endofm', seed=0, step=0)
                local_rows = metrics(pairs, arrays, config['negative_quantile'], raw_metadata)
                atomic_write_json(target_raw / 'summary.json', dict(status='COMPLETE', rows=local_rows, checks=checks, pairs=len(pairs)))
                rows.extend(local_rows)
                del local, pooled
            raw_done.add(job['fold'])
        atomic_write_json(root / 'progress.json', dict(completed=number + 1, total=len(jobs), method=job['name']))
        print('LOCAL_MODELS', number + 1, '/', len(jobs), flush=True)
    assert identity['inputs'] == {str(p): digest(p) for p in paths}
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', smoke=smoke, rows=rows, jobs=len(jobs),
        seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        completed_at=training.shared.now(), identity=identity))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    root = run / ('smoke' if smoke else 'evaluation')
    receipt = read_json(root / 'summary.json')
    assert receipt['status'] == 'COMPLETE'
    table = pd.DataFrame(receipt['rows'])
    fields = ['recall', 'negative_retention', 'auroc', 'cross_interval_recall']
    seeds = table.groupby(['partition', 'method', 'readout', 'seed'])[fields].mean().reset_index()
    means = seeds.groupby(['partition', 'method', 'readout'])[fields].mean().reset_index()
    paired = table[table.readout == 'symmetric_maxsim'].merge(table[table.readout == 'pooled_cosine'],
        on=['partition', 'method', 'seed', 'fold', 'video'], suffixes=('_local', '_pooled'), validate='one_to_one')
    for field in fields:
        paired[field + '_difference'] = paired[field + '_local'] - paired[field + '_pooled']
    output = run / ('smoke_analysis' if smoke else 'analysis')
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'procedures.csv', index=False)
    seeds.to_csv(output / 'seeds.csv', index=False)
    means.to_csv(output / 'summary.csv', index=False)
    paired.to_csv(output / 'paired_differences.csv', index=False)
    names = ['raw_endofm', 'projected_token_sparse', 'projected_token_dense', 'direct_token_sparse', 'direct_token_dense']
    labels = ['Raw EndoFM', 'Projected sparse', 'Projected dense', 'Direct sparse', 'Direct dense']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, field in zip(axes, ['recall', 'auroc', 'cross_interval_recall'], strict=True):
        for index, name in enumerate(names):
            for offset, readout, color in [(-.12, 'pooled_cosine', '#0072B2'), (.12, 'symmetric_maxsim', '#D55E00')]:
                subset = seeds[(seeds.partition == 'validation') & (seeds.method == name) & (seeds.readout == readout)]
                ax.scatter(np.full(len(subset), index + offset), subset[field], color=color, s=25,
                           label=readout if index == 0 else None)
        ax.set_xticks(range(len(names)), labels, rotation=25, ha='right')
        ax.set_title(field.replace('_', ' '))
        ax.set_ylim(0, 1.03)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Frozen local matching | exposed validation' + (' | real smoke' if smoke else ''))
    fig.text(.02, .02, 'Dots: seed-level procedure means. Raw control has no training seed. Descriptive 99% negative-retention thresholds.\nGT-region identity evidence; complete-video prompting requires separate evaluation.', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(output / 'local_matching.png', dpi=170)
    plt.close(fig)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', results=means.to_dict('records'),
        input_sha256=digest(root / 'summary.json'), source_sha256=digest(__file__)))
    atomic_write_json(run / 'summary_progress.json', dict(completed=1, total=1))
    print(means.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'summary'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'evaluate':
        evaluate(args.run, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke)
