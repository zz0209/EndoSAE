import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import ot
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import torch
from torch.nn import functional as F

from evaluate_query_conditioned_components import boundary, measurements
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest
from src.token_identity_sae import TokenIdentitySAE


SCORES = ['pooled_cosine', 'local_mean', 'symmetric_maxsim', 'uniform_transport',
          'cross_independent', 'cross_transport']


def now():
    return datetime.now().astimezone().isoformat()


def transport(matrix, first, second, independent=False):
    cost = np.ascontiguousarray(1 - matrix, dtype=np.float64)
    a, b = first.astype(np.float64), second.astype(np.float64)
    flow, solved_ot = ot.emd(a, b, cost, numItermax=100000, numThreads=1, log=True)
    assert solved_ot['warning'] is None, solved_ot
    distance = solved_ot['cost']
    np.testing.assert_allclose(distance, a @ solved_ot['u'] + b @ solved_ot['v'], atol=1e-8, rtol=0)
    assert np.min(cost - solved_ot['u'][:, None] - solved_ot['v'][None, :]) >= -1e-8
    assert np.isfinite(flow).all() and flow.min() >= 0
    residual = max(float(np.max(np.abs(flow.sum(1) - a))), float(np.max(np.abs(flow.sum(0) - b))))
    assert residual < 1e-9, residual
    score = float(np.sum(flow.astype(float) * matrix))
    np.testing.assert_allclose(score, 1 - distance, atol=1e-8, rtol=0)
    lower = float(first @ matrix @ second)
    upper = float(min(first @ matrix.max(1), second @ matrix.max(0)))
    assert lower - 1e-8 <= score <= upper + 1e-8, (lower, score, upper)
    error = None
    if independent:
        n, m = matrix.shape
        constraints = sparse.vstack([sparse.kron(sparse.eye(n), np.ones((1, m))),
                                     sparse.kron(np.ones((1, n)), sparse.eye(m))], format='csr')
        solved = linprog(cost.ravel().astype(float), A_eq=constraints,
                         b_eq=np.r_[first, second], bounds=(0, None), method='highs')
        assert solved.success, solved.message
        error = abs(distance - solved.fun)
        assert error < 1e-8, error
    return score, residual, error


def pair_values(a, b, pooled_a, pooled_b, verify=False):
    matrix = np.clip(a @ b.T, -1., 1.)
    first, second = np.ones(len(a)) / len(a), np.ones(len(b)) / len(b)
    cross_a, cross_b = np.maximum(matrix.mean(1), 0), np.maximum(matrix.mean(0), 0)
    assert cross_a.sum() > 0 and cross_b.sum() > 0
    cross_a, cross_b = cross_a / cross_a.sum(), cross_b / cross_b.sum()
    uniform, residual_u, _ = transport(matrix, first, second)
    cross, residual_c, _ = transport(matrix, cross_a, cross_b)
    maxsim = float(.5 * (matrix.max(1).mean() + matrix.max(0).mean()))
    scores = dict(pooled_cosine=float(pooled_a @ pooled_b), local_mean=float(matrix.mean()),
                  symmetric_maxsim=maxsim, uniform_transport=uniform,
                  cross_independent=float(cross_a @ matrix @ cross_b), cross_transport=cross)
    stats = dict(source_tokens=len(a), query_tokens=len(b), maxsim_transport_gap=maxsim-uniform,
                 reused_source_fraction=float(1-len(np.unique(matrix.argmax(0)))/len(b)),
                 reused_query_fraction=float(1-len(np.unique(matrix.argmax(1)))/len(a)),
                 marginal_error=max(residual_u, residual_c))
    if verify:
        sub = matrix[:min(32, len(a)), :min(32, len(b))]
        sa, sb = np.ones(len(sub))/len(sub), np.ones(sub.shape[1])/sub.shape[1]
        _, _, stats['scipy_error'] = transport(sub, sa, sb, independent=True)
        ca, cb = np.maximum(sub.mean(1), 0), np.maximum(sub.mean(0), 0)
        assert ca.sum() > 0 and cb.sum() > 0
        _, _, stats['scipy_cross_error'] = transport(sub, ca/ca.sum(), cb/cb.sum(), independent=True)
        reversed_score, _, _ = transport(matrix[::-1, ::-1].T, second[::-1], first[::-1])
        cross_reversed, _, _ = transport(matrix[::-1, ::-1].T, cross_b[::-1], cross_a[::-1])
        np.testing.assert_allclose([reversed_score, cross_reversed], [uniform, cross], atol=1e-8, rtol=0)
        stats['reverse_permutation_error'] = max(abs(uniform-reversed_score), abs(cross-cross_reversed))
    return scores, stats


def jobs(config):
    parent = read_json(Path(config['training_run']) / 'components/summary.json')
    assert parent['status'] == 'COMPLETE'
    learned = [item for item in parent['models'] if item['condition'] == config['condition']
               and item['method'] in config['methods'] and item['seed'] in config['seeds'] and item['fold'] != 'full']
    raw = [dict(next(item for item in learned if item['fold'] == fold), method='raw_endofm', seed=0)
           for fold in sorted({item['fold'] for item in learned})]
    return learned + raw


def load_vectors(item, config):
    folder = Path(item['directory'])
    records = read_json(folder / 'records.json')
    path = folder / f"held_{config['step']:04d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in ['source', 'query', 'same_identity', 'scores', 'indices']}
    indices = sorted(set(data['source']) | set(data['query']))
    videos = sorted({records[int(i)]['video_id'] for i in indices})
    assert not set(videos) & set(item['fit_videos'])
    inputs = {str(path): digest(path), str(folder / 'records.json'): digest(folder / 'records.json')}
    if item['method'] != 'raw_endofm':
        definition = read_json(folder / 'model_config.json')
        model = TokenIdentitySAE(definition, definition['method']).to(config['device']).eval().requires_grad_(False)
        assert model.identity_space == 'code'
        model_path = folder / f"model_{config['step']:04d}.npz"
        with np.load(model_path, allow_pickle=False) as saved:
            model.load_state_dict({k: torch.from_numpy(saved[k].copy()) for k in saved.files})
        with np.load(folder / 'normalization.npz', allow_pickle=False) as saved:
            mean, scale = saved['mean'].copy(), saved['scale'].copy()
        inputs.update({str(p): digest(p) for p in [model_path, folder/'normalization.npz', folder/'model_config.json']})
    local, pooled = {}, {}
    for video in videos:
        directory = Path(config['token_root']) / video
        receipt = read_json(directory / 'complete.json')
        assert receipt['status'] == 'COMPLETE'
        token_path = directory / 'tokens.npz'
        assert digest(token_path) == receipt['tokens_sha256']
        assert digest(directory/'records.json') == receipt['records_sha256']
        originals = read_json(directory / 'records.json')
        lookup = {(r['clip_id'], r['lesion_id']): j for j, r in enumerate(originals)}
        inputs[str(token_path)] = receipt['tokens_sha256']
        inputs[str(directory/'records.json')] = receipt['records_sha256']
        with np.load(token_path, allow_pickle=False) as saved:
            raw, offsets = saved['tokens'].copy(), saved['offsets'].copy()
        for index in indices:
            record = records[int(index)]
            if record['video_id'] != video:
                continue
            position = lookup[record['clip_id'], record['lesion_id']]
            for key in ['video_id', 'split', 'start_frame', 'end_frame', 'roi_tokens_per_frame']:
                assert record[key] == originals[position][key]
            value = raw[offsets[position]:offsets[position+1]]
            assert len(value) == sum(record['roi_tokens_per_frame'])
            with torch.no_grad():
                if item['method'] != 'raw_endofm':
                    value = model.encode(torch.from_numpy(((value.astype(float)-mean)/scale).astype(np.float32)).to(config['device']))
                else:
                    value = torch.from_numpy(value)
                assert torch.isfinite(value).all() and torch.all(value.norm(dim=1) > 0)
                pooled[int(index)] = F.normalize(value.mean(0), dim=0).cpu().numpy().astype(float)
                local[int(index)] = F.normalize(value.double(), dim=1).cpu().numpy()
    before = np.array([pooled[int(a)] @ pooled[int(b)] for a, b in zip(data['source'], data['query'], strict=True)])
    if item['method'] != 'raw_endofm':
        np.testing.assert_allclose(before, data['scores'], atol=2e-6, rtol=0)
    data['videos'] = np.array([records[int(i)]['video_id'] for i in data['source']])
    data['cross_interval'] = np.array([records[int(a)]['annotation_observation'] != records[int(b)]['annotation_observation']
                                    for a, b in zip(data['source'], data['query'], strict=True)])
    for a, b, label in zip(data['source'], data['query'], data['same_identity'], strict=True):
        x, y = records[int(a)], records[int(b)]
        assert x['video_id'] == y['video_id'] and x['end_frame'] < y['start_frame']
        assert bool(label) == (x['lesion_id'] == y['lesion_id'])
    return local, pooled, data, inputs, float(np.max(np.abs(before-data['scores'])))


def evaluate(run, smoke, stop_after):
    config = read_json(run / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = run / 'smoke' if smoke else run / 'evaluation'
    root.mkdir(exist_ok=True)
    identity = dict(config=digest(run/'config.json'), protocol=digest(run/'protocol.json'),
        source=digest(__file__), device=config['device'], evaluator=digest(Path(__file__).with_name('evaluate_query_conditioned_components.py')),
        model_source=digest('src/token_identity_sae.py'), torch=str(torch.__version__), numpy=np.__version__, pot=ot.__version__)
    if (root/'identity.json').exists():
        assert read_json(root/'identity.json') == identity
    atomic_write_json(root/'identity.json', identity)
    selected = jobs(config)
    if smoke:
        selected = [j for j in selected if j['fold'] in config['smoke_folds'] and j['seed'] in [0, config['seeds'][0]]]
    outputs, start, processed = [], time.perf_counter(), 0
    for number, item in enumerate(selected):
        directory = root / f"{item['method']}_seed{item['seed']}_fold{item['fold']}"
        directory.mkdir(exist_ok=True)
        local, pooled, data, inputs, pooled_error = load_vectors(item, config)
        signature = dict(identity=identity, inputs=inputs, method=item['method'], seed=item['seed'], fold=item['fold'])
        if (directory/'complete.json').exists():
            receipt = read_json(directory/'complete.json')
            assert receipt['signature'] == signature and digest(directory/'predictions.npz') == receipt['sha256']
            outputs.append(dict(directory=str(directory), **receipt))
            processed += receipt['pairs']
            continue
        order = np.arange(len(data['source']))
        if smoke:
            largest = int(np.argmax([len(local[int(a)])*len(local[int(b)]) for a,b in zip(data['source'],data['query'],strict=True)]))
            representatives = [i for video in np.unique(data['videos']) for label in [False, True]
                               for i in np.flatnonzero((data['videos']==video)&(data['same_identity']==label))[:2]]
            order = np.unique(np.r_[order[:config['smoke_pair_limit']], representatives, largest])
        data = {key: value[order] for key, value in data.items() if key != 'indices'}
        scores = {name: np.full(len(order), np.nan) for name in SCORES}
        diagnostics, completed = [], 0
        if (directory/'progress.json').exists():
            previous = read_json(directory/'progress.json')
            assert previous['signature'] == signature
            completed, diagnostics = previous['completed'], previous['diagnostics']
            with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
                scores = {name:saved[name].copy() for name in SCORES}
            assert all(np.isfinite(value[:completed]).all() for value in scores.values())
        categories = {r['category'] for r in diagnostics if 'scipy_error' in r}
        for k in range(completed, len(order)):
            a,b = int(data['source'][k]), int(data['query'][k])
            category = 'other' if not data['same_identity'][k] else 'cross_interval' if data['cross_interval'][k] else 'same_interval'
            values, stats = pair_values(local[a],local[b],pooled[a],pooled[b],verify=category not in categories)
            categories.add(category)
            for key,value in values.items():
                scores[key][k] = value
            diagnostics.append(dict(pair=int(order[k]),video=str(data['videos'][k]),same=bool(data['same_identity'][k]), category=category,**stats))
            if (k+1)%config['checkpoint_pairs']==0 or k+1==len(order):
                np.savez_compressed(directory/'predictions.npz',**data,**scores)
                atomic_write_json(directory/'progress.json',dict(signature=signature,completed=k+1,total=len(order),diagnostics=diagnostics))
                atomic_write_json(root/'progress.json',dict(status='RUNNING',completed=number,completed_pairs=processed+k+1,model=number+1,
                    total_models=len(selected),elapsed_seconds=time.perf_counter()-start,updated_at=now()))
                print('TRANSPORT_PAIRS',number+1,'/',len(selected),k+1,'/',len(order),round(time.perf_counter()-start,2),flush=True)
                pause_after_checkpoint(directory/'progress.json')
                if stop_after and processed+k+1>=stop_after:
                    return
        receipt = dict(status='COMPLETE',signature=signature,method=item['method'],seed=item['seed'],fold=item['fold'],
            pairs=len(order),pooled_error=pooled_error if item['method']!='raw_endofm' else None,
            sha256=digest(directory/'predictions.npz'),completed_at=now(),encoder_fit_videos=item['fit_videos'])
        atomic_write_json(directory/'complete.json',receipt)
        outputs.append(dict(directory=str(directory),**receipt))
        processed+=len(order)
    atomic_write_json(root/'summary.json',dict(status='COMPLETE',outputs=outputs,pairs=processed,elapsed_seconds=time.perf_counter()-start,completed_at=now()))
    atomic_write_json(root/'progress.json',dict(status='COMPLETE',completed=len(outputs),completed_pairs=processed,models=len(outputs),updated_at=now()))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config=read_json(run/'config.json')
    root=run/'smoke' if smoke else run/'evaluation'
    summary=read_json(root/'summary.json')
    rows, mechanism, checks=[],[],[]
    for item in summary['outputs']:
        directory=Path(item['directory'])
        assert digest(directory/'predictions.npz')==item['sha256']
        with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
            data={key:saved[key].copy() for key in saved.files}
        labels=data['same_identity'].astype(bool)
        eligible=[v for v in np.unique(data['videos']) if len(set(labels[data['videos']==v]))==2]
        for target in eligible:
            held=data['videos']==target
            fit=np.isin(data['videos'],[v for v in eligible if v!=target])
            if smoke and not fit.any():
                continue
            assert fit.any() and target not in item['encoder_fit_videos']
            original_threshold=boundary(data['pooled_cosine'][fit],data['videos'][fit],labels[fit],config['negative_quantile'])
            for name in SCORES:
                threshold=boundary(data[name][fit],data['videos'][fit],labels[fit],config['negative_quantile'])
                metrics=measurements(data['pooled_cosine'][held],data[name][held],labels[held],threshold,original_threshold)
                rows.append(dict(method=item['method'],seed=item['seed'],fold=item['fold'],video=target,score=name,**metrics))
        diagnostics=read_json(directory/'progress.json')['diagnostics']
        mechanism.extend(dict(method=item['method'],seed=item['seed'],fold=item['fold'],**r) for r in diagnostics)
        checks.extend(r for r in diagnostics if 'scipy_error' in r)
    output=root/'analysis'
    output.mkdir(exist_ok=True)
    table=pd.DataFrame(rows)
    table.to_csv(output/'procedures.csv',index=False)
    pd.DataFrame(mechanism).to_csv(output/'mechanism.csv',index=False)
    metrics=['auroc','recall','protection','matched_capacity','protected99_capacity']
    counts=['corrected','repeat_damage','other_damage','repeat_gain','original_wrong']
    if len(table):
        aggregate={**{k:'mean' for k in metrics},**{k:'sum' for k in counts}}
        table.groupby(['method','seed','score']).agg(aggregate).reset_index().to_csv(output/'seeds.csv',index=False)
        table.groupby(['method','score']).agg(aggregate).reset_index().to_csv(output/'summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,method in zip(axes,['token_sparse','token_dense','raw_endofm'],strict=True):
        selected=pd.DataFrame([r for r in mechanism if r['method']==method])
        videos=sorted(selected.video.unique())
        for same,label in [(True,'same identity'),(False,'different identity')]:
            values=selected[selected.same==same].groupby('video').maxsim_transport_gap.mean().reindex(videos)
            ax.scatter(np.arange(len(values)),values,label=label,alpha=.8)
        ax.set_title(method)
        ax.set_xticks(range(len(videos)),videos,rotation=60,ha='right',fontsize=7)
        ax.set_xlabel('Procedure')
        ax.set_ylabel('MaxSim minus uniform transport')
        ax.legend(fontsize=8)
    fig.suptitle('Saved local correspondence scores | procedure means')
    fig.savefig(output/'mechanism.png',dpi=140)
    plt.close(fig)
    if len(table):
        fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
        for i,method in enumerate(['token_sparse','token_dense','raw_endofm']):
            for j,metric in enumerate(['auroc','recall','protection']):
                for seed,part in table[table.method==method].groupby('seed'):
                    values=part.groupby('score')[metric].mean().reindex(SCORES)
                    axes[i,j].plot(range(len(SCORES)),values,'o-',label=str(seed))
                axes[i,j].set_xticks(range(len(SCORES)),SCORES,rotation=35,ha='right',fontsize=7)
                axes[i,j].set_title(method+' | '+metric)
                axes[i,j].legend(fontsize=7)
        fig.suptitle('Held procedure outcomes | thresholds fitted in other encoder-excluded procedures')
        fig.savefig(output/'comparison.png',dpi=140)
        plt.close(fig)
    atomic_write_json(output/'verification.json',dict(status='PASS',models=len(summary['outputs']),pairs=summary['pairs'],
        max_marginal_error=max(r['marginal_error'] for r in mechanism),
        max_scipy_error=max(r['scipy_error'] for r in checks),
        max_scipy_cross_error=max(r['scipy_cross_error'] for r in checks),
        max_symmetry_error=max(r['reverse_permutation_error'] for r in checks),
        max_pooled_error=max(r['pooled_error'] for r in summary['outputs'] if r['pooled_error'] is not None),procedure_rows=len(rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['evaluate','summarize'],required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if args.phase=='evaluate':
        evaluate(args.run,args.smoke,args.stop_after)
    else:
        summarize(args.run,args.smoke)
