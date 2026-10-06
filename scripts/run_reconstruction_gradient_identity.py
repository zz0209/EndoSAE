import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import train_token_identity_sae as training
from train_temporal_view_identity import read_tokens
from evaluate_query_conditioned_components import boundary, measurements
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest
from src.token_identity_sae import TokenIdentitySAE


def inputs(definition):
    chunks, offsets, records, receipts = [], [0], [], []
    for added, videos in [(False, definition['fold_training_videos']),
                          (True, definition['added_training_videos'])]:
        base = Path(definition['added_training_storage'] if added else definition['original_prepared_run']) / 'tokens'
        for video in videos:
            values, bounds, local, receipt = read_tokens(base / video)
            receipts.append(dict(video=video, receipt=receipt))
            for i, row in enumerate(local):
                if added and not row['original_observation']:
                    continue
                assert row['video_id'] == video and row['split'] == 'train'
                records.append(dict(row, global_index=len(records), partition='train'))
                chunks.append(values[bounds[i]:bounds[i + 1]])
                offsets.append(offsets[-1] + bounds[i + 1] - bounds[i])
    assert len(records) == 340 and len({r['video_id'] for r in records}) == 27
    return np.concatenate(chunks), np.asarray(offsets), records, receipts


def gradient_check(definition, raw, offsets, records, fitting, output):
    mean, scale = training.normalize(raw, offsets, records, fitting)
    training_records = [dict(r, split='train' if r['video_id'] in fitting else 'excluded') for r in records]
    chosen = training.sample_batch(training_records, np.random.default_rng(20261005), definition)
    generator = np.random.default_rng(np.random.SeedSequence([20261005,71005]))
    token_indices = np.stack([generator.integers(offsets[i],offsets[i+1],size=definition['tokens_per_clip']) for i in chosen])
    values = torch.tensor((raw[token_indices] - mean) / scale, dtype=torch.float32, device=definition['device'])
    mapping = {key:i for i,key in enumerate(sorted({(r['video_id'],r['lesion_id']) for r in records}))}
    labels = torch.tensor([mapping[(records[i]['video_id'],records[i]['lesion_id'])] for i in chosen],device=definition['device'])
    results = []
    for method in ['token_sparse', 'token_dense']:
        torch.manual_seed(20261005)
        model = TokenIdentitySAE(definition, method).to(definition['device'])
        projected, decoded, local = model(values)
        task = training.supcon(projected,labels,definition['temperature'])
        encoder = tuple(model.encoder.parameters())
        decoder = tuple(model.decoder.parameters())
        expected = torch.autograd.grad(task, encoder, retain_graph=True)
        reconstruction = (decoded - values).square().mean()
        expected_decoder = torch.autograd.grad(.1 * reconstruction, decoder, retain_graph=True)
        objective = task + .1 * (model.decoder(local.detach()) - values).square().mean()
        actual = torch.autograd.grad(objective, encoder + decoder)
        error = max(float((a - b).abs().max()) for a, b in zip(actual, expected + expected_decoder, strict=True))
        assert error == 0
        results.append(dict(method=method, gradient_error=error, actual_tokens=values.shape[1]))
    atomic_write_json(output, dict(status='PASS', results=results))


def train(run, smoke, stop_after):
    config = read_json(run / 'config.json')
    parent = Path(config['parent'])
    definition = read_json(parent / 'config.json')
    root = run / 'smoke' if smoke else run
    root.mkdir(parents=True, exist_ok=True)
    raw, offsets, records, receipts = inputs(definition)
    folds = read_json(parent / 'folds.json')
    videos = {r['video_id'] for r in records}
    atomic_write_json(root / 'records.json', records)
    identity = dict(source_hashes=training.source_identity(), driver_sha256=digest(__file__),
                    protocol_sha256=digest(run / 'protocol.json'), config_sha256=digest(run / 'config.json'),
                    input_sha256=training.shared.json_digest(receipts))
    atomic_write_json(root / 'input_identity.json', identity)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if smoke:
        gradient_check(definition, raw, offsets, records, videos - set(folds[0]), root / 'gradient_verification.json')
    seeds = config['seeds'][:1] if smoke else config['seeds']
    active_folds = folds[:1] if smoke else folds
    steps = config['smoke_steps'] if smoke else config['steps']
    total = len(config['conditions']) * len(config['methods']) * len(seeds) * len(active_folds)
    outputs, start = [], time.perf_counter()
    for seed in seeds:
        for fold, held in enumerate(active_folds):
            fitting = sorted(videos - set(held))
            for method in config['methods']:
                for condition in config['conditions']:
                    folder = root / condition / method / f'seed{seed}' / f'fold{fold}'
                    local = dict(definition, reconstruction_encoder_gradient=condition == 'joint',
                        reconstruction_gradient_interval=6 if smoke else config['gradient_interval'],
                        checkpoint_every=6 if smoke else 50, save_evaluation_models=True)
                    context = dict(condition=condition, completed=len(outputs), total=total,
                                   progress_path=str(root / 'training_progress.json'))
                    result = training.fit(local, method, seed, raw, offsets, records, fitting, held,
                        folder, steps, [steps], identity, context, True, stop_after)
                    outputs.append(dict(condition=condition, method=method, seed=seed, fold=fold,
                        fit_videos=fitting, held_videos=held, directory=str(folder), summary=result))
                    atomic_write_json(root / 'progress.json', dict(completed=len(outputs), total=total,
                        elapsed_seconds=time.perf_counter() - start, updated_at=training.shared.now()))
                    if smoke and condition == 'joint':
                        reference = root / 'uninterrupted_reference' / method
                        plain = dict(local)
                        del plain['reconstruction_gradient_interval']
                        del plain['reconstruction_encoder_gradient']
                        training.fit(plain, method, seed, raw, offsets, records, fitting, held,
                            reference, steps, [steps], identity, context, True, None)
                        with np.load(folder / 'model.npz') as first, np.load(reference / 'model.npz') as second:
                            for key in first.files:
                                np.testing.assert_array_equal(first[key], second[key])
                        assert read_json(folder / 'sequence.json') == read_json(reference / 'sequence.json')
    for seed in seeds:
        for fold in range(len(active_folds)):
            assert len({r['summary']['sequence_sha256'] for r in outputs if r['seed'] == seed and r['fold'] == fold}) == 1
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', outputs=outputs, steps=steps,
        identity=identity, elapsed_seconds=time.perf_counter() - start,
        paired_sequences_exact=True, completed_at=training.shared.now()))
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=total, total=total))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = read_json(run / 'config.json')
    root = run / 'smoke' if smoke else run
    training_result = read_json(root / 'training_summary.json')
    records = read_json(root / 'records.json')
    steps = training_result['steps']
    rows, gradients, fidelity, checks = [], [], [], []
    for item in training_result['outputs']:
        folder = Path(item['directory'])
        reference = root / 'joint' / item['method'] / f"seed{item['seed']}" / f"fold{item['fold']}"
        with np.load(folder / f'held_{steps:04d}.npz') as saved, np.load(reference / f'held_{steps:04d}.npz') as original:
            data = {key: saved[key].copy() for key in saved.files}
            before = original['scores'].copy()
            for key in ['source', 'query', 'same_identity', 'indices']:
                np.testing.assert_array_equal(data[key], original[key])
        lookup = {int(v): i for i, v in enumerate(data['indices'])}
        source = np.array([lookup[int(v)] for v in data['source']])
        query = np.array([lookup[int(v)] for v in data['query']])
        unit = data['embeddings'].astype(np.float64)
        direct = np.sum(unit[source] * unit[query], 1)
        error = float(np.max(np.abs(direct - data['scores'])))
        np.testing.assert_allclose(direct, data['scores'], atol=3e-7, rtol=1e-6)
        labels = data['same_identity'].astype(bool)
        videos = np.array([records[int(i)]['video_id'] for i in data['source']])
        for i, j, same in zip(data['source'], data['query'], labels, strict=True):
            a, b = records[int(i)], records[int(j)]
            assert a['video_id'] == b['video_id'] and a['end_frame'] < b['start_frame']
            assert same == (a['lesion_id'] == b['lesion_id'])
        assert not set(videos) & set(item['fit_videos'])
        eligible = [v for v in np.unique(videos) if len(set(labels[videos == v])) == 2]
        metadata = {k: item[k] for k in ['condition', 'method', 'seed', 'fold']}
        for video in eligible:
            fit = np.isin(videos, [v for v in eligible if v != video])
            held = videos == video
            assert fit.any()
            threshold = boundary(data['scores'][fit], videos[fit], labels[fit], config['negative_quantile'])
            original_threshold = boundary(before[fit], videos[fit], labels[fit], config['negative_quantile'])
            scores = data['scores'][held]
            outcome = measurements(before[held], scores, labels[held], threshold, original_threshold)
            assert outcome['corrected'] <= outcome['original_wrong']
            assert outcome['positives'] == int(labels[held].sum())
            rows.append(dict(metadata, video=video, **outcome))
        for row in read_json(folder / f'held_{steps:04d}.json')['by_clip']:
            fidelity.append(dict(metadata, **row))
        for row in read_json(folder / 'history.json'):
            if 'encoder_gradient_cosine' in row:
                gradients.append(dict(metadata, **row))
        checks.append(dict(metadata, pairs=len(labels), score_error=error,
            prediction_sha256=digest(folder / f'held_{steps:04d}.npz')))
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'procedures.csv', index=False)
    metrics = ['auroc', 'recall', 'protection', 'matched_capacity', 'protected99_capacity']
    counts = ['corrected', 'repeat_damage', 'other_damage', 'repeat_gain', 'original_wrong']
    aggregate = {**{k:'mean' for k in metrics}, **{k:'sum' for k in counts}}
    seeds = table.groupby(['condition', 'method', 'seed']).agg(aggregate).reset_index()
    seeds.to_csv(output / 'seeds.csv', index=False)
    seeds.groupby(['condition', 'method']).agg(aggregate).reset_index().to_csv(output / 'summary.csv', index=False)
    gradient = pd.DataFrame(gradients)
    gradient['norm_ratio'] = gradient.encoder_reconstruction_gradient_norm / gradient.encoder_task_gradient_norm
    gradient.to_csv(output / 'gradients.csv', index=False)
    clip = pd.DataFrame(fidelity)
    clip.to_csv(output / 'fidelity_clips.csv', index=False)
    procedure = clip.groupby(['condition','method','seed','fold','video_id'])[['reconstruction_nmse','local_active']].mean().reset_index()
    procedure.to_csv(output / 'fidelity_procedures.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, method in enumerate(config['methods']):
        for col, metric in enumerate(['auroc','protected99_capacity','protection']):
            ax = axes[row, col]
            selected = seeds[seeds.method == method]
            for seed, values in selected.groupby('seed'):
                values = values.set_index('condition').reindex(config['conditions'])
                ax.plot([0,1], values[metric], marker='o', label=str(seed))
            ax.set_xticks([0,1], ['Joint', 'Task encoder'])
            ax.set_title(method + ' | ' + metric.replace('_',' '))
            ax.legend(fontsize=7)
    fig.suptitle('Original training procedures | fixed1000updates' if not smoke else 'Real12step smoke')
    fig.savefig(output / 'comparison.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for (condition,method), values in gradient.groupby(['condition','method']):
        trend = values.groupby('step')[['encoder_gradient_cosine','norm_ratio']].mean()
        for ax, field in zip(axes[:2], ['encoder_gradient_cosine','norm_ratio'], strict=True):
            ax.plot(trend.index, trend[field], label=method+' '+condition)
            ax.set_title(field.replace('_',' '))
            ax.set_xlabel('Training step')
    axes[0].axhline(0,color='black',linewidth=.6)
    axes[1].set_yscale('log')
    means=procedure.groupby(['condition','method']).reconstruction_nmse.mean()
    axes[2].bar(range(len(means)),means)
    axes[2].set_xticks(range(len(means)),[a+'\n'+b for a,b in means.index],rotation=30,ha='right',fontsize=7)
    axes[2].set_title('Held procedure mean reconstruction NMSE')
    axes[0].legend(fontsize=7)
    fig.savefig(output/'mechanism.png',dpi=170)
    plt.close(fig)
    atomic_write_json(output/'verification.json', dict(status='PASS', checks=checks,
        models=len(checks), procedure_rows=len(rows), paired_sequences_exact=training_result['paired_sequences_exact'],
        all_rows_encoder_excluded=True, completed_at=training.shared.now()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['train','summarize'],required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if args.phase=='train':
        train(args.run,args.smoke,args.stop_after)
    else:
        summarize(args.run,args.smoke)
