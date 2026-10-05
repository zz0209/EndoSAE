import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared
from train_temporal_shared_sae import protected_statistics
from train_temporal_view_identity import read_tokens
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.token_identity_sae import TokenIdentitySAE


def original_inputs(config):
    cohort = read_json(config['cohort_config'])
    chunks, bounds, records = [], [0], []
    for partition in ['train', 'val']:
        for video in cohort['fit_video_ids'][partition]:
            value, offsets, local, _ = read_tokens(Path(config['original_prepared_run']) / 'tokens' / video)
            base = len(records)
            records.extend(dict(row, partition=partition, global_index=base + i) for i, row in enumerate(local))
            chunks.append(value)
            bounds.extend((offsets[1:] + bounds[-1]).tolist())
    if len(records) != 336:
        raise ValueError('Original evaluation observation roster changed')
    for video in config.get('added_training_videos', []):
        value, offsets, local, _ = read_tokens(Path(config['added_training_storage']) / 'tokens' / video)
        for index, row in enumerate(local):
            if row['original_observation']:
                records.append(dict(row, partition='train', global_index=len(records)))
                chunks.append(value[offsets[index]:offsets[index + 1]])
                bounds.append(bounds[-1] + offsets[index + 1] - offsets[index])
    if len(records) != 336 + config.get('added_original_observations', 0):
        raise ValueError('Added component fitting observations differ')
    return np.concatenate(chunks), np.asarray(bounds), records


def component_means(terms, pairs):
    labels = np.asarray([row['same_identity'] for row in pairs])
    positive, negative = [], []
    for video in sorted({row['video_id'] for row in pairs}):
        mask = np.asarray([row['video_id'] == video for row in pairs])
        if np.any(mask & labels) and np.any(mask & ~labels):
            positive.append(terms[mask & labels].mean(axis=0))
            negative.append(terms[mask & ~labels].mean(axis=0))
    if not positive:
        raise ValueError('No fitting procedure supports a component contrast')
    return np.mean(positive, axis=0), np.mean(negative, axis=0)


def remove_components(unit, source, query, selected):
    first, second = unit[source], unit[query]
    products = np.sum(first * second, axis=1)
    if len(selected) == 0:
        return products
    remaining_first = np.sum(first * first, axis=1) - np.sum(first[:, selected] ** 2, axis=1)
    remaining_second = np.sum(second * second, axis=1) - np.sum(second[:, selected] ** 2, axis=1)
    if np.any(remaining_first <= 0) or np.any(remaining_second <= 0):
        raise ValueError('Component removal eliminates an entire representation')
    return (products - np.sum(first[:, selected] * second[:, selected], axis=1)) / np.sqrt(remaining_first * remaining_second)


def intervention_rows(pairs, before, after, boundary, threshold):
    labels = np.asarray([row['same_identity'] for row in pairs])
    values = []
    for video in sorted({row['video_id'] for row in pairs}):
        mask = np.asarray([row['video_id'] == video for row in pairs])
        positive, negative = mask & labels, mask & ~labels
        if not positive.any() or not negative.any():
            continue
        false = negative & (before > threshold)
        true = positive & (before > threshold)
        values.append(dict(video_id=video, boundary=boundary, threshold=float(threshold),
            recall=float(np.mean(after[positive] > threshold)),
            negative_retention=float(np.mean(after[negative] <= threshold)),
            auroc=float(roc_auc_score(labels[mask], after[mask])),
            false_positive_correction=float(np.mean(after[false] <= threshold)) if false.any() else None,
            true_positive_damage=float(np.mean(after[true] <= threshold)) if true.any() else None,
            positive_score_change=float(np.mean(after[positive] - before[positive])),
            negative_score_change=float(np.mean(after[negative] - before[negative]))))
    return values


@torch.no_grad()
def analyze(run, training_root, output, resume, input_loader=original_inputs):
    config = read_json(run / 'config.json')
    training = read_json(training_root / 'training_summary.json')
    if training['status'] != 'COMPLETE':
        raise ValueError('Training is incomplete')
    output.mkdir(parents=True, exist_ok=True)
    raw, offsets, records = input_loader(config)
    lookup = {(row['clip_id'], row['lesion_id']): i for i, row in enumerate(records)}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected_steps = {(row['condition'], row['method'], row['seed']): row['step'] for row in training['selection']}
    jobs = [item for item in training['outputs'] if item['method'] != 'raw_supcon']
    receipts = []
    for position, item in enumerate(jobs):
        step = selected_steps[item['condition'], item['method'], item['seed']]
        folder = Path(item['directory'])
        model_path = folder / f'model_{step:04d}.npz'
        relative = Path(item['condition']) / item['method'] / f"seed{item['seed']}" / f"fold{item['fold']}"
        directory = output / relative
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(model_sha256=shared.file_sha256(model_path),
            normalization_sha256=shared.file_sha256(folder / 'normalization.npz'),
            training_sha256=shared.file_sha256(training_root / 'training_summary.json'),
            source_sha256=shared.file_sha256(__file__), config_sha256=shared.file_sha256(run / 'config.json'))
        receipt_path = directory / 'complete.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if not resume or receipt['identity'] != identity:
                raise ValueError('Component analysis identity changed')
            for name, expected in receipt['assets'].items():
                if shared.file_sha256(directory / name) != expected:
                    raise ValueError('Saved component analysis changed')
            receipts.append(dict(item, analysis_directory=str(directory)))
            continue
        with np.load(folder / 'normalization.npz', allow_pickle=False) as archive:
            mean, scale = archive['mean'].copy(), archive['scale'].copy()
        model = TokenIdentitySAE(config, item['method']).to(config['device']).eval()
        with np.load(model_path, allow_pickle=False) as archive:
            model.load_state_dict({key: torch.from_numpy(archive[key].copy()).to(config['device']) for key in archive.files})
        videos = set(item['fit_videos']) | set(item['held_videos'])
        embeddings, pooled, reconstruction, counts = {}, {}, [], []
        for index, row in enumerate(records):
            if row['video_id'] not in videos:
                continue
            values = torch.from_numpy(((raw[offsets[index]:offsets[index + 1]] - mean) / scale).astype(np.float32)).to(config['device'])[None]
            projected, decoded, local = model(values)
            embeddings[index] = projected[0].cpu().numpy()
            pooled[index] = local.mean(dim=1)[0].cpu().numpy()
            reconstruction.append(dict(index=index, video_id=row['video_id'],
                scope='fit' if row['video_id'] in item['fit_videos'] else 'held',
                reconstruction_nmse=float((decoded - values).square().mean() / values.square().mean()),
                active=float((local > 0).sum(-1).float().mean())))
            counts.append(index)
        unit = np.zeros((len(records), config['latent_dim']), dtype=np.float64)
        for index in counts:
            norm = np.linalg.norm(pooled[index].astype(np.float64))
            if norm <= 0:
                raise ValueError('Empty identity code')
            unit[index] = pooled[index] / norm
        job_records = read_json(folder / 'records.json')
        with np.load(folder / f'held_{step:04d}.npz', allow_pickle=False) as archive:
            expected_indices = archive['indices'].copy()
            expected_embeddings = archive['embeddings'].copy()
            expected_scores = archive['scores'].copy()
            saved_source, saved_query = archive['source'].copy(), archive['query'].copy()
        expected_mapping = [lookup[(job_records[i]['clip_id'], job_records[i]['lesion_id'])] for i in expected_indices]
        actual_embeddings = np.stack([embeddings[i] for i in expected_mapping])
        np.testing.assert_array_equal(actual_embeddings, expected_embeddings)
        current_source = np.array([lookup[(job_records[i]['clip_id'], job_records[i]['lesion_id'])] for i in saved_source])
        current_query = np.array([lookup[(job_records[i]['clip_id'], job_records[i]['lesion_id'])] for i in saved_query])
        direct = np.array([float(embeddings[i] @ embeddings[j]) for i, j in zip(current_source, current_query, strict=True)])
        np.testing.assert_array_equal(direct, expected_scores)
        scopes, selections = {}, {}
        for scope, procedures in [('fit', item['fit_videos']), ('held', item['held_videos'])]:
            pairs = shared.chronological_pairs(records, procedures)
            source = np.array([row['source_index'] for row in pairs])
            query = np.array([row['query_index'] for row in pairs])
            terms = unit[source] * unit[query]
            positive, negative = component_means(terms, pairs)
            scopes[scope] = dict(pairs=pairs, source=source, query=query, terms=terms,
                positive=positive, negative=negative, before=terms.sum(axis=1))
        ranking = np.argsort(-(scopes['fit']['negative'] - scopes['fit']['positive']), kind='stable')
        selections['selected'] = ranking[:config['component_selection_count']]
        for i, seed in enumerate(config['component_random_seeds']):
            selections[f'random{i}'] = np.random.default_rng(seed).choice(config['latent_dim'],
                size=config['component_selection_count'], replace=False)
        held = scopes['held']
        if not np.array_equal(held['source'], current_source) or not np.array_equal(held['query'], current_query):
            raise ValueError('Original evaluation pair ordering changed')
        np.testing.assert_allclose(held['before'], expected_scores, atol=3e-7, rtol=1e-6)
        thresholds = {scope: protected_statistics(data['pairs'], data['before'], config['negative_quantile'])['threshold']
                      for scope, data in scopes.items()}
        arrays = dict(unit_codes=unit[counts], indices=np.array(counts), ranking=ranking,
            fit_positive=scopes['fit']['positive'], fit_negative=scopes['fit']['negative'],
            held_positive=held['positive'], held_negative=held['negative'])
        results = []
        for scope, data in scopes.items():
            arrays[f'{scope}_source'] = data['source']
            arrays[f'{scope}_query'] = data['query']
            arrays[f'{scope}_labels'] = np.array([row['same_identity'] for row in data['pairs']])
            for variant, features in [('before', np.array([], dtype=int)), *selections.items()]:
                after = remove_components(unit, data['source'], data['query'], features)
                arrays[f'{scope}_{variant}'] = after
                if len(features):
                    arrays[variant + '_features'] = features
                own_threshold = protected_statistics(data['pairs'], after, config['negative_quantile'])['threshold']
                for boundary, threshold in [('original_scope', thresholds[scope]), ('own_scope', own_threshold),
                                             ('original_fit', thresholds['fit'])]:
                    results.extend(dict(scope=scope, variant=variant, **row) for row in
                        intervention_rows(data['pairs'], data['before'], after, boundary, threshold))
        shared.save_npz(directory / 'effects.npz', **arrays)
        atomic_write_json(directory / 'procedures.json', results)
        atomic_write_json(directory / 'representation.json', reconstruction)
        receipt = dict(status='COMPLETE', identity=identity, selected_step=step,
            held_forward_exact=True, held_scores_exact=True, component_sum_max_error=float(np.max(np.abs(held['before'] - expected_scores))),
            assets={name: shared.file_sha256(directory / name) for name in ['effects.npz', 'procedures.json', 'representation.json']})
        atomic_write_json(receipt_path, receipt)
        receipts.append(dict(item, analysis_directory=str(directory)))
        atomic_write_json(output / 'progress.json', dict(status='RUNNING', completed=position + 1, total=len(jobs),
            phase=str(relative)))
        print('COMPONENT_MODEL', position + 1, '/', len(jobs), str(relative), flush=True)
        pause_after_checkpoint(receipt_path)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', models=receipts))
    atomic_write_json(output / 'progress.json', dict(status='COMPLETE', completed=len(jobs), total=len(jobs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.training_root or args.run
    analyze(args.run, root, args.output or root / 'components', args.resume)
