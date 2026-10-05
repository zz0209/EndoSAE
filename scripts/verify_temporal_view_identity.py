import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_temporal_view_identity import load_inputs, job_inputs
from train_frozen_identity_supcon import supcon
from verify_token_identity_sae import numpy_forward
from analyze_temporal_identity_components import remove_components
from src.checkpoint_io import atomic_write_json, read_json
from src.token_identity_sae import TokenIdentitySAE


def verify(run, recovery, input_loader=load_inputs):
    torch.set_num_threads(1)
    config, cohort, raw, offsets, records, _ = input_loader(run)
    smoke = read_json(run / 'smoke' / 'training_summary.json')
    checks, normalization_checks, restored = [], [], 0
    pair_rosters = {}
    for condition in config['conditions']:
        jobs = [item for item in smoke['outputs'] if item['condition'] == condition]
        for fold in [0, 'full']:
            batch = [item for item in jobs if item['fold'] == fold]
            fitting = batch[0]['fit_videos']
            values, bounds, local = job_inputs(raw, offsets, records, fitting, condition)
            groups = {}
            for i, row in enumerate(local):
                if row['video_id'] in fitting:
                    groups.setdefault((row['video_id'], row['lesion_id']), []).append(i)
            means, seconds = [], []
            for indices in groups.values():
                means.append(np.mean([values[bounds[i]:bounds[i + 1]].mean(0, dtype=np.float64) for i in indices], axis=0))
                seconds.append(np.mean([np.square(values[bounds[i]:bounds[i + 1]].astype(np.float64)).mean(0) for i in indices], axis=0))
            mean = np.mean(means, axis=0)
            scale = np.sqrt(np.maximum(np.mean(seconds, axis=0) - mean * mean, 0))
            scale[scale == 0] = 1
            selected = []
            for indices in list(groups.values())[:2]:
                first = indices[0]
                second = next((i for i in indices if not local[i]['original_observation']), indices[-1])
                selected.extend([first, second])
            samples = np.stack([values[np.linspace(bounds[i], bounds[i + 1] - 1, 8, dtype=int)] for i in selected])
            samples = (samples - mean) / scale
            labels = torch.tensor([0, 0, 1, 1])
            for item in batch:
                folder = Path(item['directory'])
                with np.load(folder / 'normalization.npz', allow_pickle=False) as archive:
                    np.testing.assert_allclose(archive['mean'], mean, atol=2e-13, rtol=2e-13)
                    np.testing.assert_allclose(archive['scale'], scale, atol=2e-12, rtol=2e-12)
                normalization_checks.append(dict(condition=condition, fold=fold, method=item['method'],
                    equal_lesion_observation_weights=True, fit_observations=sum(len(x) for x in groups.values())))
                model = TokenIdentitySAE(config, item['method']).double()
                with np.load(folder / 'model_0012.npz', allow_pickle=False) as archive:
                    model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files})
                expected, decoded = numpy_forward(model, samples)
                actual, reconstruction, _ = model(torch.from_numpy(samples))
                np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-11, rtol=1e-10)
                if decoded is not None:
                    np.testing.assert_allclose(reconstruction.detach().numpy(), decoded, atol=1e-11, rtol=1e-10)
                logits = expected @ expected.T / config['temperature']
                np.fill_diagonal(logits, -np.inf)
                probability = logits - logsumexp(logits, axis=1, keepdims=True)
                positive = (labels.numpy()[:, None] == labels.numpy()[None, :]) & ~np.eye(4, dtype=bool)
                expected_loss = -np.mean(probability[positive])
                if decoded is not None:
                    expected_loss += config['reconstruction_weight'] * np.mean((decoded - samples) ** 2)

                def objective():
                    output, recreated, _ = model(torch.from_numpy(samples))
                    loss = supcon(output, labels, config['temperature'])
                    return loss + config['reconstruction_weight'] * (recreated - torch.from_numpy(samples)).square().mean() if recreated is not None else loss

                loss = objective()
                np.testing.assert_allclose(float(loss.detach()), expected_loss, atol=1e-11, rtol=1e-10)
                loss.backward()
                generator = torch.Generator().manual_seed(31005)
                directions = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in model.parameters()]
                length = torch.sqrt(sum(direction.square().sum() for direction in directions))
                directions = [direction / length for direction in directions]
                derivative = float(sum((p.grad * d).sum() for p, d in zip(model.parameters(), directions)).detach())
                original = [p.detach().clone() for p in model.parameters()]
                outcomes = []
                for sign in [-1, 1]:
                    with torch.no_grad():
                        for p, value, direction in zip(model.parameters(), original, directions, strict=True):
                            p.copy_(value + sign * 1e-5 * direction)
                        outcomes.append(float(objective()))
                numerical = (outcomes[1] - outcomes[0]) / 2e-5
                np.testing.assert_allclose(derivative, numerical, atol=2e-7, rtol=2e-4)
                with np.load(folder / 'held_0012.npz', allow_pickle=False) as archive:
                    roster = [(local[i]['clip_id'], local[i]['lesion_id'], local[j]['clip_id'], local[j]['lesion_id'])
                        for i, j in zip(archive['source'], archive['query'], strict=True)]
                key = (item['method'], fold)
                if key in pair_rosters:
                    if pair_rosters[key] != roster:
                        raise ValueError('Observation conditions have different evaluation pairs')
                else:
                    pair_rosters[key] = roster
                checks.append(dict(condition=condition, fold=fold, method=item['method'],
                    forward_error=float(np.max(np.abs(actual.detach().numpy() - expected))),
                    gradient_error=abs(derivative - numerical), actual_observations=selected))
                if recovery:
                    relative = folder.relative_to(run / 'smoke')
                    for filename in ['model.npz', 'model_0012.npz', 'normalization.npz', 'held_0012.npz']:
                        with np.load(folder / filename, allow_pickle=False) as first, np.load(recovery / relative / filename, allow_pickle=False) as second:
                            if first.files != second.files:
                                raise ValueError('Recovery keys differ')
                            for name in first.files:
                                np.testing.assert_array_equal(first[name], second[name])
                                restored += 1
                    if read_json(folder / 'sequence.json') != read_json(recovery / relative / 'sequence.json'):
                        raise ValueError('Recovery sampling differs')
            del values
    components = read_json(run / 'smoke' / 'components' / 'summary.json')
    component_checks = []
    for item in components['models']:
        directory = Path(item['analysis_directory'])
        with np.load(directory / 'effects.npz', allow_pickle=False) as archive:
            sparse = archive['unit_codes'].copy()
            indices = archive['indices'].copy()
            unit = np.zeros((sum(row['original_observation'] for row in records), sparse.shape[1]))
            unit[indices] = sparse
            for scope in ['fit', 'held']:
                source, query = archive[f'{scope}_source'], archive[f'{scope}_query']
                for variant in ['selected', 'random0', 'random1']:
                    features = archive[variant + '_features']
                    changed = unit.copy()
                    changed[:, features] = 0
                    norms = np.linalg.norm(changed[indices], axis=1)
                    if np.any(norms <= 0):
                        raise ValueError('Invalid actual component fixture')
                    changed[indices] /= norms[:, None]
                    expected = np.sum(changed[source] * changed[query], axis=1)
                    actual = remove_components(unit, source, query, features)
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                    np.testing.assert_array_equal(actual, archive[f'{scope}_{variant}'])
                    component_checks.append(float(np.max(np.abs(actual - expected))))
    atomic_write_json(run / 'numerical_verification.json', dict(status='PASS', models=checks,
        normalization=normalization_checks, held_pair_rosters_identical=True, original_arrays_exact=244,
        recovery_arrays_exact=restored, component_formula_checks=len(component_checks),
        component_formula_max_error=max(component_checks)))
    print('TEMPORAL_VERIFICATION_PASS', len(checks), 'models', restored, 'recovery arrays', len(component_checks), 'component checks', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recovery', type=Path)
    args = parser.parse_args()
    verify(args.run, args.recovery)
