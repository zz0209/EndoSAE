import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_token_identity_sae as training
from analyze_temporal_identity_components import original_inputs
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def initialize(run, smoke):
    config = read_json(run / 'config.json')
    root = run / 'smoke' if smoke else run
    root.mkdir(exist_ok=True)
    seeds = config['seeds'][:1] if smoke else config['seeds']
    command = [str(training.ROOT / 'artifacts/environments/modern/Scripts/python.exe')]
    stages = []

    def stage(identifier, script, arguments, resources, units, progress, output, seconds):
        stages.append(dict(id=identifier, kind=identifier.split('_')[0], label=identifier,
            command=command + ['scripts/' + script] + arguments, resources=resources, units=units,
            unit='items', progress=str(progress), output=str(output), estimate_seconds=seconds))

    stage('train', 'run_multibudget_identity.py', ['--run', str(run), '--phase', 'train'] + (['--smoke'] if smoke else []),
          ['gpu-0'], len(seeds) * 2, root / 'training_progress.json', root / 'training_summary.json', 80)
    reference = read_json(Path(config['reference_run']) / 'config.json')
    for condition in config['conditions']:
        for budget in config['evaluation_budgets']:
            child = root / f'{condition}_k{budget}'
            child.mkdir(exist_ok=True)
            definition = dict(reference, run_id=child.name, bank_run=str(child),
                fitting_vectors_root=str(child / 'fitting_vectors'), methods=['token_sparse'], seeds=seeds,
                alternative_training_runs={'p27v4': str(root / condition)}, inference_top_k={'token_sparse': budget},
                figure_label=f'{condition} K{budget}', figure_title=f'Paired continued training | {condition} K{budget}')
            if smoke:
                definition['development_videos'] = [definition['smoke_video']]
            for name, value in [('config.json', definition), ('protocol.json', read_json(run / 'protocol.json'))]:
                path = child / name
                if path.exists():
                    assert read_json(path) == value
                else:
                    atomic_write_json(path, value)
            args = ['--run', str(child)]
            stage('prepare_' + child.name, 'prepare_inference_budget.py', args + ['--resume'], ['gpu-0'],
                  28 * len(seeds) if smoke else 32 * len(seeds), child / 'preparation/progress.json', child / 'preparation/summary.json', 50)
            for phase, resources, units, seconds in [('fit', [], 13 * len(seeds), 10),
                    ('score', ['disk-d-io', 'gpu-0'], 3822 if smoke else 33622, 30 if smoke else 550),
                    ('evaluate', [], 13 * len(seeds), 40 if smoke else 750), ('summarize', [], 1, 10)]:
                directory = {'fit': 'fit', 'score': 'scores', 'evaluate': 'evaluation', 'summarize': 'analysis'}[phase]
                stage(phase + '_' + child.name, 'evaluate_exemplar_bank_direction.py', args + ['--phase', phase, '--resume'],
                      resources, units, child / directory / 'progress.json',
                      child / ('selection.json' if phase == 'fit' else directory + '/summary.json'), seconds)
            stage('verify_' + child.name, 'verify_inference_budget.py', args, [], 1,
                  child / 'verification.json', child / 'verification.json', 10)
    stage('compare', 'run_multibudget_identity.py', ['--run', str(run), '--phase', 'compare'] + (['--smoke'] if smoke else []),
          [], 1, root / 'comparison.json', root / 'comparison.json', 10)
    plan = dict(stages=stages, budget_seconds=1200 if smoke else 6500)
    target = root / 'batch_plan.json'
    if target.exists():
        assert read_json(target) == plan
    else:
        atomic_write_json(target, plan)


def train(run, smoke, stop_after):
    config = read_json(run / 'config.json')
    root = run / 'smoke' if smoke else run
    source = Path(config['initial_training'])
    definition = read_json(source / 'fit/token_sparse' / f"seed{config['seeds'][0]}" / 'model_config.json')
    raw, offsets, records = original_inputs(definition)
    fitting = sorted({r['video_id'] for r in records if r['partition'] == 'train'})
    assert len(fitting) == 27 and sum(r['partition'] == 'train' for r in records) == 340
    identity = dict(source_hashes=training.source_identity(), refinement_source=digest(__file__),
                    protocol=digest(run / 'protocol.json'), refinement_config=digest(run / 'config.json'),
                    input_hash=training.shared.json_digest(dict(records=records, offsets=offsets.tolist())))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for condition in config['conditions']:
        for seed in config['seeds'][:1] if smoke else config['seeds']:
            initial = source / 'fit/token_sparse' / f'seed{seed}'
            local = dict(read_json(initial / 'model_config.json'), initial_model_folder=str(initial),
                         learning_rate=config['learning_rate'], checkpoint_every=6 if smoke else 50,
                         save_evaluation_models=False)
            if condition == 'joint_budget':
                local['training_budgets'] = config['evaluation_budgets']
            else:
                assert condition == 'continued_k64'
            current = dict(identity, initial_assets={name: digest(initial / name) for name in
                           ['model.npz', 'normalization.npz', 'model_config.json']})
            folder = root / condition / 'fit/token_sparse' / f'seed{seed}'
            context = dict(progress_path=str(root / 'step_progress.json'), condition=condition)
            result = training.fit(local, 'token_sparse', seed, raw, offsets, records, fitting, [], folder,
                                  12 if smoke else config['steps'], [], current, context, True, stop_after)
            if smoke and condition == 'continued_k64':
                duplicate = root / 'uninterrupted_reference'
                training.fit(local, 'token_sparse', seed, raw, offsets, records, fitting, [], duplicate,
                             12, [], current, context, True, None)
                with np.load(folder / 'model.npz') as first, np.load(duplicate / 'model.npz') as second:
                    for key in first.files:
                        np.testing.assert_array_equal(first[key], second[key])
            with np.load(folder / 'model.npz') as trained, np.load(initial / 'model.npz') as start:
                assert any(not np.array_equal(trained[key], start[key]) for key in trained.files)
            with np.load(folder / 'normalization.npz') as trained, np.load(initial / 'normalization.npz') as start:
                for key in start.files:
                    np.testing.assert_array_equal(trained[key], start[key])
            results.append(dict(condition=condition, **result))
            atomic_write_json(root / 'training_progress.json', dict(completed=len(results), total=2 * (1 if smoke else len(config['seeds']))))
    for seed in config['seeds'][:1] if smoke else config['seeds']:
        assert len({r['sequence_sha256'] for r in results if r['seed'] == seed}) == 1
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', results=results,
        paired_sequences_exact=True, initial_normalization_exact=True, training_observations=340,
        training_procedures=27, identity=identity))


def compare(run, smoke):
    config = read_json(run / 'config.json')
    root = run / 'smoke' if smoke else run
    frames, diagnostics = [], []
    for condition in config['conditions']:
        for budget in config['evaluation_budgets']:
            child = root / f'{condition}_k{budget}'
            assert read_json(child / 'verification.json')['status'] == 'PASS'
            frame = pd.read_csv(child / 'analysis/procedures.csv')
            frame['condition'], frame['budget'] = condition, budget
            frames.append(frame)
            local = pd.read_csv(child / 'preparation/diagnostics.csv')
            local['condition'], local['budget'] = condition, budget
            diagnostics.append(local)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(root / 'comparison_procedures.csv', index=False)
    metric = ['matched_retention_repeat', 'zero_loss_repeat']
    seeds = data.groupby(['condition', 'budget', 'policy', 'seed'])[metric].mean().reset_index()
    seeds.to_csv(root / 'comparison_seeds.csv', index=False)
    summary = seeds.groupby(['condition', 'budget', 'policy'])[metric].mean().reset_index()
    summary.to_csv(root / 'comparison_summary.csv', index=False)
    pd.concat(diagnostics).groupby(['condition', 'budget', 'stage', 'model'])[['changed_l0', 'changed_nmse']].mean().to_csv(root / 'comparison_fidelity.csv')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, field in zip(axes, metric):
        for condition in config['conditions']:
            selected = seeds[(seeds.condition == condition) & (seeds.policy == 'exemplar_svm')]
            for seed, local in selected.groupby('seed'):
                ax.plot(local.budget, local[field] * 100, alpha=.25)
            means = selected.groupby('budget')[field].mean()
            ax.plot(means.index, means.values * 100, marker='o', label=condition)
        ax.set_xticks(config['evaluation_budgets'], ['TopK64', 'Full ReLU'])
        ax.set_ylabel('Repeat removal (%)')
        ax.set_title(field.replace('_', ' '))
        ax.legend()
    fig.suptitle('Same initialization and samples | 250-step paired refinement' if not smoke else 'Real smoke | 12-step paired refinement')
    fig.savefig(root / 'comparison.png', dpi=170)
    plt.close(fig)
    atomic_write_json(root / 'comparison.json', dict(status='COMPLETE', summary=summary.to_dict('records'),
        development_only=True, source_sha256=digest(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['initialize', 'train', 'compare'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.phase == 'initialize':
        initialize(args.run, args.smoke)
    elif args.phase == 'train':
        train(args.run, args.smoke, args.stop_after)
    else:
        compare(args.run, args.smoke)
