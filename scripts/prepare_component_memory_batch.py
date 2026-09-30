import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--scan-seconds', type=float, default=900.)
    parser.add_argument('--evaluation-seconds', type=float, default=530.)
    parser.add_argument('--discovery-script', default='scripts/discover_component_memory.py')
    parser.add_argument('--discovery-label', default='成分作用与组合干预')
    parser.add_argument('--discovery-heads', action='store_true')
    parser.add_argument('--head-seconds', type=float, default=120.)
    args = parser.parse_args()
    run = args.run
    config = read_json(run / 'config.json')
    parent = Path(config['parent_dictionary_run'])
    reused = {}
    for base in (run, run / 'smoke'):
        target = base / 'fit/spatial_weighted'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('model.npz', 'normalization.npz', 'summary.json'):
            source = parent / 'fit/spatial_weighted' / name
            expected = digest(source)
            if (target / name).exists():
                assert digest(target / name) == expected
            else:
                shutil.copyfile(source, target / name)
                assert digest(target / name) == expected
            reused[str(target / name)] = dict(source=str(source), sha256=expected)
    atomic_write_json(run / 'baseline_reuse.json', reused)
    if args.prepare_only:
        return
    python = str(ROOT / 'artifacts/environments/modern/Scripts/python.exe')
    evaluation_run = ('\\\\?\\' if os.name == 'nt' else '') + str(run.resolve())
    stages = []
    if args.discovery_heads:
        stages.append(dict(id='discovery_heads', label='发现成分所用的独立身份模型', kind='heads',
            command=[python, args.discovery_script, '--config', str(run / 'config.json'),
                     '--phase', 'heads', '--resume'],
            progress=str(run / 'head_progress.json'), output=str(run / 'head_summary.json'),
            units=9, unit='个身份模型', estimate_seconds=args.head_seconds, resources=[]))
    for method in config['methods']:
        stages.append(dict(id='discover_' + method,
            label=('SAE · ' if method == 'sparse_edit' else '普通字典 · ') + args.discovery_label,
            kind='component_scan', method=method,
            command=[python, args.discovery_script, '--config', str(run / 'config.json'),
                     '--phase', 'fit', '--method', method, '--resume'],
            progress=str(run / 'training_progress.json'), output=str(run / f'training_summary_{method}.json'),
            units=12, unit='组字典（3 seeds × 3 folds＋完整拟合）',
            estimate_seconds=args.scan_seconds, resources=['gpu-0']))
    for seed in config['seeds']:
        stages.append(dict(id=f'evaluate_{seed}', label=f'种子 {seed} · 完整提示与等幅干预', kind='evaluate',
            command=[python, 'scripts/evaluate_component_memory.py', '--run', evaluation_run,
                     '--phase', 'evaluate', '--seed', str(seed), '--resume'],
            progress=str(run / 'evaluation' / f'seed{seed}' / 'progress.json'),
            output=str(run / 'evaluation' / f'seed{seed}' / 'summary.json'),
            units=10, unit='个 procedure', estimate_seconds=args.evaluation_seconds,
            resources=['disk-e-io', 'disk-d-io']))
    stages.append(dict(id='summary', label='成分干预与逐 procedure 结果汇总', kind='summary',
        command=[python, 'scripts/evaluate_component_memory.py', '--run', evaluation_run, '--phase', 'summary', '--resume'],
        progress=str(run / 'summary_progress.json'), output=str(run / 'summary.json'),
        units=1, unit='份完整分析', estimate_seconds=10, resources=[]))
    destination = run / 'batch_plan.json'
    if destination.exists():
        raise FileExistsError(destination)
    code = ['scripts/discover_component_memory.py', 'src/component_memory_intervention.py',
            'scripts/evaluate_component_memory.py', 'scripts/evaluate_token_memory_edit.py',
            'scripts/prepare_component_memory_batch.py']
    if args.discovery_script not in code:
        code.append(args.discovery_script)
    atomic_write_json(destination, dict(question=read_json(run / 'protocol.json')['question'], stages=stages,
        config_sha256=digest(run / 'config.json'), protocol_sha256=digest(run / 'protocol.json'),
        source_hashes={name: digest(ROOT / name) for name in code},
        estimate_basis='Real full-component smoke throughput and prior complete application evaluation times.'))
    print(destination)


if __name__ == '__main__':
    main()
