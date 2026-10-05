import argparse
from pathlib import Path

from encode_token_causal_identity import ROOT, video_inputs
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def prepare(run):
    config = read_json(run / 'config.json')
    smoke = read_json(run / 'smoke_encoding_summary.json')
    evaluation = read_json(run / 'smoke_evaluation' / f"seed{config['seeds'][0]}" / 'summary.json')
    assert smoke['status'] == evaluation['status'] == 'COMPLETE'
    counts = {video: len(video_inputs(config, video)[5]) for video in config['development_videos'] + config['extension_videos']}
    total = sum(counts.values())
    seconds = sum(row['seconds'] for row in smoke['videos']) / smoke['total_native_outputs'] * total
    python = str(ROOT / 'artifacts/environments/modern/Scripts/python.exe')
    stages = [dict(id='encode', label='完整视频 · 实际观察的局部成分编码', kind='encode',
        command=[python, 'scripts/encode_token_causal_identity.py', '--run', str(run), '--resume'],
        progress=str(run / 'encoding_progress.json'), output=str(run / 'encoding_summary.json'),
        units=total, unit='个视频时点', estimate_seconds=seconds, resources=['disk-e-io', 'disk-d-io', 'gpu-0'])]
    for seed in config['seeds']:
        stages.append(dict(id=f'evaluate_{seed}', label=f'种子 {seed} · 完整视频提示评价', kind='evaluate',
            command=[python, 'scripts/evaluate_token_causal_prompting.py', '--run', str(run), '--seed', str(seed), '--resume'],
            progress=str(run / 'evaluation' / f'seed{seed}' / 'progress.json'),
            output=str(run / 'evaluation' / f'seed{seed}' / 'summary.json'), units=10, unit='个 procedure',
            estimate_seconds=evaluation['seconds'] * 5, resources=['disk-e-io', 'disk-d-io']))
    stages.append(dict(id='summary', label='全部种子与逐procedure结果图表', kind='summary',
        command=[python, 'scripts/summarize_token_causal_prompting.py', '--run', str(run)],
        progress=str(run / 'summary_progress.json'), output=str(run / 'analysis/summary.json'),
        units=1, unit='组结果', estimate_seconds=10, resources=[]))
    path = run / 'batch_plan.json'
    if path.exists():
        raise FileExistsError(path)
    atomic_write_json(path, dict(question=read_json(run / 'protocol.json')['question'], stages=stages,
        native_counts=counts, config_sha256=digest(run / 'config.json'), protocol_sha256=digest(run / 'protocol.json'),
        estimate_basis='Measured native encoding smoke and complete-video metric evaluation on two real procedures.'))
    print(dict(outputs=total, native_counts=counts, estimated_seconds=sum(s['estimate_seconds'] for s in stages)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run)
