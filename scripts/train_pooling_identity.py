import argparse
from pathlib import Path

import train_full_cohort_identity as parent
from train_temporal_view_identity import train
from src.checkpoint_io import atomic_write_json, read_json


def train_pooling(run, smoke, resume, output, stop_after):
    config = read_json(run / 'config.json')
    root = output or (run / 'smoke' if smoke else run)
    outputs = []
    for pooling in config['poolings']:
        def load_inputs(_):
            original, cohort, raw, offsets, records, receipts = parent.load_inputs(Path(config['input_run']))
            return (dict(original, **config['training'], pooling=pooling,
                         run_id=config['run_id']), cohort, raw, offsets, records, receipts)

        def sources():
            return dict(parent.source_identity(), **{
                'scripts/train_pooling_identity.py': parent.shared.file_sha256(__file__)})

        target = root / pooling
        train(run, smoke, resume, target, stop_after, input_loader=load_inputs, source_reader=sources)
        outputs.append(dict(pooling=pooling, **read_json(target / 'training_summary.json')))
    sequences = {}
    for group in outputs:
        for item in group['outputs']:
            key = (item['method'], item['seed'], item['fold'])
            sequence = read_json(Path(item['directory']) / 'sequence.json')
            if key in sequences:
                length = min(len(sequence), len(sequences[key]))
                if sequence[:length] != sequences[key][:length]:
                    raise ValueError('Pooling conditions received different observations')
            sequences[key] = sequence
    atomic_write_json(root / 'training_summary.json', dict(status='COMPLETE', outputs=outputs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train_pooling(args.run, args.smoke, args.resume, args.output, args.stop_after)
