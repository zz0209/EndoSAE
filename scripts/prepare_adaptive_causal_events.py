import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
from encode_token_causal_identity import video_inputs
from encode_causal_detection_identity import Inputs, support
from train_acknowledgement_sae import now, save_npz, json_digest
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, tokens_to_frames


def inputs(run, smoke):
    config = read_json(run / 'config.json')
    application = read_json(Path(config['application_reference']) / 'config.json')
    reference = Path(config['event_reference']) / 'inputs'
    summary = read_json(reference / 'summary.json')
    assert summary['status'] == 'COMPLETE'
    videos = [row['video'] for row in summary['videos']]
    jobs = []
    for video in videos[:3] if smoke else videos:
        source = reference / video
        manifest = read_json(source / 'events.json')
        receipt = read_json(source / 'complete.json')
        assert receipt['status'] == 'COMPLETE'
        for name, expected in receipt['assets'].items():
            assert digest(source / name) == expected
        with np.load(source / 'tokens.npz', allow_pickle=False) as saved:
            positions = saved['positions'].copy()
        definition, _, records, tracks, available, targets, _, files = video_inputs(application, video)
        if config.get('requested_positions'):
            selection = read_json(config['requested_positions'])['videos'][video]
            positions = np.array(selection['smoke_positions'] if smoke else selection['positions'], dtype=int)
            assert np.all(available[positions])
        outputs = np.unique(np.searchsorted(tracks['offsets'][1:], positions, side='right')).tolist()
        source_outputs = sorted({int(e['click']['output_index']) for e in manifest['episodes']})
        assert np.all(available[positions])
        if not config.get('requested_positions'):
            assert set(source_outputs) <= set(outputs)
        if smoke and not config.get('requested_positions'):
            outputs = sorted({source_outputs[0], outputs[-1]})
        jobs.append(dict(video=video, source=source, manifest=manifest, definition=definition,
            records=records, tracks=tracks, available=available, outputs=outputs, requested=set(positions.tolist()),
            targets=targets, raw_root=Path(application['raw_root']) / video if config.get('requested_positions') else None,
            input_hashes={str(path): digest(path) for path in files},
            event_receipt_sha256=digest(source / 'complete.json')))
    return config, jobs


@torch.no_grad()
def prepare(run, smoke, resume, preflight):
    config, jobs = inputs(run, smoke)
    total = sum(len(job['outputs']) for job in jobs)
    if preflight:
        result = dict(status='PASS', outputs=total, procedures=len(jobs),
            events=sum(len(job['manifest']['events']) for job in jobs),
            source_count=sum(len(job['manifest']['episodes']) for job in jobs),
            estimated_prefix_bytes=total * 1569 * 768 * 4,
            selection='Requested historical detector positions.' if config.get('requested_positions') else
                'All detector positions from the existing fixed event sample and acknowledgement sources.')
        atomic_write_json(run / 'input_verification.json', result)
        print(result, flush=True)
        return
    root = Path(config['storage_root']) / ('smoke' if smoke else 'prepared')
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(config_sha256=digest(run / 'config.json'), smoke=smoke,
        sources={name: digest(ROOT / name) for name in ['scripts/prepare_adaptive_causal_events.py',
            'scripts/encode_causal_detection_identity.py', 'scripts/encode_token_causal_identity.py',
            'scripts/encode_rc27_cohort.py', 'src/evaluation/realcolon_task.py']},
        videos={job['video']: dict(inputs=job['input_hashes'], event_receipt=job['event_receipt_sha256'],
                                  outputs=job['outputs']) for job in jobs})
    signature = json_digest(identity)
    if config.get('requested_positions'):
        identity['requested_positions_sha256'] = digest(config['requested_positions'])
        signature = json_digest(identity)
    if (root / 'identity.json').exists():
        assert resume and read_json(root / 'identity.json') == identity
    else:
        atomic_write_json(root / 'identity.json', identity)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(read_json(jobs[0]['definition']['encoder_config']), torch.device('cuda'))
    captured = {}
    handle = encoder.model.blocks[8].register_forward_hook(
        lambda _module, _inputs, output: captured.update(value=output.detach()))
    receipts, completed, reused = [], 0, 0
    started = time.perf_counter()
    for job in jobs:
        assert job['definition']['encoder_config'] == jobs[0]['definition']['encoder_config']
        video, records, tracks = job['video'], job['records'], job['tracks']
        prepare_image = Inputs(job['definition']['frame_root'], video)
        with np.load(job['source'] / 'tokens.npz', allow_pickle=False) as saved:
            raw, offsets, positions = saved['tokens'].copy(), saved['offsets'].copy(), saved['positions'].copy()
        lookup = {int(position): i for i, position in enumerate(positions)}
        for output in job['outputs']:
            folder = root / video / f'{output:06d}'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'complete.json'
            if path.exists():
                receipt = read_json(path)
                assert resume and receipt['identity_sha256'] == signature
                assert digest(folder / 'prefix.npy') == receipt['prefix_sha256']
                assert digest(folder / 'roi.npz') == receipt['roi_sha256']
                reused += 1
            else:
                native, _ = encoder(prepare_image(records, output))
                prefix = captured['value'][0].cpu().numpy().copy()
                assert prefix.shape == (1569, 768) and np.isfinite(prefix).all()
                tokens = tokens_to_frames(native[None])
                historical_reference = {}
                if smoke and job['raw_root']:
                    old_receipt = read_json(job['raw_root'] / 'complete.json')
                    target_index = int(np.flatnonzero(job['targets'] == output)[0])
                    shard = next(s for s in old_receipt['shards'] if s['first'] <= target_index < s['stop'])
                    old_path = job['raw_root'] / shard['file']
                    assert digest(old_path) == shard['sha256']
                    with np.load(old_path, allow_pickle=False) as saved:
                        old_positions, old_bounds, old_tokens = saved['detection_positions'], saved['offsets'], saved['tokens']
                        for j, p in enumerate(old_positions):
                            if int(p) in job['requested']:
                                historical_reference[int(p)] = old_tokens[old_bounds[j]:old_bounds[j + 1]].copy()
                local, token_positions, indices, bounds = [], [], [], [0]
                for position in range(int(tracks['offsets'][output]), int(tracks['offsets'][output + 1])):
                    if position not in job['requested']:
                        continue
                    mask, references, reason = support(records, tracks, output, position - int(tracks['offsets'][output]))
                    assert reason == 'available' and all(row['output_index'] <= output for row in references)
                    value = tokens[mask]
                    if position in lookup:
                        j = lookup[position]
                        np.testing.assert_array_equal(value, raw[offsets[j]:offsets[j + 1]])
                    if smoke and job['raw_root']:
                        np.testing.assert_array_equal(value, historical_reference[position])
                    coordinates = np.column_stack(np.where(mask))
                    token_positions.append((1 + coordinates[:, 1] * 8 + coordinates[:, 0]).astype(np.int64))
                    local.append(value)
                    indices.append(position)
                    bounds.append(bounds[-1] + len(value))
                assert local
                if smoke:
                    replay = captured['value'].clone()
                    for block in encoder.model.blocks[9:11]:
                        replay = block(replay, 1, 8, 14)
                    np.testing.assert_array_equal(replay[0].cpu().numpy(), native)
                temporary = folder / 'prefix.pending.npy'
                np.save(temporary, prefix, allow_pickle=False)
                temporary.replace(folder / 'prefix.npy')
                save_npz(folder / 'roi.npz', native=np.concatenate(local),
                    token_positions=np.concatenate(token_positions), positions=np.array(indices), offsets=np.array(bounds))
                receipt = dict(identity_sha256=signature, video=video, output=output,
                    prefix_sha256=digest(folder / 'prefix.npy'), roi_sha256=digest(folder / 'roi.npz'),
                    detections=len(indices), native_tokens_exact=all(p in lookup for p in indices),
                    native_reference_detections=sum(p in lookup for p in indices), frozen_suffix_exact=smoke,
                    historical_native_exact=True if smoke and job['raw_root'] else None,
                    inputs_end_at_output=True, ground_truth_regions_used=False, completed_at=now())
                atomic_write_json(path, receipt)
            receipts.append(receipt)
            completed += 1
            progress = dict(status='RUNNING', completed=completed, total=total, reused=reused,
                seconds=time.perf_counter() - started, video=video, updated_at=now())
            atomic_write_json(root / 'progress.json', progress)
            if completed % 10 == 0 or smoke:
                print('CAUSAL_PREFIX', progress, flush=True)
            pause_after_checkpoint(path)
    handle.remove()
    atomic_write_json(root / 'summary.json', dict(status='COMPLETE', receipts=receipts,
        identity_sha256=signature, observations=completed, reused=reused, smoke=smoke,
        seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version), completed_at=now()))
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=completed, total=total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    prepare(args.run, args.smoke, args.resume, args.preflight)
