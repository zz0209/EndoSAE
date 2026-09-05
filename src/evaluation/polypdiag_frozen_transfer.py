"""Training-side, video-label-only transfer of frozen REAL-Colon models.

This input adapter reuses the verified LR1 encoder without fabricating masks.
Video labels are never inherited by the sampled frames or patches.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, write_json, binary_metrics


def video_scores(predictions):
    if predictions.ndim != 4 or predictions.shape[1:] != (3, 8, 196):
        raise ValueError('Expected video, three clips, eight frames, 196 patches')
    return np.sort(predictions, axis=-1)[..., -4:].mean(-1).mean(-1).max(-1)


def prepare(run, config, rows):
    from PIL import Image
    cache = Path(config['cache_dir'])
    cache.mkdir(parents=True, exist_ok=False)
    inputs = np.lib.format.open_memmap(str(cache / 'inputs.npy'), mode='w+', dtype='float32',
                                     shape=(len(rows), 3, 8, 224, 224))
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    receipts = []
    for video_index, video in enumerate(config['videos']):
        path = Path(config['data_root']) / 'videos' / video['file']
        if digest(path) != video['sha256']:
            raise ValueError('Selected video changed')
        probe = subprocess.run([config['ffprobe'], '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json', str(path)],
            check=True, capture_output=True)
        stream = json.loads(probe.stdout)['streams'][0]
        count, width, height = (int(stream[k]) for k in ('nb_frames', 'width', 'height'))
        if count < 24 or width * height > 4096 * 2160:
            raise ValueError('Video outside declared decode budget')
        starts = [int(round(fraction * (count - 8))) for fraction in config['temporal_positions']]
        indices = [s + t for s in starts for t in range(8)]
        if len(set(indices)) != 24:
            raise ValueError('Temporal samples overlap')
        selection = '+'.join('eq(n\\,{})'.format(i) for i in indices)
        decoded = subprocess.run([config['ffmpeg'], '-v', 'error', '-xerror', '-threads', '2',
            '-i', str(path), '-map', '0:v:0', '-vf', 'select=' + selection, '-fps_mode', 'passthrough',
            '-frames:v', '24', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True)
        if decoded.stderr or len(decoded.stdout) != 24 * width * height * 3:
            raise ValueError('Decode errors or sample count mismatch')
        pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(24, height, width, 3)
        for clip in range(3):
            row = rows[3 * video_index + clip]
            if row['video_id'] != video['file'] or row['temporal_position'] != config['temporal_positions'][clip]:
                raise ValueError('Video/clip manifest order changed')
            for t in range(8):
                crop = Image.fromarray(pixels[8 * clip + t]).resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
                array = np.asarray(crop, dtype=np.float32) / 255.
                inputs[3 * video_index + clip, :, t] = ((array - mean) / std).transpose(2, 0, 1)
        receipts.append({'video_id': video['file'], 'stream': stream, 'sampled_frame_indices_zero_based': indices,
                         'decoded_sample_count': 24, 'decoder_errors': None})
        write_json(run / 'decode_progress.json', {'completed_videos': video_index + 1, 'total': len(config['videos'])})
        print('PREPARE {}/{}'.format(video_index + 1, len(config['videos'])), flush=True)
    inputs.flush()
    del inputs
    write_json(run / 'decode_receipts.json', receipts)
    write_json(run / 'prepared.json', {'status': 'PREPARED_VIDEO_LABEL_ONLY',
        'assets': {'inputs.npy': digest(cache / 'inputs.npy')}, 'cache_dir': str(cache),
        'manifest_sha256': digest(run / 'clip_manifest.jsonl'), 'n_clips': len(rows),
        'no_frame_or_patch_labels': True, 'runner_sha256': digest(__file__)})


def predict(run, config, rows):
    import torch
    from src.evaluation.realcolon_fixed_confirmation import predict as frozen_predict, tokens_cuda, arrays
    from src.evaluation.realcolon_score_preserving import probabilities
    from src.sae.baselines import TopKAutoencoder
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    cache = Path(config['cache_dir'])
    encoded = json.loads((run / 'encoded.json').read_text())
    if encoded['manifest_sha256'] != digest(run / 'clip_manifest.jsonl') or digest(cache / 'tokens.npy') != encoded['tokens_sha256']:
        raise ValueError('Encoded cohort changed')
    tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(rows))), torch)
    output = run / 'predictions'
    output.mkdir(exist_ok=False)
    labels = np.array([v['label'] for v in config['videos']], dtype=bool)
    records = []
    for seed in config['seeds']:
        model_config = json.loads(Path(config['model_configs'][str(seed)]).read_text())
        norm = arrays(Path(model_config['normalization']), torch, 'cuda')
        x = (raw - norm['mean']) / norm['rms']
        state = arrays(Path(config['score_projections'][str(seed)]), torch, 'cuda')
        raw_predictions = None
        for method in config['methods']:
            destination = output / str(seed) / method
            destination.mkdir(parents=True)
            if method == 'score_pca48':
                patches = probabilities(x, state, torch).reshape(len(config['videos']), 3, 8, 196)
                error = float(np.max(np.abs(patches - raw_predictions)))
                if error > config['score_tolerance']:
                    raise ValueError('Raw score preservation failed on transferred inputs')
            else:
                patches = frozen_predict(raw, method, model_config, torch).reshape(len(config['videos']), 3, 8, 196)
                error = None
                if method == 'raw_balanced':
                    raw_predictions = patches
            scores = video_scores(patches)
            metrics = binary_metrics(labels, scores)
            metrics['sensitivity_at_0_5'] = float((scores[labels] >= .5).mean())
            metrics['fpr_at_0_5'] = float((scores[~labels] >= .5).mean())
            np.save(destination / 'patch_predictions.npy', patches, allow_pickle=False)
            np.save(destination / 'video_scores.npy', scores, allow_pickle=False)
            if method == 'raw_balanced':
                reconstruction = lambda batch: batch
            elif method.startswith('pca') or method == 'score_pca48':
                basis = state['basis'] if method == 'score_pca48' else norm['basis'][:, :model_config['models'][method]['dimension']]
                reconstruction = lambda batch: (batch @ basis) @ basis.T
            else:
                dictionary = TopKAutoencoder(768, 1536, 32).cuda()
                dictionary.load_state_dict(arrays(Path(model_config['models'][method]['dictionary']), torch, 'cuda'), strict=True)
                dictionary.eval()
                reconstruction = lambda batch: dictionary.decode(dictionary.encode_inference(batch))
            sums = []
            with torch.no_grad():
                for video in x.reshape(len(config['videos']), 3 * 8 * 196, 768):
                    sse, denominator = 0., 0.
                    for batch in video.split(4096):
                        sse += float((reconstruction(batch) - batch).double().square().sum())
                        denominator += float(batch.double().square().sum())
                    sums.append({'sse': sse, 'denominator': denominator, 'nmse': sse / denominator})
            record = {'seed': seed, 'method': method, 'metrics': metrics,
                      'per_video_reconstruction': sums, 'mean_video_nmse': float(np.mean([s['nmse'] for s in sums])),
                      'raw_score_max_absolute_error': error}
            write_json(destination / 'metrics.json', record)
            records.append(record)
            print(json.dumps({'seed': seed, 'method': method, **metrics}), flush=True)
    write_json(output / 'summary.json', {'records': records, 'n_video_files': len(labels),
        'patient_independence_established': False, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'encode', 'predict'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Frozen asset changed: ' + path)
    rows = [json.loads(line) for line in (run / 'clip_manifest.jsonl').read_text().splitlines()]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    start = time.time()
    status_path = run / (args.phase + '_status.json')
    write_json(status_path, {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        if args.phase == 'encode':
            from src.evaluation.realcolon_task import encode
            encode(run, config, rows)
        else:
            (prepare if args.phase == 'prepare' else predict)(run, config, rows)
    except Exception as error:
        write_json(status_path, {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(status_path, {'status': 'COMPLETED', 'elapsed_seconds': time.time() - start})


if __name__ == '__main__':
    main()
