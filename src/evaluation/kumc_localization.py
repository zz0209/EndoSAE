"""Frozen-model localization on explicitly recovered, sampled source sequences.

No fitting or target-dependent model selection occurs here. Bounding boxes are
coarse patch-support labels; empty or cropped-out labels are not negatives.
"""
import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, write_json, project_boxes, binary_metrics


def support_frame(frame):
    """Adapt the upstream zero-based inclusive box extent to half-open edges."""
    width, height = frame['width'], frame['height']
    boxes = []
    for obj in frame['objects']:
        x0, y0, x1, y1 = obj['raw_box']
        if not all(np.isfinite([x0, y0, x1, y1])):
            raise ValueError('Nonfinite annotation')
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width - 1, x1), min(height - 1, y1)
        if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
            raise ValueError('Invalid inclusive box')
        boxes.append({'box': [x0, y0, x1 + 1, y1 + 1], 'name': obj['name']})
    return {'width': width, 'height': height, 'boxes_xyxy': boxes}


def prepare(run, config, rows):
    from PIL import Image
    archive = Path(config['archive']['source'])
    stat = archive.stat()
    if stat.st_size != config['archive']['bytes'] or stat.st_mtime_ns != config['archive']['mtime_ns']:
        raise ValueError('Previously verified archive changed')
    cache = Path(config['cache_dir'])
    cache.mkdir(parents=True, exist_ok=False)
    inputs = np.lib.format.open_memmap(str(cache / 'inputs.npy'), mode='w+', dtype='float32',
                                      shape=(len(rows), 3, 8, 224, 224))
    masks = np.zeros((len(rows), 8, 196), dtype=bool)
    receipts = []
    mean, std = np.array([.485, .456, .406], dtype=np.float32), np.array([.229, .224, .225], dtype=np.float32)
    with archive.open('rb') as stream:
        for i, row in enumerate(rows):
            for t, frame in enumerate(row['frames']):
                stream.seek(frame['image_offset'])
                blob = stream.read(frame['image_bytes'])
                if len(blob) != frame['image_bytes']:
                    raise ValueError('Truncated image member')
                with Image.open(io.BytesIO(blob)) as image:
                    if image.size != (frame['width'], frame['height']):
                        raise ValueError('JPEG/XML size mismatch')
                    crop = image.convert('RGB').resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
                    value = np.asarray(crop, dtype=np.float32) / 255.
                inputs[i, :, t] = ((value - mean) / std).transpose(2, 0, 1)
                adapted = support_frame(frame)
                masks[i, t], fallbacks = project_boxes(adapted)
                status = ('missing_object' if not frame['objects'] else 'crop_dropped' if not masks[i, t].any()
                          else 'full_support' if masks[i, t].all() else 'mixed_support')
                receipts.append({'clip_index': i, 'time_index': t, 'image_member': frame['image_member'],
                    'image_sha256': hashlib.sha256(blob).hexdigest(), 'support_status': status,
                    'patch_count': int(masks[i, t].sum()), 'small_box_fallbacks': fallbacks,
                    'half_open_boxes': adapted['boxes_xyxy']})
            print('PREPARE {}/{}'.format(i + 1, len(rows)), flush=True)
    inputs.flush()
    del inputs
    np.save(str(cache / 'masks.npy'), masks, allow_pickle=False)
    write_json(run / 'frame_receipts.json', receipts)
    write_json(run / 'prepared.json', {'status': 'PREPARED_BOX_SUPPORT', 'n_clips': len(rows),
        'assets': {name: digest(cache / name) for name in ['inputs.npy', 'masks.npy']},
        'manifest_sha256': digest(run / 'clip_manifest.jsonl'), 'runner_sha256': digest(__file__),
        'frame_receipts_sha256': digest(run / 'frame_receipts.json')})


def localization_metrics(scores, masks, rows):
    if scores.shape != masks.shape or not np.isfinite(scores).all():
        raise ValueError('Prediction shape/finite contract failed')
    result = {}
    for video in sorted(set(row['video_id'] for row in rows)):
        indices = [i for i, row in enumerate(rows) if row['video_id'] == video]
        ap, pointing = [], []
        for target, score in zip(masks[indices].reshape(-1, 196), scores[indices].reshape(-1, 196)):
            if not target.any() or target.all():
                continue
            ap.append(binary_metrics(target, score)['average_precision'])
            # Expected hit under uniform selection among exact maxima, for all methods.
            pointing.append(float(target[score == score.max()].mean()))
        result[video] = {'frame_mean_patch_ap': float(np.mean(ap)) if ap else None,
                         'pointing_hit_rate': float(np.mean(pointing)) if pointing else None,
                         'eligible_frames': len(ap)}
    return result


def predict(run, config, rows):
    import torch
    from src.evaluation.realcolon_fixed_confirmation import predict as frozen_predict, tokens_cuda
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    cache = Path(config['cache_dir'])
    encoded = json.loads((run / 'encoded.json').read_text())
    prepared = json.loads((run / 'prepared.json').read_text())
    if (encoded['manifest_sha256'] != digest(run / 'clip_manifest.jsonl') or
            encoded['prepared_sha256'] != digest(run / 'prepared.json') or
            digest(cache / 'tokens.npy') != encoded['tokens_sha256'] or
            digest(cache / 'masks.npy') != prepared['assets']['masks.npy']):
        raise ValueError('Encoded input/target binding changed')
    masks = np.load(str(cache / 'masks.npy'), allow_pickle=False)
    tokens = np.load(str(cache / 'tokens.npy'), mmap_mode='r', allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(rows))), torch)
    output = run / 'predictions'
    output.mkdir(exist_ok=False)
    yy, xx = np.meshgrid(np.arange(14) * 2 - 13, np.arange(14) * 2 - 13, indexing='ij')
    priors = {'center': -(xx * xx + yy * yy).ravel(), 'uniform': np.zeros(196)}
    records = []
    for seed in config['seeds']:
        model_config = json.loads(Path(config['model_configs'][str(seed)]).read_text())
        for method in config['methods']:
            scores = frozen_predict(raw, method, model_config, torch).reshape(masks.shape)
            destination = output / str(seed) / method
            destination.mkdir(parents=True)
            np.save(str(destination / 'patch_predictions.npy'), scores, allow_pickle=False)
            records.append({'seed': seed, 'method': method, 'per_video': localization_metrics(scores, masks, rows)})
    for method, prior in priors.items():
        scores = np.broadcast_to(prior.astype(np.float32), masks.shape).copy()
        destination = output / method
        destination.mkdir()
        np.save(str(destination / 'patch_predictions.npy'), scores, allow_pickle=False)
        records.append({'seed': None, 'method': method, 'per_video': localization_metrics(scores, masks, rows)})
    write_json(output / 'summary.json', {'records': records, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'independent_patient_count_established': False, 'model_fitting': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'encode', 'predict'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for path, expected in config['asset_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen asset changed: ' + path)
    rows = [json.loads(line) for line in (run / 'clip_manifest.jsonl').read_text().splitlines()]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    started = time.time()
    status = run / (args.phase + '_status.json')
    write_json(status, {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        if args.phase == 'encode':
            from src.evaluation.realcolon_task import encode
            encode(run, config, rows)
        else:
            (prepare if args.phase == 'prepare' else predict)(run, config, rows)
    except Exception as error:
        write_json(status, {'status': 'FAILED', 'error': repr(error)})
        raise
    write_json(status, {'status': 'COMPLETED', 'elapsed_seconds': time.time() - started})


if __name__ == '__main__':
    main()
