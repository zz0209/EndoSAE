import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encode_rc27_cohort import Encoder
from run_continuous_confirmation import prepare_input
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames, write_json


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(config, video):
    prediction = Path(config['predictions']) / video
    receipt = read(prediction / 'complete.json')
    assert receipt['status'] == 'complete' and not receipt['is_smoke']
    records = [json.loads(line) for line in (prediction / 'detections.jsonl').read_text().splitlines()]
    assert len(records) == receipt['inputs']
    assert all(r['video_id'] == video for r in records)
    frames = np.array([r['frame_index'] for r in records])
    assert np.all(np.diff(frames) == 1)
    directory = Path(config['tracking_roots'][video]) / video
    summary = read(directory / 'summary.json')
    assert summary['input_identity'][str(prediction / 'detections.jsonl')] == digest(prediction / 'detections.jsonl')
    with np.load(directory / 'tracks.npz') as source:
        tracks = {key: value.copy() for key, value in source.items()}
    assert np.array_equal(tracks['frame_indices'], frames)
    expected = np.cumsum([0] + [len(r['detections']) for r in records])
    assert np.array_equal(tracks['offsets'], expected)
    original = read(directory / 'click_input.json')
    click = {'available': original['available']}
    if click['available']:
        click.update({key: original[key] for key in ['output_index', 'detection_index', 'time', 'input_frame']})
        assert records[click['output_index']]['frame_index'] == click['input_frame']
        assert original['detection'] == records[click['output_index']]['detections'][click['detection_index']]
    paths = [prediction / 'detections.jsonl', prediction / 'complete.json', directory / 'tracks.npz', directory / 'click_input.json']
    return records, tracks, click, {str(path): digest(path) for path in paths}


def support(records, tracks, output, detection_index):
    masks = np.zeros((8, 196), dtype=bool)
    references = []
    if output < 7:
        return masks, references, 'insufficient_past_images'
    position = int(tracks['offsets'][output]) + detection_index
    identity = int(tracks['track_ids'][position])
    for past in range(output - 7, output + 1):
        indices = []
        if past == output:
            indices = [detection_index]
        elif identity >= 0:
            start, end = tracks['offsets'][past:past + 2]
            indices = np.flatnonzero(tracks['track_ids'][start:end] == identity).tolist()
            assert len(indices) <= 1
        if not indices:
            continue
        index = indices[0]
        record = records[past]
        box = record['detections'][index]['xyxy']
        mask, fallback = project_boxes(dict(width=record['width'], height=record['height'], boxes_xyxy=[dict(box=box)]))
        masks[past - output + 7] = mask
        references.append(dict(output_index=past, frame_index=record['frame_index'], detection_index=index,
                               xyxy=box, tokens=int(mask.sum()), small_box_fallback=fallback))
    assert all(row['output_index'] <= output for row in references)
    reason = 'available' if masks[-1].any() else 'current_box_outside_backbone_crop'
    return masks, references, reason


class Inputs:
    def __init__(self, frame_root, video):
        self.directory = Path(frame_root) / video
        self.cache = {}
        self.bytes_read = 0
        self.mean = np.array([.485, .456, .406], dtype=np.float32)
        self.std = np.array([.229, .224, .225], dtype=np.float32)

    def __call__(self, records, output):
        selected = records[output - 7:output + 1]
        assert len(selected) == 8
        keep = {row['frame_index'] for row in selected}
        self.cache = {key: value for key, value in self.cache.items() if key in keep}
        for row in selected:
            index = row['frame_index']
            if index in self.cache:
                continue
            path = self.directory / ('%06d.jpg' % index)
            with Image.open(path) as image:
                assert image.size == (row['width'], row['height'])
                crop = image.convert('RGB').resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
                value = np.asarray(crop, dtype=np.float32) / 255.
            self.cache[index] = ((value - self.mean) / self.std).transpose(2, 0, 1)
            self.bytes_read += path.stat().st_size
        return np.stack([self.cache[row['frame_index']] for row in selected], axis=1)


class Representations:
    def __init__(self, directory):
        directory = Path(directory)
        with np.load(directory / 'normalization.npz') as data:
            self.mean, self.scale = data['mean'].copy(), data['scale'].copy()
        self.model = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128)).eval().requires_grad_(False)
        with np.load(directory / 'model.npz') as data:
            self.model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in data.items()}, strict=True)

    def __call__(self, raw):
        standardized = (raw - self.mean) / self.scale
        with torch.no_grad():
            projected = F.normalize(self.model(torch.from_numpy(standardized.astype(np.float32))), dim=-1).numpy()
        return dict(raw_l2=(raw / np.linalg.norm(raw, axis=-1, keepdims=True)).astype(np.float32),
                    standardized_l2=(standardized / np.linalg.norm(standardized, axis=-1, keepdims=True)).astype(np.float32),
                    supcon_l2=projected)


def run(config, config_path, destination, video, device, smoke):
    records, tracks, click, inputs = load_inputs(config, video)
    assert not smoke or click['available']
    targets = list(range(click['output_index'], min(len(records), click['output_index'] + 3))) if smoke else list(range(len(records)))
    destination.mkdir(parents=True, exist_ok=True)
    identity = dict(config_sha256=digest(config_path), source_sha256=digest(__file__), input_identity=inputs,
        model_identity={str(Path(config['projection_fit']) / name): digest(Path(config['projection_fit']) / name)
                        for name in ['model.npz', 'normalization.npz']}, video_id=video, device=str(device),
        smoke=smoke, outputs=targets, python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
    if (destination / 'identity.json').exists():
        assert read(destination / 'identity.json') == identity
    else:
        write_json(destination / 'identity.json', identity)
        write_json(destination / 'config.json', config)
        shutil.copyfile(__file__, destination / 'source.py')
    if (destination / 'complete.json').exists():
        print('REUSE_COMPLETE', video, flush=True)
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(read(config['encoder_config']), device)
    transforms = Representations(config['projection_fit'])
    prepare = Inputs(config['frame_root'], video)
    progress_path = destination / 'progress.json'
    processed = read(progress_path)['processed_outputs'] if progress_path.exists() else 0
    count = int(tracks['offsets'][-1])
    arrays = {}
    shapes = {'raw_mean': (count, 768), 'raw_l2': (count, 768), 'standardized_l2': (count, 768), 'supcon_l2': (count, 128),
              'available': (count,), 'status_code': (count,), 'support_tokens': (count, 8), 'inference_seconds': (len(records),), 'preprocessing_seconds': (len(records),)}
    for name, shape in shapes.items():
        path = destination / (name + '.npy')
        dtype = np.float64 if name == 'raw_mean' else np.bool_ if name == 'available' else np.int16 if name in ['status_code', 'support_tokens'] else np.float32
        arrays[name] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=dtype, shape=shape)
        if processed == 0:
            arrays[name][:] = False if name == 'available' else 0 if name in ['status_code', 'support_tokens'] else np.nan
    if not (destination / 'indices.npz').exists():
        np.savez(destination / 'indices.npz', **tracks)
        write_json(destination / 'click.json', click)
    started = time.perf_counter()
    for number in range(processed, len(targets)):
        output = targets[number]
        rows = records[output]['detections']
        supports = [support(records, tracks, output, i) for i in range(len(rows))]
        valid = [i for i, (_, _, reason) in enumerate(supports) if reason == 'available']
        offset = int(tracks['offsets'][output])
        for i, (mask, _, reason) in enumerate(supports):
            arrays['status_code'][offset + i] = {'available': 1, 'insufficient_past_images': 2, 'current_box_outside_backbone_crop': 3}[reason]
            arrays['support_tokens'][offset + i] = mask.sum(axis=1)
        if valid:
            begin = time.perf_counter()
            tensor = prepare(records, output)
            arrays['preprocessing_seconds'][output] = time.perf_counter() - begin
            begin = time.perf_counter()
            native, _ = encoder(tensor)
            arrays['inference_seconds'][output] = time.perf_counter() - begin
            tokens = tokens_to_frames(native[None])
            means = np.stack([tokens[supports[i][0]].mean(axis=0, dtype=np.float64) for i in valid])
            values = transforms(means)
            assert np.isfinite(means).all() and all(np.isfinite(value).all() for value in values.values())
            positions = offset + np.array(valid)
            arrays['raw_mean'][positions] = means
            for key, value in values.items():
                arrays[key][positions] = value
            arrays['available'][positions] = True
            if smoke and output == click['output_index']:
                frames = [dict(row, boxes_xyxy=[]) for row in records[output - 7:output + 1]]
                reference, _ = prepare_input(dict(video_id=video, frames=frames), Path(config['frame_root']))
                assert np.array_equal(tensor, reference)
                np.save(destination / 'smoke_native.npy', native)
                write_json(destination / 'smoke_checks.json', dict(preprocessing_exact=True,
                    pooling='All selected ROI tokens across available temporal positions have equal weight',
                    frame_indices=[r['frame_index'] for r in frames], source_support=supports[click['detection_index']][1],
                    past_only=all(r['frame_index'] <= records[output]['frame_index'] for r in frames)))
        if click['available'] and output == click['output_index']:
            selected = click['detection_index']
            position = offset + selected
            recipe = dict(click=click, detection=rows[selected], track_id=int(tracks['track_ids'][position]),
                status=supports[selected][2], support=supports[selected][1], fixed_at_frame=records[output]['frame_index'])
            write_json(destination / 'source_memory.json', recipe)
            np.savez(destination / 'source_memory.npz', **{key: np.array(arrays[key][position]) for key in ['raw_mean', 'raw_l2', 'standardized_l2', 'supcon_l2', 'available', 'support_tokens']})
        if (number + 1) % 64 == 0 or number + 1 == len(targets):
            for array in arrays.values():
                array.flush()
            write_json(progress_path, dict(processed_outputs=number + 1, total_outputs=len(targets), last_frame=records[output]['frame_index']))
            print('DESCRIPTORS', video, number + 1, '/', len(targets), 'camera outputs', round(time.perf_counter() - started, 3), 'seconds', flush=True)
    measured = np.isfinite(arrays['inference_seconds'])
    available = arrays['available']
    available_frames = (arrays['support_tokens'][available] > 0).sum(axis=1)
    status = arrays['status_code']
    processed_detections = int((status > 0).sum())
    if click['available']:
        with np.load(destination / 'source_memory.npz') as fixed:
            position = int(tracks['offsets'][click['output_index']]) + click['detection_index']
            for key in fixed.files:
                assert np.array_equal(fixed[key], arrays[key][position], equal_nan=True)
    result = dict(status='PASS_REAL_SMOKE' if smoke else 'COMPLETE', video_id=video, camera_outputs=len(targets),
        raw_detections=count, available_descriptors=int(available.sum()), encoded_outputs=int(measured.sum()),
        processed_detections=processed_detections,
        descriptor_status_counts={name: int((status == code).sum()) for name, code in
            [('pending', 0), ('available', 1), ('insufficient_past_images', 2), ('current_box_outside_backbone_crop', 3)]},
        crop_excluded_fraction=float((status == 3).sum() / processed_detections) if processed_detections else None,
        available_support_frame_counts={str(n): int((available_frames == n).sum()) for n in range(1, 9)},
        available_unassigned_detections=int((available & (tracks['track_ids'] < 0)).sum()),
        mean_support_frames=float(available_frames.mean()) if len(available_frames) else None,
        inference_median_seconds=float(np.nanmedian(arrays['inference_seconds'])),
        preprocessing_median_seconds=float(np.nanmedian(arrays['preprocessing_seconds'])),
        elapsed_this_invocation_seconds=time.perf_counter() - started, jpeg_bytes_read=prepare.bytes_read,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        source_memory_fixed=True, identity_sha256=digest(destination / 'identity.json'),
        descriptor_unavailable_policy='Retain display; unavailable arrays are NaN and available=False',
        timing='Offline causal replay; model and preprocessing times measured. No additional-latency clinical claim.')
    write_json(destination / 'complete.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    assert args.video in config['videos']
    run(config, args.config, args.output / args.video, args.video, torch.device(args.device), args.smoke)


if __name__ == '__main__':
    main()
