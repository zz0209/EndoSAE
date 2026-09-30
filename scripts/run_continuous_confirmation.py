import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import math
import platform
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_realcolon_visibility import FRAME_RE, parse_xml
from scripts.build_realcolon_visibility_pack import extract_task_manifest
from src.evaluation.realcolon_task import project_boxes, digest, write_json
from verify_backbone_runtime import head_arrays
from verify_tail_runtime import FrozenTail, arrays, load_role_sae
import run_model_port_parity as parity


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def prepare_metadata(run, config):
    videos = {row['video_id']: row for row in read(config['annotation_summary'])['videos']}
    destination = run / 'metadata'
    destination.mkdir(exist_ok=True)
    records = []
    selected_videos = config.get('videos', config.get('reserved_videos'))
    detailed = config.get('preserve_frame_timeline', False)
    window_identity = {key: config[key] for key in ['window_seconds', 'pre_lesion_seconds', 'window_rule', 'annotation_rule']}
    window_identity['videos' if 'videos' in config else 'reserved_videos'] = selected_videos
    if detailed:
        window_identity.update(video_splits=config['video_splits'], preserve_frame_timeline=True)
    for video in selected_videos:
        manifest, receipt = destination / (video + '.jsonl'), destination / (video + '.json')
        if receipt.exists():
            record = read(receipt)
            assert record['window_identity'] == window_identity and record['manifest_sha256'] == digest(manifest)
            records.append(record)
            continue
        info = videos[video]
        fps = info['fps']
        duration = math.floor(config['window_seconds'] * fps)
        first_lesion = min(lesion['first_frame'] for lesion in info['lesions'])
        start = min(max(first_lesion - math.floor(config['pre_lesion_seconds'] * fps), 0), info['frame_count'] - duration)
        end = start + duration
        assert 0 <= start < end <= info['frame_count']
        frames, exclusions = {}, []
        archive_path = Path(config['annotation_directory']) / (video + '_annotations.tar.gz')
        assert digest(archive_path) == info['archive']['sha256']
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.xml'):
                    continue
                match = FRAME_RE.search(member.name)
                assert match is not None
                index = int(match.group(1))
                if index < start or index >= end:
                    continue
                assert index not in frames
                payload = archive.extractfile(member).read()
                width, height, boxes, anomalies, _ = parse_xml(payload)
                frame = dict(frame_index=index, xml_member=member.name, width=width, height=height,
                             boxes_xyxy=[dict(lesion_id=uid, box=list(box)) for uid, box in boxes], seconds=index / fps)
                mask, fallback_count = project_boxes(frame)
                frame['class'] = 1 if mask.any() else (0 if not boxes else -1)
                frame['small_box_fallback_count'] = fallback_count
                frame['eligible'] = not anomalies and frame['class'] >= 0
                if detailed:
                    xml = ET.fromstring(payload)
                    frame['original_boxes_xyxy'] = [dict(lesion_id=node.findtext('unique_id'),
                        box=[float(node.findtext('bndbox/' + coordinate)) for coordinate in ['xmin', 'ymin', 'xmax', 'ymax']])
                        for node in xml.findall('object')]
                    frame['annotation_status'] = 'invalid_annotation' if anomalies else ('target_outside_crop' if frame['class'] < 0 else ('target_present' if frame['class'] else 'annotated_target_absent'))
                    frame['annotation_anomalies'] = anomalies
                    frame['visible_lesion_ids'] = [box['lesion_id'] for box in frame['boxes_xyxy']
                        if project_boxes(dict(frame, boxes_xyxy=[box]))[0].any()]
                    if anomalies:
                        frame['class'] = -1
                frames[index] = frame
                if anomalies or frame['class'] < 0:
                    exclusions.append(dict(frame_index=index, reason='annotation_anomaly' if anomalies else 'target_outside_crop', details=anomalies))
        missing = sorted(set(range(start, end)) - set(frames))
        exclusions.extend(dict(frame_index=index, reason='missing_annotation') for index in missing)
        if detailed:
            for index in missing:
                frames[index] = dict(frame_index=index, seconds=index / fps, annotation_status='missing_annotation',
                    eligible=False, **{'class': -1}, boxes_xyxy=None, original_boxes_xyxy=None, visible_lesion_ids=None)
        clips, excluded_clips = [], []
        segment, previous_end = -1, None
        for first in range(start, end - 7, 8):
            indices = list(range(first, first + 8))
            invalid = [i for i in indices if i not in frames or not frames[i]['eligible']]
            if invalid:
                excluded = dict(first_frame=first, excluded_frames=invalid)
                if detailed:
                    excluded.update(end_frame_exclusive=first + 8, frames=[frames[index] for index in indices], interrupts_continuity=True)
                excluded_clips.append(excluded)
                previous_end = None
                continue
            clip = dict(clip_id=video + '_' + str(first), video_id=video, split=config.get('video_splits', {}).get(video, 'independent_confirmation'),
                        fps=fps, frames=[frames[index] for index in indices])
            if detailed:
                contiguous = previous_end == first
                if not contiguous:
                    segment += 1
                clip.update(continuity_segment=segment, previous_clip_contiguous=contiguous,
                            available_time_seconds=(first + 7) / fps, first_frame=first, end_frame_exclusive=first + 8)
            clips.append(clip)
            previous_end = first + 8
        assert clips
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in clips), encoding='utf-8')
        record = dict(video_id=video, fps=fps, first_lesion_frame=first_lesion, start_frame=start, end_frame_exclusive=end,
                      clips=len(clips), eligible_frames=len(clips) * 8, terminal_frames=list(range(start + duration // 8 * 8, end)),
                      frame_exclusions=exclusions, excluded_clips=excluded_clips,
                      full_video_prior_annotation_anomaly_count=info['annotation_anomaly_count'],
                      annotation_sha256=digest(archive_path), manifest_sha256=digest(manifest), window_identity=window_identity)
        if detailed:
            record.update(split=config['video_splits'][video], requested_frames=duration,
                duration_seconds=duration / fps, eligible_duration_seconds=len(clips) * 8 / fps,
                continuity_segments=segment + 1, terminal_frame_records=[frames[index] for index in record['terminal_frames']],
                frame_status_counts={status: sum(frame['annotation_status'] == status for frame in frames.values())
                    for status in ['target_present', 'annotated_target_absent', 'target_outside_crop', 'invalid_annotation', 'missing_annotation']},
                lesion_ids_in_window=sorted({box['lesion_id'] for frame in frames.values() for box in (frame['original_boxes_xyxy'] or [])}))
        write_json(receipt, record)
        records.append(record)
        print('METADATA', video, 'clips', len(clips), 'excluded_clips', len(excluded_clips), flush=True)
    write_json(destination / 'summary.json', dict(videos=records, total_clips=sum(r['clips'] for r in records), pixels_accessed=False))


def prepare_input(row, frame_root):
    value = np.empty((3, 8, 224, 224), dtype=np.float32)
    masks = []
    mean, std = np.array([.485, .456, .406], dtype=np.float32), np.array([.229, .224, .225], dtype=np.float32)
    for t, frame in enumerate(row['frames']):
        with Image.open(frame_root / row['video_id'] / ('%06d.jpg' % frame['frame_index'])) as image:
            assert image.size == (frame['width'], frame['height'])
            crop = image.convert('RGB').resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
            array = np.asarray(crop, dtype=np.float32) / 255.
        value[:, t] = ((array - mean) / std).transpose(2, 0, 1)
        masks.append(project_boxes(frame)[0])
    return value, np.asarray(masks, dtype=bool)


class Consumer:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.source = read(config['source_config'])
        sparse_run = Path(config['sparse_selection_run'])
        sparse_config, selected = read(sparse_run / 'config.json'), read(sparse_run / 'fit/summary.json')['methods']
        statistics = Path(self.source['source_folds'][0]['source_fold']) / 'input_statistics.npz'
        self.stats, self.tail = arrays(statistics, device), FrozenTail(self.source['state_exchange'], device)
        norm, weight, bias = head_arrays(self.source)
        self.norm = {key: torch.from_numpy(value).to(device) for key, value in norm.items()}
        self.weight, self.bias = torch.from_numpy(weight).to(device), torch.from_numpy(np.asarray(bias)).to(device)
        self.models, self.specs = {}, {'factual': dict(gamma=0, threshold=selected['task_only']['factual_threshold'])}
        self.identities = {str(statistics): digest(statistics), str(sparse_run / 'fit/summary.json'): digest(sparse_run / 'fit/summary.json')}
        for name in config['sparse_models']:
            self.models[name] = load_role_sae(self.source, sparse_config['models'][name], device)
            self.identities[sparse_config['models'][name]] = digest(sparse_config['models'][name])
            for key in [name, name + '_raw_control']:
                method = selected[key]
                candidate = next(v for v in method['training_candidates'] if v['gamma'] == method['selected_gamma'])
                self.specs[key] = dict(gamma=candidate['gamma'], threshold=candidate['threshold'], source=name, raw=key.endswith('_raw_control'))
        if config['dense_selection_run'] is not None:
            from train_dense_task_control import load_rank96
            dense_path = Path(config['dense_selection_run']) / 'fit/summary.json'
            dense = read(dense_path)['methods']['rank96_task']
            candidate = next(v for v in dense['training_candidates'] if v['gamma'] == dense['selected_gamma'])
            self.models['rank96_task'] = load_rank96(config['dense_initial_pca'], config['dense_checkpoint'], device)
            self.specs['rank96_task'] = dict(gamma=candidate['gamma'], threshold=candidate['threshold'], source='rank96_task', raw=False)
            self.identities[str(dense_path)] = digest(dense_path)
            self.identities[config['dense_checkpoint']] = digest(config['dense_checkpoint'])
        assert parity.sha256_file(self.source['state_exchange']) == self.source['state_exchange_sha256']
        module = parity.load_timesformer(str(ROOT / 'third_party/Endo-FM/models'))
        self.backbone = parity.build_model(torch, module)
        self.backbone.load_state_dict(parity.load_state_exchange(self.source['state_exchange'], torch, np), strict=True)
        self.backbone.to(device).eval()
        for model in [self.backbone, self.tail] + list(self.models.values()):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.captured = {}
        self.handle = self.backbone.blocks[10].register_forward_hook(lambda _module, _inputs, output: self.captured.update(block10=output.detach()))

    def output(self, native):
        final = self.tail(native)
        tokens = final[:, 1:].reshape(-1, 196, 8, 768).permute(0, 2, 1, 3)
        return ((tokens - self.norm['mean']) / self.norm['rms']).double() @ self.weight + self.bias

    def from_native(self, native):
        x = (native[0, 1:] - self.stats['mean']) / self.stats['rms']
        deltas = {name: model.encode_inference(x)[:, 32:64] @ model.decoder.weight[:, 32:64].T for name, model in self.models.items()}
        batch = native.expand(len(self.specs), -1, -1).clone()
        for index, (name, spec) in enumerate(self.specs.items()):
            if spec['gamma'] == 0:
                continue
            delta = deltas[spec['source']]
            if spec['raw']:
                delta = x * (delta.square().mean(-1, keepdim=True).sqrt() / x.square().mean(-1, keepdim=True).sqrt())
            batch[index, 1:] += spec['gamma'] * delta * self.stats['rms']
        assert torch.equal(batch[:, :1], native[:, :1].expand(len(self.specs), -1, -1))
        result = self.output(batch).cpu().numpy()
        assert np.isfinite(result).all()
        return result

    def __call__(self, inputs):
        with torch.no_grad():
            self.backbone.forward_features(torch.from_numpy(inputs[None]).to(self.device), get_all=True)
            return self.from_native(self.captured['block10'])


def save_chunk(path, **payload):
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, **payload)
    temporary.replace(path)


def smoke(run, config, device):
    destination = run / 'smoke'
    destination.mkdir(exist_ok=False)
    source = Path(config['smoke_source_run'])
    manifest, cache = rows(source / 'clip_manifest.jsonl'), Path(read(source / 'execution_config.json')['cache_dir'])
    frame_root = Path(read(source / 'extraction_summary.json')['output_dir'])
    old_inputs, old_native = np.load(cache / 'inputs.npy', mmap_mode='r'), np.load(cache / 'block10.npy', mmap_mode='r')
    consumer = Consumer(config, device)
    reports, outputs = [], []
    started = time.perf_counter()
    for index in config['smoke_indices']:
        inputs, labels = prepare_input(manifest[index], frame_root)
        assert np.array_equal(inputs, old_inputs[index])
        assert np.array_equal(labels, np.load(cache / 'masks.npy', mmap_mode='r')[index] > 0)
        actual = consumer(inputs)
        with torch.no_grad():
            reference = consumer.from_native(torch.from_numpy(np.array(old_native[index:index + 1], copy=True)).to(device))
        difference = np.abs(actual - reference)
        thresholds = np.array([v['threshold'] for v in consumer.specs.values()])[:, None, None]
        decision_difference = int(np.count_nonzero((actual >= thresholds) != (reference >= thresholds)))
        saved_errors = {}
        for method_index, (method, spec) in enumerate(consumer.specs.items()):
            selection_run = Path(config['dense_selection_run'] if method == 'rank96_task' else config['sparse_selection_run'])
            saved_method = 'task_only' if method == 'factual' else method
            gamma_index = read(selection_run / 'config.json')['gammas'].index(spec['gamma'])
            with np.load(selection_run / 'fit' / (saved_method + '_dev.npz'), allow_pickle=False) as saved:
                expected = saved['maps'][index, gamma_index]
            saved_errors[method] = float(np.abs(reference[method_index] - expected).max())
            assert saved_errors[method] < 1e-4
        frame_decisions = int(np.count_nonzero((actual.max(-1) >= thresholds[:, :, 0]) != (reference.max(-1) >= thresholds[:, :, 0])))
        target_actual = np.where(labels[None], actual, -np.inf).max(-1)
        target_reference = np.where(labels[None], reference, -np.inf).max(-1)
        target_decisions = int(np.count_nonzero((target_actual >= thresholds[:, :, 0]) != (target_reference >= thresholds[:, :, 0])))
        reports.append(dict(index=index, preprocessing_equal=True, map_max_error=float(difference.max()), map_mean_error=float(difference.mean()),
                            threshold_token_decision_differences=decision_difference, frame_alarm_decision_differences=frame_decisions,
                            target_detection_decision_differences=target_decisions, saved_consumer_max_error=saved_errors))
        outputs.append(actual)
        print('SMOKE', json.dumps(reports[-1]), flush=True)
    save_chunk(destination / 'chunk_00000.npz', maps=np.asarray(outputs), methods=np.array(list(consumer.specs)), indices=np.array(config['smoke_indices']))
    with np.load(destination / 'chunk_00000.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['maps'], np.asarray(outputs))
    write_json(destination / 'summary.json', dict(cases=reports, methods=consumer.specs, seconds=time.perf_counter() - started,
               peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
               code_sha256=digest(__file__), config_sha256=digest(run / 'config.json')))


def infer(run, config, device):
    assert config['dense_selection_run'] is not None
    consumer, storage = Consumer(config, device), Path(config['storage_root'])
    identity = dict(config_sha256=digest(run / 'config.json'), code_sha256=digest(__file__), models=consumer.identities,
                    methods=consumer.specs, python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    source_config_sha256=digest(config['source_config']), state_exchange_sha256=consumer.source['state_exchange_sha256'],
                    annotation_manifests={video: digest(run / 'metadata' / (video + '.jsonl')) for video in config['reserved_videos']},
                    extraction_receipts={video: digest(storage / 'extraction' / video / ('extracted_' + video + '.json')) for video in config['reserved_videos']})
    identity_path = run / 'inference_identity.json'
    if identity_path.exists():
        assert read(identity_path) == identity
    else:
        write_json(identity_path, identity)
    identity_hash = digest(identity_path)
    started = time.perf_counter()
    for video in config['reserved_videos']:
        metadata = run / 'metadata' / (video + '.jsonl')
        video_rows = rows(metadata)
        destination = storage / 'predictions' / video
        destination.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(video_rows), config['chunk_clips']):
            subset = video_rows[start:start + config['chunk_clips']]
            output = destination / ('chunk_%05d.npz' % start)
            if output.exists():
                with np.load(output, allow_pickle=False) as saved:
                    assert saved['identity_sha256'].item() == identity_hash
                    assert saved['clip_ids'].tolist() == [row['clip_id'] for row in subset]
                continue
            maps, masks, frame_indices, seconds = [], [], [], []
            for row in subset:
                value, truth = prepare_input(row, storage / 'frames')
                maps.append(consumer(value))
                masks.append(truth)
                frame_indices.append([frame['frame_index'] for frame in row['frames']])
                seconds.append([frame['seconds'] for frame in row['frames']])
            save_chunk(output, maps=np.asarray(maps), masks=np.asarray(masks), frame_indices=np.asarray(frame_indices), seconds=np.asarray(seconds),
                       clip_ids=np.array([row['clip_id'] for row in subset]), methods=np.array(list(consumer.specs)), identity_sha256=np.array(identity_hash))
            progress = dict(video=video, completed_clips=start + len(subset), total_clips=len(video_rows), seconds=time.perf_counter() - started)
            write_json(run / 'progress.json', progress)
            print('INFERENCE', json.dumps(progress), flush=True)
        write_json(destination / 'complete.json', dict(video=video, clips=len(video_rows), manifest_sha256=digest(metadata), identity_sha256=identity_hash))
    write_json(run / 'inference_complete.json', dict(videos=config['reserved_videos'], seconds=time.perf_counter() - started,
               storage_root=str(storage), peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['metadata', 'extract', 'smoke', 'infer'], required=True)
    parser.add_argument('--video')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    config = read(args.run / 'config.json')
    if args.phase == 'metadata':
        prepare_metadata(args.run, config)
        return
    if args.phase == 'extract':
        assert args.video in config['reserved_videos']
        storage = Path(config['storage_root'])
        extract_task_manifest(args.run / 'metadata' / (args.video + '.jsonl'), Path(config['frame_archive_directory']),
                              storage / 'frames', storage / 'extraction' / args.video)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.phase == 'smoke':
        smoke(args.run, config, device)
    else:
        infer(args.run, config, device)


if __name__ == '__main__':
    main()
