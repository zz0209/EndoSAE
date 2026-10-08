import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import collections
import hashlib
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_rc27_cohort import Encoder
from run_continuous_confirmation import prepare_input
from train_acknowledgement_sae import now, save_npz, json_digest
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.kumc_localization import support_frame
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames


def select(run):
    config = read_json(run / 'config.json')
    source = Path(config['metadata_run'])
    groups = read_json(source / 'all_source_metadata.json')
    archive = read_json(source / 'metadata_plan.json')['archive']
    candidates, exclusions = collections.defaultdict(list), []
    for key, frames in groups.items():
        counts = collections.Counter(f['original_frame_index'] for f in frames)
        eligible = []
        for frame in frames:
            reason = None
            if counts[frame['original_frame_index']] != 1:
                reason = 'ambiguous_original_frame'
            elif len(frame['objects']) != 1:
                reason = 'not_exactly_one_annotated_object'
            elif not project_boxes(support_frame(frame))[0].any():
                reason = 'no_visible_patch_support'
            if reason:
                exclusions.append(dict(source=key, frame=frame['original_frame_index'], reason=reason))
            else:
                eligible.append(frame)
        eligible.sort(key=lambda f: f['original_frame_index'])
        if len(eligible) >= config['observations_per_source'] + 7:
            candidates[key.rsplit('/', 1)[1]].append((key, eligible))
    records, clips, selected_sources = [], [], []
    for family in sorted(candidates, key=int):
        key, frames = sorted(candidates[family], key=lambda item: (-len(item[1]), item[0]))[0]
        video = 'KUMC_' + key.split('/')[1] + '_' + family
        selected_sources.append(dict(source=key, family=family, video_id=video,
            eligible_frames=len(frames), alternative_sources=[item[0] for item in candidates[family] if item[0] != key]))
        endpoints = np.linspace(7, len(frames) - 1, config['observations_per_source']).round().astype(int)
        assert len(set(endpoints.tolist())) == config['observations_per_source']
        for j, endpoint in enumerate(endpoints):
            chosen = frames[endpoint - 7:endpoint + 1]
            formatted = []
            for frame in chosen:
                adapted = support_frame(frame)
                for box in adapted['boxes_xyxy']:
                    box['lesion_id'] = 'source_lesion'
                formatted.append(dict(adapted, frame_index=frame['original_frame_index'],
                    image_member=frame['image_member'], xml_sha256=frame['xml_sha256']))
            assert all(a['frame_index'] < b['frame_index'] for a, b in zip(formatted, formatted[1:]))
            clip_id = video + '_%02d' % j
            clips.append(dict(clip_id=clip_id, video_id=video, frames=formatted))
            records.append(dict(index=len(records), video_id=video, lesion_id='source_lesion',
                split='supplemental_train', clip_id=clip_id, source_sequence_key=key, source_family=family,
                original_observation=True, roi_tokens_per_frame=[int(project_boxes(f)[0].sum()) for f in formatted],
                frame_indices=[f['frame_index'] for f in formatted]))
    assert records and len(selected_sources) == len({r['source_family'] for r in records})
    manifest = dict(config_sha256=digest(run / 'config.json'), archive=archive,
        metadata_sha256=digest(source / 'all_source_metadata.json'), records=records, clips=clips,
        sources=selected_sources, excluded_frames=exclusions,
        sequence_semantics='Original source sequence provides within-sequence lesion supervision. One sequence per numeric family; patient identities remain uncertified.',
        temporal_semantics='Eight chronological released images; original capture intervals and frame rate are unknown. No claim of an eight-frame camera-time window.',
        exposure='Author-training archive only. Previously used localization development families are allowed as supplemental training. REAL-Colon development identities remain unchanged.')
    destination = run / 'selection.json'
    if destination.exists():
        assert read_json(destination) == manifest
    else:
        atomic_write_json(destination, manifest)
    print('SELECTION', dict(sources=len(selected_sources), observations=len(records),
        excluded_frames=len(exclusions), unique_images=len({f['image_member'] for c in clips for f in c['frames']})), flush=True)
    return config, manifest


@torch.no_grad()
def prepare(run, smoke, resume):
    config, manifest = select(run)
    records, clips = manifest['records'], manifest['clips']
    archive = Path(manifest['archive']['source'])
    stat = archive.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (manifest['archive']['bytes'], manifest['archive']['mtime_ns'])
    selected = list(range(len(records)))
    if smoke:
        videos = [manifest['sources'][i]['video_id'] for i in [0, len(manifest['sources']) // 2, len(manifest['sources']) - 1]]
        selected = [index for video in videos for index in [i for i, r in enumerate(records) if r['video_id'] == video][::31]]
        assert len(selected) == 6
    root = Path(config['storage_root']) / ('smoke' if smoke else 'prepared')
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(selection_sha256=digest(run / 'selection.json'), selected_indices=selected,
        source_hashes={name: digest(ROOT / name) for name in ['scripts/prepare_supplemental_identity.py',
            'scripts/encode_rc27_cohort.py', 'scripts/run_continuous_confirmation.py',
            'src/evaluation/kumc_localization.py', 'src/evaluation/realcolon_task.py']})
    signature = json_digest(identity)
    if (root / 'identity.json').exists():
        assert resume and read_json(root / 'identity.json') == identity
    else:
        atomic_write_json(root / 'identity.json', identity)
    atomic_write_json(root / 'records.json', records)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = Encoder(read_json(config['encoding_config']), torch.device('cuda'))
    captured = {}
    handle = encoder.model.blocks[8].register_forward_hook(lambda _m, _i, out: captured.update(value=out.detach()))
    started, receipts, preview = time.perf_counter(), [], []
    frame_root = root / 'frames'
    with tarfile.open(archive, 'r:') as tar:
        for completed, index in enumerate(selected, 1):
            row, clip = records[index], clips[index]
            folder = root / 'observations' / f'{index:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            receipt_path = folder / 'complete.json'
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert resume and receipt['identity_sha256'] == signature
                assert digest(folder / 'prefix.npy') == receipt['prefix_sha256']
                assert digest(folder / 'roi.npz') == receipt['roi_sha256']
            else:
                frame_receipts = []
                for frame in clip['frames']:
                    destination = frame_root / row['video_id'] / ('%06d.jpg' % frame['frame_index'])
                    member = tar.getmember(frame['image_member'])
                    assert member.isfile()
                    blob = tar.extractfile(member).read()
                    sha = hashlib.sha256(blob).hexdigest()
                    if destination.exists():
                        assert digest(destination) == sha
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(blob)
                    frame_receipts.append(dict(frame_index=frame['frame_index'], sha256=sha))
                image, masks = prepare_input(clip, frame_root)
                np.testing.assert_array_equal(masks.sum(1), row['roi_tokens_per_frame'])
                native, _ = encoder(image)
                prefix = captured['value'][0].cpu().numpy().copy()
                assert prefix.shape == (1569, 768) and np.isfinite(prefix).all()
                positions = np.column_stack(np.where(masks)).astype(np.int16)
                value = tokens_to_frames(native[None])[masks]
                suffix_error = None
                if smoke:
                    replay = captured['value'].clone()
                    for block in encoder.model.blocks[9:11]:
                        replay = block(replay, 1, 8, 14)
                    np.testing.assert_array_equal(replay[0].cpu().numpy(), native)
                    suffix_error = float(np.abs(replay[0].cpu().numpy() - native).max())
                    linear = 1 + positions[:, 1].astype(np.int64) * 8 + positions[:, 0]
                    np.testing.assert_array_equal(native[linear], value)
                temporary = folder / 'prefix.pending.npy'
                np.save(temporary, prefix, allow_pickle=False)
                temporary.replace(folder / 'prefix.npy')
                save_npz(folder / 'roi.npz', native=value, positions=positions)
                receipt = dict(identity_sha256=signature, index=index, prefix_sha256=digest(folder / 'prefix.npy'),
                    roi_sha256=digest(folder / 'roi.npz'), suffix_max_error=suffix_error,
                    frames=frame_receipts, backbone_sha256=encoder.state_sha256, completed_at=now())
                atomic_write_json(receipt_path, receipt)
            receipts.append(receipt)
            if smoke:
                frame = clip['frames'][-1]
                with Image.open(frame_root / row['video_id'] / ('%06d.jpg' % frame['frame_index'])) as original:
                    tile = original.convert('RGB')
                draw = ImageDraw.Draw(tile)
                draw.rectangle(frame['boxes_xyxy'][0]['box'], outline='yellow', width=3)
                tile.thumbnail((400, 300))
                preview.append((tile, row['video_id'] + ' frame ' + str(frame['frame_index'])))
            progress = dict(status='RUNNING', completed=completed, total=len(selected),
                seconds=time.perf_counter() - started, video=row['video_id'], updated_at=now())
            atomic_write_json(root / 'progress.json', progress)
            if completed % 10 == 0 or smoke:
                print('SUPPLEMENTAL_PREFIX', progress, flush=True)
            pause_after_checkpoint(receipt_path)
    handle.remove()
    if smoke:
        canvas = Image.new('RGB', (800, 990), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (tile, label) in enumerate(preview):
            x, y = i % 2 * 400, i // 2 * 330
            canvas.paste(tile, (x, y + 25))
            draw.text((x + 5, y + 5), label, fill='black')
        canvas.save(run / 'smoke_inputs.png')
    summary = dict(status='COMPLETE', observations=len(selected), sources=len(manifest['sources']),
        seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        identity_sha256=signature, receipts=receipts, completed_at=now(),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version, tf32=False))
    atomic_write_json(root / 'summary.json', summary)
    atomic_write_json(root / 'progress.json', dict(status='COMPLETE', completed=len(selected), total=len(selected)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.select_only:
        select(args.run)
    else:
        prepare(args.run, args.smoke, args.resume)
