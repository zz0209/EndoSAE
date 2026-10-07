import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encode_token_causal_identity import video_inputs
from evaluate_endomind_observation import full_frame_metadata, overlap
from train_acknowledgement_sae import save_npz, now, json_digest
from src.checkpoint_io import atomic_write_json, read_json, pause_after_checkpoint
from src.evaluation.realcolon_task import digest, project_boxes


def prepare(run, smoke):
    config = read_json(run / 'config.json')
    reference = read_json(Path(config['prefix_run']) / 'config.json')
    application = read_json(Path(reference['application_reference']) / 'config.json')
    prefix = Path(reference['storage_root']) / ('smoke' if smoke else 'prepared')
    ready = read_json(prefix / 'summary.json')
    assert ready['status'] == 'COMPLETE'
    target = run / ('smoke_regions' if smoke else 'regions')
    target.mkdir(parents=True, exist_ok=True)
    identity = dict(config=digest(run / 'config.json'), prefix=digest(prefix / 'summary.json'),
        source=digest(__file__), projection=digest(ROOT / 'src/evaluation/realcolon_task.py'))
    signature = json_digest(identity)
    start = time.perf_counter()
    receipts, diagnostics = [], []
    for video in sorted({r['video'] for r in ready['receipts']}):
        phase = next(p for p in ['development', 'extension'] if video in application[p + '_videos'])
        settings = read_json(Path(application[phase + '_base']) / 'config.json')
        metadata = Path(settings['metadata'])
        metadata_hashes = {str(metadata / (video + suffix)): digest(metadata / (video + suffix))
                           for suffix in ['.json', '.jsonl']}
        _, _, frames, _, _ = full_frame_metadata(metadata, video)
        definition, _, records, tracks, _, _, _, _ = video_inputs(application, video)
        for item in [r for r in ready['receipts'] if r['video'] == video]:
            output = item['output']
            destination = target / video / f'{output:06d}'
            destination.mkdir(parents=True, exist_ok=True)
            completed = destination / 'complete.json'
            if completed.exists():
                receipt = read_json(completed)
                assert receipt['signature'] == signature and digest(destination / 'roi.npz') == receipt['roi_sha256']
            else:
                path = prefix / video / f'{output:06d}' / 'roi.npz'
                assert digest(path) == item['roi_sha256']
                with np.load(path, allow_pickle=False) as saved:
                    positions, offsets = saved['positions'].copy(), saved['offsets'].copy()
                    original = saved['token_positions'].copy()
                known = all(records[p]['frame_index'] in frames for p in range(output - 7, output + 1))
                current = frames.get(records[output]['frame_index'])
                matches = overlap(records[output]['detections'], current['original_boxes_xyxy'])['matches'] if current else []
                lesions = {m['prediction_index']: m['lesion_id'] for m in matches}
                selected, bounds, local = [], [0], []
                for j, position in enumerate(positions):
                    old = original[offsets[j]:offsets[j + 1]]
                    index = int(position - tracks['offsets'][output])
                    lesion = lesions.get(index)
                    status = 'unmatched_detection' if lesion is None else 'unknown_annotation'
                    chosen = old
                    if lesion is not None and known:
                        masks = []
                        for past in range(output - 7, output + 1):
                            frame = frames[records[past]['frame_index']]
                            boxes = [b for b in frame['original_boxes_xyxy'] if b['lesion_id'] == lesion]
                            masks.append(project_boxes(dict(frame, boxes_xyxy=boxes))[0])
                        mask = np.stack(masks)
                        if mask[-1].any():
                            temporal, spatial = np.where(mask)
                            chosen = (1 + spatial * 8 + temporal).astype(np.int64)
                            status = 'annotated_support'
                        else:
                            status = 'annotation_outside_crop'
                    assert len(chosen) and chosen.min() >= 1 and chosen.max() <= 1568
                    overlap_count = len(np.intersect1d(old, chosen))
                    local.append(dict(video=video, output=output, position=int(position), lesion=lesion,
                        status=status, original_tokens=len(old), annotated_tokens=len(chosen),
                        original_target_fraction=overlap_count / len(old),
                        support_jaccard=overlap_count / len(np.union1d(old, chosen)),
                        changed=not np.array_equal(np.sort(old), np.sort(chosen))))
                    selected.append(chosen)
                    bounds.append(bounds[-1] + len(chosen))
                save_npz(destination / 'roi.npz', positions=positions, offsets=np.array(bounds),
                    token_positions=np.concatenate(selected))
                receipt = dict(signature=signature, video=video, output=output, diagnostics=local,
                    roi_sha256=digest(destination / 'roi.npz'), original_roi_sha256=item['roi_sha256'],
                    metadata=metadata_hashes, completed_at=now())
                atomic_write_json(completed, receipt)
            receipts.append(receipt)
            diagnostics.extend(receipt['diagnostics'])
            progress = dict(status='RUNNING', completed=len(receipts), total=len(ready['receipts']),
                seconds=time.perf_counter() - start, updated_at=now())
            atomic_write_json(target / 'progress.json', progress)
            if smoke or len(receipts) % 100 == 0:
                print('ANNOTATED_CAUSAL_REGIONS', progress, flush=True)
            pause_after_checkpoint(completed)
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(target / 'diagnostics.csv', index=False)
    atomic_write_json(target / 'summary.json', dict(status='COMPLETE', identity=identity, receipts=receipts,
        status_counts=frame.status.value_counts().to_dict(), changed=int(frame.changed.sum()),
        seconds=time.perf_counter() - start, completed_at=now()))
    atomic_write_json(target / 'progress.json', dict(status='COMPLETE', completed=len(receipts), total=len(receipts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.run, args.smoke)
