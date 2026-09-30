import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.realcolon_task import digest, project_boxes, tokens_to_frames, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def roi(native, clip, lesion):
    masks = np.stack([project_boxes(dict(frame, boxes_xyxy=[b for b in frame['boxes_xyxy'] if b['lesion_id'] == lesion]))[0] for frame in clip['frames']])
    assert masks.shape == (8, 196) and masks.any(axis=1).all()
    features = tokens_to_frames(native[None])
    selected = features[masks]
    mean = selected.mean(axis=0, dtype=np.float64)
    norm = np.linalg.norm(mean)
    assert np.isfinite(selected).all() and norm > 0
    return mean / norm, masks, norm


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()), minimum=float(a.min()), q25=float(np.quantile(a,.25)), median=float(np.median(a)), q75=float(np.quantile(a,.75)), maximum=float(a.max())) if len(a) else dict(n=0)


def statistics(rows):
    labels = np.array([p['same_identity'] for p in rows], dtype=int)
    scores = np.array([p['cosine'] for p in rows])
    result = dict(pairs=len(rows), same=describe(scores[labels==1]), different=describe(scores[labels==0]), auroc=None)
    if len(set(labels)) == 2:
        result['auroc'] = float(roc_auc_score(labels,scores))
        fpr,tpr,thresholds = roc_curve(labels,scores,drop_intermediate=False)
        result['roc'] = dict(fpr=fpr.tolist(),tpr=tpr.tolist(),thresholds=[float(x) if np.isfinite(x) else None for x in thresholds])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['smoke','full'], required=True)
    args = parser.parse_args()
    config = read(args.config)
    run = Path(args.config).parent / args.phase
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / 'summary.json').exists()
    started = time.perf_counter()
    annotation = {v['video_id']:v for v in read(config['annotation_summary'])['videos']}
    descriptors, records, identities = [], [], []
    videos = config['videos'][:1] if args.phase == 'smoke' else config['videos']
    for video in videos:
        directory = Path(config['cache_root']) / video
        identity, complete = read(directory/'identity.json'),read(directory/'complete.json')
        metadata = Path(config['metadata_root']) / (video+'.json')
        clips = read(metadata)['clips']
        assert complete['status']=='COMPLETE' and identity['split']==config['split']=='train'
        assert complete['identity_sha256']==digest(directory/'identity.json')
        assert identity['metadata_sha256']==digest(metadata)
        assert identity['clip_ids']==[c['clip_id'] for c in clips]
        native = np.load(directory/'block10.npy',mmap_mode='r')
        cached_masks = np.load(directory/'masks.npy',mmap_mode='r')
        frame_indices = np.load(directory/'frame_indices.npy')
        assert native.shape==(len(clips),1569,768)
        assert np.array_equal(frame_indices,[[f['frame_index'] for f in c['frames']] for c in clips])
        lesion_first = {x['lesion_id']:x['first_frame'] for x in annotation[video]['lesions']}
        identities.append(dict(video_id=video,cache_identity=identity,complete=complete,metadata_sha256=digest(metadata)))
        count = 0
        for index,clip in enumerate(clips):
            lesions = clip['sampling_lesion_ids']
            if not lesions:
                continue
            block = np.asarray(native[index])
            actual_union = np.stack([project_boxes(f)[0] for f in clip['frames']])
            assert np.array_equal(actual_union,cached_masks[index])
            for lesion in lesions:
                descriptor,mask,norm = roi(block,clip,lesion)
                positions,times = np.nonzero(mask.T)
                direct = block[1+positions*8+times].mean(axis=0,dtype=np.float64)
                direct /= np.linalg.norm(direct)
                assert np.allclose(descriptor,direct,rtol=0,atol=1e-12)
                assert abs(np.linalg.norm(descriptor)-1)<1e-12
                records.append(dict(descriptor_index=len(descriptors),video_id=video,lesion_id=lesion,clip_id=clip['clip_id'],cache_clip_index=index,start_frame=clip['start_frame'],end_frame=clip['end_frame'],fps=clip['fps'],lesion_first_frame=lesion_first[lesion],roi_token_count=int(mask.sum()),roi_tokens_per_frame=mask.sum(axis=1).tolist(),native_mean_norm=float(norm)))
                descriptors.append(descriptor)
                count += 1
                if args.phase=='smoke':
                    break
            if args.phase=='smoke':
                break
        print('DESCRIPTORS',video,count,'lesion clips',flush=True)
    write_json(run/'descriptors.json',records)
    np.save(run/'descriptors.npy',np.stack(descriptors))
    write_json(run/'identities.json',identities)
    if args.phase=='smoke':
        write_json(run/'summary.json',dict(status='PASS',actual_clip=records[0],checks=['Real GT union equals cached mask','Spatial-major temporal-minor direct indexing equals mature conversion','Native values finite','L2 norm equals one'],seconds=time.perf_counter()-started))
        print('SMOKE_PASS',records[0]['clip_id'],flush=True)
        return
    pairs=[]
    sources={}
    for row in sorted(records,key=lambda r:r['start_frame']):
        sources.setdefault(row['lesion_id'],row)
    for lesion,source in sources.items():
        for query in records:
            if query['video_id'] != source['video_id'] or query['start_frame'] <= source['end_frame']:
                continue
            same = lesion==query['lesion_id']
            if not same and query['lesion_first_frame'] <= source['end_frame']:
                continue
            cosine=float(np.dot(descriptors[source['descriptor_index']],descriptors[query['descriptor_index']]))
            pairs.append(dict(source_lesion_id=lesion,query_lesion_id=query['lesion_id'],video_id=source['video_id'],source_clip_id=source['clip_id'],query_clip_id=query['clip_id'],source_end_frame=source['end_frame'],query_start_frame=query['start_frame'],elapsed_seconds=(query['start_frame']-source['end_frame'])/source['fps'],same_identity=same,cosine=cosine))
    by_video={v:statistics([p for p in pairs if p['video_id']==v]) for v in config['videos']}
    by_lesion={l:dict(source=sources[l],**statistics([p for p in pairs if p['source_lesion_id']==l])) for l in sources}
    result=dict(status='COMPLETE',generated_at_utc=datetime.now(timezone.utc).isoformat(),config=config,config_sha256=digest(args.config),source_sha256=digest(__file__),projection_source_sha256=digest(ROOT/'src/evaluation/realcolon_task.py'),runtime=dict(python=platform.python_version(),numpy=np.__version__,device='cpu',threads=1),seconds=time.perf_counter()-started,procedures=len(videos),lesions=len(sources),descriptors=len(records),pooled=statistics(pairs),by_video=by_video,by_source_lesion=by_lesion,macro_procedure_auroc=float(np.mean([v['auroc'] for v in by_video.values() if v['auroc'] is not None])),macro_evaluable_source_auroc=float(np.mean([v['auroc'] for v in by_lesion.values() if v['auroc'] is not None])))
    write_json(run/'predictions.json',pairs)
    result['descriptor_clips_per_second']=len(records)/result['seconds']
    result['maximum_native_clip_bytes_read']=len(records)*1569*768*4
    write_json(run/'summary.json',result)
    lines=['# GT-ROI identity association diagnostic','',config['information'],'','| Procedure | Same pairs | Different pairs | Same median | Different median | AUROC |','|---|---:|---:|---:|---:|---:|']
    for video,row in by_video.items():
        lines.append(f"| {video} | {row['same']['n']} | {row['different']['n']} | {row['same']['median']:.4f} | {row['different']['median']:.4f} | {row['auroc']:.4f} |")
    lines += ['',f"{len(sources)} source lesions; {len(records)} GT-ROI descriptors; {len(pairs)} within-procedure chronological comparisons. Procedure-mean AUROC {result['macro_procedure_auroc']:.4f}. Source lesions with no future different identity have an undefined individual AUROC and remain in the saved counts.",'','All descriptor identities, box-support token counts, individual similarities, temporal distances and complete ROC coordinates are saved. Source clips use all eight frames and are available only at clip end. This measures identity association in sampled GT regions; continuous recurrence, detector-box quality and feedback outcomes require their own evaluation.','']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(procedures=result['procedures'],lesions=result['lesions'],descriptors=result['descriptors'],by_video=by_video,seconds=result['seconds']),indent=2),flush=True)


if __name__=='__main__':
    main()
