import os

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnose_gt_roi_identity import statistics
from src.evaluation.realcolon_task import digest,project_boxes,tokens_to_frames,write_json


def read(path):
    return json.loads(Path(path).read_text())


def roster(config):
    rows=[json.loads(line) for line in (Path(config['metadata_root'])/'clip_manifest.jsonl').read_text().splitlines()]
    result={v:rows_for_video for v in sorted({r['video_id'] for r in rows}) if (rows_for_video:=[r for r in rows if r['video_id']==v])}
    assert all(r['split'] in ['train','val'] for r in rows)
    for split,expected in config.get('official_cohort_expected',config['expected']).items():
        group=[r for r in rows if r['split']==split]
        assert len({r['video_id'] for r in group})==expected['videos']
        assert len({l for r in group for l in r['sampling_lesion_ids']})==expected['lesions']
        assert sum(len(r['sampling_lesion_ids']) for r in group)==expected['descriptors']
    return result


def observation_interval(video,lesion,clip):
    info=next(r for r in video['lesions'] if r['lesion_id']==lesion)
    cursor=info['first_frame']
    intervals=[]
    for gap in sorted([r for r in video['gaps'] if r['lesion_id']==lesion],key=lambda r:r['gap_start_frame']):
        intervals.append((cursor,gap['gap_start_frame']-1))
        cursor=gap['gap_end_frame']+1
    intervals.append((cursor,info['last_frame']))
    assert sum(b-a+1 for a,b in intervals)==info['annotated_frames']
    matched=[(i,a,b) for i,(a,b) in enumerate(intervals) if a<=clip['start_frame'] and b>=clip['end_frame']]
    assert len(matched)==1
    i,a,b=matched[0]
    return dict(interval_id=f'{lesion}:annotation_interval_{i}',start_frame=a,end_frame=b,lesion_first_frame=info['first_frame'])


def update_status(config,run,rows):
    status={}
    for split in ['train','val']:
        videos=[v for v,c in rows.items() if c[0]['split']==split]
        completed=[v for v in videos if (run/'descriptors'/v/'complete.json').exists()]
        status[split]=dict(expected_videos=len(videos),completed=completed,pending=[v for v in videos if v not in completed],cache_pending=[v for v in videos if not (Path(config['cache_root'])/v/'complete.json').exists()])
    ready=all(not s['pending'] for s in status.values())
    write_json(run/'extraction_status.json',dict(status='READY' if ready else 'PENDING',splits=status))
    return ready


def extract(config,run,video_filter):
    rows=roster(config)
    annotations={v['video_id']:v for v in read(config['annotation_summary'])['videos']}
    selected=[video_filter] if video_filter else list(rows)
    assert all(v in rows for v in selected)
    for video in selected:
        cache=Path(config['cache_root'])/video
        out=run/'descriptors'/video
        if (out/'complete.json').exists():
            print('REUSE',video,flush=True)
            continue
        if not (cache/'complete.json').exists():
            print('PENDING_CACHE',video,flush=True)
            continue
        identity=read(cache/'identity.json')
        complete=read(cache/'complete.json')
        clips=rows[video]
        metadata=Path(config['metadata_root'])/(video+'.json')
        assert complete['status']=='COMPLETE' and identity['metadata_sha256']==digest(metadata)
        assert complete['identity_sha256']==digest(cache/'identity.json')
        assert identity['clip_ids']==[c['clip_id'] for c in clips]
        native=np.load(cache/'block10.npy',mmap_mode='r')
        assert native.shape==(len(clips),1569,768)
        raw,normalized,records=[],[],[]
        start=time.perf_counter()
        for i,clip in enumerate(clips):
            if not clip['sampling_lesion_ids']:
                continue
            tokens=tokens_to_frames(np.asarray(native[i])[None])
            for lesion in clip['sampling_lesion_ids']:
                masks=np.stack([project_boxes(dict(frame,boxes_xyxy=[b for b in frame['boxes_xyxy'] if b['lesion_id']==lesion]))[0] for frame in clip['frames']])
                assert masks.any(axis=1).all()
                mean=tokens[masks].mean(axis=0,dtype=np.float64)
                length=np.linalg.norm(mean)
                assert np.isfinite(mean).all() and length>0
                interval=observation_interval(annotations[video],lesion,clip)
                records.append(dict(index=len(raw),video_id=video,split=clip['split'],lesion_id=lesion,clip_id=clip['clip_id'],cache_clip_index=i,start_frame=clip['start_frame'],end_frame=clip['end_frame'],fps=clip['fps'],roi_tokens_per_frame=masks.sum(axis=1).tolist(),original_mean_norm=float(length),annotation_observation=interval,lesion_first_frame=interval['lesion_first_frame']))
                raw.append(mean)
                normalized.append(mean/length)
            print('EXTRACT',video,len(raw),'lesion clips',flush=True)
        expected=sum(len(c['sampling_lesion_ids']) for c in clips)
        assert len(records)==expected
        out.mkdir(parents=True,exist_ok=True)
        np.savez(out/'descriptors.npz',raw=np.stack(raw),l2=np.stack(normalized))
        write_json(out/'records.json',records)
        write_json(out/'complete.json',dict(status='COMPLETE',video_id=video,split=clips[0]['split'],descriptors=len(records),lesions=len({r['lesion_id'] for r in records}),cache_identity_sha256=digest(cache/'identity.json'),metadata_sha256=digest(metadata),source_sha256=digest(__file__),descriptor_sha256=digest(out/'descriptors.npz'),record_sha256=digest(out/'records.json'),seconds=time.perf_counter()-start))
        del native
    ready=update_status(config,run,rows)
    print('COHORT_STATUS','READY' if ready else 'PENDING',flush=True)


def load_descriptors(config,run,smoke):
    rows=roster(config)
    descriptor_root=Path(config.get('input_descriptor_root',str(run/'descriptors')))
    if 'fit_video_ids' in config:
        selected=[]
        for split,video_ids in config['fit_video_ids'].items():
            assert split in ['train','val'] and len(video_ids)==len(set(video_ids))
            assert len(video_ids)==config['expected'][split]['videos']
            assert all(v in rows and rows[v][0]['split']==split for v in video_ids)
            selected.extend(video_ids)
        assert len(selected)==len(set(selected))
    else:
        selected=list(rows)
    if not smoke and 'fit_video_ids' not in config:
        assert update_status(config,run,rows),'Formal fit requires every train/val export'
    arrays=[]
    records=[]
    inputs=[]
    for video in ([config['smoke_video']] if smoke else selected):
        assert video in selected
        directory=descriptor_root/video
        receipt=read(directory/'complete.json')
        assert receipt['status']=='COMPLETE' and receipt['video_id']==video
        assert receipt['descriptor_sha256']==digest(directory/'descriptors.npz')
        assert receipt['record_sha256']==digest(directory/'records.json')
        local=read(directory/'records.json')
        assert all(r['video_id']==video and r['split']==rows[video][0]['split'] for r in local)
        with np.load(directory/'descriptors.npz') as a:
            arrays.append(a['raw'].copy())
        records.extend(local)
        inputs.append(dict(video_id=video,split=rows[video][0]['split'],directory=str(directory),receipt=receipt))
    raw=np.concatenate(arrays)
    assert len(raw)==len(records)
    if not smoke:
        for split,expected in config['expected'].items():
            group=[r for r in records if r['split']==split]
            assert len(group)==expected['descriptors'] and len({r['video_id'] for r in group})==expected['videos'] and len({r['lesion_id'] for r in group})==expected['lesions']
        assert {r['lesion_id'] for r in records if r['split']=='train'}.isdisjoint({r['lesion_id'] for r in records if r['split']=='val'})
        assert {r['video_id'] for r in records if r['split']=='train'}.isdisjoint({r['video_id'] for r in records if r['split']=='val'})
    write_json(run/('smoke_descriptor_inputs.json' if smoke else 'fit_descriptor_inputs.json'),inputs)
    return raw,records


def make_model(config):
    a,b,c=config['projection']
    return nn.Sequential(nn.Linear(a,b),nn.ReLU(),nn.Linear(b,c))


def supcon(embeddings,labels,temperature):
    n=len(labels)
    logits=embeddings@embeddings.T/temperature
    diag=torch.eye(n,dtype=torch.bool,device=logits.device)
    positive=(labels[:,None]==labels[None,:])&~diag
    assert positive.any(dim=1).all()
    log_probability=logits-torch.logsumexp(logits.masked_fill(diag,-torch.inf),dim=1,keepdim=True)
    return -(torch.where(positive,log_probability,0).sum(dim=1)/positive.sum(dim=1)).mean()


def sample_batch(records,rng,config):
    videos=sorted({r['video_id'] for r in records if r['split']=='train'})
    chosen=rng.choice(videos,size=min(config['batch_procedures'],len(videos)),replace=False)
    indices=[]
    for video in chosen:
        lesions=sorted({r['lesion_id'] for r in records if r['video_id']==video})
        for lesion in rng.choice(lesions,size=min(config['lesions_per_procedure'],len(lesions)),replace=False):
            available=[i for i,r in enumerate(records) if r['lesion_id']==lesion]
            indices.extend(rng.choice(available,size=config['views_per_lesion'],replace=False).tolist())
    assert len(indices)==len(set(indices))
    return indices


def evaluate(embeddings,records):
    sources={}
    for i,r in sorted(enumerate(records),key=lambda x:x[1]['start_frame']):
        sources.setdefault(r['lesion_id'],i)
    pairs=[]
    for lesion,i in sources.items():
        s=records[i]
        for j,q in enumerate(records):
            if q['video_id']!=s['video_id'] or q['start_frame']<=s['end_frame']:
                continue
            same=q['lesion_id']==lesion
            if not same and q['lesion_first_frame']<=s['end_frame']:
                continue
            pairs.append(dict(video_id=s['video_id'],source_lesion_id=lesion,query_lesion_id=q['lesion_id'],source_clip_id=s['clip_id'],query_clip_id=q['clip_id'],same_identity=same,cosine=float(embeddings[i]@embeddings[j]),source_end_frame=s['end_frame'],query_start_frame=q['start_frame'],same_annotation_interval=s['annotation_observation']['interval_id']==q['annotation_observation']['interval_id']))
    by_video={v:statistics([p for p in pairs if p['video_id']==v]) for v in sorted({r['video_id'] for r in records})}
    by_source={l:statistics([p for p in pairs if p['source_lesion_id']==l]) for l in sources}
    macro=float(np.mean([r['auroc'] for r in by_video.values() if r['auroc'] is not None]))
    return dict(procedures=len(by_video),lesions=len(sources),evaluable_procedures=sum(r['auroc'] is not None for r in by_video.values()),macro_procedure_auroc=macro,by_video=by_video,by_source_lesion=by_source,pairs=pairs)


def save_model(path,model):
    np.savez(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})


def fit(config,run,smoke):
    raw,records=load_descriptors(config,run,smoke)
    train=np.array([r['split']=='train' for r in records])
    scaler=StandardScaler().fit(raw[train])
    x=torch.from_numpy(scaler.transform(raw).astype(np.float32))
    torch.manual_seed(config['seed'])
    rng=np.random.default_rng(config['seed'])
    model=make_model(config)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    labels={l:i for i,l in enumerate(sorted({r['lesion_id'] for r in records}))}
    y=torch.tensor([labels[r['lesion_id']] for r in records])
    output=run/('cpu_smoke' if smoke else 'fit')
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'summary.json').exists()
    np.savez(output/'normalization.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_samples_seen=scaler.n_samples_seen_)
    unit_raw=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    for split in (['train'] if smoke else ['train','val']):
        selected=np.array([r['split']==split for r in records])
        write_json(output/f'raw_{split}_diagnostic.json',evaluate(unit_raw[selected],[r for i,r in enumerate(records) if selected[i]]))
    history=[]
    evaluations=[]
    sequence=[]
    steps=2 if smoke else config['steps']
    started=time.perf_counter()
    for step in range(1,steps+1):
        indices=sample_batch(records,rng,config)
        sequence.append(indices)
        z=F.normalize(model(x[indices]),dim=1)
        loss=supcon(z,y[indices],config['temperature'])
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm']))
        assert grad_norm>0
        optimizer.step()
        batch=[records[i] for i in indices]
        internal_negatives=sum(a['video_id']==b['video_id'] and a['lesion_id']!=b['lesion_id'] for a in batch for b in batch)
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad_norm,batch_size=len(indices),same_procedure_negative_pairs=internal_negatives))
        if step%50==0 or smoke:
            print('TRAIN',step,steps,float(loss.detach()),flush=True)
        if step in config['checkpoints'] or step==steps:
            save_model(output/f'model_step{step:04d}.npz',model)
            with torch.no_grad():
                embeddings=F.normalize(model(x),dim=1).numpy()
            chosen=np.ones(len(records),dtype=bool) if smoke else ~train
            evaluation=evaluate(embeddings[chosen],[r for i,r in enumerate(records) if chosen[i]])
            write_json(output/f'validation_step{step:04d}.json',evaluation)
            evaluations.append(dict(step=step,macro_procedure_auroc=evaluation['macro_procedure_auroc']))
    assert any(not torch.equal(original[k],v) for k,v in model.state_dict().items())
    best=max(evaluations,key=lambda r:(r['macro_procedure_auroc'],-r['step']))
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in np.load(output/f"model_step{best['step']:04d}.npz").items()},strict=True)
    with torch.no_grad():
        encoded=F.normalize(model(x),dim=1).numpy()
    assert np.isfinite(encoded).all() and np.allclose(np.linalg.norm(encoded,axis=1),1,atol=1e-5)
    save_model(output/'model.npz',model)
    rebuilt=make_model(config)
    rebuilt.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in np.load(output/'model.npz').items()},strict=True)
    with torch.no_grad():
        assert np.array_equal(encoded,F.normalize(rebuilt(x),dim=1).numpy())
    np.save(output/'embeddings.npy',encoded)
    for split in (['train'] if smoke else ['train','val']):
        selected=np.array([r['split']==split for r in records])
        write_json(output/f'selected_{split}_diagnostic.json',evaluate(encoded[selected],[r for i,r in enumerate(records) if selected[i]]))
    write_json(output/'records.json',records)
    write_json(output/'history.json',history)
    write_json(output/'sequence.json',sequence)
    write_json(output/'summary.json',dict(status='PASS_REAL_CPU_SMOKE' if smoke else 'COMPLETE',smoke=smoke,steps=steps,selected_checkpoint=best,evaluations=evaluations,seconds=time.perf_counter()-started,normalization_training_samples=int(scaler.n_samples_seen_),descriptors=len(records),procedures=len({r['video_id'] for r in records}),lesions=len(labels),model_reload_exact=True,source_sha256=digest(__file__),config_sha256=digest(run/'config.json'),runtime=dict(torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__,device='cpu',threads=1)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['extract','smoke','fit','status'],required=True)
    parser.add_argument('--video')
    args=parser.parse_args()
    config=read(args.config)
    run=Path(args.config).parent
    torch.set_num_threads(1)
    if args.phase=='extract':
        extract(config,run,args.video)
    elif args.phase=='status':
        print(update_status(config,run,roster(config)))
    else:
        fit(config,run,args.phase=='smoke')


if __name__=='__main__':
    main()
