"""Target-native reconstruction dictionary and equal-budget box-support heads."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from src.evaluation.realcolon_task import digest,write_json


def execute(run,c,torch):
    from src.evaluation.realcolon_fixed_confirmation import tokens_cuda,arrays
    from src.evaluation.realcolon_task_cv import train_dictionary
    from src.evaluation.realcolon_task_supervised import feature_stats
    from src.evaluation.kumc_head_cv import FrameSupportSampler
    from src.evaluation.kumc_localization import localization_metrics
    source=Path(c['source_run']);sc=json.loads((source/'config.json').read_text())
    rows=[json.loads(s) for s in (source/'clip_manifest.jsonl').read_text().splitlines()]
    masks=np.load(source/'masks.npy',allow_pickle=False)
    train=np.array([r['split']=='train' for r in rows]);parts=[]
    for spec in sc['input_exchanges']:
        src=Path(spec['run']);ec=json.loads((src/'config.json').read_text())
        tokens=np.load(Path(ec['cache_dir'])/'tokens.npy',mmap_mode='r',allow_pickle=False)
        identity=c['cache_identity'][str(Path(ec['cache_dir'])/'tokens.npy')]
        st=(Path(ec['cache_dir'])/'tokens.npy').stat()
        assert st.st_size==identity['bytes'] and st.st_mtime_ns==identity['mtime_ns']
        assert digest(Path(ec['cache_dir'])/'tokens.npy')==identity['sha256']
        parts.append(tokens_cuda(tokens,list(range(len(tokens))),torch))
    raw=torch.cat(parts);del parts
    fit_indices=torch.tensor(np.flatnonzero(np.repeat(train,1568)),device='cuda')
    fit_raw=raw[fit_indices]
    mean=fit_raw.double().mean(0).float()
    rms=(fit_raw.double()-mean.double()).square().mean().sqrt().float().clamp_min(1e-6)
    x=(raw-mean)/rms;del raw,fit_raw
    fit_x=x[fit_indices]
    with torch.no_grad():
        covariance=fit_x.double().T@fit_x.double()/len(fit_x)
        values,vectors=torch.linalg.eigh(covariance);basis=vectors[:,-48:].flip(1).float()
    np.savez(run/'normalization.npz',mean=mean.cpu().numpy(),rms=rms.cpu().numpy(),basis=basis.cpu().numpy(),eigenvalues=values.cpu().numpy())
    print('Training target-native TopK: '+str(c['topk_steps'])+' steps',flush=True)
    sae,history=train_dictionary(fit_x,c,torch);sae.eval()
    for p in sae.parameters():p.requires_grad_(False)
    np.savez(run/'dictionary.npz',**{k:v.detach().cpu().numpy() for k,v in sae.state_dict().items()})
    write_json(run/'dictionary_training.json',{'history':history,'fit_clip_indices':np.flatnonzero(train).tolist(),'fit_token_count':len(fit_indices),'labels_used':False})
    print('Dictionary saved; fitting three predeclared heads',flush=True)
    sampler=FrameSupportSampler(rows,masks,train,torch,'cuda')
    write_json(run/'fit_scope.json',{'training_clip_indices':np.flatnonzero(train).tolist(),'development_clip_indices':np.flatnonzero(~train).tolist(),
        'fit_frame_indices':sampler.fit_frame_indices,'training_families':sorted({r['video_id'].split('/')[-1] for r,u in zip(rows,train) if u}),
        'development_families':sorted({r['video_id'].split('/')[-1] for r,u in zip(rows,train) if not u}),
        'statistics_and_dictionary_fit_token_count':len(fit_indices)})
    models=run/'models';models.mkdir(exist_ok=False)
    for method in c['methods']:
        transform=(lambda batch:batch) if method=='raw_native' else ((lambda batch:batch@basis) if method=='pca48_native' else sae.encode_inference)
        fm,fs=feature_stats(x,fit_indices,transform,torch)
        head=torch.nn.Linear(len(fm),1,device='cuda')
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        optimizer=torch.optim.AdamW(head.parameters(),lr=c['head_learning_rate'],weight_decay=c['head_weight_decay'])
        generator=torch.Generator(device='cuda').manual_seed(c['seed'])
        history=[]
        for step in range(c['head_steps']):
            indices,labels=sampler.sample(c['head_batch_size'],generator)
            with torch.no_grad():features=(transform(x[indices])-fm)/fs
            logits=head(features).squeeze(-1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite head loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            if (step+1)%50==0:history.append({'step':step+1,'bce':float(loss.detach())})
        dest=models/method;dest.mkdir()
        np.savez(dest/'head.npz',weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy(),feature_mean=fm.cpu().numpy(),feature_scale=fs.cpu().numpy())
        write_json(dest/'training.json',{'history':history,'head_sha256':digest(dest/'head.npz'),'head_parameters':sum(p.numel() for p in head.parameters()),
            'initialization':'all_zero_equal_probability','normalization_fit_clip_indices':np.flatnonzero(train).tolist(),'dictionary_frozen':True})
        print('FROZEN '+method,flush=True)
    freeze={'normalization.npz':digest(run/'normalization.npz'),'dictionary.npz':digest(run/'dictionary.npz'),'fit_scope.json':digest(run/'fit_scope.json')}
    freeze.update({'models/'+m+'/head.npz':digest(models/m/'head.npz') for m in c['methods']})
    write_json(run/'all_models_frozen.json',{'assets':freeze,'development_outputs_evaluated':False})
    records=[]
    for method in c['methods']:
        head=arrays(models/method/'head.npz',torch,'cuda')
        scores=[];sse=[];den=[];l0=[]
        with torch.no_grad():
            for batch in x.split(4096):
                if method=='raw_native':features=batch;reconstruction=batch
                elif method=='pca48_native':features=batch@basis;reconstruction=features@basis.T
                else:features=sae.encode_inference(batch);reconstruction=sae.decode(features)
                logits=torch.nn.functional.linear((features-head['feature_mean'])/head['feature_scale'],head['weight'],head['bias'])
                scores.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
                sse.append((reconstruction-batch).double().square().sum(1).cpu().numpy())
                den.append(batch.double().square().sum(1).cpu().numpy());l0.append((features!=0).sum(1).cpu().numpy())
        scores=np.concatenate(scores).reshape(masks.shape)
        dest=models/method;np.save(dest/'patch_predictions.npy',scores,allow_pickle=False)
        np.savez(dest/'reconstruction_statistics.npz',**{k:np.concatenate(v).reshape(masks.shape) for k,v in [('sse',sse),('denominator',den),('l0',l0)]})
        record={'method':method,'train':localization_metrics(scores[train],masks[train],[r for r,u in zip(rows,train) if u]),
            'development':localization_metrics(scores[~train],masks[~train],[r for r,u in zip(rows,train) if not u])}
        records.append(record);print(json.dumps({'method':method,'development_ap':float(np.mean([v['frame_mean_patch_ap'] for v in record['development'].values()]))}),flush=True)
    assert all(digest(run/p)==h for p,h in freeze.items())
    write_json(run/'metrics.json',{'records':records})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);run=p.parse_args().run_dir.resolve()
    c=json.loads((run/'config.json').read_text())
    for path,h in c['asset_sha256'].items():
        if digest(path)!=h:raise ValueError('Frozen asset changed: '+path)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    started=time.time();write_json(run/'status.json',{'status':'RUNNING','pid':os.getpid()})
    (run/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:execute(run,c,torch)
    except Exception as e:
        write_json(run/'status.json',{'status':'FAILED','error':repr(e)});raise
    write_json(run/'status.json',{'status':'COMPLETED','elapsed_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
