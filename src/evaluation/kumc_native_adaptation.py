"""Native-initialized task/reconstruction adaptation with fixed source grouping."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from src.evaluation.realcolon_task import digest,write_json


def execute(run,c,torch):
    from src.evaluation.realcolon_fixed_confirmation import arrays,tokens_cuda
    from src.evaluation.kumc_head_cv import FrameSupportSampler
    from src.evaluation.kumc_dictionary_cv import LinearBottleneck,task_loss,evaluate
    from src.evaluation.kumc_localization import localization_metrics
    from src.sae.baselines import TopKAutoencoder
    from src.sae.low_rank_topk import LowRankTopK
    native=Path(c['native_run']);nc=json.loads((native/'config.json').read_text())
    cohort=Path(nc['source_run']);cc=json.loads((cohort/'config.json').read_text())
    rows=[json.loads(s) for s in (cohort/'clip_manifest.jsonl').read_text().splitlines()]
    masks=np.load(cohort/'masks.npy',allow_pickle=False);train=np.array([r['split']=='train' for r in rows])
    assert sorted({r['video_id'].split('/')[-1] for r,u in zip(rows,train) if u})==c['training_families']
    assert sorted({r['video_id'].split('/')[-1] for r,u in zip(rows,train) if not u})==c['development_families']
    assert set(c['training_families']).isdisjoint(c['development_families'])
    parts=[]
    for spec in cc['input_exchanges']:
        ec=json.loads((Path(spec['run'])/'config.json').read_text());path=Path(ec['cache_dir'])/'tokens.npy'
        identity=nc['cache_identity'][str(path)];st=path.stat()
        assert st.st_size==identity['bytes'] and st.st_mtime_ns==identity['mtime_ns']
        tokens=np.load(path,mmap_mode='r',allow_pickle=False);parts.append(tokens_cuda(tokens,list(range(len(tokens))),torch))
    raw=torch.cat(parts);del parts
    norm=arrays(native/'normalization.npz',torch,'cuda');x=(raw-norm['mean'])/norm['rms'];del raw
    fit_indices=torch.tensor(np.flatnonzero(np.repeat(train,1568)),device='cuda')
    sampler=FrameSupportSampler(rows,masks,train,torch,'cuda')
    write_json(run/'fit_scope.json',{'fit_clip_indices':np.flatnonzero(train).tolist(),'reconstruction_fit_clip_indices':np.flatnonzero(train).tolist(),
        'fit_frame_indices':sampler.fit_frame_indices,'training_families':c['training_families'],'development_families':c['development_families'],
        'normalization_fit_new_inputs':False,'inherited_normalization':str(native/'normalization.npz')})
    output=run/'models';output.mkdir(exist_ok=False)
    for method in c['methods']:
        parent='raw_native' if method=='raw_continue' else 'pca48_native' if method=='linear48_task' else 'topk32_native'
        initial=arrays(native/'models'/parent/'head.npz',torch,'cuda');mean,scale=initial['feature_mean'],initial['feature_scale']
        model=None
        if method=='linear48_task':model=LinearBottleneck(norm['basis'])
        elif method.startswith('topk'):
            base=TopKAutoencoder(768,1536,32).cuda();base.load_state_dict(arrays(native/'dictionary.npz',torch,'cuda'),strict=True)
            torch.manual_seed(c['seed']+200000);model=LowRankTopK(base,c['update_rank'])
        head=torch.nn.Linear(len(mean),1,device='cuda')
        head.load_state_dict({k:initial[k] for k in ['weight','bias']})
        with torch.no_grad():
            f=x[fit_indices] if model is None else model.encode_inference(x[fit_indices])
            replay=torch.sigmoid(head((f-mean)/scale)).squeeze(-1).cpu().numpy();del f
        original=np.load(native/'models'/parent/'patch_predictions.npy',allow_pickle=False)[train].ravel()
        error=float(np.max(abs(replay-original)));assert error<=1e-6
        parameters=[{'params':head.parameters(),'lr':c['head_learning_rate'],'weight_decay':c['head_weight_decay']}]
        if model is not None:parameters.append({'params':[p for p in model.parameters() if p.requires_grad],
            'lr':c['linear_learning_rate'] if method=='linear48_task' else c['low_rank_learning_rate'],'weight_decay':0.})
        optimizer=torch.optim.AdamW(parameters)
        generator=torch.Generator(device='cuda').manual_seed(c['seed']);rg=torch.Generator(device='cuda').manual_seed(c['seed']+100000)
        history=[];joint=method!='topk_reconstruction_only'
        for step in range(c['steps']):
            indices,labels=sampler.sample(c['batch_size'],generator)
            if model is None:
                bce=torch.nn.functional.binary_cross_entropy_with_logits(head((x[indices]-mean)/scale).squeeze(-1),labels);mse=x.new_tensor(0.)
            else:
                bce=task_loss(model,head,x[indices],labels,mean,scale,joint)
                selected=fit_indices[torch.randint(len(fit_indices),(c['batch_size'],),device='cuda',generator=rg)]
                batch=x[selected];mse=(model.decode(model.encode_inference(batch))-batch).square().mean()
            loss=bce+mse
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if model is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%50==0:history.append({'step':step+1,'bce':float(bce.detach()),'mse':float(mse.detach())})
        dest=output/method;dest.mkdir()
        np.savez(dest/'head.npz',weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy(),feature_mean=mean.cpu().numpy(),feature_scale=scale.cpu().numpy())
        files=['head.npz']
        if model is not None:
            np.savez(dest/'representation.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()});files.append('representation.npz')
        if method.startswith('topk'):
            assert model.base_unchanged()
            np.savez(dest/'materialized_representation.npz',**{k:v.cpu().numpy() for k,v in model.materialized_state().items()});files.append('materialized_representation.npz')
        write_json(dest/'training.json',{'history':history,'parent':parent,'task_updates_representation':model is not None and joint,
            'source_warm_probability_max_difference':error,'optimizer_state':'fresh AdamW from saved parameters',
            'representation_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0,
            'head_parameters':sum(p.numel() for p in head.parameters()),'base_unchanged':model.base_unchanged() if method.startswith('topk') else None,
            'assets':{p:digest(dest/p) for p in files}})
        print('FROZEN '+method,flush=True)
    freeze={m:json.loads((output/m/'training.json').read_text())['assets'] for m in c['methods']}
    write_json(run/'all_models_frozen.json',{'models':freeze,'fit_scope_sha256':digest(run/'fit_scope.json'),'development_outputs_evaluated':False})
    records=[]
    for method in c['methods']:
        dest=output/method;h=arrays(dest/'head.npz',torch,'cuda');head=torch.nn.Linear(len(h['feature_mean']),1,device='cuda')
        head.load_state_dict({k:h[k] for k in ['weight','bias']})
        if method=='raw_continue':
            with torch.no_grad():scores=torch.sigmoid(head((x-h['feature_mean'])/h['feature_scale'])).squeeze(-1).cpu().numpy().reshape(masks.shape)
            stats={'sse':np.zeros(masks.shape),'denominator':x.double().square().sum(1).cpu().numpy().reshape(masks.shape),'l0':np.full(masks.shape,768)}
        else:
            if method=='linear48_task':
                model=LinearBottleneck(norm['basis']);model.load_state_dict(arrays(dest/'representation.npz',torch,'cuda'),strict=True)
            else:
                model=TopKAutoencoder(768,1536,32).cuda();model.load_state_dict(arrays(dest/'materialized_representation.npz',torch,'cuda'),strict=True)
            scores,stats=evaluate(model,head,h['feature_mean'],h['feature_scale'],x,masks.shape)
        np.save(dest/'patch_predictions.npy',scores,allow_pickle=False);np.savez(dest/'reconstruction_statistics.npz',**stats)
        record={'method':method,'train':localization_metrics(scores[train],masks[train],[r for r,u in zip(rows,train) if u]),
            'development':localization_metrics(scores[~train],masks[~train],[r for r,u in zip(rows,train) if not u])}
        records.append(record);print(json.dumps({'method':method,'development_ap':float(np.mean([v['frame_mean_patch_ap'] for v in record['development'].values()]))}),flush=True)
    for method,assets in freeze.items():
        for p,h in assets.items():assert digest(output/method/p)==h
    write_json(run/'metrics.json',{'records':records})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);run=parser.parse_args().run_dir.resolve()
    c=json.loads((run/'config.json').read_text())
    for p,h in c['asset_sha256'].items():
        if digest(p)!=h:raise ValueError('Frozen asset changed: '+p)
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
