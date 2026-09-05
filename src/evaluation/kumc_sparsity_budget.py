"""Bounded sparsity budget comparison with training-only moments and shared heads."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

from src.evaluation.realcolon_task import digest,write_json


def fit(run,config,torch):
    from src.evaluation.realcolon_fixed_confirmation import arrays,tokens_cuda
    from src.evaluation.kumc_head_cv import FrameSupportSampler
    from src.evaluation.kumc_dictionary_cv import LinearBottleneck,evaluate,task_loss
    from src.sae.baselines import TopKAutoencoder
    from src.sae.low_rank_topk import LowRankTopK
    rows=[json.loads(line) for line in (run/'clip_manifest.jsonl').read_text().splitlines()]
    raw_parts=[];mask_parts=[];receipts=[]
    offset=0
    for spec in config['input_exchanges']:
        source=Path(spec['run']);old=json.loads((source/'config.json').read_text())
        old_rows=[json.loads(line) for line in (source/'clip_manifest.jsonl').read_text().splitlines()]
        prepared=json.loads((source/'prepared.json').read_text());encoded=json.loads((source/'encoded.json').read_text())
        assert encoded['manifest_sha256']==digest(source/'clip_manifest.jsonl')==prepared['manifest_sha256']
        assert encoded['prepared_sha256']==digest(source/'prepared.json')
        assert encoded['state_exchange_sha256']==config['state_exchange_sha256']
        cache=Path(old['cache_dir'])
        assert digest(cache/'tokens.npy')==encoded['tokens_sha256']
        assert digest(cache/'masks.npy')==prepared['assets']['masks.npy']
        for i,row in enumerate(old_rows):
            current=rows[offset+i]
            for key in ['clip_id','video_id','frames','original_frame_ids','original_frame_gaps']:
                assert current[key]==row[key]
        tokens=np.load(cache/'tokens.npy',mmap_mode='r',allow_pickle=False)
        raw_parts.append(tokens_cuda(tokens,list(range(len(old_rows))),torch))
        mask_parts.append(np.load(cache/'masks.npy',allow_pickle=False))
        receipts.extend(json.loads((source/'frame_receipts.json').read_text()))
        offset+=len(old_rows)
    assert offset==len(rows)
    # Exact duplicated pixels cannot be silently counted across newly separated sources.
    hashes=[r['image_sha256'] for r in receipts]
    hash_families={}
    for i,value in enumerate(hashes):
        hash_families.setdefault(value,set()).add(rows[i//8]['video_id'].split('/')[-1])
    if any(len(values)>1 for values in hash_families.values()):
        raise ValueError('Exact duplicate pixels across source families require explicit grouping')
    raw=torch.cat(raw_parts);del raw_parts
    masks=np.concatenate(mask_parts)
    np.save(run/'masks.npy',masks,allow_pickle=False)
    train=np.array([r['split']=='train' for r in rows])
    fit_families={r['video_id'].split('/')[-1] for r,use in zip(rows,train) if use}
    held_families={r['video_id'].split('/')[-1] for r,use in zip(rows,train) if not use}
    assert fit_families.isdisjoint(held_families)
    assert sorted(fit_families)==config['training_families'] and sorted(held_families)==config['development_families']
    model_config=json.loads(Path(config['model_config']).read_text())
    norm=arrays(Path(model_config['normalization']),torch,'cuda');x=(raw-norm['mean'])/norm['rms'];del raw
    output=run/'models';output.mkdir(exist_ok=False)
    (output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    fit_indices=torch.tensor(np.flatnonzero(np.repeat(train,8*196)),device='cuda')
    fit_clips=np.flatnonzero(train).tolist()
    sampler=FrameSupportSampler(rows,masks,train,torch,'cuda')
    write_json(run/'fit_scope.json',{'training_clip_indices':fit_clips,'development_clip_indices':np.flatnonzero(~train).tolist(),
        'training_families':sorted(fit_families),'development_families':sorted(held_families),'fit_frame_indices':sampler.fit_frame_indices,
        'selected_unique_image_payloads':len(set(hashes)),'previous_clips_reused':config.get('reused_clip_count',config['old_clip_count'])})
    from src.evaluation.realcolon_task_supervised import feature_stats
    with torch.no_grad():
        covariance=torch.zeros((768,768),dtype=torch.float64,device='cuda')
        for ids in fit_indices.split(4096):
            batch=x[ids].double();covariance+=batch.T@batch
        values,vectors=torch.linalg.eigh(covariance/len(fit_indices))
        basis=vectors[:,-192:].flip(1).float()
    np.savez(run/'linear_initialization.npz',basis=basis.cpu().numpy(),eigenvalues=values.cpu().numpy())
    # The shared input origin stays transferred; this is an uncentered second-moment basis.
    saved=[]
    for method in config['methods']:
        sparse=method.startswith('topk');model=None
        if method.startswith('linear'):
            width=int(method.removeprefix('linear'));model=LinearBottleneck(basis[:,:width])
        elif sparse:
            k=int(method.removeprefix('topk'));base=TopKAutoencoder(768,1536,k).cuda()
            base.load_state_dict(arrays(Path(model_config['models']['topk_frozen_balanced']['dictionary']),torch,'cuda'),strict=True)
            base.k.fill_(k)
            torch.manual_seed(config['seed']+200000);model=LowRankTopK(base,16)
        transform=(lambda batch:batch) if model is None else model.encode_inference
        mean,scale=feature_stats(x,fit_indices,transform,torch)
        head=torch.nn.Linear(len(mean),1,device='cuda')
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        dest=output/method;dest.mkdir()
        np.savez(dest/'initial_head.npz',weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy(),feature_mean=mean.cpu().numpy(),feature_scale=scale.cpu().numpy())
        if model is not None:
            np.savez(dest/'initial_representation.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        groups=[{'params':head.parameters(),'lr':config['head_learning_rate'],'weight_decay':1.}]
        if model is not None:
            groups.append({'params':[p for p in model.parameters() if p.requires_grad],'lr':config['low_rank_learning_rate'] if sparse else config['linear_learning_rate'],'weight_decay':0.})
        optimizer=torch.optim.AdamW(groups)
        generator=torch.Generator(device='cuda').manual_seed(config['seed'])
        rg=torch.Generator(device='cuda').manual_seed(config['seed']+100000)
        history=[]
        for step in range(config['steps']):
            indices,labels=sampler.sample(config['batch_size'],generator)
            if model is None:
                bce=torch.nn.functional.binary_cross_entropy_with_logits(head((x[indices]-mean)/scale).squeeze(-1),labels);mse=x.new_tensor(0.)
            else:
                bce=task_loss(model,head,x[indices],labels,mean,scale,True)
                chosen=fit_indices[torch.randint(len(fit_indices),(config['batch_size'],),generator=rg,device='cuda')]
                batch=x[chosen];mse=(model.decode(model.encode_inference(batch))-batch).square().mean()
            loss=bce+mse
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if model is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%50==0:history.append({'step':step+1,'bce':float(bce.detach()),'mse':float(mse.detach())})
        np.savez(dest/'head.npz',weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy(),feature_mean=mean.cpu().numpy(),feature_scale=scale.cpu().numpy())
        files=['head.npz','initial_head.npz']
        if model is not None:
            np.savez(dest/'representation.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()});files+=['representation.npz','initial_representation.npz']
        if sparse:
            assert model.base_unchanged()
            np.savez(dest/'materialized_representation.npz',**{k:v.cpu().numpy() for k,v in model.materialized_state().items()});files+=['materialized_representation.npz']
        write_json(dest/'training.json',{'history':history,'statistics_fit_token_count':len(fit_indices),'initial_head':'zero_weights_and_bias',
            'before_development_evaluation':True,'representation_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0,
            'representation_total_parameters':sum(p.numel() for p in model.parameters()) if model else 0,'head_parameters':sum(p.numel() for p in head.parameters()),
            'coefficient_payload_bytes':int(method.removeprefix('topk'))*6 if sparse else (int(method.removeprefix('linear'))*4 if model else 768*4),
            'assets':{p:digest(dest/p) for p in files}})
        saved.append(method);print('FROZEN '+method,flush=True)
    write_json(run/'all_models_frozen.json',{'methods':saved,'model_hashes':{m:json.loads((output/m/'training.json').read_text())['assets'] for m in saved},
        'development_predictions_evaluated':False,'fit_scope_sha256':digest(run/'fit_scope.json'),'linear_initialization_sha256':digest(run/'linear_initialization.npz')})
    from src.evaluation.kumc_localization import localization_metrics
    records=[]
    for method in config['methods']:
        dest=output/method;h=arrays(dest/'head.npz',torch,'cuda');head=torch.nn.Linear(len(h['feature_mean']),1,device='cuda')
        head.load_state_dict({k:h[k] for k in ['weight','bias']})
        if method=='raw':
            with torch.no_grad():scores=torch.sigmoid(head((x-h['feature_mean'])/h['feature_scale'])).squeeze(-1).cpu().numpy().reshape(masks.shape)
            stats={'sse':np.zeros(masks.shape),'denominator':x.double().square().sum(-1).cpu().numpy().reshape(masks.shape),'l0':np.full(masks.shape,768)}
        else:
            if method.startswith('linear'):
                model=LinearBottleneck(basis[:,:int(method.removeprefix('linear'))]);model.load_state_dict(arrays(dest/'representation.npz',torch,'cuda'),strict=True)
            else:
                model=TopKAutoencoder(768,1536,int(method.removeprefix('topk'))).cuda();model.load_state_dict(arrays(dest/'materialized_representation.npz',torch,'cuda'),strict=True)
            scores,stats=evaluate(model,head,h['feature_mean'],h['feature_scale'],x,masks.shape)
        np.save(dest/'patch_predictions.npy',scores,allow_pickle=False);np.savez(dest/'reconstruction_statistics.npz',**stats)
        records.append({'method':method,'train':localization_metrics(scores[train],masks[train],[r for r,u in zip(rows,train) if u]),
            'development':localization_metrics(scores[~train],masks[~train],[r for r,u in zip(rows,train) if not u])})
        print(json.dumps({'method':method,'development_ap':float(np.mean([v['frame_mean_patch_ap'] for v in records[-1]['development'].values()]))}),flush=True)
    for method in saved:
        for p,hsh in json.loads((output/method/'training.json').read_text())['assets'].items():assert digest(output/method/p)==hsh
    write_json(run/'metrics.json',{'records':records,'status':'COMPLETED'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    run=parser.parse_args().run_dir.resolve();config=json.loads((run/'config.json').read_text())
    for path,expected in config['asset_sha256'].items():
        if digest(path)!=expected:raise ValueError('Frozen asset changed: '+path)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    started=time.time();write_json(run/'status.json',{'status':'RUNNING','pid':os.getpid()})
    try:fit(run,config,torch)
    except Exception as error:
        write_json(run/'status.json',{'status':'FAILED','error':repr(error)});raise
    write_json(run/'status.json',{'status':'COMPLETED','elapsed_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
