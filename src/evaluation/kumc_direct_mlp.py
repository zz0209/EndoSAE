"""Directly trained residual MLP control on frozen EndoFM token features."""
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
    from src.evaluation.pairwise_support_loss import pairwise_support_loss
    base_run=Path(config['base_run'])
    bh=arrays(base_run/'models/raw/head.npz',torch,'cuda')
    with torch.no_grad():base_logits=torch.nn.functional.linear((x-bh['feature_mean'])/bh['feature_scale'],bh['weight'],bh['bias']).squeeze(-1)
    original=np.load(base_run/'models/raw/patch_predictions.npy',allow_pickle=False)
    error=float(np.max(abs(torch.sigmoid(base_logits).cpu().numpy().reshape(masks.shape)-original)));assert error<=1e-6
    np.save(run/'base_logits.npy',base_logits.cpu().numpy().reshape(masks.shape),allow_pickle=False)
    np.save(run/'base_predictions.npy',torch.sigmoid(base_logits).cpu().numpy().reshape(masks.shape),allow_pickle=False)
    mean,scale=feature_stats(x,fit_indices,lambda b:b,torch)
    np.savez(run/'input_statistics.npz',feature_mean=mean.cpu().numpy(),feature_scale=scale.cpu().numpy())
    saved=[]
    for seed in config['seeds']:
        torch.manual_seed(seed)
        hidden=torch.nn.Linear(768,config['hidden_width'],device='cuda')
        head=torch.nn.Linear(config['hidden_width'],1,device='cuda')
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        np.savez(run/('initial_'+str(seed)+'.npz'),hidden_weight=hidden.weight.detach().cpu().numpy(),hidden_bias=hidden.bias.detach().cpu().numpy(),weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy())
        optimizer=torch.optim.AdamW([{'params':hidden.parameters(),'lr':config['hidden_learning_rate'],'weight_decay':0.},
            {'params':head.parameters(),'lr':config['head_learning_rate'],'weight_decay':config['head_weight_decay']}])
        generator=torch.Generator(device='cuda').manual_seed(seed);history=[]
        for step in range(config['steps']):
            ids,labels=sampler.sample(config['batch_size'],generator)
            features=(x[ids]-mean)/scale
            correction=head(torch.relu(hidden(features))).squeeze(-1)
            logits=base_logits[ids]+correction
            ranking=pairwise_support_loss(logits,labels,ids);penalty=correction.square().mean()
            loss=ranking+config['residual_l2']*penalty
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.);torch.nn.utils.clip_grad_norm_(hidden.parameters(),1.)
            optimizer.step()
            if (step+1)%50==0:history.append({'step':step+1,'total':float(loss.detach()),'pairwise_logistic':float(ranking.detach()),'residual_squared':float(penalty.detach())})
            if step+1 in config['snapshot_steps']:
                method='mlp'+str(config['hidden_width'])+'_seed'+str(seed)+'_step'+str(step+1)
                dest=output/method;dest.mkdir()
                np.savez(dest/'model.npz',hidden_weight=hidden.weight.detach().cpu().numpy(),hidden_bias=hidden.bias.detach().cpu().numpy(),weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy())
                write_json(dest/'training.json',{'seed':seed,'steps':step+1,'history':history.copy(),'statistics_fit_token_count':len(fit_indices),
                    'initial_residual':'identically_zero','base_replay_max_difference':error,'before_development_evaluation':True,
                    'trainable_parameters':sum(p.numel() for m in [hidden,head] for p in m.parameters()),
                    'assets':{'model.npz':digest(dest/'model.npz')}})
                saved.append(method);print('FROZEN '+method,flush=True)
    assert saved==config['methods']
    frozen={m:json.loads((output/m/'training.json').read_text())['assets'] for m in saved}
    write_json(run/'all_models_frozen.json',{'methods':saved,'model_hashes':frozen,'development_predictions_evaluated':False,
        'fit_scope_sha256':digest(run/'fit_scope.json'),'input_statistics_sha256':digest(run/'input_statistics.npz'),
        'base_logits_sha256':digest(run/'base_logits.npy'),'initial_models_sha256':{str(s):digest(run/('initial_'+str(s)+'.npz')) for s in config['seeds']}})
    from src.evaluation.kumc_localization import localization_metrics
    records=[]
    for method in ['base']+saved:
        if method=='base':scores=np.load(run/'base_predictions.npy',allow_pickle=False)
        else:
            dest=output/method;h=arrays(dest/'model.npz',torch,'cuda');parts=[]
            with torch.no_grad():
                for batch in x.split(4096):
                    hidden_values=torch.relu(torch.nn.functional.linear((batch-mean)/scale,h['hidden_weight'],h['hidden_bias']))
                    parts.append(torch.nn.functional.linear(hidden_values,h['weight'],h['bias']).squeeze(-1))
                residual=torch.cat(parts);scores=torch.sigmoid(base_logits+residual).cpu().numpy().reshape(masks.shape)
            np.save(dest/'residual_logits.npy',residual.cpu().numpy().reshape(masks.shape),allow_pickle=False)
            np.save(dest/'patch_predictions.npy',scores,allow_pickle=False)
        records.append({'method':method,'train':localization_metrics(scores[train],masks[train],[r for r,u in zip(rows,train) if u]),
            'development':localization_metrics(scores[~train],masks[~train],[r for r,u in zip(rows,train) if not u])})
        print(json.dumps({'method':method,'development_ap':float(np.mean([v['frame_mean_patch_ap'] for v in records[-1]['development'].values()]))}),flush=True)
    for method in saved:
        for p,hsh in frozen[method].items():assert digest(output/method/p)==hsh
    for p,hsh in config['asset_sha256'].items():assert digest(p)==hsh
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
