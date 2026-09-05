"""Frozen residual models on prospectively selected, unused source families."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from src.evaluation.realcolon_task import digest, write_json


def execute(run, c, torch):
    from src.evaluation.realcolon_fixed_confirmation import arrays, tokens_cuda
    from src.evaluation.kumc_localization import localization_metrics
    from src.sae.baselines import TopKAutoencoder
    rows=[json.loads(s) for s in (run/'clip_manifest.jsonl').read_text().splitlines()]
    families={r['video_id'].split('/')[-1] for r in rows}
    assert families==set(c['confirmation_families'])
    assert families.isdisjoint(c['previously_exposed_families'])
    prepared=json.loads((run/'prepared.json').read_text());encoded=json.loads((run/'encoded.json').read_text())
    assert prepared['manifest_sha256']==encoded['manifest_sha256']==digest(run/'clip_manifest.jsonl')
    assert encoded['prepared_sha256']==digest(run/'prepared.json')
    assert encoded['state_exchange_sha256']==c['state_exchange_sha256']
    cache=Path(c['cache_dir'])
    assert digest(cache/'tokens.npy')==encoded['tokens_sha256']
    assert digest(cache/'masks.npy')==prepared['assets']['masks.npy']
    receipts=json.loads((run/'frame_receipts.json').read_text())
    used=set(c['previous_image_sha256'])
    assert not used.intersection(r['image_sha256'] for r in receipts), 'Confirmation duplicates exposed pixels'
    new_hashes=[r['image_sha256'] for r in receipts]
    assert len(set(new_hashes))==len(new_hashes), 'Duplicated confirmation pixels need grouping review'
    masks=np.load(cache/'masks.npy',allow_pickle=False)
    raw=tokens_cuda(np.load(cache/'tokens.npy',mmap_mode='r',allow_pickle=False),list(range(len(rows))),torch)
    norm=arrays(Path(c['normalization']),torch,'cuda');x=(raw-norm['mean'])/norm['rms'];del raw
    base=arrays(Path(c['base_head']),torch,'cuda')
    with torch.no_grad():logits=torch.nn.functional.linear((x-base['feature_mean'])/base['feature_scale'],base['weight'],base['bias']).squeeze(-1)
    st=arrays(Path(c['sparse_dictionary']),torch,'cuda');sparse=TopKAutoencoder(768,1536,128).cuda();sparse.load_state_dict(st,strict=True)
    sparse.eval()
    for p in sparse.parameters():p.requires_grad_(False)
    linear=arrays(Path(c['linear_representation']),torch,'cuda');random=arrays(Path(c['random_projection']),torch,'cuda')
    transforms={'raw':lambda b:b,'linear192':lambda b:torch.nn.functional.linear(b,linear['encoder.weight']),
                'topk128':sparse.encode_inference,'random_relu1536':lambda b:torch.relu(torch.nn.functional.linear(b,random['weight']))}
    out=run/'predictions';out.mkdir(exist_ok=False)
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'prediction_scope.json',{'model_fitting':False,'normalization_fitting':False,'confirmation_families':sorted(families),
        'n_unique_images':len(set(new_hashes)),'model_assets_frozen_before_pixels':c['created_before_new_pixels_and_outputs']})
    records=[]
    for spec in c['models']:
        m=spec['name']
        with torch.no_grad():
            if m=='base':residual=torch.zeros_like(logits)
            else:
                h=arrays(Path(spec['head']),torch,'cuda');parts=[]
                for b in x.split(4096):
                    features=transforms[spec['features']](b)
                    parts.append(torch.nn.functional.linear((features-h['feature_mean'])/h['feature_scale'],h['weight'],h['bias']).squeeze(-1))
                residual=torch.cat(parts)
            scores=torch.sigmoid(logits+residual).cpu().numpy().reshape(masks.shape)
        dest=out/m;dest.mkdir();np.save(dest/'patch_predictions.npy',scores,allow_pickle=False)
        np.save(dest/'residual_logits.npy',residual.cpu().numpy().reshape(masks.shape),allow_pickle=False)
        metrics=localization_metrics(scores,masks,rows)
        records.append({'method':m,'per_source':metrics})
        print(json.dumps({'method':m,'confirmation_ap':float(np.mean([v['frame_mean_patch_ap'] for v in metrics.values()]))}),flush=True)
    np.save(out/'base_logits.npy',logits.cpu().numpy().reshape(masks.shape),allow_pickle=False)
    for path,h in c['asset_sha256'].items():assert digest(path)==h
    write_json(run/'metrics.json',{'records':records,'model_fitting':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    run=parser.parse_args().run_dir.resolve();c=json.loads((run/'config.json').read_text())
    for p,h in c['asset_sha256'].items():assert digest(p)==h,p
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    started=time.time();write_json(run/'status.json',{'status':'RUNNING','pid':os.getpid()})
    try:execute(run,c,torch)
    except Exception as e:write_json(run/'status.json',{'status':'FAILED','error':repr(e)});raise
    write_json(run/'status.json',{'status':'COMPLETED','elapsed_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
