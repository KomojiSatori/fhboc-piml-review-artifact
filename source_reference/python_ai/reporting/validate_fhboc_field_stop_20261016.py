"""Post-freeze, independent evaluation of cheap full-domain stopping."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.stats import beta
from threadpoolctl import threadpool_limits
import research_fhboc_field_stop_20261016 as research
import research_fhboc_stopping_20261016 as teacher
from research_fhboc_cheap_stop_20261016 import ROOT as DATA,features as probe_features,positive
from fhboc_sparse_checkpoint_20261016 import FieldCheckpoint
from scipy.special import logsumexp

ROOT=research.ROOT;BASE=research.BASE;dump=research.dump;sha=research.sha

def freeze():
    target=ROOT/'validation_freeze.json'
    files=[Path(__file__),Path(research.__file__),BASE/'python_ai/reporting/fhboc_sparse_checkpoint_20261016.py',
        ROOT/'development/frozen_controller.joblib',ROOT/'development/controller_config.json',
        teacher.MODEL,BASE/'matlab_code/pipeline/export_fhboc_cheap_stop_20261016.m']
    hashes={str(p.relative_to(BASE)):sha(p) for p in files}
    if target.exists():
        value=json.loads(target.read_text(encoding='utf8'));assert value['sha256']==hashes,'Validation source/model changed after freeze';return value
    methods=json.loads((ROOT/'development/method_results.json').read_text());base=methods[0]
    eligible=[r for r in methods if r['method'].startswith('manual-evidence-') and
        max(r['detection_losses'],r['handoff_losses'],r['loss_vs64_detection'],r['loss_vs64_handoff'])<=1 and r['false_alarms']<=base['false_alarms']]
    manual=min(eligible,key=lambda r:r['field_and_feature_seconds']+.036518*r['projection_mac_ratio']/.02458337986749716)
    threshold=float(manual['method'].removeprefix('manual-evidence-'))
    # Freeze source identities using seeds/code epochs, before reading test outcomes.
    ledger=[];dev_seeds=set()
    for s,c,g in teacher.GROUPS:
        p=teacher.SOURCE/s/f'condition_{c}_geometry_{g}/metadata.json'
        dev_seeds.update(int(r['noise_seed']) for r in json.loads(p.read_text()))
    for c in (1,2):
        for g in (1,2):
            for folder in ['fhboc_independent_validation_760_20261014','fhboc_6frame_extension_20261014']:
                p=BASE/'results/diagnostics'/folder/f'allgrid/condition_{c}_geometry_{g}/metadata.json'
                m=json.loads(p.read_text());seeds={int(r['noise_seed']) for r in m}
                assert not seeds&dev_seeds
                ledger.append(dict(source=str(p.relative_to(BASE)),sha256=sha(p),observations=len(m),seeds=sorted(seeds)))
    value=dict(sha256=hashes,manual_threshold=threshold,manual_development=manual,
        controller_config=json.loads((ROOT/'development/controller_config.json').read_text()),
        records=1000,H1=400,calibration_H0=200,test_H0=400,source_ledger=ledger,
        source_seed_overlap_with_training=0,holdout_outcomes_used_for_selection=0)
    dump(target,value);return value

def geometry(ci,gi):
    frozen=freeze();folder=DATA/f'holdout_c{ci}_g{gi}';out=folder/'field_evaluation';out.mkdir(exist_ok=True)
    if (out/'completion.json').exists():return
    d=loadmat(folder/'geometry.mat',simplify_cells=True);p=loadmat(folder/'probes.mat',simplify_cells=True)
    model=json.loads(teacher.MODEL.read_text());fd,tau=teacher.reference_coordinates(d['states'],d['cfg'],.0025);plan=teacher.WindowPlan(fd,tau)
    checkpoint=FieldCheckpoint(d,model);ids=np.asarray(p['probe_ids'],int)-1;q=d['template_energy'][ids]
    prior=.9*np.exp(d['log_aided'][ids])+.1*np.exp(d['log_uniform'][ids]);lw=np.log(prior/np.asarray(p['weights'])[ids]);lw-=logsumexp(lw)
    O=len(d['metadata']);name=folder.name;started=time.perf_counter()
    for stage in [0,1,3]:
        if all((out/f's{stage}_{o:03d}.json').exists() for o in range(O)):
            if stage!=0 or (out/'features.npz').exists():continue
        if stage<3:
            b=loadmat(folder/f'rank_{[32,64][stage]}.mat',variable_names=['projected'])['projected'];ps=None
        else:
            f=BASE/'results/diagnostics/fhboc_independent_validation_760_20261014/allgrid'/f'condition_{ci}_geometry_{gi}/allgrid.mat'
            a=loadmat(f,variable_names=['projected','path_scores']);f=BASE/'results/diagnostics/fhboc_6frame_extension_20261014/allgrid'/f'condition_{ci}_geometry_{gi}/allgrid.mat'
            z=loadmat(f,variable_names=['projected','path_scores']);b=np.concatenate([a['projected'],z['projected']],axis=2);ps=np.concatenate([a['path_scores'],z['path_scores']],axis=1);del a,z
        for o,m in enumerate(d['metadata']):
            target=out/f's{stage}_{o:03d}.json'
            if target.exists():continue
            if stage==0:
                clock=time.perf_counter()
                cheap=probe_features(p['probe_projected'][:,:,o,:],q,d['observation_energy'][:,o],d['dimensions'],d['states'][ids],lw,p['probe_retention'],p['latent_energy'][o])
                x=np.r_[cheap,checkpoint(b[:,:,o],d['observation_energy'][:,o])];elapsed=time.perf_counter()-clock
                dump(out/f'x_{o:03d}.json',dict(features=x.tolist(),seconds=elapsed))
            scores=ps[:,o,:] if ps is not None else teacher.path_scores(b[:,:,o],d['template_energy'],d['observation_energy'][:,o],d['dimensions'])
            r=teacher.infer_field(d,b[:,:,o],scores,model,plan,fd,tau,None,o,stage,None)
            r.update(group=name,run='holdout',condition=ci,geometry=gi,observation=o+1,role=m['role'],signal=bool(m['signal']),
                cn0_db_hz=float(m['cn0_db_hz']) if m['signal'] else None,noise_seed=int(m['noise_seed']))
            dump(target,r)
        if stage==0:
            packet=[json.loads((out/f'x_{o:03d}.json').read_text()) for o in range(O)]
            np.savez(out/'features.npz',X=[r['features'] for r in packet],seconds=[r['seconds'] for r in packet])
        del b,ps
        if stage<3:(folder/f'rank_{[32,64][stage]}.mat').unlink()
        print('INDEPENDENT FIELD STOP',name,'stage',stage,'records',O,flush=True)
    dump(out/'completion.json',dict(complete=True,observations=O,seconds=time.perf_counter()-started,
        frozen_sha256=sha(ROOT/'validation_freeze.json')))

def dataset():
    traces=[];meta=[];cost={};X=[];seconds=[]
    for ci in (1,2):
        for gi in (1,2):
            name=f'holdout_c{ci}_g{gi}';folder=DATA/name;out=folder/'field_evaluation'
            assert (out/'completion.json').exists(),name
            d=loadmat(folder/'geometry.mat',simplify_cells=True,variable_names=['states','dimensions','metadata'])
            H=len(d['states']);D=np.sum(d['dimensions'])
            cost[name]=dict(full_projection_complex_macs=float(H*D),compressed_projection_complex_macs=[float(6*H*r+8*D*r) for r in [32,64]])
            f=np.load(out/'features.npz');X.extend(f['X']);seconds.extend(f['seconds'])
            for o in range(len(d['metadata'])):
                tr=[json.loads((out/f's{s}_{o:03d}.json').read_text()) for s in [0,1,1,3]];traces.append(tr);meta.append(tr[0])
    gg=np.array([r['group'] for r in meta]);cal=np.array([r['role']=='calibration_H0' for r in meta]);sig=np.array([r['signal'] for r in meta])
    score=np.array([[r['score'] for r in tr] for tr in traces]);margin=score.copy();groups=sorted(cost)
    for g in groups:
        ix=gg==g;margin[ix]-=score[ix&cal].max(axis=0)
    det=margin>0;hit=det&sig[:,None]&np.array([[r['localized'] for r in tr] for tr in traces])
    return dict(traces=traces,meta=meta,cost=cost,gg=gg,cal=cal,signal=sig,score=score,margin=margin,groups=groups,
        detected=det,hit=hit,cheap_seconds=np.array(seconds)),np.array(X)

def evaluate():
    f=freeze();d,X=dataset();bundle=joblib.load(ROOT/'development/frozen_controller.joblib');config=bundle['config']
    pc=positive(bundle['models']['continue'],X);pr=positive(bundle['models']['refine'],X)
    choices={'full':np.full(len(X),3),'fixed-32':np.zeros(len(X),int),'fixed-64':np.ones(len(X),int),
        'manual-stopping':np.where(X[:,37]<=f['manual_threshold'],-1,1),
        'learned-stopping':np.where(pc<=config['reject_threshold'],-1,np.where(pr<=config['refine_threshold'],0,1))}
    results=[];events={}
    for method,choice in choices.items():
        r,e=research.describe(d,choice,method);results.append(r);events[method]=e
        for field in ['detection_losses','handoff_losses','loss_vs64_detection','loss_vs64_handoff']:
            k=r[field];r[field+'_upper95']=float(beta.ppf(.95,k+1,r['n_H1']-k)) if k<r['n_H1'] else 1.
    out=ROOT/'independent';out.mkdir(exist_ok=True)
    dump(out/'method_results.json',results);dump(out/'events.json',events)
    np.savez(out/'controller_inputs.npz',X=X,pc=pc,pr=pr)
    learned=results[-1];accept=dict(performance=all(learned[k]<=8 for k in ['detection_losses','handoff_losses','loss_vs64_detection','loss_vs64_handoff']),
        no_additional_false_alarm_excess=learned['false_alarms']<=results[0]['false_alarms']+4,
        per_cell=all(max(c['detection_losses'],c['handoff_losses'])<=3 for c in learned['cells']))
    dump(out/'completion_status.json',dict(complete=True,records=len(X),frozen_sha256=sha(ROOT/'validation_freeze.json'),acceptance=accept))
    print('INDEPENDENT STOP RESULTS',json.dumps(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--geometry',nargs=2,type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.freeze:freeze()
        if a.geometry:geometry(*a.geometry)
        if a.evaluate:evaluate()
