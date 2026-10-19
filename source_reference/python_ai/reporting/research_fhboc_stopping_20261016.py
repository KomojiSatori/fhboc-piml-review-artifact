"""Fixed-observation, nested-rank stopping research; frozen PIML is the teacher.

All output is isolated. Complete state fields are evaluated at every rank.
Run --prepare, --infer NAME, --evaluate, or --batch (bounded temporary disk).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from fhboc_allgrid_inference_20261013 import (WindowPlan, reference_coordinates,
    state_features, angular_log_bf, standardized_features)
from fhboc_complex_consistency_20261014 import state_features as complex_features, aggregate_features
from fhboc_amplitude_prior_20261013 import robust_path_prior, model_log_bf

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'results/diagnostics/fhboc_adaptive_stopping_20261016'
SOURCE=BASE/'results/diagnostics/fhboc_allgrid_path_learning_20261013'
MODEL=BASE/'results/diagnostics/fhboc_core_ablation_20261015/frozen_model.json'
GROUPS=[(s,c,g) for s,gg in [('pilot',[1,2]),('validation',[1]),('fresh',[1])]
        for c in (1,2) for g in gg]
RANKS=[32,64,96,3264]
FEATURES=['rank_fraction','score','log_volume','window_mass','competitor_mass',
 'window_gap','entropy','rate_std','adjacent_agreement','minimum_loo_agreement',
 'missing_energy','fd_action_change','tau_action_change','score_change','mass_change']

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def name_of(s,c,g):return f'{s}_c{c}_g{g}'

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    inputs=[MODEL,Path(__file__),BASE/'matlab_code/pipeline/export_fhboc_stopping_traces_20261016.m',
        BASE/'python_ai/reporting/fhboc_allgrid_inference_20261013.py',
        BASE/'python_ai/reporting/fhboc_complex_consistency_20261014.py',
        BASE/'python_ai/reporting/fhboc_amplitude_prior_20261013.py']
    hashes={str(p.relative_to(BASE)):sha(p) for p in inputs}
    path=ROOT/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        assert previous['frozen_inputs_sha256']==hashes,'Research source changed after protocol freeze'
        return previous
    records=[];seed_groups={}
    for s,c,g in GROUPS:
        folder=SOURCE/s/f'condition_{c}_geometry_{g}'
        m=json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        for row in m:
            seed=int(row['noise_seed']);key=name_of(s,c,g)
            seed_groups.setdefault(seed,set()).add(key)
        records.append(dict(name=name_of(s,c,g),observations=len(m),
            signal=sum(bool(r['signal']) for r in m),
            source=str((folder/'allgrid.mat').relative_to(BASE)),
            source_sha256=sha(folder/'allgrid.mat'),metadata_sha256=sha(folder/'metadata.json')))
    collision={str(k):sorted(v) for k,v in seed_groups.items() if len(v)>1}
    assert not collision,collision
    protocol=dict(version=1,frames=6,reference_time_s=.0025,ranks=RANKS,
        regions=8,training_templates_per_region=128,groups=records,
        observations=sum(r['observations'] for r in records),new_observations=0,
        frozen_inputs_sha256=hashes,features=FEATURES,
        source_seed_group_collisions=collision,
        objective='Reduce fixed-observation computation while preserving detection and correct handoff',
        controller='Depth-3 risk tree, grouped out-of-fold; complete teacher available at last stage',
        calibration='Per-geometry stage H0 maxima; complete adaptive policy recalibrated on calibration H0',
        split='Leave one whole code/hopping/aiding geometry out; paired strengths and stages remain together',
        scope='Computational development on existing detector-development records; teacher coefficients fixed',
        cost='Incremental projection MAC model plus measured full-field inference/control time; cold setup separate')
    dump(path,protocol);return protocol

def path_scores(b,q,E,D):
    dots=np.asarray(b,complex)*np.sqrt(q*(E/D)[None,:])
    result=[]
    for groups in [[np.array([k]) for k in range(6)],
                   [np.arange(k,k+2) for k in (0,2,4)],[np.arange(6)]]:
        score=np.zeros(len(q))
        for ix in groups:
            c=abs(dots[:,ix].sum(axis=1))**2/(q[:,ix].sum(axis=1)*E[ix].sum())
            score-=D[ix].sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps))
        result.append(score)
    return np.stack(result,axis=-1)

def infer_field(d,b,ps,model,plan,fd,tau,retention,o,stage,previous):
    start=time.perf_counter();q=d['template_energy'];D=np.asarray(d['dimensions'],float)
    E=d['observation_energy'][:,o];H=len(fd);totalD=D.sum()
    packet={k:d[k] for k in ('cfg','axes','aiding','template_energy','states')}
    packet.update(projected=b[:,:,None],observation_energy=E[:,None],dimensions=D,
                  path_scores=ps[:,None,:])
    g=ps[:,2];raw=state_features(packet,angular_log_bf(g[:,None],totalD))[0]
    x=standardized_features(raw,model)
    prior=robust_path_prior(d['log_aided'],d['log_uniform'],.1)
    base=prior+model_log_bf(g,totalD,model['amplitude_model'])
    ld=base+x@np.asarray(model['beta_detection']);ll=base+x@np.asarray(model['beta_localization'])
    shape=tuple(len(np.atleast_1d(d['axes'][k])) for k in ('fd_hz','tau_samples','rate_hz_s'))
    cf=complex_features(b,q,E,D,d['states'],shape);phi=aggregate_features(cf,ld)
    score=float(logsumexp(ld)+np.clip((phi-model['complex_feature_mean'])/model['complex_feature_scale'],-4,4)@np.asarray(model['complex_feature_beta']))
    weights=np.exp(ll-logsumexp(ll));mass=plan.mass(weights);idx=int(np.argmax(mass))
    infer_seconds=time.perf_counter()-start;start=time.perf_counter()
    disjoint=(abs(fd-fd[idx])>500)|(abs(tau-tau[idx])>2)
    other=float(mass[disjoint].max()) if disjoint.any() else 0.
    rate=np.asarray(d['states'])[:,2];rm=float(weights@rate)
    rs=float(np.sqrt(weights@(rate-rm)**2)/max(np.ptp(rate),1))
    entropy=float(-np.sum(weights*np.log(np.maximum(weights,1e-300)))/np.log(H))
    missing=float(weights@np.maximum(1-retention.mean(axis=1),0)) if retention is not None else 0.
    fdelta=0. if previous is None else abs(float(fd[idx])-previous['fd_hat_hz'])/250
    tdelta=0. if previous is None else abs(float(tau[idx])-previous['tau_hat_samples'])
    sdelta=0. if previous is None else score-previous['score']
    mdelta=0. if previous is None else float(mass[idx])-previous['window_mass']
    features=[RANKS[stage]/3264,score,float(logsumexp(ld)-ld.max()),float(mass[idx]),other,
        float(mass[idx])-other,entropy,rs,float(phi[2]),float(phi[4]),missing,fdelta,tdelta,sdelta,mdelta]
    truth=np.asarray(d['truth_reference']);df=float(fd[idx]-truth[0]);dt=float(tau[idx]-truth[1])
    return dict(stage=stage,rank=RANKS[stage],score=score,fd_hat_hz=float(fd[idx]),tau_hat_samples=float(tau[idx]),
        fd_error_hz=df,tau_error_samples=dt,localized=bool(abs(df)<=250 and abs(dt)<=1),
        window_mass=float(mass[idx]),features=features,inference_seconds=infer_seconds,
        controller_feature_seconds=time.perf_counter()-start)

def infer(name,cleanup=False):
    folder=ROOT/name;done=folder/'inference_complete.json'
    if done.exists():return
    d=loadmat(folder/'geometry.mat',simplify_cells=True);info=d['information'];s=info['run'];c=int(info['condition']);gi=int(info['geometry'])
    source=SOURCE/s/f'condition_{c}_geometry_{gi}'
    prior=np.load(source/'geometry.npz');d.update(log_aided=prior['log_aided'],log_uniform=prior['log_uniform'])
    model=json.loads(MODEL.read_text(encoding='utf8'));fd,tau=reference_coordinates(d['states'],d['cfg'],.0025);plan=WindowPlan(fd,tau)
    meta=list(d['metadata']);O=len(meta);rows=[];prev=[None]*O;maxgap=0.
    for stage,rank in enumerate(RANKS):
        if all((folder/f'event_s{stage}_{o+1:03d}.json').exists() for o in range(O)):
            for o in range(O):
                r=json.loads((folder/f'event_s{stage}_{o+1:03d}.json').read_text(encoding='utf8'))
                rows.append(r);prev[o]=r
            continue
        if stage<3:
            b=loadmat(folder/f'rank_{rank}.mat',variable_names=['projected'])['projected'];teacher=None
        else:
            teacher=loadmat(source/'allgrid.mat',variable_names=['projected','path_scores']);b=teacher['projected']
        for o,m in enumerate(meta):
            output=folder/f'event_s{stage}_{o+1:03d}.json'
            if output.exists():
                r=json.loads(output.read_text(encoding='utf8'));rows.append(r);prev[o]=r;continue
            ps=path_scores(b[:,:,o],d['template_energy'],d['observation_energy'][:,o],np.asarray(d['dimensions'],float))
            if teacher is not None:
                gap=float(np.max(abs(ps-teacher['path_scores'][:,o,:])));maxgap=max(maxgap,gap)
                assert gap<2e-5,('Teacher score reconstruction',gap)
                ps=teacher['path_scores'][:,o,:]
            retention=d['retention'][:,:,stage] if stage<3 else None
            r=infer_field(d,b[:,:,o],ps,model,plan,fd,tau,retention,o,stage,prev[o])
            r.update(group=name,condition=c,geometry=gi,run=s,observation=o+1,role=m['role'],
                signal=bool(m['signal']),cn0_db_hz=float(m['cn0_db_hz']) if m['signal'] else None,
                noise_seed=int(m['noise_seed']))
            dump(output,r);rows.append(r);prev[o]=r
        print('STOPTRACE inferred',name,'rank',rank,'records',O,flush=True)
        del b,teacher
        if cleanup and stage<3:(folder/f'rank_{rank}.mat').unlink()
    q=np.asarray(d['template_energy']);ret=np.asarray(d['retention']);H=len(q);D=np.asarray(d['dimensions'],float)
    summary=dict(complete=True,observations=O,teacher_reconstructed_path_score_max_gap=maxgap,
        retention=[dict(rank=RANKS[j],minimum=float(ret[:,:,j].min()),mean=float(ret[:,:,j].mean()),
            fifth_percentile=float(np.percentile(ret[:,:,j],5))) for j in range(3)],
        full_projection_complex_macs=float(H*D.sum()),
        compressed_projection_complex_macs=[float(6*H*r+8*D.sum()*r) for r in RANKS[:3]],
        cold_timing_seconds=np.asarray(d['timing']).sum(axis=0).tolist(),
        timing_columns=info['timing_columns'].tolist() if hasattr(info['timing_columns'],'tolist') else info['timing_columns'],
        raw_replay_seconds=float(info['replay_seconds']),
        checks=np.asarray(d['checks']).tolist(),train_state_count=8*128,
        singular_ratio_min=float(np.min(d['singular_ratio'])))
    dump(folder/'events.json',rows);dump(done,summary)

def batch():
    prepare()
    matlab=r'D:\MATLAB_YX\MATLAB\R2025b\bin\matlab.exe'
    for s,c,g in GROUPS:
        name=name_of(s,c,g);folder=ROOT/name
        if (folder/'inference_complete.json').exists():continue
        if not (folder/'export_complete.json').exists():
            command=f"addpath(genpath('matlab_code')); export_fhboc_stopping_traces_20261016('{s}',{c},{g})"
            subprocess.run([matlab,'-batch',command],cwd=BASE,check=True)
        infer(name,cleanup=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--infer');p.add_argument('--batch',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:prepare()
        if a.infer:infer(a.infer,cleanup=True)
        if a.batch:batch()
        if a.evaluate:
            from evaluate_fhboc_stopping_20261016 import evaluate
            evaluate()
