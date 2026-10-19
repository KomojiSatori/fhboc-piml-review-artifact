"""Cheap physical checkpoints and one-readout, cost-sensitive stopping.

Training uses the earlier development traces. Holdout coefficients and policy
are frozen before accessing holdout outcomes. No candidate state is pruned.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from research_fhboc_stopping_20261016 import BASE,MODEL,GROUPS,name_of,dump,sha
from evaluate_fhboc_stopping_20261016 import load as old_load

ROOT=BASE/'results/diagnostics/fhboc_cheap_field_stopping_20261016'
OLD=BASE/'results/diagnostics/fhboc_adaptive_stopping_20261016'
NAMES=['gmax16','gmax32','evidence16','evidence32','gmean32','g90_32','g99_32',
 'entropy16','entropy32','evidence_change','gdelta_mean','gdelta_max','gdelta_weighted',
 'posterior_TV','peak_fd_change','peak_tau_change','coarse_window_mass',
 'adjacent_phase','coherence','missing16','missing32','missing32_max',
 'latent16_mean','latent16_max','latent16_std','latent32_mean','latent32_max','latent32_std']

PROBE_NAMES=NAMES.copy()
from fhboc_sparse_checkpoint_20261016 import FieldCheckpoint,GLOBAL_NAMES
NAMES=NAMES+GLOBAL_NAMES

def features(b,q,E,D,states,logw,retention,latent):
    """512 fixed probes; no full-grid PIML score or teacher feature is used."""
    D=np.asarray(D,float);sigma=E.sum()/D.sum();scale=np.sqrt(q*(E/D)[None,:])
    dots=np.asarray(b,complex)*scale[:,:,None];total=dots.sum(axis=1)
    c=abs(total)**2/(q.sum(axis=1)[:,None]*E.sum())
    g=-D.sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps))
    bf=-np.log(17)-D.sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps)*16/17)
    logs=bf+logw[:,None];evidence=logsumexp(logs,axis=0);p=np.exp(logs-evidence)
    entropy=-(p*np.log(np.maximum(p,1e-300))).sum(axis=0)/np.log(len(q))
    delta=g[:,1]-g[:,0];best=np.argmax(logs,axis=0)
    fd=states[:,0]+.0025*states[:,2];tau=states[:,1] # drift is common at this checkpoint's coarse scale
    support=(abs(fd-fd[best[1]])<=250)&(abs(tau-tau[best[1]])<=1)
    a=dots[:,:,1]/np.sqrt(q*sigma)
    adjacent=np.real(np.sum(a[:,:-1]*a[:,1:].conj(),axis=1))/np.maximum(
        np.sqrt(np.sum(abs(a[:,:-1])**2,axis=1)*np.sum(abs(a[:,1:])**2,axis=1)),1e-30)
    coherence=abs(total[:,1])**2/np.maximum(6*np.sum(abs(dots[:,:,1])**2,axis=1),1e-30)
    missing=np.maximum(1-np.asarray(retention).mean(axis=1),0)
    values=[g[:,0].max(),g[:,1].max(),*evidence,float(p[:,1]@g[:,1]),*np.quantile(g[:,1],[.9,.99]),
        *entropy,evidence[1]-evidence[0],np.mean(abs(delta)),np.max(abs(delta)),p[:,1]@abs(delta),
        .5*np.sum(abs(p[:,1]-p[:,0])),abs(fd[best[1]]-fd[best[0]])/250,
        abs(tau[best[1]]-tau[best[0]]),p[support,1].sum(),p[:,1]@adjacent,p[:,1]@coherence,
        p[:,0]@missing[:,0],p[:,1]@missing[:,1],missing[:,1].max()]
    for j,r in enumerate((16,32)):
        u=np.asarray(latent)[:,:,j]/r
        values.extend([u.mean(),u.max(),u.std()])
    result=np.asarray(values,float);assert len(result)==len(PROBE_NAMES) and np.isfinite(result).all()
    return result

def prepare_features(name):
    folder=ROOT/name;target=folder/'cheap_features.npz'
    if target.exists():return
    d=loadmat(folder/'geometry.mat',simplify_cells=True);p=loadmat(folder/'probes.mat',simplify_cells=True)
    ids=np.asarray(p['probe_ids'],int)-1;q=d['template_energy'][ids];states=d['states'][ids]
    # Importance-weighted physical summary; used as a feature, not a detector statistic.
    prior=.9*np.exp(d['log_aided'][ids])+.1*np.exp(d['log_uniform'][ids])
    logw=np.log(prior/np.asarray(p['weights'])[ids]);logw-=logsumexp(logw)
    field=FieldCheckpoint(d,json.loads(MODEL.read_text(encoding='utf8')));full_b=loadmat(folder/'rank_32.mat',variable_names=['projected'])['projected']
    rows=[];seconds=[]
    for o in range(len(d['metadata'])):
        started=time.perf_counter()
        rows.append(features(p['probe_projected'][:,:,o,:],q,d['observation_energy'][:,o],d['dimensions'],
            states,logw,p['probe_retention'],p['latent_energy'][o]))
        rows[-1]=np.r_[rows[-1],field(full_b[:,:,o],d['observation_energy'][:,o])]
        seconds.append(time.perf_counter()-started)
    np.savez(target,x=rows,seconds=seconds,probe_ids=ids,logw=logw)
    del full_b
    (folder/'rank_32.mat').unlink()
    dump(folder/'feature_status.json',dict(complete=True,observations=len(rows),feature_names=NAMES,
        mean_seconds=float(np.mean(seconds)),maximum_seconds=max(seconds),inputs_sha256={
            n:sha(folder/n) for n in ['probes.mat','geometry.mat']},source_sha256=sha(Path(__file__))))

def classifier(kind):
    if kind=='tree':return DecisionTreeClassifier(max_depth=3,min_samples_leaf=15,class_weight='balanced',random_state=20261016)
    return make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=500))

def positive(model,x):
    p=model.predict_proba(x);classes=list(model.classes_)
    return p[:,classes.index(1)] if 1 in classes else np.zeros(len(x))

def describe(d,choice,label):
    """choice: -1=early H0, 0=rank32 readout, 1=rank64 readout, 3=teacher."""
    n=len(choice);ii=np.arange(n);real=np.maximum(choice,0)
    margin=d['margin'][ii,real].copy();margin[choice<0]=-1e6
    # A nonpositive adaptive gate preserves the calibrated rank-gate origin.
    for group in d['groups']:
        group_ix=d['gg']==group;gate=max(0.,float(margin[group_ix&d['cal']].max()))
        margin[group_ix]-=gate
    det=margin>0;localized=np.array([tr[int(max(c,0))]['localized'] for tr,c in zip(d['traces'],choice)])
    handoff=det&localized&d['signal'];base_det=d['detected'][:,3];base_hit=d['hit'][:,3]
    ref64=d['detected'][:,1];hit64=d['hit'][:,1];test=~d['cal'];bg=test&~d['signal'];sig=d['signal']
    extra=d.get('cheap_seconds',np.zeros(n));cost=[];mac=[]
    for i,c in enumerate(choice):
        packet=d['cost'][d['gg'][i]];H=packet['full_projection_complex_macs']/(6*3264)
        r=32 if c<0 else (3264 if c==3 else [32,64][c])
        # Initial 32 latent dimensions + 16/32 probe projections; reuse latents if continuing.
        cmac=6*8*3264*max(32,r if r<1000 else 32)+6*H*r+6*512*(16+32)
        if label.startswith('fixed-') or label=='full':cmac=packet['full_projection_complex_macs'] if c==3 else packet['compressed_projection_complex_macs'][c]
        mac.append(cmac/packet['full_projection_complex_macs'])
        field=0. if c<0 else d['traces'][i][c]['inference_seconds']
        cost.append(field+(0 if label.startswith('fixed-') or label=='full' else extra[i]))
    result=dict(method=label,pd=float(det[sig].mean()),ph=float(handoff[sig].mean()),pfa=float(det[bg].mean()),
        n_H1=int(sig.sum()),n_H0=int(bg.sum()),detected=int(det[sig].sum()),handoff=int(handoff[sig].sum()),false_alarms=int(det[bg].sum()),
        detection_losses=int(np.sum(sig&base_det&~det)),handoff_losses=int(np.sum(base_hit&~handoff)),
        loss_vs64_detection=int(np.sum(sig&ref64&~det)),loss_vs64_handoff=int(np.sum(hit64&~handoff)),
        additional_false_alarms=int(np.sum(bg&~d['detected'][:,3]&det)),
        projection_mac_ratio=float(np.mean(np.array(mac)[test])),field_and_feature_seconds=float(np.mean(np.array(cost)[test])),
        reject_count=int(np.sum((choice<0)&test)),rank32_count=int(np.sum((choice==0)&test)),rank64_count=int(np.sum((choice==1)&test)))
    result['cells']=[]
    for ci in (1,2):
        for cn in (32,36):
            cell=sig&np.array([r['condition']==ci and r['cn0_db_hz']==cn for r in d['meta']])
            result['cells'].append(dict(condition=ci,cn0=cn,n=int(cell.sum()),pd=float(det[cell].mean()),ph=float(handoff[cell].mean()),
                detection_losses=int(np.sum(cell&base_det&~det)),handoff_losses=int(np.sum(cell&base_hit&~handoff))))
    return result,dict(choice=choice.tolist(),detected=det.tolist(),handoff=handoff.tolist())

def development():
    d=old_load();assert len(d['groups'])==8
    X=[];seconds=[]
    for row in d['meta']:
        folder=ROOT/row['group'];prepare_features(row['group'])
        z=np.load(folder/'cheap_features.npz');o=row['observation']-1;X.append(z['x'][o]);seconds.append(z['seconds'][o])
    X=np.array(X);d['cheap_seconds']=np.array(seconds);noncal=~d['cal'];n=len(X)
    fd=np.array([[r['fd_hat_hz'] for r in tr] for tr in d['traces']]);tau=np.array([[r['tau_hat_samples'] for r in tr] for tr in d['traces']])
    drift=(abs(fd[:,0]-fd[:,3])>62.5)|(abs(tau[:,0]-tau[:,3])>.25)
    low=d['unsafe'][:,0]|(abs(d['margin'][:,0]-d['margin'][:,3])>.5*abs(d['margin'][:,0]))|(drift&(d['detected'][:,0]|d['detected'][:,3]))
    reject=d['signal']&(d['detected'][:,3]|d['detected'][:,1])
    pred={};folds=[]
    for target,y in [('continue',reject),('refine',low)]:
        for kind in ['tree','linear']:
            pp=np.zeros(n)
            for held in d['groups']:
                train=(d['gg']!=held)&noncal;test=d['gg']==held
                model=classifier(kind);model.fit(X[train],y[train]);pp[test]=positive(model,X[test])
                folds.append(dict(target=target,kind=kind,held_out=held,train_groups=sorted(set(d['gg'][train])),train_records=int(train.sum())))
            pred[target,kind]=pp
    out=ROOT/'development';out.mkdir(exist_ok=True);methods=[];events={};configs={}
    def add(choice,name,config):
        r,e=describe(d,choice,name);methods.append(r);events[name]=e;configs[name]=config
    for c,label in [(3,'full'),(0,'fixed-32'),(1,'fixed-64')]:add(np.full(n,c),label,{})
    # Thresholds are development selections; all candidates are retained for audit.
    for kind in ['tree','linear']:
        pc=pred['continue',kind];pr=pred['refine',kind]
        for rth in [0.,.05,.1,.2,.35,.5,.75,1.]:
            choice=np.where(pr<=rth,0,1)
            add(choice,f'precision-{kind}-{rth:g}',dict(mode='precision',kind=kind,refine_threshold=rth,reject_threshold=-1.))
        for th in [-1.,0.,.01,.03,.05,.1,.15,.2,.3,.4,.5,.7]:
            for rth in [-1.,.1,.35,.5,.75]:
                choice=np.where(pc<=th,-1,np.where(pr<=rth,0,1))
                add(choice,f'stop-{kind}-{th:g}-{rth:g}',dict(mode='joint',kind=kind,refine_threshold=rth,reject_threshold=th))
    # Same cheap physics, scalar manual stopping controls.
    for th in np.quantile(X[:,37],np.linspace(0,.7,20)):
        add(np.where(X[:,37]<=th,-1,1),f'manual-evidence-{th:.6g}',dict(mode='manual',reject_threshold=float(th)))
    for th in np.quantile(X[:,12],np.linspace(0,1,20)):
        add(np.where(X[:,12]<=th,0,1),f'manual-refine-{th:.6g}',dict(mode='manual-refine',threshold=float(th)))
    baseline=methods[0];eligible=[]
    for r in methods:
        if not r['method'].startswith(('stop-','precision-')):continue
        if max(r['detection_losses'],r['handoff_losses'],r['loss_vs64_detection'],r['loss_vs64_handoff'])>1:continue
        if r['false_alarms']>baseline['false_alarms']:continue
        # Measured field/feature time and a fixed, pre-existing dense-kernel cost model.
        # Physical projection dominates 0.04 * fraction/fixed64 + field readout.
        projected=.036518*r['projection_mac_ratio']/.02458337986749716
        r['selection_cost_seconds']=r['field_and_feature_seconds']+projected
        eligible.append(r)
    assert eligible,'No controller meeting one-record loss guard'
    chosen=min(eligible,key=lambda r:r['selection_cost_seconds']);config=configs[chosen['method']]
    kind=config['kind'];fitted={}
    for target,y in [('continue',reject),('refine',low)]:
        model=classifier(kind);model.fit(X[noncal],y[noncal]);fitted[target]=model
    bundle=dict(config=config,models=fitted,feature_names=NAMES)
    joblib.dump(bundle,out/'frozen_controller.joblib')
    dump(out/'controller_config.json',dict(**config,feature_names=NAMES,selection=chosen,
        all_held_out_outcomes_used=0,controller_sha256=sha(out/'frozen_controller.joblib'),
        source_sha256=sha(Path(__file__)),exporter_sha256=sha(BASE/'matlab_code/pipeline/export_fhboc_cheap_field_20261016.m'),
        holdout_plan=dict(groups=4,observations=1000,H1=400,calibration_H0=200,test_H0=400,
            tolerance_overall_percentage_points=2,tolerance_cell_percentage_points=3,additional_Pfa_percentage_points=1)))
    dump(out/'grouped_folds.json',folds);dump(out/'method_results.json',methods);dump(out/'policy_events.json',events)
    np.savez(out/'features.npz',X=X,seconds=seconds,continue_labels=reject,refine_labels=low)
    print('CHEAP STOP DEVELOPMENT',json.dumps(chosen),flush=True)

def prepare_protocol():
    ROOT.mkdir(exist_ok=True,parents=True);path=ROOT/'protocol.json'
    value=dict(frames=6,probe_states=512,probe_ranks=[16,32],final_ranks=[32,64],
        full_domain_final_readout=True,feature_names=NAMES,development_records=600,
        independent_records=1000,model_sha256=sha(MODEL),exporter_sha256=sha(BASE/'matlab_code/pipeline/export_fhboc_cheap_field_20261016.m'),
        feature_source_sha256=sha(Path(__file__)),allowed_early_actions=['H0','rank32_readout','rank64_readout'])
    if path.exists():assert json.loads(path.read_text())==value
    else:dump(path,value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--features');p.add_argument('--train',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:prepare_protocol()
        if a.features:prepare_features(a.features)
        if a.train:development()
