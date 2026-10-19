"""Actual six-frame banks, one capture per call, identical optimized PIML heads."""
import argparse
import json
import time
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits,threadpool_info
from research_fhboc_stopping_20261016 import (BASE,MODEL,path_scores,state_features,angular_log_bf,
    standardized_features,complex_features,aggregate_features,WindowPlan,reference_coordinates,dump,sha)
from research_fhboc_cheap_stop_20261016 import ROOT as DATA,features as probe_features
from fhboc_sparse_checkpoint_20261016 import FieldCheckpoint
from fhboc_fast_checkpoint_20261016 import FastFieldCheckpoint
from validate_fhboc_field_stop_20261016 import ROOT,freeze

class OnlineBank:
    def __init__(self,folder):
        self.d=loadmat(folder/'geometry.mat',simplify_cells=True)
        self.bank=loadmat(folder/'online_bank.mat',simplify_cells=True)
        self.probe=loadmat(folder/'probes.mat',simplify_cells=True)
        self.model=json.loads(MODEL.read_text());self.checkpoint=FastFieldCheckpoint(self.d,self.model)
        self.fd,self.tau=reference_coordinates(self.d['states'],self.d['cfg'],.0025);self.plan=WindowPlan(self.fd,self.tau)
        self.q=self.d['template_energy'];self.D=np.asarray(self.d['dimensions'],float)
        self.ids=np.asarray(self.probe['probe_ids'],int)-1;self.groups=[np.asarray(x,int)-1 for x in self.bank['region_ids']]
        self.probe_group=[]
        for ix in self.groups:
            pos=np.flatnonzero(np.isin(self.ids,ix));self.probe_group.append((pos,np.searchsorted(ix,self.ids[pos])))
        # Two rank bands are prepared with the bank, avoiding strided BLAS copies
        # at each capture. Their combined storage equals one 64-rank library.
        self.bands={}
        for k in range(6):
            for g in range(8):
                C=np.asarray(self.bank['coefficients'][k,g])
                self.bands[k,g,0]=np.asfortranarray(C[:32])
                self.bands[k,g,32]=np.asfortranarray(C[32:64])
        prior=.9*np.exp(self.d['log_aided'][self.ids])+.1*np.exp(self.d['log_uniform'][self.ids])
        self.logw=np.log(prior/np.asarray(self.probe['weights'])[self.ids]);self.logw-=logsumexp(self.logw)
        self.prior=np.logaddexp(self.d['log_aided']+np.log(.9),self.d['log_uniform']+np.log(.1))
    def projection(self,o,rank=64,need_probes=False,previous=None):
        H=len(self.q);inner=np.zeros((H,6),complex) if previous is None else previous.copy()
        p16=np.zeros((512,6),complex);latent=np.zeros((6,8,2));start=0 if previous is None else 32
        E=self.d['observation_energy'][:,o];scale=np.sqrt(E/self.D)
        for k in range(6):
            z=np.asarray(self.bank['whitened_observations'][k])[:,o]
            for g,ix in enumerate(self.groups):
                U=np.asarray(self.bank['basis'][k,g]);C=np.asarray(self.bank['coefficients'][k,g])
                h=z.conj()@U[:,start:rank]
                block=self.bands[k,g,start] if rank-start==32 else C
                inner[ix,k]+=h@block
                if need_probes:
                    pos,local=self.probe_group[g];p16[pos,k]=(h[:16]@self.bands[k,g,0][:16,local])/scale[k]
                    for j,r in enumerate((16,32)):latent[k,g,j]=np.sum(abs(h[:r])**2)/(E[k]/self.D[k])
        b=(inner/scale).astype(np.complex64)
        return b,inner,p16,latent
    def cheap(self,o,b,p16,latent,probe=True):
        E=self.d['observation_energy'][:,o]
        glob=self.checkpoint(b,E)
        if not probe:return glob
        bb=np.stack([p16,b[self.ids]],axis=-1)
        x=probe_features(bb,self.q[self.ids],E,self.D,self.d['states'][self.ids],self.logw,self.probe['probe_retention'],latent)
        return np.r_[x,glob]
    def readout(self,o,b,gate=None):
        """Gaussian-mixture LUT shared by all comparators; exact complex heads."""
        E=self.d['observation_energy'][:,o];ps=path_scores(b,self.q,E,self.D);g=ps[:,2]
        packet={k:self.d[k] for k in ('cfg','axes','aiding','template_energy','states')}
        packet.update(projected=b[:,:,None],observation_energy=E[:,None],dimensions=self.D,path_scores=ps[:,None,:])
        x=standardized_features(state_features(packet,angular_log_bf(g[:,None],self.D.sum()))[0],self.model)
        bf=self.checkpoint.bf(g)
        base=self.prior+bf;ld=base+x@np.asarray(self.model['beta_detection'])
        cf=complex_features(b,self.q,E,self.D,self.d['states'],self.checkpoint.shape);phi=aggregate_features(cf,ld)
        score=float(logsumexp(ld)+np.clip((phi-self.model['complex_feature_mean'])/self.model['complex_feature_scale'],-4,4)@np.asarray(self.model['complex_feature_beta']))
        if gate is not None and score<=gate:return dict(score=score,detected=False)
        ll=base+x@np.asarray(self.model['beta_localization']);idx=int(np.argmax(self.plan.mass(np.exp(ll-ll.max()))))
        return dict(score=score,detected=gate is None or score>gate,fd_hat_hz=float(self.fd[idx]),tau_hat_samples=float(self.tau[idx]))

def run(folder,repeats=2,all_records=False):
    frozen=freeze();bank=OnlineBank(folder);d=bank.d
    bundle=joblib.load(ROOT/'development/frozen_controller.joblib');config=bundle['config']
    pipe=bundle['models']['continue'];scaler=pipe[0];lr=pipe[1]
    weights=lr.coef_[0]/scaler.scale_;bias=float(lr.intercept_[0]-scaler.mean_@weights-np.log(config['reject_threshold']/(1-config['reject_threshold'])))
    dump(ROOT/'development/compiled_controller.json',dict(weights=weights.tolist(),bias=bias,
        action='stop H0 when dot(features,weights)+bias <= 0; otherwise complete rank64',
        real_MAC=len(weights),source_sha256=sha(ROOT/'development/frozen_controller.joblib')))
    # Fixed role/record selection, independent of any score or outcome.
    meta=list(d['metadata']);selected=[]
    for role,cn,count in [('H1',32,12),('H1',36,12),('holdout_H0',None,24)]:
        selected += [i for i,m in enumerate(meta) if m['role']==role and (cn is None or m['cn0_db_hz']==cn)][:count]
    ci=int(d['information']['condition']);gi=int(d['information']['geometry'])
    if all_records:selected=[i for i,m in enumerate(meta) if m['role']!='calibration_H0']
    out=ROOT/('actual_bank_benchmark_all' if all_records else 'actual_bank_benchmark')/f'condition_{ci}_geometry_{gi}';out.mkdir(exist_ok=True,parents=True)
    scored=DATA/f'holdout_c{ci}_g{gi}'/'field_evaluation';s64=[json.loads((scored/f's1_{o:03d}.json').read_text()) for o in range(len(meta))]
    gate=max(r['score'] for r,m in zip(s64,meta) if m['role']=='calibration_H0')
    rows=[];checks=[];feature_gaps=[];policy_matches=[];manual_matches=[]
    reference_features=np.load(scored/'features.npz')['X']
    reference_stops=pipe.predict_proba(reference_features)[:,1]<=config['reject_threshold']
    # Warm libraries and numerical equivalence before timing.
    o=selected[0];b,*_=bank.projection(o);bank.readout(o,b,gate)
    for repeat in range(repeats):
        for o in selected:
            for method in np.roll(['fixed64-lazy','manual-stop','learned-stop'],(o+repeat)%3):
                started=time.perf_counter();decision_time=0.;checkpoint_time=0.;projection_time=0.;readout_time=0.
                if method=='fixed64-lazy':
                    t=time.perf_counter();b,_,_,_=bank.projection(o,64);projection_time+=time.perf_counter()-t
                    t=time.perf_counter();answer=bank.readout(o,b,gate);readout_time=time.perf_counter()-t
                    stopped=False
                else:
                    t=time.perf_counter();b,inner,p16,latent=bank.projection(o,32,need_probes=method=='learned-stop');projection_time+=time.perf_counter()-t
                    t=time.perf_counter();x=bank.cheap(o,b,p16,latent,probe=method=='learned-stop');checkpoint_time=time.perf_counter()-t
                    t=time.perf_counter();stopped=bool(x@weights+bias<=0) if method=='learned-stop' else bool(x[9]<=frozen['manual_threshold']);decision_time=time.perf_counter()-t
                    if stopped:answer=dict(detected=False)
                    else:
                        t=time.perf_counter();b,_,_,_=bank.projection(o,64,previous=inner);projection_time+=time.perf_counter()-t
                        t=time.perf_counter();answer=bank.readout(o,b,gate);readout_time=time.perf_counter()-t
                elapsed=time.perf_counter()-started
                rows.append(dict(observation=o+1,repeat=repeat,method=str(method),signal=bool(meta[o]['signal']),cn0=meta[o]['cn0_db_hz'] if meta[o]['signal'] else None,
                    seconds=elapsed,projection_seconds=projection_time,checkpoint_seconds=checkpoint_time,decision_seconds=decision_time,readout_seconds=readout_time,
                    stopped=stopped,detected=answer['detected']))
                if method=='fixed64-lazy':checks.append(abs(answer['score']-s64[o]['score']))
                if method=='learned-stop':
                    feature_gaps.append(float(np.max(abs(x-reference_features[o]))));policy_matches.append(stopped==reference_stops[o])
                if method=='manual-stop':manual_matches.append(stopped==(reference_features[o,37]<=frozen['manual_threshold']))
            if (selected.index(o)+1)%12==0:print('ACTUAL BANK BENCH',repeat,selected.index(o)+1,flush=True)
    summary=[]
    for method in ['fixed64-lazy','manual-stop','learned-stop']:
        rr=[r for r in rows if r['method']==method];times=np.array([r['seconds'] for r in rr])
        summary.append(dict(method=method,n=len(rr),mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),p95_seconds=float(np.percentile(times,95)),
            stop_fraction=float(np.mean([r['stopped'] for r in rr])),projection_seconds=float(np.mean([r['projection_seconds'] for r in rr])),
            checkpoint_seconds=float(np.mean([r['checkpoint_seconds'] for r in rr])),decision_seconds=float(np.mean([r['decision_seconds'] for r in rr]))))
    assert all(policy_matches) and all(manual_matches),'Online stopping differs from frozen independent evaluation'
    dump(out/'records.json',rows);dump(out/'summary.json',dict(methods=summary,score_gap_max=max(checks),feature_gap_max=max(feature_gaps),
        all_frozen_policy_decisions_match=True,all_frozen_manual_decisions_match=True,
        physical_bank=str(folder.relative_to(BASE)),records_per_repeat=len(selected),role_mixture='1/2 H0, 1/4 32dB-Hz, 1/4 36dB-Hz',
        source_sha256=sha(__import__('pathlib').Path(__file__)),
        scope='Actual cached basis/coefficients and whitened PMF observations, single capture, 2 BLAS threads; bank construction and calibration separate',threadpools=threadpool_info()))
    print('ACTUAL BANK TIMING',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=2);p.add_argument('--condition',type=int,default=1);p.add_argument('--geometry',type=int,default=1);p.add_argument('--all-records',action='store_true');a=p.parse_args()
    name=f'holdout_c{a.condition}_g{a.geometry}'
    folder=DATA/name if (a.condition,a.geometry)==(1,1) else BASE/'results/diagnostics/fhboc_cheap_high_benchmark_20261016'/name
    with threadpool_limits(limits=2):run(folder,a.repeats,a.all_records)
