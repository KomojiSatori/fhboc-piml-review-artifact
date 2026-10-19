"""Readout regression: mathematical counterexamples and fixed capture replay."""
import gc
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from campaign_fhboc_large_20261017 import BASE,BULK,ROOT as SOURCE,read,write,sha,verify_main
from benchmark_fhboc_field_stop_20261016 import OnlineBank,WindowPlan
from fhboc_centered_window_readout_20261018 import CenteredOnlineBank,select_window_action

OUT=BASE/'results/diagnostics/fhboc_receiver_audit_20261018'


def mathematical_gate():
    fd=np.arange(-500.,501.,62.5);tau=np.arange(-2.,2.1,.5)
    f,t=np.meshgrid(fd,tau,indexing='ij');f=f.ravel();t=t.ravel();plan=WindowPlan(f,t)
    w=np.zeros(len(f));peak=np.flatnonzero((f==0)&(t==0))[0];w[peak]=1.
    ll=np.log(np.maximum(w,1e-300));mass=plan.mass(w)
    assert np.count_nonzero(mass==mass.max())>1
    assert select_window_action(mass,ll)==peak
    # A MAP substitution would fail this two-mode counterexample. The maximum
    # handoff mass must remain primary even when its centre has low density.
    w[:]=0;w[(f==-187.5)&(t==0)]=.6;w[(f==187.5)&(t==0)]=.4
    ll=np.log(np.maximum(w,1e-300));mass=plan.mass(w);i=select_window_action(mass,ll)
    assert mass[i]==1. and mass[np.argmax(ll)]<1.
    # Strictly different masses are not merged by a numerical tie tolerance.
    assert select_window_action(np.array([1.,1.-1e-14]),np.array([-100.,100.]))==0
    # A unique secondary optimum is independent of candidate enumeration.
    mass=np.array([.5,.9,.9]);ll=np.array([20.,1.,2.]);perm=np.array([2,0,1])
    assert perm[select_window_action(mass[perm],ll[perm])]==select_window_action(mass,ll)
    return dict(single_peak_passed=True,two_mode_primary_objective_passed=True,
                near_ties_remain_distinct=True,permutation_equivariance_passed=True)


def replay(c,g):
    name=f'c{c}_g{g}';target=OUT/f'{name}_centered_replay.json'
    if target.exists():return read(target)
    bank=CenteredOnlineBank(BULK/name);meta=list(bank.d['metadata'])
    selected=[]
    for cn in (32,36,38):
        selected += [i for i,m in enumerate(meta) if m['signal'] and int(m['cn0_db_hz'])==cn][:2]
    selected += [i for i,m in enumerate(meta) if m['role']=='holdout_H0'][:2]
    archived={}
    for line in (SOURCE/name/'events.jsonl').open(encoding='utf8'):
        r=json.loads(line)
        if r['method']=='PIML-64' and r['observation']-1 in selected:archived[r['observation']-1]=r
    gate=read(SOURCE/name/'complete.json')['gates']['PIML-64']
    rows=[]
    def label(answer,truth):
        if 'fd_hat_hz' not in answer:return None
        df=float(answer['fd_hat_hz']-truth[0]);dt=float(answer['tau_hat_samples']-truth[1])
        return dict(fd_error_hz=df,tau_error_samples=dt,localized=abs(df)<=250 and abs(dt)<=1)
    truth=bank.d['truth_reference']
    for o in selected:
        b=bank.projection(o,64)[0]
        old=OnlineBank.readout(bank,o,b,gate);new=bank.readout(o,b,gate)
        assert old['score']==new['score'] and old['detected']==new['detected']
        assert abs(old['score']-archived[o]['score'])<2e-6
        for field in ('fd_hat_hz','tau_hat_samples'):
            if field in old:
                error=archived[o]['fd_error_hz' if field=='fd_hat_hz' else 'tau_error_samples']
                if meta[o]['signal']:assert abs(old[field]-truth[0 if field=='fd_hat_hz' else 1]-error)<1e-8
        rows.append(dict(observation=o+1,signal=bool(meta[o]['signal']),
                         cn0=int(meta[o]['cn0_db_hz']) if meta[o]['signal'] else None,
                         score=new['score'],detected=new['detected'],old=label(old,truth),new=label(new,truth),
                         readout_changed=old.get('fd_hat_hz')!=new.get('fd_hat_hz') or old.get('tau_hat_samples')!=new.get('tau_hat_samples')))
    direct=loadmat(OUT/f'{name}_direct.mat',simplify_cells=True)
    o=0
    for k,z in enumerate(bank.bank['whitened_observations']):
        z[:,o]=direct['clean_whitened'][k]
        bank.d['observation_energy'][k,o]=np.vdot(z[:,o],z[:,o]).real
    b=bank.projection(o,64)[0]
    old=OnlineBank.readout(bank,o,b);new=bank.readout(o,b)
    assert old['score']==new['score']
    assert label(new,truth)['localized']
    result=dict(complete=True,geometry=name,noisy_records=rows,noiseless_old=label(old,truth),
                noiseless_new=label(new,truth),all_detection_scores_unchanged=True,
                source_sha256=sha(Path(__file__)))
    write(target,result);print('CENTERED REPLAY',name,'noisy_changed',sum(r['readout_changed'] for r in rows),flush=True)
    return result


def main():
    audit=read(OUT/'piml_audit_complete.json')
    assert audit['complete'] and audit['implementation_gates_passed']
    verify_main();gate=mathematical_gate();results=[]
    with threadpool_limits(limits=2):
        for c in (1,2):
            for g in range(1,10):results.append(replay(c,g));gc.collect()
    rows=[r for result in results for r in result['noisy_records']]
    changed=[r for r in rows if r['readout_changed']]
    h=[r for r in rows if r['signal'] and r['detected']]
    write(OUT/'centered_replay_complete.json',dict(complete=True,mathematical_gate=gate,
          geometries=18,noiseless_correct=sum(r['noiseless_new']['localized'] for r in results),
          noisy_records=len(rows),noisy_signal_records=sum(r['signal'] for r in rows),
          noisy_readouts_changed=len(changed),all_detection_scores_unchanged=True,
          noisy_handoff_gains=sum(r['new']['localized'] and not r['old']['localized'] for r in h),
          noisy_handoff_losses=sum(r['old']['localized'] and not r['new']['localized'] for r in h),
          scope='Fixed regression selection; original full campaign events and paper unchanged.',
          readout_sha256=sha(BASE/'python_ai/reporting/fhboc_centered_window_readout_20261018.py'),
          source_sha256=sha(Path(__file__))))
    verify_main()


if __name__=='__main__':main()
