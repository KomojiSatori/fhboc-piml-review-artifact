"""Independent ~24 h FHBOC campaign: frozen seven-method, full-domain comparison.

Commands: prepare, profile, run, status, summarize. Main sample count is selected
from profiling time alone, before any main capture is generated.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import beta
from threadpoolctl import threadpool_limits, threadpool_info

from benchmark_fhboc_field_stop_20261016 import OnlineBank
from fhboc_fair_baselines_20261015 import TrajectoryNCI
from audit_basic_receiver_20261016 import BlockAdapter
from research_fhboc_stopping_20261016 import BASE, MODEL, sha

ROOT=BASE/'results/diagnostics/fhboc_large_validation_20261017'
PLAN=ROOT/'execution_plan.json'
BULK=Path.home()/'fhboc_research_runs'/'fhboc_large_validation_20261017'
FROZEN=BASE/'results/diagnostics/fhboc_cheap_field_stopping_20261016/development'
METHODS=['PIML-full','PIML-64','manual-stop','learned-stop','NCI','NCI-compensated','BASIC-quarter-ms']
MATLAB=r'D:\MATLAB_YX\MATLAB\R2025b\bin\matlab.exe'


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)


def read(p):return json.loads(p.read_text(encoding='utf8'))
def now():return datetime.now(timezone.utc).isoformat()
def tag(c,g,pilot=False):return f'profile_c{c}_g{g}' if pilot else f'c{c}_g{g}'


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);BULK.mkdir(parents=True,exist_ok=True)
    if PLAN.exists() and (ROOT/'seed_audit.json').exists():return read(PLAN)
    plan=dict(campaign='fhboc_large_validation_20261017',created_utc=now(),budget_seconds=24*3600,
        bulk_root=str(BULK),result_root=str(ROOT),cn0_db_hz=list(range(30,39)),conditions=[1,2],
        geometry_candidates=[12,9,6],H1_per_strength_bounds=[50,200],H1_step=10,
        geometries_per_condition=0,H1_per_strength=0,calibration_H0=400,test_H0=800,
        frames=6,sample_rate_hz=100000000,reference_time_s=.0025,target_pfa=.02,
        grid=dict(fd_half_width_hz=2500,fd_step_hz=62.5,tau_half_width_chips=1.5,tau_step_samples=.5,
            rate_bounds_hz_s=[-250000,250000],rate_step_hz_s=25000),
        response=dict(template_tau_step_samples=.125,batch_size=128),
        seeds=dict(hop=1100000000,aiding=1200000000,truth=1300000000,pod=1400000000,probe=1500000000,noise=1700000000),
        methods=METHODS,controller_sha256=sha(FROZEN/'frozen_controller.joblib'),
        controller_weights_sha256=sha(FROZEN/'compiled_controller.json'),PIML_sha256=sha(MODEL),
        thresholds=dict(learned_stop=.1,manual_stop=.154072),
        calibration='Each reference method: ascending order ceil((n+1)*(1-Pfa)), strict >; stopped methods inherit PIML-64 gate',
        partition='g mod 3 = 0: transfer stratum; remaining: nominal. Report strata separately.',
        nominal_dynamics=dict(low=[50000,3230],high=[1500000,200230]),
        transfer_dynamics='New geometry: fd +/-10%, rate magnitude +/-15%, alternating rate sign. Same observations for every method.',
        pairing='Same full six-frame standardized H1 noise across all nine C/N0 values; H0 counted once',
        baseline=dict(NCI='Existing static aided energy statistic',compensated_NCI='Frozen full-grid TrajectoryNCI 4096 FFT',
            BASIC='Frozen corrected quarter-ms BlockAdapter, 500 Hz coarse bank, 2048/2048 FFT'),
        statistics='Per geometry and stratum counts; noise-cluster pairing; paired method differences; Ph conditional errors; measured warm-bank costs',
        model_updates=0,manuscript_updates=False,
        interpretation='Weak-signal operating characteristics and incremental stopping tradeoff; lower C/N0 alone is not attributed to model generalization',
        selection='Choose geometry count and H1 count solely from a separate high-dynamic profile g99; no main outcome inspection')
    write(PLAN,plan);write(ROOT/'protocol_initial.json',plan)
    # Audit actual historical stream identities before new observations exist.
    listed=subprocess.run(['rg','--files','--no-ignore','results','-g','metadata.json','-g','*seed*.csv'],cwd=BASE,capture_output=True,text=True,check=True).stdout.splitlines()
    historical=set()
    for name in listed:
        path=BASE/name
        if ROOT in path.parents:continue
        if path.suffix=='.json':
            try:
                data=read(path)
                if isinstance(data,list):
                    historical.update(int(float(r['noise_seed'])) for r in data if isinstance(r,dict) and r.get('noise_seed') is not None)
            except (ValueError,TypeError,KeyError):continue
        else:
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);keys=[k for k in reader.fieldnames or [] if 'seed' in k.lower()]
                for row in reader:
                    for k in keys:
                        try:historical.add(int(float(row[k])))
                        except (ValueError,TypeError):pass
    ledger=[];seeds=set()
    for c,g in [(c,g) for c in (1,2) for g in range(1,13)]+[(2,99)]:
        offset=c*1000000+g*10000
        for role in ('hop','aiding','truth','pod','probe'):
            seed=plan['seeds'][role]+offset;ledger.append(dict(condition=c,geometry=g,role=role,record='',seed=seed));seeds.add(seed)
        for role,records in [('H1',range(1,201)),('calibration_H0',range(2001,2401)),('holdout_H0',range(4001,4801))]:
            for record in records:
                seed=plan['seeds']['noise']+offset+record
                ledger.append(dict(condition=c,geometry=g,role=role,record=record,seed=seed));seeds.add(seed)
    assert len(seeds)==len(ledger) and max(seeds)<2**32 and not seeds&historical
    with (ROOT/'seed_ledger_maximum.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    write(ROOT/'seed_audit.json',dict(passed=True,historical_files=len(listed),historical_values=len(historical),reserved_streams=len(seeds),collisions=[]))
    write(ROOT/'status.json',dict(state='prepared',updated_utc=now()))
    return plan


def hashes():
    files=[Path(__file__),MODEL,FROZEN/'frozen_controller.joblib',FROZEN/'compiled_controller.json']
    files+=list((BASE/'matlab_code/pipeline').glob('*.m'))
    files += [BASE/'python_ai/reporting'/name for name in (
        'benchmark_fhboc_field_stop_20261016.py','fhboc_sparse_checkpoint_20261016.py','fhboc_fast_checkpoint_20261016.py',
        'fhboc_fair_baselines_20261015.py','audit_basic_receiver_20261016.py','basic_yang2008.py',
        'research_fhboc_stopping_20261016.py','research_fhboc_cheap_stop_20261016.py',
        'fhboc_allgrid_inference_20261013.py','fhboc_amplitude_prior_20261013.py','fhboc_complex_consistency_20261014.py')]
    return {str(p.relative_to(BASE)):sha(p) for p in files}


def check_models(plan):
    assert sha(MODEL)==plan['PIML_sha256']
    assert sha(FROZEN/'frozen_controller.joblib')==plan['controller_sha256']
    assert sha(FROZEN/'compiled_controller.json')==plan['controller_weights_sha256']


def export(c,g,pilot=False):
    p=read(PLAN);check_models(p);folder=BULK/tag(c,g,pilot)
    if not (folder/'export_complete.json').exists():
        assert __import__('shutil').disk_usage(BULK).free>24*1024**3,'Need 24 GiB temporary space on bulk drive'
        command=f"addpath(genpath('matlab_code'));build_fhboc_large_validation_20261017('{PLAN.as_posix()}',{c},{g},{str(pilot).lower()});"
        with (ROOT/f'{tag(c,g,pilot)}_matlab.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen([MATLAB,'-batch',command],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            write(ROOT/'heartbeat.json',dict(phase='physical_export',condition=c,geometry=g,pilot=pilot,child_pid=child.pid,updated_utc=now()))
            code=child.wait()
        assert code==0,f'MATLAB failed: {log.name}'
    return folder


def quantile_gate(scores,alpha=.02,profile=False):
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if profile:rank=min(rank,len(scores))
    assert rank<=len(scores),'Insufficient calibration H0 for the requested tail'
    return float(np.sort(scores)[rank-1])


class Receiver:
    def __init__(self,folder):
        self.bank=OnlineBank(folder);self.d=self.bank.d;self.meta=list(self.d['metadata'])
        self.front=loadmat(folder/'frontend.mat',simplify_cells=True)
        self.comp=TrajectoryNCI(self.front['frontend'],6,nfft=4096);self.basic=BlockAdapter(self.front['frontend'],6,4)
        np.testing.assert_allclose(self.comp.states,self.d['states'],rtol=0,atol=1e-8)
        H=len(self.d['states']);self.full=[];self.phase=[]
        for k in range(6):
            f=folder/f'full_frame_{k+1}.c64';D=int(self.d['dimensions'][k]);assert f.stat().st_size==H*D*8
            self.full.append(np.memmap(f,dtype=np.complex64,mode='r',shape=(H,D)))
            s=self.d['states'];t=k*.001
            self.phase.append(np.exp(2j*np.pi*(s[:,0]*t+.5*s[:,2]*t*t))/np.sqrt(self.bank.q[:,k]))
        controller=read(FROZEN/'compiled_controller.json');self.w=np.array(controller['weights']);self.bias=controller['bias']
        self.gates={}
        errors=[]
        for k in range(6):
            z=self.bank.bank['whitened_observations'][k][:,:3]
            reference=np.asarray(self.d['full_precision_probe'][k]).T@z.conj()
            actual=self.full[k][:8]@z.astype(np.complex64).conj()
            errors.append(float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)))
        assert max(errors)<2e-5,'Full dictionary single-precision numerical identity failed'
        self.precision_errors=errors
    def pmf(self,o):return [v[:,o] for v in self.front['observations']]
    def full_projection(self,o):
        b=np.empty((len(self.bank.q),6),np.complex64)
        for k in range(6):
            z=np.asarray(self.bank.bank['whitened_observations'][k][:,o],np.complex64)
            inner=self.full[k]@z.conj()
            b[:,k]=inner*self.phase[k]/np.sqrt(self.d['observation_energy'][k,o]/self.bank.D[k])
        return b
    def infer(self,o,method,gate=None):
        stopped=False;x=None
        if method=='PIML-full':answer=self.bank.readout(o,self.full_projection(o),gate)
        elif method=='PIML-64':
            b,*_=self.bank.projection(o,64);answer=self.bank.readout(o,b,gate)
        elif method in ('manual-stop','learned-stop'):
            learned=method=='learned-stop'
            b,inner,p16,latent=self.bank.projection(o,32,need_probes=learned)
            x=self.bank.cheap(o,b,p16,latent,probe=learned)
            stopped=bool(x@self.w+self.bias<=0) if learned else bool(x[9]<=.154072)
            if stopped:answer=dict(score=None,detected=False)
            else:
                b,*_=self.bank.projection(o,64,previous=inner);answer=self.bank.readout(o,b,gate)
        elif method=='NCI':
            answer=dict(score=float(self.front['nci_scores'][o]),fd_hat_hz=float(self.front['nci_estimates'][o,0]),tau_hat_samples=float(self.front['nci_estimates'][o,1]))
        elif method=='NCI-compensated':answer=self.comp.infer(self.pmf(o))
        elif method=='BASIC-quarter-ms':answer=self.basic.infer(self.pmf(o))
        else:raise ValueError(method)
        return answer,stopped,x


def make_event(receiver,o,method,answer,stopped,seconds,gate,calibration=False):
    m=receiver.meta[o];score=answer.get('score');detected=bool(not stopped and score is not None and score>gate)
    fd=answer.get('fd_hat_hz');tau=answer.get('tau_hat_samples');truth=receiver.d['truth_reference']
    signal=bool(m['signal']);df=float(fd-truth[0]) if fd is not None else None;dt=float(tau-truth[1]) if tau is not None else None
    localized=bool(signal and df is not None and abs(df)<=250 and abs(dt)<=1)
    return dict(condition=int(receiver.d['information']['condition']),geometry=int(receiver.d['information']['geometry']),
        stratum='transfer' if receiver.d['information']['transfer'] else 'nominal',observation=o+1,
        role=m['role'],signal=signal,cn0=float(m['cn0_db_hz']) if signal else None,noise_seed=int(m['noise_seed']),
        method=method,score=score,threshold=gate,stopped=stopped,detected=detected,localized=localized,handoff=detected and localized,
        fd_error_hz=df if signal else None,tau_error_samples=dt if signal else None,seconds=float(seconds),
        timing_scope='MATLAB static NCI including FFT/scoring' if method=='NCI' else 'Python warm physical-bank backend',
        calibration=calibration)


def score(c,g,pilot=False):
    p=read(PLAN);folder=BULK/tag(c,g,pilot);out=ROOT/tag(c,g,pilot);out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return read(out/'complete.json')
    started=time.perf_counter();r=Receiver(folder);cal=[o for o,m in enumerate(r.meta) if m['role']=='calibration_H0']
    cal_file=out/'calibration.json'
    if cal_file.exists():calibrated=read(cal_file)
    else:
        samples=[]
        for o in cal:
            for method in ['PIML-full','PIML-64','NCI','NCI-compensated','BASIC-quarter-ms']:
                tick=time.perf_counter();a,stop,_=r.infer(o,method)
                seconds=time.perf_counter()-tick if method!='NCI' else float(r.front['nci_seconds'][o])
                samples.append(dict(observation=o,method=method,answer=a,seconds=seconds))
            if len(samples)%50==0:
                write(ROOT/'heartbeat.json',dict(phase='calibration',condition=c,geometry=g,pilot=pilot,calibration_done=len(samples)//5,updated_utc=now()))
        gates={m:quantile_gate([s['answer']['score'] for s in samples if s['method']==m],p['target_pfa'],pilot) for m in ['PIML-full','PIML-64','NCI','NCI-compensated','BASIC-quarter-ms']}
        gates['manual-stop']=gates['learned-stop']=gates['PIML-64']
        calibrated=dict(gates=gates,records=samples);write(cal_file,calibrated)
    r.gates=calibrated['gates'];eventfile=out/'events.jsonl';existing=[]
    if eventfile.exists():
        # Commit whole observations; a partial final line/block is discarded on resume.
        text=eventfile.read_text(encoding='utf8');valid=[]
        for line in text.splitlines():
            try:valid.append(json.loads(line))
            except json.JSONDecodeError:break
        grouped={}
        for a in valid:grouped.setdefault(a['observation'],[]).append(a)
        existing=[a for oo,v in grouped.items() if len(v)==7 and len({x['method'] for x in v})==7 for a in v]
        eventfile.write_text(''.join(json.dumps(a)+'\n' for a in existing),encoding='utf8')
    done={a['observation'] for a in existing};selected=[o for o,m in enumerate(r.meta) if m['role']!='calibration_H0']
    # Calibration already warmed all six complete response libraries.
    with eventfile.open('a',encoding='utf8',buffering=1) as stream:
        for n,o in enumerate(selected):
            if o+1 in done:continue
            records=[];answers={};features=None
            for method in np.roll(METHODS,o%len(METHODS)):
                tick=time.perf_counter();a,stop,x=r.infer(o,str(method),r.gates[method])
                seconds=time.perf_counter()-tick if method!='NCI' else float(r.front['nci_seconds'][o])
                e=make_event(r,o,str(method),a,stop,seconds,r.gates[method]);records.append(e);answers[str(method)]=e
                if method=='learned-stop':features=x
            ref=answers['PIML-64']
            for method in ('manual-stop','learned-stop'):
                e=answers[method];assert e['detected']==(ref['detected'] and not e['stopped'])
                if not e['stopped']:assert abs(e['score']-ref['score'])<2e-6
            # Features retained separately as diagnostic evidence, with no fitting.
            with (out/'controller_inputs.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(observation=o+1,features=features.tolist()))+'\n')
            for e in records:stream.write(json.dumps(e,allow_nan=False)+'\n')
            stream.flush()
            if (n+1)%10==0:
                write(ROOT/'heartbeat.json',dict(phase='scoring',condition=c,geometry=g,pilot=pilot,done=n+1,total=len(selected),updated_utc=now()))
                print('LARGE SCORE',tag(c,g,pilot),n+1,'/',len(selected),flush=True)
    elapsed=time.perf_counter()-started
    result=dict(complete=True,condition=c,geometry=g,pilot=pilot,records=len(selected),calibration_H0=len(cal),
        seconds=elapsed,bulk_folder=str(folder),gates=r.gates,threadpools=threadpool_info(),event_sha256=sha(eventfile),
        full_projection_precision_errors=r.precision_errors)
    write(out/'complete.json',result)
    for name in ('geometry.mat','probes.mat','export_complete.json'):
        __import__('shutil').copy2(folder/name,out/name)
    del r;gc.collect();release_full_templates(folder)
    return result


def release_full_templates(folder):
    folder=folder.resolve();assert folder.is_relative_to(BULK.resolve()) and folder!=BULK.resolve()
    for k in range(1,7):
        p=folder/f'full_frame_{k}.c64'
        if p.exists():p.unlink()


def profile():
    p=prepare();check_models(p)
    write(ROOT/'profile_source_snapshot.json',dict(created_utc=now(),sha256=hashes(),outcomes_used_for_model_selection=False))
    t=time.perf_counter();export(2,99,True)
    with threadpool_limits(limits=2):score(2,99,True)
    rows=[json.loads(x) for x in (ROOT/'profile_c2_g99/events.jsonl').read_text(encoding='utf8').splitlines()]
    info=read(BULK/'profile_c2_g99/export_complete.json')
    # Use worst measured class mean to budget variable C/N0 and H0 mixtures.
    keys=[None,*p['cn0_db_hz']];class_cost=[]
    for cn in keys:
        rr=[r for r in rows if r['cn0']==cn]
        class_cost.append(sum(np.mean([r['seconds'] for r in rr if r['method']==m]) for m in METHODS))
    per_record=float(max(class_cost)*1.15);setup=float(info['setup_seconds']*1.20)
    available=p['budget_seconds']-(time.perf_counter()-t)-1800
    choice=None
    for G in p['geometry_candidates']:
        n=int((available/(2*G)-setup)/per_record-1200)//9
        n=min(200,(n//10)*10)
        if n>=50:choice=(G,n);break
    assert choice is not None,'24-hour estimate needs a smaller predeclared campaign tier'
    G,n=choice;p.update(geometries_per_condition=G,H1_per_strength=n,
        profiling=dict(source='profile_c2_g99',setup_seconds=setup,conservative_seconds_per_record=per_record,
            elapsed_seconds=time.perf_counter()-t,estimated_main_seconds=2*G*(setup+per_record*(9*n+1200))),
        main_observations=2*G*(9*n+1200),main_H1=2*G*9*n,main_calibration_H0=2*G*400,main_test_H0=2*G*800)
    write(PLAN,p);write(ROOT/'main_freeze.json',dict(created_utc=now(),execution_plan_sha256=sha(PLAN),sha256=hashes(),
        profile_sha256=sha(ROOT/'profile_c2_g99/complete.json'),selection_inputs='timing only; main observations do not yet exist'))
    write(ROOT/'status.json',dict(state='ready',updated_utc=now(),plan=p,completed_pairs=0))
    print('MAIN PLAN FROZEN',G,n,p['main_observations'],p['profiling'],flush=True)


def verify_main():
    p=read(PLAN);f=read(ROOT/'main_freeze.json');assert f['execution_plan_sha256']==sha(PLAN)
    assert f['sha256']==hashes(),'A frozen source changed; main run stopped before using changed code'
    check_models(p);return p


def summarize():
    records=[]
    for f in sorted(ROOT.glob('c*_g*/events.jsonl')):
        if not (f.parent/'complete.json').exists():continue
        records.extend(json.loads(x) for x in f.read_text(encoding='utf8').splitlines())
    if not records:return
    grouped={}
    for r in records:
        key=(r['condition'],r['stratum'],r['cn0'],r['method'])
        grouped.setdefault(key,[]).append(r)
    summary=[]
    for (c,s,cn,m),rr in grouped.items():
        n=len(rr);k=sum(r['detected'] for r in rr);h=sum(r['handoff'] for r in rr)
        summary.append(dict(condition=c,stratum=s,cn0=cn,method=m,n=n,detected=k,handoff=h,
            Pd=k/n if cn is not None else None,Ph=h/n if cn is not None else None,Pfa=k/n if cn is None else None,
            early_stops=sum(r['stopped'] for r in rr),mean_seconds=float(np.mean([r['seconds'] for r in rr]))))
    write(ROOT/'summary_partial.json',dict(updated_utc=now(),method_events=len(records),cells=summary,
        timing_scope='NCI MATLAB timing separate; other methods counterbalanced Python warm-bank timing',manuscript_updated=False))
    with (ROOT/'summary_partial.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)


def run():
    if not (ROOT/'main_freeze.json').exists():profile()
    p=verify_main();status=read(ROOT/'status.json');started=status.get('run_started_epoch',time.time())
    deadline=started+p['budget_seconds']-p['profiling']['elapsed_seconds'];completed=[]
    write(ROOT/'status.json',dict(state='running',pid=os.getpid(),updated_utc=now(),run_started_epoch=started,
        deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),completed_pairs=0,target_pairs=p['geometries_per_condition']))
    for g in range(1,p['geometries_per_condition']+1):
        pair_finished=all((ROOT/f'c{c}_g{g}/complete.json').exists() for c in (1,2))
        if not pair_finished and completed and time.time()>deadline:
            break
        for c in (1,2):
            verify_main()
            if not (ROOT/f'c{c}_g{g}/complete.json').exists():
                export(c,g)
                with threadpool_limits(limits=2):score(c,g)
            write(ROOT/'status.json',dict(state='running',pid=os.getpid(),updated_utc=now(),run_started_epoch=started,
                condition=c,geometry=g,completed_pairs=len(completed),target_pairs=p['geometries_per_condition'],
                planned_records=p['main_observations'],deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat()))
        completed.append(g);summarize()
        if (ROOT/'stop_after_pair.request').exists():break
    summarize();state='complete' if len(completed)==p['geometries_per_condition'] else 'balanced_partial'
    write(ROOT/'status.json',dict(state=state,updated_utc=now(),run_started_epoch=started,completed_pairs=len(completed),target_pairs=p['geometries_per_condition'],elapsed_seconds=time.time()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','profile','run','status','summarize']);a=p.parse_args()
    try:
        if a.action=='prepare':print(json.dumps(prepare(),indent=2))
        elif a.action=='profile':profile()
        elif a.action=='run':run()
        elif a.action=='status':print(json.dumps(dict(status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else None,heartbeat=read(ROOT/'heartbeat.json') if (ROOT/'heartbeat.json').exists() else None),indent=2))
        else:summarize()
    except Exception as e:
        write(ROOT/'failure.json',dict(at_utc=now(),exception=repr(e),action=a.action,pid=os.getpid()))
        raise
    else:
        if a.action!='status' and (ROOT/'failure.json').exists():
            with (ROOT/'resolved_failures.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(read(ROOT/'failure.json'))+'\n')
            (ROOT/'failure.json').unlink()


if __name__=='__main__':main()
