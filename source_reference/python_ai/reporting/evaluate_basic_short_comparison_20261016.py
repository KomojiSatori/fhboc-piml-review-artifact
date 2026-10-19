"""Complete the quarter-ms BASIC comparison on all frozen conditions.

Only BASIC scores are replaced in this new archive; other detectors are reused
unchanged. This retrospective receiver revision shares the original records.
"""
import csv
import hashlib
import json
import time
import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from audit_basic_receiver_20261016 import BlockAdapter,SOURCE,OUT as AUDIT,BASE,save

OUT=BASE/'results/diagnostics/fhboc_basic_short_comparison_20261016'


def readcsv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def writecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def flag(r,key):return str(r[key]).lower() in ('true','1')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol.json',dict(date='2026-10-16',variant='BASIC quarter-ms (four 16-PMF subcorrelations per frame)',
        coarse_frequency_spacing_hz=500,rate_fft=2048,residual_frequency_fft=2048,conditions=[1,2],geometries=[1,2],
        frames=[6,10],cn0={'6':[32,36],'10':[32]},calibration='50 independent H0 maximum per method/geometry, strict >',
        provenance='Retrospective adapter correction on the existing paired records; all PIML/NCI outputs frozen',
        selection='Quarter-ms fixed by 1ms scalar-response mismatch; no fitting',new_raw_observations=0))
    events=[r for r in readcsv(SOURCE/'method_events.csv') if r['method']!='BASIC'];new=[]
    for ci in (1,2):
        for gi in (1,2):
            d=loadmat(SOURCE/f'condition_{ci}_geometry_{gi}'/'shared_frontend.mat',simplify_cells=True)
            for M in (6,10):
                truth=np.asarray(d['evaluation']['truth_six' if M==6 else 'truth_ten']);plan=BlockAdapter(d['frontend'],M,4)
                path=OUT/f'condition_{ci}_geometry_{gi}_{M}.json'
                if path.exists():new+=json.loads(path.read_text(encoding='utf8'));continue
                previous={}
                if ci==2:
                    previous={(r['role'],r['record_index']):r for r in json.loads((AUDIT/f'geometry_{gi}_{M}_4.json').read_text(encoding='utf8'))}
                group=[];start=time.perf_counter()
                for oi,meta in enumerate(d['evaluation']['metadata']):
                    role=meta['role']
                    if role=='development_H0':continue
                    signal=bool(meta['signal']);record=int(meta['record_index'])
                    for strength in ((32,36) if M==6 and signal else (32,) if signal else (0,)):
                        old=previous.get((role,record)) if strength!=36 else None
                        if old is not None:p={k:old[k] for k in ['score','fd_hat_hz','tau_hat_samples','rate_hat_hz_s','state_index']}
                        else:
                            pmf=[d['observations'][k,oi] for k in range(M)]
                            if strength==36:pmf=[d['clean_observations'][k]+(v-d['clean_observations'][k])*10**(-4/20) for k,v in enumerate(pmf)]
                            p=plan.infer(pmf)
                        df=float(p['fd_hat_hz']-truth[0]);dt=float(p['tau_hat_samples']-truth[1])
                        group.append(dict(condition=ci,geometry=gi,frames=M,method='BASIC',variant='quarter-ms-500Hz',observation=oi+1,
                            record_index=record,role=role,signal=int(signal),cn0_db_hz=strength if signal else '',noise_seed=int(meta['noise_seed']),
                            **p,fd_error_hz=df if signal else '',tau_error_samples=dt if signal else '',
                            localized=bool(signal and abs(df)<=250 and abs(dt)<=1),runtime_s=''))
                    if len(group)%50==0:print('BASIC short full comparison',ci,gi,M,len(group),round(time.perf_counter()-start,1),flush=True)
                save(path,group);new+=group
    assert len(new)==1800
    for ci in (1,2):
        for gi in (1,2):
            for M in (6,10):
                group=[r for r in new if (r['condition'],r['geometry'],r['frames'])==(ci,gi,M)]
                gate=max(r['score'] for r in group if r['role']=='calibration_H0')
                for r in group:r.update(threshold=gate,detected=r['score']>gate,handoff=bool(r['signal'] and r['localized'] and r['score']>gate))
    events+=new;assert len(events)==9000;writecsv(OUT/'method_events.csv',events)
    summary=[];roc=[];decomp=[]
    for M in (6,10):
        for ci in (1,2):
            for cn0 in ((32,36) if M==6 else (32,)):
                for method in ['NCI','NCI-compensated','BASIC','PIML','Physical-independent-GLRT']:
                    group=[r for r in events if (int(r['frames']),int(r['condition']),r['method'])==(M,ci,method)]
                    h=[r for r in group if flag(r,'signal') and float(r['cn0_db_hz'])==cn0];bg=[r for r in group if r['role']=='holdout_H0']
                    assert len(h)==100 and len(bg)==200
                    counts={k:sum(flag(r,f) for r in data) for k,f,data in [('detected_count','detected',h),('handoff_count','handoff',h),('localized_count','localized',h),('false_count','detected',bg)]}
                    summary.append(dict(frames=M,condition=ci,cn0_db_hz=cn0,method=method,H1_n=100,test_H0_n=200,**counts,
                        Pd=counts['detected_count']/100,Ph=counts['handoff_count']/100,Pfa=counts['false_count']/200))
                    decomp.append(dict(frames=M,condition=ci,cn0_db_hz=cn0,method=method,
                        correct_missed=sum(flag(r,'localized') and not flag(r,'detected') for r in h),wrong_detected=sum(not flag(r,'localized') and flag(r,'detected') for r in h)))
                    null=sorted([float(r['score'])-float(r['threshold']) for r in bg],reverse=True)
                    for k in (2,4,10):
                        cutoff=null[k];accepted=[r for r in h if float(r['score'])-float(r['threshold'])>cutoff]
                        roc.append(dict(frames=M,condition=ci,cn0_db_hz=cn0,method=method,false_count=k,Pfa=k/200,detected_count=len(accepted),Pd=len(accepted)/100,
                            handoff_count=sum(flag(r,'localized') for r in accepted),Ph=sum(flag(r,'localized') for r in accepted)/100,offset_to_calibrated_gate=cutoff))
    for name,rows in [('summary',summary),('empirical_matched_pfa',roc),('error_decomposition',decomp)]:writecsv(OUT/(name+'.csv'),rows)
    save(OUT/'completion_status.json',dict(complete=True,events=9000,new_BASIC_events=1800,new_raw_observations=0,
        original_input_sha256=hashlib.sha256((SOURCE/'method_events.csv').read_bytes()).hexdigest(),
        changed_detector='Only BASIC receiver aperture and coarse bank; other methods exactly reused'))
    print(json.dumps([r for r in summary if r['method']=='BASIC']),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
