"""Frozen six-frame factor ablation on the complete 100-H1 validation cases."""
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from fhboc_allgrid_inference_20261013 import WindowPlan,standardized_features
from fhboc_amplitude_prior_20261013 import robust_path_prior,model_log_bf
from fhboc_complex_consistency_20261014 import state_features,aggregate_features
from research_fhboc_weak_mechanism_20261013 import BASE,read_json,write_json,write_csv,sha256

ROOT=BASE/'results/diagnostics/fhboc_core_ablation_20261015'
OLD=BASE/'results/diagnostics/fhboc_independent_validation_760_20261014'
EXT=BASE/'results/diagnostics/fhboc_6frame_extension_20261014'
CONF=BASE/'results/diagnostics/fhboc_complex_consistency_confirmation_760_20261014'
VARIANTS=['G_ind','G_coh','P_MAP','P_WIN','A_MAP','A_WIN','S_MAP','S_WIN','F_MAP','F_WIN']
DESCRIPTIONS={
 'G_ind':'逐帧独立复幅度／噪声尺度GLRT；最大状态读出',
 'G_coh':'六帧共同复幅度／噪声尺度GLRT；最大状态读出',
 'P_MAP':'等权幅度混合物理证据；MAP状态读出',
 'P_WIN':'等权幅度混合物理证据；交接窗口读出',
 'A_MAP':'冻结学习幅度权重；MAP状态读出',
 'A_WIN':'冻结学习幅度权重；交接窗口读出',
 'S_MAP':'A＋检测／定位状态修正；MAP状态读出',
 'S_WIN':'A＋检测／定位状态修正；交接窗口读出',
 'F_MAP':'完整冻结检测分数；MAP状态读出',
 'F_WIN':'完整冻结PIML；交接窗口读出'}


def rows(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def check():
    for path,value in read_json(ROOT/'freeze_manifest.json')['inputs'].items():assert sha256(BASE/path)==value,path


def prepare():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'freeze_manifest.json').exists():check();return
    model=read_json(EXT/'frozen_model.json');write_json(ROOT/'frozen_model.json',model)
    p=dict(date='2026-10-15',frames=6,conditions=[1,2],strengths=[32,36],H1_per_condition_strength=100,
        calibration_H0_per_geometry=50,test_H0_per_geometry=100,variants=DESCRIPTIONS,
        fixed_amplitude_weights=[.2]*5,rho=model['amplitude_model']['rho'],path_uniform_fraction=.1,
        learned_coefficients='All retained unchanged; interventions only switch the listed components',
        baseline_physics='P has fixed equal amplitude weights and zero state/record correction',
        statistical_reference='G_ind and G_coh compare phase-group nuisance constraints on the same templates and full state grid',
        calibration='Each variant/geometry maximum of the same calibration H0; strict >',
        comparison='P_MAP vs P_WIN, A_MAP vs A_WIN, S_MAP vs S_WIN, F_MAP vs F_WIN isolate readout at identical scores',
        data='Existing old+extension paired records; no new observations or fitting')
    write_json(ROOT/'frozen_protocol.json',p)
    paths=[Path(__file__),ROOT/'frozen_model.json',ROOT/'frozen_protocol.json',EXT/'method_events.csv',
        CONF/'confirmation_events.csv',CONF/'confirmation_features.csv']
    paths+=[BASE/'python_ai/reporting'/x for x in ['fhboc_allgrid_inference_20261013.py','fhboc_amplitude_prior_20261013.py','fhboc_complex_consistency_20261014.py']]
    for source in (OLD,EXT):
        for ci in (1,2):
            for gi in (1,2):
                folder=source/'allgrid'/f'condition_{ci}_geometry_{gi}'
                paths += [folder/x for x in ['feature_manifest.json','geometry.npz','metadata.json','features.npy']]
                paths.append(source/folder.name/'pilot.mat')
    write_json(ROOT/'freeze_manifest.json',dict(inputs={str(p.relative_to(BASE)):sha256(p) for p in paths},model_fitting=False,new_observations=0))
    snap=ROOT/'source_snapshots';snap.mkdir(exist_ok=True)
    import shutil
    for path in paths:
        if path.suffix=='.py':shutil.copy2(path,snap/path.name)
    print('Core ablation frozen: ten variants, same 1000 six-frame observations.',flush=True)


def packet(logs,fd,tau,plan,window):
    score=float(logsumexp(logs));weights=np.exp(logs-score)
    index=int(np.argmax(plan.mass(weights))) if window else int(np.argmax(logs))
    return score,float(fd[index]),float(tau[index]),index


def score_geometry(ci,gi):
    check();name=f'condition_{ci}_geometry_{gi}';folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'completion_status.json').exists():return
    model=read_json(ROOT/'frozen_model.json');fixed=dict(model['amplitude_model']);fixed['weights']=[.2]*5
    confirmation={(int(r['condition']),int(r['geometry']),int(r['observation'])):r for r in rows(CONF/'confirmation_features.csv')}
    retained=[r for r in rows(EXT/'method_events.csv') if int(r['condition'])==ci and int(r['geometry'])==gi and r['method']=='PIML']
    full={(r['source'],int(r['observation'])):r for r in retained}
    events=[];score_gap=location_gap=0.;phi_checks=0
    for source,tag in ((OLD,'original_760'),(EXT,'extension')):
        cache=source/'allgrid'/name;g=np.load(cache/'geometry.npz');meta=read_json(cache/'metadata.json');raw=np.load(cache/'features.npy',mmap_mode='r')
        aux=loadmat(cache/'allgrid.mat',variable_names=['dimensions','axes'],simplify_cells=True)
        D=float(np.asarray(aux['dimensions'],float).sum());fd,tau=g['fd'],g['tau'];prior=robust_path_prior(g['log_aided'],g['log_uniform'],.1);plan=WindowPlan(fd,tau)
        native=loadmat(source/name/'pilot.mat',variable_names=['states','result'],simplify_cells=True)
        if tag=='extension':
            complex_data=loadmat(cache/'allgrid.mat',variable_names=['projected','template_energy','observation_energy','dimensions','states','axes'],simplify_cells=True)
            shape=tuple(len(np.atleast_1d(complex_data['axes'][k])) for k in ('fd_hz','tau_samples','rate_hz_s'))
        for o,m in enumerate(meta):
            target=folder/f'{tag}_{o+1:03d}.json'
            if target.exists():
                saved=read_json(target);events.extend(saved['events']);score_gap=max(score_gap,saved['score_gap']);location_gap=max(location_gap,saved['location_gap']);phi_checks+=1;continue
            b=prior+model_log_bf(g['glrt'][o],D,model['amplitude_model']);p=prior+model_log_bf(g['glrt'][o],D,fixed)
            x=standardized_features(raw[o],model);ld=b+x@np.asarray(model['beta_detection']);ll=b+x@np.asarray(model['beta_localization']);TD=float(logsumexp(ld))
            original=full[tag,o+1]
            if tag=='original_760':phi=np.array([float(confirmation[ci,gi,o+1][k]) for k in model['feature_names']])
            else:
                z=state_features(complex_data['projected'][:,:,o],complex_data['template_energy'],complex_data['observation_energy'][:,o],
                    np.asarray(complex_data['dimensions'],float),complex_data['states'],shape)
                phi=aggregate_features(z,ld)
            correction=float(np.clip((phi-np.array(model['complex_feature_mean']))/np.array(model['complex_feature_scale']),-4,4)@np.asarray(model['complex_feature_beta']))
            final=TD+correction;sg=abs(final-float(original['score']));score_gap=max(score_gap,sg);assert sg<2e-9,sg
            result={}
            for variant,logits in (('P',p),('A',b),('S',ll),('F',ll)):
                for suffix,window in (('MAP',False),('WIN',True)):
                    sc,f,t,i=packet(logits,fd,tau,plan,window)
                    if variant=='S':sc=TD
                    elif variant=='F':sc=final
                    result[variant+'_'+suffix]=(sc,f,t,i)
            lg=max(abs(result['F_WIN'][1]-float(original['fd_hat_hz'])),abs(result['F_WIN'][2]-float(original['tau_hat_samples'])))
            location_gap=max(location_gap,lg);assert lg<1e-8,lg
            for variant,column in (('G_ind',0),('G_coh',2)):
                i=int(native['result']['selected_indices'][column,o])-1
                result[variant]=(float(native['result']['scores'][column,o]),float(fd[i]),float(tau[i]),i)
            output=[]
            for variant in VARIANTS:
                sc,f,t,i=result[variant];signal=bool(m['signal']);df=f-g['truth'][0];dt=t-g['truth'][1]
                output.append(dict(condition=ci,geometry=gi,frames=6,source=tag,observation=o+1,record_index=int(m['record_index']),
                    role=m['role'],signal=int(signal),cn0_db_hz=m['cn0_db_hz'] if signal else '',noise_seed=int(m['noise_seed']),method=variant,
                    score=sc,fd_hat_hz=f,tau_hat_samples=t,state_index=i,fd_error_hz=float(df) if signal else '',tau_error_samples=float(dt) if signal else '',
                    localized=bool(signal and abs(df)<=250 and abs(dt)<=1)))
            write_json(target,dict(events=output,score_gap=sg,location_gap=lg));events.extend(output);phi_checks+=1
            if (o+1)%10==0:print('CORE ABLATION',ci,gi,tag,o+1,'/',len(meta),flush=True)
        if tag=='extension':del complex_data
        del g,raw,native
    assert len(events)==2500;write_json(folder/'method_events.json',events)
    write_json(folder/'completion_status.json',dict(complete=True,method_events=2500,full_score_max_gap=score_gap,full_readout_max_gap=location_gap,
        full_reconstruction_checks=phi_checks,events_sha256=sha256(folder/'method_events.json')))
    check();print('CORE ABLATION COMPLETE',ci,gi,flush=True)


def summarize():
    check();events=[];per=[];summary=[]
    for ci in (1,2):
        for gi in (1,2):
            folder=ROOT/f'condition_{ci}_geometry_{gi}';assert read_json(folder/'completion_status.json')['complete'];events+=read_json(folder/'method_events.json')
    for ci in (1,2):
        for gi in (1,2):
            for method in VARIANTS:
                s=[r for r in events if (r['condition'],r['geometry'],r['method'])==(ci,gi,method)]
                cal=[r for r in s if r['role']=='calibration_H0'];bg=[r for r in s if r['role']=='holdout_H0'];assert len(cal)==50 and len(bg)==100
                gate=max(r['score'] for r in cal)
                for r in s:r.update(threshold=gate,detected=r['score']>gate,handoff=bool(r['signal'] and r['localized'] and r['score']>gate))
                for cn0 in (32,36):
                    h=[r for r in s if r['signal'] and r['cn0_db_hz']==cn0];assert len(h)==50
                    per.append(dict(condition=ci,geometry=gi,cn0_db_hz=cn0,method=method,threshold=gate,H1_n=50,test_H0_n=100,
                        detected_count=sum(r['detected'] for r in h),handoff_count=sum(r['handoff'] for r in h),localized_count=sum(r['localized'] for r in h),false_count=sum(r['detected'] for r in bg)))
    for cn0 in (32,36):
        for ci in (1,2):
            for method in VARIANTS:
                s=[r for r in per if (r['cn0_db_hz'],r['condition'],r['method'])==(cn0,ci,method)]
                a={k:sum(r[k] for r in s) for k in ('detected_count','handoff_count','localized_count','false_count')}
                summary.append(dict(frames=6,condition=ci,cn0_db_hz=cn0,method=method,H1_n=100,test_H0_n=200,**a,Pd=a['detected_count']/100,Ph=a['handoff_count']/100,Pfa=a['false_count']/200))
    assert len(events)==10000;write_csv(ROOT/'method_events.csv',events);write_csv(ROOT/'summary.csv',summary);write_csv(ROOT/'per_geometry_summary.csv',per)
    write_json(ROOT/'completion_status.json',dict(complete=True,method_events=10000,observations=1000,new_observations=0,trained_new_parameters=0,variants=VARIANTS))
    print('Core ablation summarized.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','geometry','summarize']);p.add_argument('--condition',type=int);p.add_argument('--geometry',type=int);a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if a.action=='prepare':prepare()
        elif a.action=='geometry':score_geometry(a.condition,a.geometry)
        else:summarize()
