"""Reviewer-facing feature, stopping-threshold and handoff-tolerance diagnostics.

All fits use the historical eight-geometry development data. Completed campaign
captures are reused for retrospective sensitivity evaluation, with frozen
detector scores/readouts and no new waveform/noise generation.
"""
import csv
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from evaluate_fhboc_stopping_20261016 import load
from research_fhboc_field_stop_20261016 import classifier, positive, NAMES

BASE = Path(__file__).resolve().parents[2]
DEV = BASE/'results/diagnostics/fhboc_cheap_field_stopping_20261016/development'
DATA = BASE/'results/diagnostics/fhboc_large_validation_20261017'
OUT = BASE/'results/diagnostics/fhboc_revision_readiness_20261018'
BLOCKS = {'probes': list(range(28)), 'globals': list(range(28,42)),
          'complex': list(range(42,50)), 'state': list(range(50,58))}


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def csvwrite(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def campaign():
    features=[]; records=[]; inputs={}
    for c in (1,2):
        for g in range(1,10):
            folder=DATA/f'c{c}_g{g}'; events={}
            for line in (folder/'events.jsonl').open(encoding='utf8'):
                e=json.loads(line)
                if not e['calibration']: events.setdefault(e['observation'], {})[e['method']]=e
            for line in (folder/'controller_inputs.jsonl').open(encoding='utf8'):
                row=json.loads(line); packet=events[row['observation']]
                e=packet['PIML-64']; features.append(row['features']); records.append(e)
            for name in ('events.jsonl','controller_inputs.jsonl'):
                inputs[str((folder/name).relative_to(BASE))]=digest(folder/name)
    X=np.asarray(features, float)
    assert X.shape==(27360,58)
    return X, records, inputs


def summary(prob, threshold, signal, detected, handoff):
    stop=prob<=threshold; bg=~signal
    return dict(threshold=float(threshold), continued=int((~stop).sum()),
                continued_fraction=float((~stop).mean()),
                detection_losses=int((signal & detected & stop).sum()),
                handoff_losses=int((handoff & stop).sum()),
                Pd=float((detected & ~stop)[signal].mean()),
                Ph=float((handoff & ~stop)[signal].mean()),
                Pfa=float((detected & ~stop)[bg].mean()))


def safe_threshold(prob, signal, det, hit, budget):
    # Equal-probability groups remain indivisible. Allow a threshold below all
    # records as the valid no-stop policy if no observed score meets the guard.
    candidates=np.unique(prob); order=np.argsort(prob, kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(prob[order])!=0),len(prob)-1]
    ld=np.cumsum((signal & det)[order])[ends]
    lh=np.cumsum(hit[order])[ends]
    good=np.flatnonzero((ld<=budget)&(lh<=budget))
    return float(candidates[good[-1]]) if len(good) else float(np.nextafter(prob.min(), -np.inf))


def interval(prob, threshold, relevant):
    values=np.unique(prob[relevant]); below=values[values<=threshold]; above=values[values>threshold]
    return dict(lower_inclusive=float(below[-1]) if len(below) else None,
                upper_exclusive=float(above[0]) if len(above) else None)


def tolerance_audit():
    rows=[]; aggregate={}
    for c in (1,2):
        for g in range(1,10):
            for line in (DATA/f'c{c}_g{g}'/'events.jsonl').open(encoding='utf8'):
                e=json.loads(line)
                if not e['signal'] or not e['detected']: continue
                if e['method'] not in ('PIML-full','PIML-64','NCI','NCI-compensated','BASIC-quarter-ms'): continue
                assert e['handoff']==(abs(e['fd_error_hz'])<=250 and abs(e['tau_error_samples'])<=1)
                key=(c,e['stratum'],int(e['cn0']),e['method'])
                aggregate.setdefault(key, []).append((abs(e['fd_error_hz']),abs(e['tau_error_samples'])))
    for (c,stratum,cn0,method), values in sorted(aggregate.items()):
        a=np.asarray(values)
        n=480 if stratum=='nominal' else 240
        # Freeze each method's original estimate; only the success rectangle changes.
        for f in (125.,187.5,250.,312.5,375.):
            for t in (.5,.75,1.,1.25,1.5):
                h=int(((a[:,0]<=f)&(a[:,1]<=t)).sum())
                rows.append(dict(condition=c,stratum=stratum,cn0=cn0,method=method,
                                 frequency_tolerance_hz=f,delay_tolerance_samples=t,
                                 H1=n,detections=len(a),handoffs=h,Ph=h/n))
    csvwrite('tolerance_sensitivity_fixed_readout.csv',rows)
    constants=dict(sample_rate_hz=1e8, code_rate_hz=10.23e6,
                   frequency_tolerance_hz=250.,delay_tolerance_samples=1.,
                   delay_tolerance_seconds=1e-8,delay_tolerance_chips=.1023,
                   frequency_grid_step_hz=62.5,delay_grid_step_samples=.5,
                   frequency_tolerance_grid_steps=4,delay_tolerance_grid_steps=2,
                   one_ms_frequency_only_coherent_power_loss_db=float(10*np.log10(np.sinc(.25)**2)),
                   one_ms_differential_phase_radians=float(2*np.pi*250*.001),
                   one_ms_four_quadrant_FLL_unaliased_half_width_hz=500.,
                   provenance='Existing receiver acceptance design; supporting scales and sensitivity, not a measured tracking convergence basin.',
                   sensitivity='Fixed original estimates/detection thresholds, changed evaluation tolerance only; no reoptimized window readout.')
    write(OUT/'tolerance_basis.json',constants)
    return rows


def estimation_audit():
    """Paired error statistics on explicitly shared detection/success subsets."""
    pairs={}
    for c in (1,2):
        for g in range(1,10):
            for line in (DATA/f'c{c}_g{g}'/'events.jsonl').open(encoding='utf8'):
                e=json.loads(line)
                if e['signal'] and e['method'] in ('PIML-full','BASIC-quarter-ms'):
                    pairs.setdefault((c,g,e['observation']),{})[e['method']]=e
    groups={}
    for packet in pairs.values():
        a,b=packet['PIML-full'],packet['BASIC-quarter-ms']
        for scope,keep in (('common_detection',a['detected'] and b['detected']),
                           ('common_handoff',a['handoff'] and b['handoff'])):
            if not keep: continue
            key=(a['condition'],a['stratum'],int(a['cn0']),scope)
            groups.setdefault(key,[]).append((a,b))
    rows=[]
    for (c,stratum,cn0,scope), packets in sorted(groups.items()):
        for i,method in enumerate(('PIML-full','BASIC-quarter-ms')):
            error=np.array([[pair[i]['fd_error_hz'],pair[i]['tau_error_samples']] for pair in packets])
            rows.append(dict(condition=c,stratum=stratum,cn0=cn0,subset=scope,method=method,n=len(error),
                frequency_RMSE_hz=float(np.sqrt(np.mean(error[:,0]**2))),
                delay_RMSE_samples=float(np.sqrt(np.mean(error[:,1]**2))),
                frequency_bias_hz=float(error[:,0].mean()),delay_bias_samples=float(error[:,1].mean())))
    csvwrite('paired_parameter_error_decomposition.csv',rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    d=load(); data=np.load(DEV/'features.npz'); X=data['X']; y=data['continue_labels']
    noncal=~d['cal']; assert X.shape==(600,58)
    assert np.array_equal(y,d['signal'] & (d['detected'][:,3]|d['detected'][:,1]))
    XC, events, inputs=campaign()
    signal=np.array([r['signal'] for r in events],bool)
    det=np.array([r['detected'] for r in events],bool)
    hit=np.array([r['handoff'] for r in events],bool)
    condition=np.array([r['condition'] for r in events]); strength=np.array([r['cn0'] or 0 for r in events])
    variants={'all58':list(range(58))}
    for name,columns in BLOCKS.items(): variants['without_'+name]=[i for i in range(58) if i not in columns]
    variants.update(globals14=BLOCKS['globals'],probes28=BLOCKS['probes'],
                    global_complex22=BLOCKS['globals']+BLOCKS['complex'],
                    global_state22=BLOCKS['globals']+BLOCKS['state'],surrogate1=[37])
    fixed=read(DEV/'controller_config.json'); frozen=joblib.load(DEV/'frozen_controller.joblib')['models']['continue']
    rows=[]; curves=[]; devcurves=[]; selection={}; prob_store={}; oof_store={}
    devsignal=d['signal'][noncal]; devdet=d['detected'][noncal,1]; devhit=d['hit'][noncal,1]
    for name,cols in variants.items():
        pred=np.zeros(len(X))
        for held in d['groups']:
            train=noncal & (d['gg']!=held); test=d['gg']==held
            model=classifier('linear'); model.fit(X[train][:,cols],y[train])
            pred[test]=positive(model,X[test][:,cols])
        model=classifier('linear');model.fit(X[noncal][:,cols],y[noncal])
        pc=positive(model,XC[:,cols]);prob_store[name]=pc;oof_store[name]=pred
        if name=='all58':
            assert np.allclose(pc,positive(frozen,XC),atol=1e-10,rtol=1e-10)
        joblib.dump(dict(columns=cols,feature_names=[NAMES[i] for i in cols],model=model),OUT/'models'/f'{name}.joblib')
        chosen={}
        for budget in (0,1):
            # Work point selected using out-of-geometry development predictions.
            th=safe_threshold(pred[noncal],devsignal,devdet,devhit,budget)
            chosen[str(budget)]=th
            rows.append(dict(variant=name,dimensions=len(cols),policy=f'development_budget_{budget}',
                             **summary(pc,th,signal,det,hit)))
        rows.append(dict(variant=name,dimensions=len(cols),policy='common_probability_0.1',
                         **summary(pc,.1,signal,det,hit)))
        selection[name]=dict(columns=cols,thresholds=chosen,grouped_folds=len(d['groups']))
        print(name,'complete',flush=True)
    # Recover the published discrete development sweep, then provide finer
    # retrospective curves with the identical frozen model on campaign inputs.
    for r in read(DEV/'method_results.json'):
        if r['method'].startswith('stop-linear-') and r['method'].endswith('--1'):
            devcurves.append({k:r[k] for k in ('method','detection_losses','handoff_losses','loss_vs64_detection',
                'loss_vs64_handoff','reject_count','rank64_count','pd','ph','pfa','field_and_feature_seconds','projection_mac_ratio')})
    csvwrite('original_development_threshold_sweep.csv',devcurves)
    manual_rows=[]
    for th in (-.1,0.,.05,.1,.154072,.2,.3,.4,.5,.75,1.):
        manual_rows.append(summary(XC[:,37],th,signal,det,hit))
    csvwrite('frozen_manual_threshold_sensitivity.csv',manual_rows)
    manual_original=[r for r in read(DEV/'method_results.json') if r['method'].startswith('manual-evidence-')]
    csvwrite('original_manual_development_sweep.csv',
             [{k:r[k] for k in ('method','detection_losses','handoff_losses','loss_vs64_detection',
                'loss_vs64_handoff','reject_count','rank64_count','pd','ph','pfa')} for r in manual_original])
    pp=prob_store['all58']
    compiled=read(DEV/'compiled_controller.json')
    assert np.allclose(pp,expit(XC@np.asarray(compiled['weights'])+compiled['bias']+logit(.1)),atol=1e-10)
    for th in np.unique(np.r_[np.linspace(.005,.3,60),.01,.03,.05,.1,.15,.2,.3,.4,.5,.7]):
        for c in (0,1,2):
            mask=np.ones(len(pp),bool) if c==0 else condition==c
            curves.append(dict(condition=c,**summary(pp[mask],th,signal[mask],det[mask],hit[mask])))
    csvwrite('feature_block_ablation.csv',rows);csvwrite('frozen_threshold_sensitivity.csv',curves)
    # Exact intervals quantify decision/count plateaus, rather than asserting
    # robustness from a rounded x-axis or a single chosen threshold.
    oo=oof_store['all58'][noncal]
    plateau=dict(development_OOF_success_counts=interval(oo,.1,devsignal & devdet),
                 campaign_success_counts=interval(pp,.1,signal & det),
                 campaign_all_decisions=interval(pp,.1,np.ones(len(pp),bool)),
                 probability_threshold=.1,original_development_selection=fixed['selection'],
                 probability_interpretation='Class-balanced logistic score; threshold is a cost/retention working point, not an asserted calibrated signal posterior.')
    plateau['manual_campaign_success_counts']=interval(XC[:,37],.154072,signal & det)
    plateau['manual_development_success_counts']=interval(X[noncal,37],.154072,devsignal & devdet)
    write(OUT/'threshold_selection_and_plateaus.json',plateau)
    write(OUT/'feature_ablation_protocol.json',dict(
        source_development_records=600,source_development_noncal=int(noncal.sum()),
        campaign_records=len(XC),fit='StandardScaler + class-balanced LogisticRegression, C=0.3; eight geometry folds',
        threshold_selection='Development-only grouped OOF predictions; rank64 detection/handoff loss budgets 0 and 1',
        fixed_detector='Frozen rank64 score, geometry threshold and readout',
        interpretation='Retrospective feature-input ablation on completed evaluation records; no new independent confirmation.',
        feature_dependencies='Global surrogate_score/residual summaries already use state and complex features. Removing explicit blocks tests incremental input value, not elimination of upstream feature computation.',
        selected=selection))
    tol=tolerance_audit();estimation_audit()
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(8,3.2))
        for c,label in ((1,'Low dynamics'),(2,'High dynamics')):
            rr=[r for r in curves if r['condition']==c]
            axes[0].plot([r['threshold'] for r in rr],[100*r['continued_fraction'] for r in rr],label=label)
            axes[1].plot([r['threshold'] for r in rr],[100*r['handoff_losses']/6480 for r in rr],label=label)
        for ax in axes:ax.axvline(.1,color='k',ls='--',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Frozen continuation-score threshold')
        axes[0].set_ylabel('Continuation calls (%)');axes[1].set_ylabel('Lost handoffs / all H1 (percentage points)');axes[0].legend(frameon=False)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'threshold_sensitivity.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
        for c,ax in enumerate(axes,1):
            for method,color in (('PIML-full','C3'),('NCI','C0'),('NCI-compensated','C1'),('BASIC-quarter-ms','C2')):
                rr=[r for r in tol if r['condition']==c and r['stratum']=='nominal' and r['cn0']==34 and r['method']==method and r['delay_tolerance_samples']==1]
                ax.plot([r['frequency_tolerance_hz'] for r in rr],[100*r['Ph'] for r in rr],marker='o',ms=3,label=method,color=color)
            ax.axvline(250,color='k',ls='--',lw=.8);ax.set_title(('Low','High')[c-1]+' dynamics / 34 dB-Hz');ax.set_xlabel('Frequency tolerance (Hz)');ax.grid(alpha=.2)
        axes[0].set_ylabel('Correct handoff (%)');axes[1].legend(frameon=False,fontsize=7)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'tolerance_sensitivity.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    inputs.update({str((DEV/n).relative_to(BASE)):digest(DEV/n) for n in ('features.npz','frozen_controller.joblib','controller_config.json','method_results.json')})
    write(OUT/'completion.json',dict(complete=True,new_observations=0,new_detector_training=0,
        development_controller_refits=len(variants)*(len(d['groups'])+1),main_detector_modified=False,
        source_sha256=digest(Path(__file__)),inputs_sha256=inputs))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
