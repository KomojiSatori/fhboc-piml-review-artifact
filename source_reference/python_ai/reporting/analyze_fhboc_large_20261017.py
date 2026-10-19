"""Standalone analysis of the independent campaign; never writes manuscripts."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from campaign_fhboc_large_20261017 import ROOT, METHODS, read, write, sha


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def csvout(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    plan=read(ROOT/'execution_plan.json');rows=[];included=[]
    for g in range(1,plan['geometries_per_condition']+1):
        paths=[ROOT/f'c{c}_g{g}/complete.json' for c in (1,2)]
        if not all(p.exists() for p in paths):continue
        for c,p in zip((1,2),paths):
            packet=read(p);f=p.parent/'events.jsonl';assert sha(f)==packet['event_sha256']
            rr=[json.loads(x) for x in f.read_text(encoding='utf8').splitlines()];rows+=rr
            included.append(dict(condition=c,geometry=g,records=packet['records'],event_sha256=packet['event_sha256']))
    assert rows,'No complete balanced geometry pairs yet'
    lookup={(r['condition'],r['geometry'],r['observation'],r['method']):r for r in rows}
    assert len(lookup)==len(rows)
    grouped=defaultdict(list)
    for r in rows:grouped[r['condition'],r['stratum'],r['cn0'],r['method']].append(r)
    summary=[];paired=[]
    for (c,s,cn,m),rr in grouped.items():
        n=len(rr);k=sum(r['detected'] for r in rr);h=sum(r['handoff'] for r in rr)
        detected=[r for r in rr if r['detected'] and r['fd_error_hz'] is not None]
        summary.append(dict(condition=c,stratum=s,cn0=cn,method=m,n=n,detected=k,handoff=h,
            Pd=k/n if cn is not None else None,Ph=h/n if cn is not None else None,Pfa=k/n if cn is None else None,
            binomial_interval95=interval(k,n),handoff_binomial_interval95=interval(h,n) if cn is not None else None,
            handoff_given_detection=h/k if k and cn is not None else None,wrong_detected=k-h if cn is not None else None,
            detected_fd_rmse=float(np.sqrt(np.mean([r['fd_error_hz']**2 for r in detected]))) if detected else None,
            detected_delay_rmse=float(np.sqrt(np.mean([r['tau_error_samples']**2 for r in detected]))) if detected else None,
            early_stop_fraction=sum(r['stopped'] for r in rr)/n,mean_seconds=float(np.mean([r['seconds'] for r in rr]))))
        if cn is None:continue
        for reference in ('PIML-full','PIML-64'):
            if m==reference:continue
            ref=[lookup[r['condition'],r['geometry'],r['observation'],reference] for r in rr]
            delta=np.array([r['handoff']-b['handoff'] for r,b in zip(rr,ref)],float)
            # Resample complete physical geometries; comparisons remain paired.
            geos=sorted({r['geometry'] for r in rr});by=[delta[[i for i,r in enumerate(rr) if r['geometry']==g]] for g in geos]
            rng=np.random.default_rng(2026101700+c);boot=[]
            for _ in range(2000):
                ids=rng.integers(0,len(by),len(by))
                boot.append(float(np.mean(np.concatenate([rng.choice(by[j],len(by[j]),replace=True) for j in ids]))))
            paired.append(dict(condition=c,stratum=s,cn0=cn,method=m,reference=reference,n=n,
                detection_losses=sum(b['detected'] and not r['detected'] for r,b in zip(rr,ref)),
                detection_gains=sum(r['detected'] and not b['detected'] for r,b in zip(rr,ref)),
                handoff_losses=sum(b['handoff'] and not r['handoff'] for r,b in zip(rr,ref)),
                handoff_gains=sum(r['handoff'] and not b['handoff'] for r,b in zip(rr,ref)),
                Ph_difference=float(delta.mean()),paired_geometry_noise_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist()))
    csvout('performance.csv',summary);csvout('paired_comparisons.csv',paired)
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    colors={m:f'C{i}' for i,m in enumerate(METHODS)}
    for stratum in ('nominal','transfer'):
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for c in (1,2):
            for col,metric in enumerate(('Pd','Ph')):
                a=axes[c-1,col]
                for method in METHODS:
                    rr=sorted([r for r in summary if r['condition']==c and r['stratum']==stratum and r['method']==method and r['cn0'] is not None],key=lambda r:r['cn0'])
                    if not rr:continue
                    a.plot([r['cn0'] for r in rr],[r[metric] for r in rr],marker='o',ms=3,label=method,color=colors[method])
                a.set(title=f'{"Low" if c==1 else "High"} dynamics: {metric}',ylim=(-.02,1.02));a.grid(alpha=.2)
                if c==2:a.set_xlabel(r'$C/N_0$ (dB-Hz)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
        fig.tight_layout(rect=[0,.09,1,1])
        for ext in ('pdf','png'):fig.savefig(figures/f'{stratum}_performance.{ext}',dpi=220)
        plt.close(fig)
    notes=['# Independent large-scale FHBOC validation','',
        f'Completed balanced geometry pairs: {len(included)//2}. Method events: {len(rows)}. Models and stopping thresholds remain frozen.','',
        'Seven methods share raw PMF observations, aiding and independent calibration/test backgrounds. Nominal and perturbed/sign-reversed dynamic strata are reported separately. H1 noise is paired across nine signal strengths. Background records are counted once.','',
        'performance.csv contains Pd, Ph, Pfa, conditional handoff, and detected-event RMSE. paired_comparisons.csv contains paired losses/gains and geometry/noise bootstrap intervals. Binomial cell intervals describe the sampled records; paired geometry bootstrap intervals retain physical-geometry variation.','',
        'Timing: PIML variants use prepared whitened PMF observations and response libraries. NCI static timings come from MATLAB FFT/scoring; compensated NCI/BASIC timings include their Python FFT/adapter stages. Timing comparisons across PIML variants share preprocessing. Full receiver cost also includes raw PMF and any method-specific preprocessing.','',
        'Weak-signal changes are reported as performance curves and incremental compute/performance tradeoffs. Any generalization diagnosis requires comparing methods at the same C/N0 and paired physical realization.','',
        'No manuscript files are updated by this analysis.','']
    (ROOT/'analysis_readme.md').write_text('\n'.join(notes),encoding='utf8')
    write(ROOT/'analysis_complete.json',dict(complete=True,balanced_pairs=len(included)//2,included=included,
        method_events=len(rows),cells=len(summary),paired_comparisons=len(paired),source_sha256=sha(__import__('pathlib').Path(__file__)),manuscript_updated=False))


if __name__=='__main__':main()
