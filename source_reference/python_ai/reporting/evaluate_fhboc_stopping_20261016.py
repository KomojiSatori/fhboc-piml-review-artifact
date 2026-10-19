"""Grouped stopping-risk learning, adaptive H0 calibration, and cost audit."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from research_fhboc_stopping_20261016 import ROOT,RANKS,FEATURES,dump,sha

def load():
    rows=[];cost={}
    for f in sorted(ROOT.glob('*/inference_complete.json')):
        cost[f.parent.name]=json.loads(f.read_text(encoding='utf8'))
        rows+=json.loads((f.parent/'events.json').read_text(encoding='utf8'))
    groups=sorted(cost);keys=sorted({(r['group'],r['observation']) for r in rows})
    lookup={(r['group'],r['observation'],r['stage']):r for r in rows}
    traces=[[lookup[g,o,s] for s in range(4)] for g,o in keys]
    scores=np.array([[r['score'] for r in tr] for tr in traces]);margin=np.zeros_like(scores)
    X=np.array([[r['features'] for r in tr] for tr in traces]);gates={}
    meta=[t[0] for t in traces];gg=np.array([r['group'] for r in meta]);
    cal=np.array([r['role']=='calibration_H0' for r in meta]);signal=np.array([r['signal'] for r in meta]);
    for g in groups:
        ix=gg==g;gate=scores[ix&cal].max(axis=0);margin[ix]=scores[ix]-gate;gates[g]=gate.tolist()
    X[:,:,1]=margin
    detected=margin>0;localized=np.array([[r['localized'] for r in t] for t in traces]);hit=localized&detected&signal[:,None]
    unsafe=(signal[:,None]&detected[:,3:4]&~detected)| (hit[:,3:4]&~hit) | ((~signal[:,None])&~detected[:,3:4]&detected)
    return dict(traces=traces,cost=cost,meta=meta,groups=groups,gg=gg,cal=cal,signal=signal,
        score=scores,margin=margin,X=X,detected=detected,hit=hit,unsafe=unsafe,gates=gates)

def train(X,y):
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,class_weight='balanced',random_state=20261016)
    tree.fit(X,y.astype(int));return tree

def risk(tree,X):
    pred=tree.predict_proba(X)
    return pred[:,list(tree.classes_).index(1)] if 1 in tree.classes_ else np.zeros(len(X))

def selected(risks,threshold,stages=(0,1,2,3)):
    n=len(risks);choice=np.full(n,stages[-1],int)
    active=np.ones(n,bool)
    for s in stages[:-1]:
        stop=active&(risks[:,s]<=threshold);choice[stop]=s;active[stop]=False
    return choice

def performance(d,choice,method,stages=(0,1,2,3)):
    n=len(choice);ii=np.arange(n);marg=d['margin'][ii,choice].copy();terminal_gate={}
    # Calibrate the entire selected terminal statistic, including all looks.
    for g in d['groups']:
        ix=d['gg']==g;threshold=float(marg[ix&d['cal']].max());marg[ix]-=threshold;terminal_gate[g]=threshold
    detected=marg>1e-12
    localized=np.array([d['traces'][i][choice[i]]['localized'] for i in range(n)])
    hit=detected&localized&d['signal'];keep=~d['cal'];bg=keep&~d['signal'];sig=d['signal']
    base_det=d['detected'][:,3];base_hit=d['hit'][:,3]
    macratio=[];infer=[];head=[];projection=[]
    for i,(tr,s) in enumerate(zip(d['traces'],choice)):
        cost=d['cost'][tr[0]['group']];full=cost['full_projection_complex_macs'];visited=[j for j in stages if j<=s]
        if s==3:
            approximate=max([j for j in stages if j<3],default=-1)
            cmac=full+(cost['compressed_projection_complex_macs'][approximate] if approximate>=0 else 0)
        else:cmac=cost['compressed_projection_complex_macs'][s]
        macratio.append(cmac/full);projection.append(cmac)
        infer.append(sum(tr[j]['inference_seconds'] for j in visited))
        checks=[j for j in stages[:-1] if j<=s] if method.startswith(('learned-','manual-')) else []
        head.append(sum(tr[j]['controller_feature_seconds'] for j in checks))
    macratio=np.array(macratio);infer=np.array(infer);head=np.array(head)
    def subset(mask):
        return dict(n=int(mask.sum()),projection_mac_ratio=float(macratio[mask].mean()),
            field_inference_seconds=float(infer[mask].mean()),control_feature_seconds=float(head[mask].mean()),
            full_fallback_fraction=float(np.mean(choice[mask]==3))) if mask.any() else None
    result=dict(method=method,n_signal=int(sig.sum()),detections=int(detected[sig].sum()),handoffs=int(hit[sig].sum()),
        false_alarms=int(detected[bg].sum()),n_test_H0=int(bg.sum()),
        pd=float(detected[sig].mean()),ph=float(hit[sig].mean()),pfa=float(detected[bg].mean()),
        detection_losses=int(np.sum(sig&base_det&~detected)),detection_gains=int(np.sum(sig&~base_det&detected)),
        handoff_losses=int(np.sum(base_hit&~hit)),handoff_gains=int(np.sum(~base_hit&hit)),
        new_false_alarms=int(np.sum(bg&~base_det&detected)),
        average= subset(keep),H0=subset(bg),H1_32=subset(sig&np.array([r['cn0_db_hz']==32 for r in d['meta']])),
        H1_36=subset(sig&np.array([r['cn0_db_hz']==36 for r in d['meta']])),
        stop_counts={str(RANKS[s]):int(np.sum(keep&(choice==s))) for s in range(4)},
        policy_terminal_calibration=terminal_gate)
    result['cells']=[]
    for c in (1,2):
        for cn0 in (32,36):
            ix=sig&np.array([r['condition']==c and r['cn0_db_hz']==cn0 for r in d['meta']])
            if ix.any():result['cells'].append(dict(condition=c,cn0=cn0,n=int(ix.sum()),pd=float(detected[ix].mean()),ph=float(hit[ix].mean()),
                handoff_losses=int(np.sum(ix&base_hit&~hit)),detection_losses=int(np.sum(ix&base_det&~detected))))
    return result,dict(choice=choice.tolist(),detected=detected.tolist(),handoff=hit.tolist(),terminal_margin=marg.tolist())

def evaluate():
    d=load();assert len(d['groups'])>=2,'At least two complete geometries required'
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    n=len(d['meta']);folds=[];noncal=~d['cal'];controller_times=[];predictions={}
    margin_guard=(abs(d['margin']-d['margin'][:,3:4])>.5*abs(d['margin']))|d['unsafe']
    fd=np.array([[r['fd_hat_hz'] for r in tr] for tr in d['traces']])
    tau=np.array([[r['tau_hat_samples'] for r in tr] for tr in d['traces']])
    action_drift=(abs(fd-fd[:,3:4])>62.5)|(abs(tau-tau[:,3:4])>.25)
    action_relevant=d['detected']|d['detected'][:,3:4]
    targets={'failure':d['unsafe'],'margin-guard':margin_guard,'joint-guard':margin_guard|action_drift,
        'decision-guard':margin_guard|(action_drift&action_relevant)}
    for target,y in targets.items():
        pred=np.zeros((n,3))
        for held in d['groups']:
            train_ix=(d['gg']!=held)&noncal;test_ix=d['gg']==held
            tree=train(d['X'][train_ix,:3].reshape(-1,len(FEATURES)),y[train_ix,:3].reshape(-1))
            clock=time.perf_counter();pred[test_ix]=risk(tree,d['X'][test_ix,:3].reshape(-1,len(FEATURES))).reshape(-1,3)
            controller_times.append((time.perf_counter()-clock)/(3*test_ix.sum()))
            folds.append(dict(target=target,held_out=held,train_groups=sorted(set(d['gg'][train_ix])),
                train_records=int(train_ix.sum()),train_checkpoints=int(3*train_ix.sum()),
                unsafe_checkpoints=int(y[train_ix,:3].sum()),tree=export_text(tree,feature_names=FEATURES)))
        predictions[target]=pred
    thresholds=[0.,.01,.03,.05,.1,.2,.35,.5,.75,1.]
    results=[];policies={}
    def add(choice,name,stages=(0,1,2,3)):
        r,p=performance(d,choice,name,stages);results.append(r);policies[name]=p
    add(np.full(n,3),'full-PIML',stages=(3,))
    for s in range(3):add(np.full(n,s),f'fixed-rank-{RANKS[s]}',stages=(s,))
    for stages,label in [((0,1,2,3),'nested'),((0,3),'32-full'),((0,1),'32-64')]:
        oracle=np.full(n,stages[-1],int)
        for s in reversed(stages[:-1]):oracle[~d['unsafe'][:,s]]=s
        add(oracle,'oracle-'+label,stages)
        for target,pred in predictions.items():
            for threshold in thresholds:
                add(selected(pred,threshold,stages),f'learned-{label}-{target}-risk-{threshold:g}',stages)
        # A small non-learning reference: stable window and adequate gate distance.
        for distance in (.02,.05,.1,.2,.5,1.,2.,4.):
            manual=np.full((n,3),np.inf)
            for s in stages[:-1]:
                enough=np.abs(d['margin'][:,s])>=distance
                stable=np.ones(n,bool) if s==0 else ((d['X'][:,s,11]<=.25)&(d['X'][:,s,12]<=.25))
                manual[enough&stable,s]=0.
            add(selected(manual,0.,stages),f'manual-{label}-margin-{distance:g}',stages)
    baseline=results[0]
    # Working point selected on grouped development predictions. No holdout claim.
    eligible=[r for r in results if r['method'].startswith('learned-') and r['handoff_losses']==0 and r['detection_losses']==0 and r['false_alarms']<=baseline['false_alarms']]
    chosen=min(eligible,key=lambda r:r['average']['projection_mac_ratio']) if eligible else None
    chosen_target=next((target for target in targets if chosen and f'-{target}-risk-' in chosen['method']),'failure')
    final=train(d['X'][noncal,:3].reshape(-1,len(FEATURES)),targets[chosen_target][noncal,:3].reshape(-1))
    model=dict(feature_names=FEATURES,feature_transform='Feature 1 is score minus current rank calibration maximum; other features unchanged',
        children_left=final.tree_.children_left.tolist(),children_right=final.tree_.children_right.tolist(),
        feature=final.tree_.feature.tolist(),threshold=final.tree_.threshold.tolist(),
        classes=final.classes_.tolist(),value=final.tree_.value.tolist(),depth=final.get_depth(),
        nodes=final.tree_.node_count,training_records=int(noncal.sum()),
        selected_policy=chosen['method'] if chosen else None,training_target=chosen_target,
        selected_rank_path=([32,64] if chosen and '-32-64-' in chosen['method'] else
            [32,3264] if chosen and '-32-full-' in chosen['method'] else RANKS),
        selection_phase='Grouped controller development',source_sha256=sha(Path(__file__)))
    dump(out/'frozen_controller.json',model);dump(out/'grouped_folds.json',folds)
    dump(out/'method_results.json',results);dump(out/'policy_events.json',policies)
    dump(out/'stage_gates.json',d['gates'])
    summary=dict(complete=len(d['groups'])==8,groups=d['groups'],observations=n,
        signal=int(d['signal'].sum()),calibration_H0=int(d['cal'].sum()),test_H0=int((~d['signal']&~d['cal']).sum()),
        baseline=baseline,selected=chosen,controller_model=model,
        controller_seconds_per_checkpoint_median=float(np.median(controller_times)),
        caveat='Geometry-held-out controller development; fixed PIML teacher previously used these development records',
        cost_note='Projection MAC model and measured field/feature time are separate; no inferred end-to-end speedup',
        unsafe_per_rank=[int(d['unsafe'][noncal,s].sum()) for s in range(3)],
        check_identity_max=np.max([np.max(v['checks'],axis=0) for v in d['cost'].values()],axis=0).tolist())
    dump(out/'summary.json',summary)
    flat=[]
    for r in results:
        flat.append({k:r[k] for k in ('method','pd','ph','pfa','detection_losses','handoff_losses','new_false_alarms')}|
            {'projection_mac_ratio':r['average']['projection_mac_ratio'],'field_seconds':r['average']['field_inference_seconds'],
             'feature_seconds':r['average']['control_feature_seconds'],'fallback_fraction':r['average']['full_fallback_fraction']})
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    plot(results,out)
    print(json.dumps(dict(groups=len(d['groups']),observations=n,baseline=baseline,selected=chosen),ensure_ascii=False),flush=True)

def plot(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for prefix,color,marker in [('fixed','tab:blue','s'),('manual-32-full','tab:gray','^'),('learned-32-full','tab:orange','o'),('learned-nested','tab:green','.'),('learned-32-64','tab:purple','D'),('oracle','tab:red','*')]:
        rr=[r for r in results if r['method'].startswith(prefix)]
        if not rr:continue
        axes[0].scatter([r['average']['projection_mac_ratio'] for r in rr],[100*r['ph'] for r in rr],label=prefix,c=color,marker=marker)
        axes[1].scatter([r['average']['field_inference_seconds'] for r in rr],[100*r['ph'] for r in rr],label=prefix,c=color,marker=marker)
    for ax in axes:
        ax.axhline(100*results[0]['ph'],ls='--',c='black',lw=1,label='Full PIML');ax.set_ylabel('Handoff probability (%)');ax.grid(alpha=.2)
    axes[0].set_xlabel('Projection complex MACs / full projection');axes[1].set_xlabel('Measured state-field inference time (s / record)')
    axes[0].legend(fontsize=7);fig.savefig(out/'stopping_tradeoff.png',dpi=180);fig.savefig(out/'stopping_tradeoff.pdf');plt.close(fig)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):evaluate()
