"""Core ablation tables, matched-Pfa readings, paired module effects and audit."""
import csv
from collections import defaultdict
import numpy as np
from scipy.stats import beta,binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_fhboc_core_ablation_20261015 import ROOT,EXT,OLD,VARIANTS,DESCRIPTIONS,check,rows
from research_fhboc_weak_mechanism_20261013 import BASE,read_json,write_json,write_csv,sha256

CONTRASTS=[('G_ind','G_coh','跨帧共同复幅度／噪声尺度约束'),('G_coh','P_MAP','物理证据边缘化与先验'),
 ('P_MAP','P_WIN','固定物理模型的窗口读出'),('P_WIN','A_WIN','幅度混合权重学习'),
 ('A_WIN','S_WIN','状态级检测／定位修正'),('S_WIN','F_WIN','观测级复一致性修正'),
 ('A_MAP','A_WIN','幅度学习模型的窗口读出'),('S_MAP','S_WIN','状态修正模型的窗口读出'),('F_MAP','F_WIN','完整模型的窗口读出')]


def flag(r,k):return str(r[k]).lower() in ('true','1')


def main():
    check();assert read_json(ROOT/'completion_status.json')['complete'];events=rows(ROOT/'method_events.csv');summary=rows(ROOT/'summary.csv')
    assert len(events)==10000;group=defaultdict(list);index={};intervals=[];roc=[];effects=[]
    for r in events:
        group[int(r['condition']),int(r['geometry']),r['method']].append(r)
        key=(int(r['condition']),int(r['geometry']),r['source'],int(r['observation']),r['method']);assert key not in index;index[key]=r
        assert flag(r,'detected')==(float(r['score'])>float(r['threshold']))
        assert flag(r,'handoff')==(flag(r,'signal') and flag(r,'localized') and flag(r,'detected'))
        if flag(r,'signal'):assert flag(r,'localized')==(abs(float(r['fd_error_hz']))<=250 and abs(float(r['tau_error_samples']))<=1)
    for selected in group.values():
        cal=[r for r in selected if r['role']=='calibration_H0'];bg=[r for r in selected if r['role']=='holdout_H0']
        assert len(cal)==50 and len(bg)==100;gate=max(float(r['score']) for r in cal)
        assert all(float(r['threshold'])==gate for r in selected)
        assert not {r['noise_seed'] for r in cal}&{r['noise_seed'] for r in selected if r['role']!='calibration_H0'}
    for s in summary:
        ci=int(s['condition']);cn0=int(s['cn0_db_hz']);method=s['method'];selected=[r for r in events if int(r['condition'])==ci and r['method']==method]
        h=[r for r in selected if flag(r,'signal') and float(r['cn0_db_hz'])==cn0];bg=[r for r in selected if r['role']=='holdout_H0'];assert len(h)==100 and len(bg)==200
        counts=dict(detected_count=sum(flag(r,'detected') for r in h),handoff_count=sum(flag(r,'handoff') for r in h),localized_count=sum(flag(r,'localized') for r in h),false_count=sum(flag(r,'detected') for r in bg))
        for k,v in counts.items():assert int(s[k])==v
        np.testing.assert_allclose([float(s[k]) for k in ('Pd','Ph','Pfa')],[counts['detected_count']/100,counts['handoff_count']/100,counts['false_count']/200],rtol=0,atol=1e-14)
        for metric,key,n in [('Pd','detected_count',100),('Ph','handoff_count',100),('Pfa','false_count',200)]:
            k=int(s[key]);intervals.append(dict(condition=ci,cn0_db_hz=cn0,method=method,metric=metric,k=k,n=n,value=k/n,
                lower_95=float(beta.ppf(.025,k,n-k+1)) if k else 0.,upper_95=float(beta.ppf(.975,k+1,n-k)) if k<n else 1.))
        null=sorted([float(r['score'])-float(r['threshold']) for r in bg],reverse=True)
        for k in (2,4,10):
            cutoff=null[k];chosen=[r for r in h if float(r['score'])-float(r['threshold'])>cutoff];assert sum(v>cutoff for v in null)==k
            roc.append(dict(frames=6,condition=ci,cn0_db_hz=cn0,method=method,Pfa=k/200,false_count=k,
                detected_count=len(chosen),handoff_count=sum(flag(r,'localized') for r in chosen),Pd=len(chosen)/100,Ph=sum(flag(r,'localized') for r in chosen)/100,
                offset_to_calibrated_gate=cutoff,role='Descriptive test-H0 ROC'))
    # A pure readout intervention must preserve every record's score and decision.
    unchanged=0
    for r in events:
        if r['method'] not in ('P_MAP','A_MAP','S_MAP','F_MAP'):continue
        b=index[int(r['condition']),int(r['geometry']),r['source'],int(r['observation']),r['method'].replace('MAP','WIN')]
        assert float(r['score'])==float(b['score']) and flag(r,'detected')==flag(b,'detected');unchanged+=1
    assert unchanged==4000
    original=rows(EXT/'method_events.csv');full_replay=0
    for r in original:
        if r['method']!='PIML':continue
        b=index[int(r['condition']),int(r['geometry']),r['source'],int(r['observation']),'F_WIN']
        assert abs(float(r['score'])-float(b['score']))<2e-9
        assert abs(float(r['threshold'])-float(b['threshold']))<2e-9
        for k in ('detected','handoff'):assert flag(r,k)==flag(b,k)
        if flag(r,'signal'):assert flag(r,'localized')==flag(b,'localized')
        full_replay+=1
    assert full_replay==1000
    # Existing noncoherent physical baseline is reproduced as an independent reference.
    prior=rows(BASE/'results/diagnostics/fhboc_fair_baseline_comparison_20261015/method_events.csv')
    old_index={(int(r['condition']),int(r['geometry']),r['role'],int(r['record_index']),float(r['cn0_db_hz']) if r['cn0_db_hz'] else 0):r for r in prior if r['frames']=='6' and r['method']=='Physical-independent-GLRT'}
    for r in events:
        if r['method']=='G_ind':
            b=old_index[int(r['condition']),int(r['geometry']),r['role'],int(r['record_index']),float(r['cn0_db_hz']) if r['cn0_db_hz'] else 0]
            assert float(r['score'])==float(b['score']) and flag(r,'handoff')==flag(b,'handoff')
    for ci in (1,2):
        for cn0 in (32,36):
            for first,last,label in CONTRASTS:
                selected=[r for r in events if int(r['condition'])==ci and r['method']==first and flag(r,'signal') and float(r['cn0_db_hz'])==cn0];wins=losses=0
                for r in selected:
                    b=index[ci,int(r['geometry']),r['source'],int(r['observation']),last]
                    wins+=flag(b,'handoff') and not flag(r,'handoff');losses+=flag(r,'handoff') and not flag(b,'handoff')
                effects.append(dict(condition=ci,cn0_db_hz=cn0,first=first,last=last,component=label,gains=wins,losses=losses,Ph_delta=(wins-losses)/100,
                    exact_McNemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    write_csv(ROOT/'empirical_matched_pfa.csv',roc);write_csv(ROOT/'component_effects.csv',effects);write_csv(ROOT/'binomial_intervals.csv',intervals)
    maxscore=max(read_json(ROOT/f'condition_{ci}_geometry_{gi}'/'completion_status.json')['full_score_max_gap'] for ci in (1,2) for gi in (1,2))
    write_json(ROOT/'audit_results.json',dict(passed=True,events=10000,variant_geometry_calibrations=40,summary_rows=40,
        same_score_readout_checks=4000,frozen_full_model_replays=1000,physical_independent_GLRT_replays=1000,full_score_max_gap=maxscore,
        new_observations=0,parameter_fits=0))
    cells=[(32,1),(32,2),(36,1),(36,2)]
    for name,data in [('ablation_calibrated',summary),('ablation_matched_pfa002',[r for r in roc if r['false_count']==4])]:
        matrix=np.array([[float(next(r for r in data if (int(r['cn0_db_hz']),int(r['condition']),r['method'])==(cn0,ci,m))['Ph']) for cn0,ci in cells] for m in VARIANTS])
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=np.arange(4),xticklabels=['32 low','32 high','36 low','36 high'],yticks=np.arange(10),yticklabels=VARIANTS,
            title='Six-frame Ph; '+('empirical Pfa = 2%' if 'matched' in name else 'independent calibration'))
        for i in range(10):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]*100:.0f}/100',ha='center',va='center',color='white' if matrix[i,j]>.55 else 'black')
        for ext in ('png','pdf','svg'):fig.savefig(ROOT/(name+'.'+ext),dpi=180)
        plt.close(fig)
    lines=['# PIML核心消融：三种检测范式下的证据与读出贡献','','日期：2026-10-15。六帧32／36 dB-Hz，低／高动态每格100条H1。','',
        '## 组件定义','','共用1000条既有观测、完整状态网格、复响应、白化算子与辅助路径先验。每方法／几何由50条校准H0形成门限，以100条独立测试H0评价。','',
        '| 变体 | 定义 |','|---|---|']
    lines += ['| '+m+' | '+DESCRIPTIONS[m]+' |' for m in VARIANTS]
    lines+=['','P的幅度先验为ρ={4,8,16,32,64}上的等权高斯幅度混合，A使用冻结的五个学习权重；S加入八维检测与八维定位系数，F再加入八维观测级复一致性修正。所有变体的路径先验均为90%辅助高斯单元质量＋10%均匀单元质量。','',
        '## 正确交接率Ph','','| 变体 | 32低 | 32高 | 36低 | 36高 |','|---|---:|---:|---:|---:|']
    for m in VARIANTS:lines.append('| '+m+' | '+' | '.join(str(next(r for r in summary if (int(r['cn0_db_hz']),int(r['condition']),r['method'])==(cn0,ci,m))['handoff_count'])+'/100' for cn0,ci in cells)+' |')
    lines+=['','![独立校准消融](../results/diagnostics/fhboc_core_ablation_20261015/ablation_calibrated.png)','',
        '## 经验Pfa=2%时的Ph','','| 变体 | 32低 | 32高 | 36低 | 36高 |','|---|---:|---:|---:|---:|']
    for m in VARIANTS:lines.append('| '+m+' | '+' | '.join(str(next(r for r in roc if (r['cn0_db_hz'],r['condition'],r['method'],r['false_count'])==(cn0,ci,m,4))['handoff_count'])+'/100' for cn0,ci in cells)+' |')
    lines+=['','![同虚警消融](../results/diagnostics/fhboc_core_ablation_20261015/ablation_matched_pfa002.png)','',
        '## 逐组件配对变化','','下表使用各变体的独立校准门限；每项为新增交接数／损失交接数。','',
        '| 干预 | 32低 | 32高 | 36低 | 36高 |','|---|---:|---:|---:|---:|']
    for a,b,label in CONTRASTS:
        lines.append('| '+a+'→'+b+'：'+label+' | '+' | '.join(f"{r['gains']}/{r['losses']}" for cn0,ci in cells for r in effects if (r['condition'],r['cn0_db_hz'],r['first'],r['last'])==(ci,cn0,a,b))+' |')
    lines+=['','## 机制结论','',
        '共同复幅度与噪声尺度的联合约束是本轮最清楚的跨帧增益：经验2%虚警下，G_ind→G_coh在32低／高动态由17%／18%变为19%／28%，36低／高动态由88%／86%变为97%／94%。',
        '固定物理P_WIN与完整F_WIN的四格经验Ph分别为22／20／98／91和22／21／97／90。幅度、状态及观测级修正改变噪声尾部和信号排序；四格结果给出各组件的实际增量。高动态弱信号中，共同幅度最大统计G_coh达到28%，完整边缘证据F_WIN为21%。',
        '窗口读出在低动态32 dB-Hz下产生直接交接收益：P为20%→22%，A和S为21%→24%，F为21%→22%；各MAP／WIN配置保持逐记录检测分数和门限一致。',
        '论文主线据此聚焦完整物理响应与跨帧联合推断的检测器范式，将幅度学习、状态修正、复一致性和窗口读出作为可独立量化的实现组件。','',
        '## 核验与产物','',f'10000条事件、40组门限和40行汇总重算通过。4000次MAP／窗口干预保持分数与检测判决完全一致；1000条完整模型重构与冻结PIML一致，最大分数差异{maxscore:.3g}。逐帧物理GLRT的1000条记录复现既有对照。',
        'summary.csv与per_geometry_summary.csv保存Pd、Ph、Pfa及计数；component_effects.csv保存配对变化与检验；binomial_intervals.csv保存二项区间；method_events.csv保存逐记录结果。实验未拟合新参数，新增观测为0。','',
        '复现：scripts/run_fhboc_core_ablation_20261015.ps1。','']
    text='\n'.join(lines);(ROOT/'report.md').write_text(text,encoding='utf-8');(BASE/'docs/2026-10-15_piml_core_ablation_results.md').write_text(text,encoding='utf-8')
    print('Core ablation report and audit passed.',flush=True)


if __name__=='__main__':main()
