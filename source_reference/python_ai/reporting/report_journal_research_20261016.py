"""Measured-data comparison, matched-backbone ablation and research report."""
import os,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta,binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_basic_short_comparison_20261016 import writecsv

BASE=Path(__file__).resolve().parents[2];ROOT=BASE/'results/diagnostics/fhboc_journal_research_20261016'
VARIANT=os.environ.get('FHBOC_JOURNAL_VARIANT','original');assert VARIANT in ['original','moment']
SEMI=ROOT/('semiphysical_moment' if VARIANT=='moment' else 'semiphysical');LEARN=ROOT/('learning_moment' if VARIANT=='moment' else 'learning')
OUT=ROOT/('report_moment' if VARIANT=='moment' else 'report_original')


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.item()),encoding='utf8')


def calculate(events):
    summary=[];roc=[];paired=[];per=[]
    for source in ['cttc','texbat']:
        names=sorted({r['method'] for r in events if r['source']==source})
        for method in names:
            group=[r for r in events if r['source']==source and r['method']==method]
            gates={g:max(r['score'] for r in group if r['geometry']==g and r['role']=='calibration_H0') for g in [201,202]}
            for r in group:r.update(threshold=gates[r['geometry']],detected=r['score']>gates[r['geometry']],handoff=bool(r['localized'] and r['score']>gates[r['geometry']]))
            bg=[r for r in group if r['role']=='holdout_H0'];cut=sorted([r['score']-r['threshold'] for r in bg],reverse=True)[4];assert len(bg)==200
            for level in [-34,-30]:
                h=[r for r in group if r['signal'] and r['sbr_db']==level];assert len(h)==100
                row=dict(source=source,sbr_db=level,method=method,H1=100,test_H0=200,detected=sum(r['detected'] for r in h),handoff=sum(r['handoff'] for r in h),localized=sum(r['localized'] for r in h),false_alarms=sum(r['detected'] for r in bg))
                row.update(Pd=row['detected']/100,Ph=row['handoff']/100,Pfa=row['false_alarms']/200)
                for metric,k,n in [('Ph',row['handoff'],100),('Pfa',row['false_alarms'],200)]:
                    row[metric+'_lower95']=float(beta.ppf(.025,k,n-k+1)) if k else 0.;row[metric+'_upper95']=float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
                for name,rr in [('all',h),('handoff',[r for r in h if r['handoff']])]:
                    for field in ['fd_error_hz','tau_error_samples']:row[name+'_'+field+'_RMSE']=float(np.sqrt(np.mean([r[field]**2 for r in rr]))) if rr else None
                summary.append(row);accepted=[r for r in h if r['score']-r['threshold']>cut]
                roc.append(dict(source=source,sbr_db=level,method=method,false_alarms=4,Pfa=.02,detected=len(accepted),handoff=sum(r['localized'] for r in accepted),Pd=len(accepted)/100,Ph=sum(r['localized'] for r in accepted)/100,offset=cut))
                for gi in [201,202]:
                    hh=[r for r in h if r['geometry']==gi];bb=[r for r in bg if r['geometry']==gi]
                    per.append(dict(source=source,geometry=gi,sbr_db=level,method=method,H1=50,test_H0=100,threshold=gates[gi],handoff=sum(r['handoff'] for r in hh),false_alarms=sum(r['detected'] for r in bb)))
    for source in ['cttc','texbat']:
        for level in [-34,-30]:
            candidates=[r for r in events if r['source']==source and r['signal'] and r['sbr_db']==level]
            names=sorted({r['method'] for r in candidates});idx={(r['geometry'],r['record_index'],r['method']):r for r in candidates}
            comparisons=([('Gcoh-WIN',m) for m in names if m in ['PIML-field-neural','PIML-field-linear','PIML-frozen']] if 'Gcoh-WIN' in names else [('P_WIN','A_WIN'),('A_WIN','S_WIN'),('S_WIN','F_WIN')])
            for a,b in comparisons:
                gains=losses=0
                for gi in [201,202]:
                    for record in range(1,51):
                        left=idx[gi,record,a]['handoff'];right=idx[gi,record,b]['handoff'];gains+=right and not left;losses+=left and not right
                paired.append(dict(source=source,sbr_db=level,first=a,last=b,gains=gains,losses=losses,Ph_difference=(gains-losses)/100,exact_McNemar_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.))
    return summary,roc,paired,per


def empirical_pairs(events,roc):
    """Paired differences at the descriptive ROC point, separate from calibration."""
    rows=[]
    for source in ['cttc','texbat']:
        for level in [-34,-30]:
            group=[r for r in events if r['source']==source and r['signal'] and r['sbr_db']==level]
            names={r['method'] for r in group}
            cuts={r['method']:r['offset'] for r in roc if r['source']==source and r['sbr_db']==level}
            idx={(r['geometry'],r['record_index'],r['method']):bool(r['localized'] and r['score']-r['threshold']>cuts[r['method']]) for r in group}
            pairs=([('Gcoh-WIN',m) for m in ['PIML-frozen','PIML-field-linear','PIML-field-neural']] if 'Gcoh-WIN' in names else [('P_WIN','A_WIN'),('A_WIN','S_WIN'),('S_WIN','F_WIN')])
            for a,b in pairs:
                gains=losses=0
                for gi in [201,202]:
                    for record in range(1,51):
                        left,right=idx[gi,record,a],idx[gi,record,b]
                        gains+=right and not left;losses+=left and not right
                rows.append(dict(source=source,sbr_db=level,first=a,last=b,gains=gains,losses=losses,Ph_difference=(gains-losses)/100,
                    exact_McNemar_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.,
                    operating_point='descriptive empirical 4/200 test backgrounds'))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);events=[];ablation=[];gates=[]
    for source in ['cttc','texbat']:
        for gi in [201,202]:
            f=SEMI/f'{source}_holdout_g{gi}';status=json.loads((f/'evaluation/completion.json').read_text(encoding='utf8'));assert status['complete'] and status['models_unchanged']
            if VARIANT=='moment':
                gate=json.loads((f/'completion_status.json').read_text(encoding='utf8'))['gates'];assert gate['passed'];gates.append(gate)
            events+=json.loads((f/'evaluation/method_events.json').read_text(encoding='utf8'));ablation+=json.loads((f/'core_ablation.json').read_text(encoding='utf8'))
    assert len(events)==8000 and len(ablation)==10000
    s,r,p,g=calculate(events);a,ar,ap,ag=calculate(ablation)
    for filename,data in [('method_events',events),('summary',s),('empirical_matched_pfa',r),('paired_comparisons',p),('paired_empirical_comparisons',empirical_pairs(events,r)),('per_geometry',g),('ablation_events',ablation),('ablation_summary',a),('ablation_empirical',ar),('ablation_pairs',ap),('ablation_empirical_pairs',empirical_pairs(ablation,ar))]:writecsv(OUT/(filename+'.csv'),data)
    def find(data,source,level,method):return next(x for x in data if x['source']==source and x['sbr_db']==level and x['method']==method)
    # The same physical state/readout must be retained by both new learning heads.
    for source in ['cttc','texbat']:
        for level in [-34,-30]:
            for method in ['PIML-field-linear','PIML-field-neural']:assert find(s,source,level,method)['localized']==find(s,source,level,'Gcoh-WIN')['localized']
    names=['NCI','NCI-compensated','BASIC-short','Gcoh-WIN','PIML-frozen','PIML-field-linear','PIML-field-neural']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#999999','#d17a00','#3c8758','#68639d','#2463a4','#929040','#a24f72']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,source in zip(axes,['cttc','texbat']):
        for method,color in zip(names,colors):ax.plot([-34,-30],[100*find(r,source,level,method)['Ph'] for level in [-34,-30]],marker='o',label=method,color=color)
        ax.set(title=source.upper(),xticks=[-34,-30],xlabel='Injected signal/background power (dB)',ylim=(0,100));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Valid initialization at empirical Pfa 2% (%)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=8);fig.subplots_adjust(bottom=.27,left=.08,right=.98,wspace=.12)
    fig.savefig(OUT/'measured_background_comparison.png',dpi=300);fig.savefig(OUT/'measured_background_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    variants=['G_coh','P_WIN','A_WIN','S_WIN','F_WIN']
    for ax,source in zip(axes,['cttc','texbat']):
        for j,level in enumerate([-34,-30]):
            vals=[100*find(ar,source,level,m)['Ph'] for m in variants];ax.plot(range(len(variants)),vals,marker='o',label=str(level)+' dB SBR')
        ax.set(title=source.upper(),xticks=range(len(variants)),xticklabels=['Gcoh-MAP','P_WIN','A_WIN','S_WIN','F_WIN'],ylim=(0,100));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Valid initialization at empirical Pfa 2% (%)');fig.tight_layout();fig.savefig(OUT/'measured_core_ablation.png',dpi=300);fig.savefig(OUT/'measured_core_ablation.pdf');plt.close(fig)
    lines=['# 面向期刊的新增研究：学习、公开背景与复杂度','',f'日期：2026-10-16。主结果响应版本：{VARIANT}。','',
        '## 已完成的工作','',
        '新增CTTC与TEXBAT两来源半物理验证：每来源、每强度100条H1，另100条校准背景与200条测试背景；共1000条六帧观测。开发另380条实测背景观测，结合600条既有高／低动态开发观测，学习按10个完整几何组进行五折留出。强度配对共享背景窗口；原始字节区间含重采样保护间隔互不重叠。','',
        '信号按0.04倍缩放：4 MS/s、409.2 kcps、40.92 kHz×1…15跳频副载波；已知发射端低通滤波器阶数64、截止1.35 MHz。低动态50 kHz初始频移、3.23 kHz/s变化率；全部方法保持相同辅助信息、完整状态域和观测。注入档位为−34/−30 dB信号／完整实测背景功率比。','',
        'TEXBAT原25 MS/s片段按官方scott.m交织I/Q读取，带保护间隔重采样至4 MS/s；16 KB前缀与提供方HTTP Range逐字节相等。CTTC使用30 s以后的开发／校准与70 s以后的测试时段，原采样率4 MS/s、交织int16 IQ。目标H0为背景中未注入FHBOC/P信号，背景仍保留真实GNSS及接收机特性。','',
        '## 同骨干学习研究','',
        '旧学习修正的交叉消融在1000条既有六帧观测上完成：Gcoh与Gcoh＋冻结状态修正在32 dB-Hz下均为19%/28%（MAP）或21%/28%（窗口），低／高动态顺序；36 dB-Hz低动态仅97%→98%。这直接测量旧状态修正的增量。','',
        '新增学习候选以Gcoh最大状态分数为物理锚点，固定Gcoh窗口动作；21维完整证据场统计表征温度尺度积分、熵、窗口支持、频率／延迟分散及物理一致性。22参数线性模型与185参数21→8→1网络分别学习有界±3的记录证据修正，零修正精确回到物理骨干。目标为正确初始化标签的成本加权二元交叉熵，校准背景记录参与评价、不参与拟合，强度与动态标签仅用于开发计数。','',
        '以下独立门限计数与经验工作点共同呈现候选的实际作用。模型在读取留出逐事件结果之前冻结；段内矩物理修正通过模型误差门触发，保留相同网络结构、目标与选择规则。','',
        '| 来源 | SBR | 方法 | Pd | Ph | Pfa | 经验2% Ph |','|---|---|---|---|---|---|---|']
    for source in ['cttc','texbat']:
        for level in [-34,-30]:
            for method in names:
                x=find(s,source,level,method);y=find(r,source,level,method)
                lines.append(f'| {source} | {level} | {method} | {100*x["Pd"]:g} | {100*x["Ph"]:g} | {100*x["Pfa"]:g} | {100*y["Ph"]:g} |')
    lines+=['','## 学习组件的独立迁移增量','',
        'P_WIN采用固定等权幅度混合与零学习修正，A_WIN只改变学习幅度权重，S_WIN增加旧状态修正，F_WIN增加旧复一致性修正。全部系数沿用高斯开发训练值。下表比较相同门限协议下各组件，配对得失与精确McNemar值另存ablation_pairs.csv。','',
        '| 来源 | SBR | Gcoh-MAP | 固定P | 学习A | 状态S | 完整F |','|---|---|---|---|---|---|---|']
    for source in ['cttc','texbat']:
        for level in [-34,-30]:lines.append(f'| {source} | {level} | '+' | '.join(f'{100*find(ar,source,level,m)["Ph"]:g}' for m in variants)+' |')
    lines+=['','本轮学习结论：新增证据场网络在两来源−30 dB工作点均低于原冻结PIML，其与同输入线性模型的Ph相同；185参数容量本身没有带来稳定增量。固定P至学习A在四个点的Ph一致。TEXBAT −30 dB中复一致性修正S→F在经验同虚警点带来6个百分点提升，而独立校准门限下为2条新增、1条丢失，净增1条。两种口径分别说明噪声尾部排序和实际冻结门限的作用。核心消融中的Gcoh采用MAP读出，主比较Gcoh-WIN采用窗口读出。','',
        '后续学习目标据此转向复投影残差、接收背景协方差与跨帧失配的受限修正；当前发布模型继续采用原冻结PIML。下一轮先在开发几何上建立复残差与误检事件的关联，再决定是否增加学习模块。','',
        '## BASIC、辅助域与复杂度','',
        '原BASIC已知pilot块间共轭积／变化率FFT／去啁啾频率估计通过独立公式检查。短相关接口由相关积分带宽失配导出，原高斯实验为发现问题后的回顾性修订；本次公开背景实验前已固定每帧四个16段子相关、500 Hz粗频率间隔和2048点变化率／残余频率网格。所有强度、来源使用同一配置。','',
        '论文参数表改写为绝对动态、辅助中心与残余搜索三层：f₀=f_a+Δf、τ₀=τ_a+Δτ，Δf∈±2500 Hz、Δτ∈±1.5 chip，变化率搜索±250 kHz/s。1.5 MHz为去辅助载波之前的绝对频移，±2500 Hz为辅助中心附近的残余不确定性。','',
        '理论运算与四套实际3264维稀疏因子已核算；包括PMF、FFT、全状态查询、BASIC两阶段、物理投影、候选白化、学习头与实际窗口支持重复计数。复用q缓存的对偶投影恒等式误差约10⁻¹⁵，冷启动与在线代价分列。详见docs/2026-10-16_detector_complexity_analysis.md。','',
        '## 物理响应修正与证据文件','',
        '限带低采样率下，旧段中心模型的所选状态评分抽查最大误差约10.3%。新增段内一、二阶载波相位矩，在相同原始PMF上重算物理证据；最终四几何的所选状态精确响应门限为2%。原近似版本与完整台账保存在semiphysical和report_original，主表使用semiphysical_moment。','',
        f'最终评分抽查最大相对误差：{max((g["maximum_relative_score_gap"] for g in gates),default=float("nan")):.6g}。这是所选状态抽查，完整网格继续使用二阶矩响应模型。','',
        '原始逐事件、配对差异、二项区间、全部核心消融及逐几何门限均随本目录保存。新学习候选和原PIML分别报告；算法采用由开发留出确定，独立测试给出其作用和反例。','']
    report_path=BASE/'docs/2026-10-16_journal_research_results.md'
    if VARIANT=='moment':report_path.write_text('\n'.join(lines),encoding='utf8')
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    save(OUT/'completion_status.json',dict(complete=True,response_version=VARIANT,observations=1000,comparison_events=len(events),ablation_events=len(ablation),
        source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LEARN/'frozen_field_model.json',LEARN/'frozen_linear_field_model.json',SEMI/'protocol.json']},selected_response_gates=gates))
    print('Research report complete',VARIANT,flush=True);print(json.dumps([x for x in r if x['method'] in ['PIML-frozen','Gcoh-WIN','PIML-field-neural']],ensure_ascii=True))


if __name__=='__main__':main()
