"""Independent mechanism development: seed audit, complete-state diagnostics, figures.

No detector fitting, holdout threshold tuning, or dynamic-NCI simulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from scipy.linalg import expm
from scipy.stats import gamma

BASE = Path(__file__).resolve().parents[2]
PROTOCOL = BASE / 'configs/fhboc_weak_mechanism_development_v1_20261013.json'
DEFAULT_ROOT = BASE / 'results/diagnostics/fhboc_weak_mechanism_20261013'


def projection_covariance(template_coherence, dimension):
    """Covariance of two normalized projections of one isotropic complex vector."""
    assert dimension > 1 and 0 <= template_coherence <= 1
    return (dimension*template_coherence-1)/(dimension**2*(dimension+1))


def fixed_state_null_tail(score, dimensions):
    """Exact independent-frame survival; NEVER the full-grid false-alarm tail."""
    d=np.asarray(dimensions,dtype=float).reshape(-1)
    assert len(d)>0 and np.all(d>1) and np.isfinite(d).all()
    if score<=0:return 1.0
    rates=(d-1)/d
    if np.all(rates==rates[0]):return float(gamma.sf(score,a=len(d),scale=1/rates[0]))
    # Sequential exponential phases avoid unstable partial fractions for near-equal d.
    generator=np.diag(-rates)+np.diag(rates[:-1],1)
    return float(np.clip(expm(generator*score)[0].sum(),0,1))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def validate_frozen_manifest(previous,current):
    # Analysis may evolve; frozen generator/configuration/source data may not.
    changed=[name for name,value in previous.items()
        if not name.endswith('.py') and current.get(name)!=value]
    if changed:
        raise RuntimeError('Frozen generation inputs changed; use a new protocol/output root: '+', '.join(changed))


def seed_ledger(p):
    rows = []
    for condition in p['conditions']:
        for gi in range(1, 7):
            offset = p['condition_seed_stride'] * condition + p['geometry_seed_stride'] * gi
            for frame in range(1, 7):
                stride = frame * p['frame_seed_stride']
                for kind, seed in (
                    ('hop', p['seed_bases']['hop'] + offset + stride),
                    ('H1_shared_noise', p['seed_bases']['noise'] + offset + stride),
                    ('H0_noise', p['seed_bases']['noise'] + offset + stride + p['h0_noise_offset']),
                ):
                    rows.append(dict(condition=condition, geometry=gi, frame=frame, kind=kind, seed=seed))
            rows.append(dict(condition=condition, geometry=gi, frame=0, kind='aiding',
                             seed=p['seed_bases']['aiding'] + offset))
    seeds = [r['seed'] for r in rows]
    assert len(seeds) == len(set(seeds)) and max(seeds) < 2**32
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    p = read_json(PROTOCOL)
    ledger = seed_ledger(p)
    candidates = {r['seed'] for r in ledger}
    # Check BOTH record seeds and their six actual frame RNG seeds.
    history = set()
    inventory = []
    def retain(value):
        try:
            v = float(value)
        except (ValueError, TypeError):
            return
        if np.isfinite(v) and v == int(v) and 0 <= v < 2**32:
            history.add(int(v))
            history.update(int(v) + k * p['frame_seed_stride'] for k in range(1, 7))
    def walk(value, field=''):
        if isinstance(value, dict):
            for key, val in value.items():
                walk(val, key)
        elif isinstance(value, list):
            for val in value:
                walk(val, field)
        elif 'seed' in field.lower():
            retain(value)
    for directory in [BASE/'configs', BASE/'results/diagnostics', BASE/'results/frozen_protocols']:
        for path in sorted(directory.rglob('*.json')):
            if path == PROTOCOL or root in path.parents:
                continue
            try:
                walk(read_json(path))
            except (json.JSONDecodeError, UnicodeError):
                raise RuntimeError(f'Cannot audit retained seed registry: {path}')
            inventory.append(str(path.relative_to(BASE)))
        for path in sorted(directory.rglob('*.csv')):
            if root in path.parents:
                continue
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = [f for f in (reader.fieldnames or []) if 'seed' in f.lower()]
                if not fields:
                    continue
                for row in reader:
                    for f in fields:
                        retain(row[f])
            inventory.append(str(path.relative_to(BASE)))
    for source in p['sources']:
        records = loadmat(BASE/source, variable_names=['records'], simplify_cells=True)['records']
        for record in records:
            walk(record['trial'])
        inventory.append(source)
    # Historical source literals also cover seed bases used by previous scripts.
    number = re.compile(r'(?<![\w.])(?:[1-4]\d{9}|[1-4](?:\.\d+)?[eE]\+?9)(?![\w.])')
    for directory, extension in [(BASE/'matlab_code/pipeline', '*.m'), (BASE/'python_ai/reporting', '*.py')]:
        for path in sorted(directory.glob(extension)):
            if '20261013' in path.name:
                continue
            for value in number.findall(path.read_text(encoding='utf-8-sig')):
                retain(value)
    collisions = sorted(candidates & history)
    assert not collisions, f'Historical RNG collision: {collisions}'
    write_csv(root/'seed_ledger.csv', ledger)
    write_json(root/'seed_audit.json', dict(passed=True, protocol=p,
        unique_rng_seeds=len(candidates), historical_values_and_frame_expansions=len(history),
        collisions=collisions, historical_registry_files=inventory,
        scope='Retained diagnostic JSON/seed CSV registries, configuration JSON, both source-record MAT files, historical pipeline/reporting seed literals. Not a global inventory of all legacy binary artifacts.',
        paired_H1='32 and 36 dB-Hz intentionally use the same frame-noise seeds'))
    sources = [PROTOCOL, Path(__file__)] + list((BASE/'matlab_code/pipeline').glob('*mechanism*20261013.m'))
    sources += [BASE/source for source in p['sources']]
    sources += [BASE/'matlab_code/pipeline'/f'{n}.m' for n in (
        'physical_path_grid_prior', 'fhboc_shared_rho_log_bayes_factor', 'fhboc_pcode_signal',
        'make_dynamic_frame_trial', 'dynamic_truth_at_time', 'candidate_local_search_reference',
        'candidate_capture_config_for_reference', 'fhboc_template_complex_surface',
        'fhboc_template_waveform_batch', 'fhboc_pmf_observation_operator', 'fhboc_pmf_whiten',
        'fhboc_pmf_response_bank', 'fhboc_pmf_bank_response', 'fhboc_physical_state_scores')]
    manifest = {str(path.relative_to(BASE)): sha256(path) for path in sources}
    if (root/'input_manifest.json').exists():
        validate_frozen_manifest(read_json(root/'input_manifest.json'),manifest)
    write_json(root/'input_manifest.json', manifest)
    write_json(root/'frozen_protocol.json', p)
    print(f'Seed audit passed: {len(candidates)} distinct RNG seeds; {len(inventory)} retained registries.')


def progress(root):
    p=read_json(PROTOCOL);completed=[]
    for ci in p['conditions']:
        for gi in range(1,7):
            path=root/f'condition_{ci}_geometry_{gi:02d}'/'completion_status.json'
            if path.exists() and read_json(path)['complete']:
                completed.append(dict(condition=ci,geometry=gi,observations=3))
    status=dict(complete=False,observations=len(completed)*3,expected_observations=36,
        completed_geometries=completed,role=p['role'],stage='generation or report pending')
    final=root/'completion_status.json'
    if final.exists() and read_json(final).get('complete'):
        status=read_json(final)
    else:
        write_json(final,status)
    print(json.dumps(status,ensure_ascii=False))


def reference_errors(states, truth, fs, rf, t):
    eta = -fs/rf
    fd = states[:, 0] + states[:, 2]*t
    tau = states[:, 1] + eta*(states[:, 0]*t + .5*states[:, 2]*t*t)
    return np.abs(fd-truth[0]), np.abs(tau-truth[1])


def verify_statistics(data):
    """Recompute every retained MATLAB statistic in Python, including MAP density."""
    c=np.minimum(np.asarray(data['c']),1-np.finfo(float).eps)
    d=np.asarray(data['dimensions']).reshape(-1)
    lu=np.asarray(data['lu']).reshape(-1);la=np.asarray(data['la']).reshape(-1)
    p=data['protocol'];rho=np.asarray(p['amplitude_rho'],dtype=float);weights=np.asarray(p['amplitude_weights'],float)
    weights=weights/weights.sum();H=len(lu)
    assert abs(np.exp(lu).sum()-1)<1e-10 and abs(np.exp(la).sum()-1)<1e-10
    largest_error=0.0
    for o in range(3):
        co=c[:,:,o];glrt=-np.sum(d*np.log1p(-co),axis=1)
        terms=np.empty((H,len(rho)))
        for i,r in enumerate(rho):
            terms[:,i]=-len(d)*np.log1p(r)-np.sum(d*np.log1p(-r/(1+r)*co),axis=1)+np.log(weights[i])
        bf=logsumexp(terms,axis=1)
        scores=np.array([glrt.max(),logsumexp(lu+bf),logsumexp(la+bf)])
        selected=np.array([np.argmax(glrt),np.argmax(data['density']['log_uniform']+bf),
                           np.argmax(data['density']['log_aided']+bf)])+1
        r=data['result']
        error=max(float(np.max(abs(glrt-r['path_glrt'][:,o]))),
            float(np.max(abs(bf-r['path_shared_log_bf'][:,o]))),
            float(np.max(abs(scores-r['scores'][:,o]))))
        assert error<1e-8
        np.testing.assert_array_equal(selected,r['selected_indices'][:,o])
        largest_error=max(largest_error,error)
    return largest_error


def record_diagnostics(data, observation):
    p = data['protocol'];states = np.asarray(data['states'], dtype=float)
    r = data['result']; diag = data['diagnostic'];cfg = data['cfg']
    values = np.asarray(r['path_glrt'])[:, observation]
    fd_error, tau_error = reference_errors(states, diag['truth_reference'], cfg['signal']['fs'],
        cfg['signal']['rf_carrier_hz'], p['reference_time_s'])
    good = (fd_error <= 250) & (tau_error <= 1)
    assert good.any() and (~good).any()
    gi = np.flatnonzero(good)[np.argmax(values[good])]
    bi = np.flatnonzero(~good)[np.argmax(values[~good])]
    selected = np.asarray(r['selected_indices'], dtype=int)[:, observation]-1
    rates = np.unique(states[:, 2]);true_rate = diag['truth_initial'][2]
    nearest_rate = rates[np.argmin(abs(rates-true_rate))]
    oracle_mask = states[:, 2] == nearest_rate
    oracle = np.flatnonzero(oracle_mask)[np.argmax(values[oracle_mask])]
    row = dict(condition=int(data['provenance']['condition']), geometry=int(data['provenance']['geometry_index']),
        observation=p['observations_per_geometry'][observation], signal=bool(p['signal_present'][observation]),
        cn0_db_hz=int(p['cn0_db_hz'][observation]), hypotheses=len(states),
        score_gap_good_minus_bad=float(values[gi]-values[bi]),
        best_good_rank=int(1+np.sum(values>values[gi])), glrt_localization=bool(good[selected[0]]),
        uniform_localization=bool(good[selected[1]]), aided_localization=bool(good[selected[2]]),
        oracle_rate_localization=bool(good[oracle]), selected_rate_hz_s=float(states[selected[0],2]),
        rate_boundary=bool(states[selected[0],2] in (rates[0],rates[-1])),
        fd_error_hz=float(fd_error[selected[0]]), code_error_samples=float(tau_error[selected[0]]),
        glrt_score=float(r['scores'][0,observation]), uniform_log_evidence=float(r['scores'][1,observation]),
        aided_log_evidence=float(r['scores'][2,observation]),
        posterior_entropy_uniform=float(r['posterior_entropy'][0,observation]),
        posterior_entropy_aided=float(r['posterior_entropy'][1,observation]))
    profiles=[]
    for rate in rates:
        mask=states[:,2]==rate;winner=np.flatnonzero(mask)[np.argmax(values[mask])]
        profiles.append(dict(condition=row['condition'],geometry=row['geometry'],observation=row['observation'],
            rate_hz_s=float(rate), maximum_glrt=float(values[winner]),fd_error_hz=float(fd_error[winner]),
            code_error_samples=float(tau_error[winner]),localization=bool(good[winner])))
    return row,profiles


def null_geometry_check(data):
    """Conditional H0 moments; state samples within one record are correlated."""
    c=np.asarray(data['c'],dtype=float)[:,:,2];d=np.asarray(data['dimensions'],dtype=float).reshape(-1)
    statistic=-d*np.log1p(-c)
    maximum=float(np.max(statistic.sum(axis=1)))
    return dict(condition=int(data['provenance']['condition']),geometry=int(data['provenance']['geometry_index']),
        mean_dimension_scaled_coherence=float(np.mean(d*c)),
        expected_dimension_scaled_coherence=1.0,
        mean_fixed_state_score=float(statistic.mean()),
        expected_fixed_state_score=float(np.mean(d/(d-1))),
        full_grid_max=maximum,
        fixed_state_tail_at_grid_max=fixed_state_null_tail(maximum,d),
        union_bound_at_grid_max=min(1,len(c)*fixed_state_null_tail(maximum,d)),
        correlated_state_count=len(c),
        warning='Mean over a correlated template field, not independent H0 trials or a KS sample')


def plot_outputs(root,records,profiles,loaded):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    figures=root/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ci in (1,2):
        for cn0,color in [(32,'#a4512c'),(36,'#286f99')]:
            rows=[r for r in records if r['condition']==ci and r['signal'] and r['cn0_db_hz']==cn0]
            ax[ci-1].plot([r['geometry'] for r in rows],[r['score_gap_good_minus_bad'] for r in rows],
                'o-',color=color,label=f'{cn0} dB-Hz')
        ax[ci-1].axhline(0,color='gray',ls='--');ax[ci-1].set(xlabel='Independent geometry',
            title=['Low-rate / 50 kHz','High-rate / 1.5 MHz'][ci-1]);ax[ci-1].legend()
    ax[0].set_ylabel('Best acceptable - best unacceptable GLRT score')
    fig.tight_layout();fig.savefig(figures/'good_bad_score_gap.png',dpi=220);fig.savefig(figures/'good_bad_score_gap.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for ci in (1,2):
        for o,label in enumerate(['H1_cn032','H1_cn036']):
            a=axs[ci-1,o]
            for gi in range(1,7):
                rows=[r for r in profiles if r['condition']==ci and r['geometry']==gi and r['observation']==label]
                y=np.array([r['maximum_glrt'] for r in rows]);a.plot(np.array([r['rate_hz_s'] for r in rows])/1000,y-y.max(),label=str(gi))
            a.set(xlabel='Rate (kHz/s)',ylabel='Profile score - global maximum',title=f'Condition {ci}, {label}')
            a.legend(title='Geometry',ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(figures/'rate_profiles.png',dpi=220);fig.savefig(figures/'rate_profiles.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    fisher_rows=[]
    for data in loaded:
        ci=int(data['provenance']['condition']);gi=int(data['provenance']['geometry_index'])
        for f in data['diagnostic']['fisher']:
            sd=np.asarray(f['reference_sd']);cn0=int(f['cn0_db_hz'])
            axs[ci-1].scatter(gi,sd[2]/1000,color='#a4512c' if cn0==32 else '#286f99',
                marker='o' if cn0==32 else 's',label=f'{cn0} dB-Hz' if gi==1 else None)
            fisher_rows.append(dict(condition=ci,geometry=gi,cn0_db_hz=cn0,fd_sd_hz=float(sd[0]),
                tau_sd_samples=float(sd[1]),rate_sd_hz_s=float(sd[2]),rank=int(f['rank']),condition_number=float(f['condition_number'])))
    for ci in (1,2):
        axs[ci-1].set(xlabel='Independent geometry',ylabel='Local rate CRLB SD (kHz/s)',title=f'Condition {ci}')
        axs[ci-1].legend()
    fig.tight_layout();fig.savefig(figures/'local_rate_information.png',dpi=220);fig.savefig(figures/'local_rate_information.pdf');plt.close(fig)
    write_csv(root/'local_information.csv',fisher_rows)
    # Actual template coherence: max over frequency/rate, no assumption of a fixed sidepeak offset.
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for data in loaded:
        ci=int(data['provenance']['condition']);states=np.asarray(data['states']);truth=data['diagnostic']['truth_initial']
        coh=np.asarray(data['diagnostic']['template_coherence']).mean(axis=1)
        taus=np.unique(states[:,1]);curve=[coh[states[:,1]==tau].max() for tau in taus]
        axs[ci-1].plot(taus-truth[1],curve,alpha=.65)
    for ci in (1,2):axs[ci-1].set(xlabel='Initial delay - truth (samples)',ylabel='Max mean template coherence',title=f'Condition {ci}')
    fig.tight_layout();fig.savefig(figures/'template_ambiguity.png',dpi=220);fig.savefig(figures/'template_ambiguity.pdf');plt.close(fig)


def report(root):
    p=read_json(PROTOCOL);records=[];profiles=[];null=[];gates=[];loaded=[]
    executed_sources=None
    for ci in p['conditions']:
        for gi in range(1,7):
            folder=root/f'condition_{ci}_geometry_{gi:02d}'
            status=read_json(folder/'completion_status.json');assert status['complete']
            data=loadmat(folder/'mechanism.mat',simplify_cells=True)
            assert plain(data['protocol'])==p,'Checkpoint protocol must match every frozen field.'
            assert np.asarray(data['c']).shape==(len(data['states']),6,3)
            assert all(np.isfinite(np.asarray(data['result']['scores']).ravel()))
            assert bool(status['numeric_gates_passed'])
            captured=dict(zip(plain(data['names']),plain(data['source_text'])))
            if executed_sources is None:executed_sources=captured
            assert captured==executed_sources,'Mixed MATLAB implementations across geometry artifacts.'
            gate=dict(condition=ci,geometry=gi,**data['gates'])
            gate['python_full_statistic_gap']=verify_statistics(data)
            gate['grid_checked']=bool(data['convergence']['performed'])
            gate['grid_passed']=bool(status.get('grid_convergence_passed',True))
            if gate['grid_checked']:
                gate['reference_marginal_max_gap']=float(np.max(abs(np.asarray(data['convergence']['score_delta'])[1:,:])))
            else:gate['reference_marginal_max_gap']=''
            gates.append(gate);loaded.append(data)
            for o in range(3):
                row,profile=record_diagnostics(data,o);records.append(row);profiles.extend(profile)
            null.append(null_geometry_check(data))
    assert len(records)==36 and len({(r['condition'],r['geometry'],r['observation']) for r in records})==36
    expected=seed_ledger(p);expected_map={(r['condition'],r['geometry'],r['frame'],r['kind']):r['seed'] for r in expected}
    for data in loaded:
        ci=int(data['provenance']['condition']);gi=int(data['provenance']['geometry_index'])
        ns=np.asarray(data['provenance']['frame_noise_seeds'],dtype=np.int64)
        assert np.array_equal(ns[:,0],ns[:,1]) and not np.any(ns[:,0]==ns[:,2])
        for k in range(6):
            assert int(ns[k,0])==expected_map[(ci,gi,k+1,'H1_shared_noise')]
            assert int(ns[k,2])==expected_map[(ci,gi,k+1,'H0_noise')]
            assert int(data['provenance']['frame_hop_seeds'][k])==expected_map[(ci,gi,k+1,'hop')]
    write_csv(root/'record_diagnostics.csv',records);write_csv(root/'rate_profiles.csv',profiles)
    write_csv(root/'null_field_moments.csv',null);write_csv(root/'numeric_gates.csv',gates)
    write_csv(root/'research_component_times.csv',[
        dict(condition=int(data['provenance']['condition']),geometry=int(data['provenance']['geometry_index']),
             three_observation_build_and_checks_s=float(data['provenance']['build_s']),
             shared_three_observation_scoring_and_probe_s=float(data['provenance']['search_s']))
        for data in loaded])
    plot_outputs(root,records,profiles,loaded)
    lines=['# FHBOC 六帧弱信号机制研究：36条独立开发观测','',
        '12种新副本／辅助误差几何，每种包含32与36 dB-Hz的配对H1及1条独立H0。仅用于机制开发，没有拟合检测门限；定位合格数量不是Pd或Ph。', '',
        '| 条件 | C/N₀ | H1数 | GLRT定位合格 | 均匀边缘定位合格 | 辅助边缘定位合格 | oracle-rate定位合格 | GLRT边界命中 | 中位正确−错误分数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summaries=[]
    for ci in p['conditions']:
        for cn0 in (32,36):
            rows=[r for r in records if r['condition']==ci and r['cn0_db_hz']==cn0 and r['signal']]
            counts=[sum(r[f] for r in rows) for f in ('glrt_localization','uniform_localization','aided_localization','oracle_rate_localization','rate_boundary')]
            gap=float(np.median([r['score_gap_good_minus_bad'] for r in rows]))
            lines.append(f"| {ci} | {cn0} | 6 | {' | '.join(map(str,counts))} | {gap:.4f} |")
            summaries.append(dict(condition=ci,cn0_db_hz=cn0,n=6,glrt_localization=counts[0],
                uniform_localization=counts[1],aided_localization=counts[2],oracle_rate_localization=counts[3],
                boundary_hits=counts[4],median_good_bad_gap=gap))
    convergence_pass=all(g['grid_passed'] for g in gates)
    write_json(root/'summary.json',dict(groups=summaries,numerical_gates=gates,grid_convergence_passed=convergence_pass))
    write_json(root/'analysis_manifest.json',{
        'analysis_source':str(Path(__file__).relative_to(BASE)),
        'analysis_sha256':sha256(Path(__file__)),
        'protocol_sha256':sha256(PROTOCOL),
        'executed_matlab_source_text_sha256':{name:hashlib.sha256(value.encode('utf-8')).hexdigest()
                                            for name,value in executed_sources.items()},
        'code_continuity_audit_sha256':sha256(root/'code_continuity_audit.json') if (root/'code_continuity_audit.json').exists() else None,
        'geometry_artifacts':{str((root/f'condition_{ci}_geometry_{gi:02d}'/'mechanism.mat').relative_to(root)):
            sha256(root/f'condition_{ci}_geometry_{gi:02d}'/'mechanism.mat') for ci in p['conditions'] for gi in range(1,7)},
    })
    lines+=['','## 证据与限制','',
        'oracle-rate仅限制到最接近真实变化率的网格层，属于带额外真值信息的条件诊断，不能部署或用于选择检测器搜索域；有限网格下不保证它是定位通过率的上界。',
        '均匀／辅助边缘统计原样采用rho={1,4,16,64}及原有积分质量与连续密度MAP读出。三个分数的数值尺度不同，不能共用门限或由更大的数值认定更好。',
        '局部Fisher消去每帧独立复幅度，使用生成器噪声尺度只作理论评价；不包含全局错峰或检测失败。',
        'H0固定模板投影的Beta分布与指数变换具有解析依据；本轮H0只有12个独立六帧记录，同记录内的所有状态共享噪声，不能把约2.7万状态当作独立样本。',
        f'正式细网格与参考网格的两几何收敛检查：{convergence_pass}。逐项见numeric_gates.csv，若未通过，不据此冻结新检测器。',
        '动态率与初始频移沿用两个既有联合条件；本轮不能独立归因动态率。动态补偿NCI继续暂缓，未运行新NCI/BASIC检测统计。','',
        '## 产物','',
        'record_diagnostics.csv：36条观测的完整域定位与竞争；rate_profiles.csv：变化率剖面；local_information.csv：局部信息；null_field_moments.csv：相关H0场矩；numeric_gates.csv：数值门控；figures/：四组可导出PNG/PDF图。',
        '每个geometry目录中的mechanism.mat保存逐状态评分、逐帧观测和模板重合度、维数、状态轴、积分先验、原配置、种子、来源以及实际运行MATLAB源文本。']
    lines+=['','research_component_times.csv中的时间包含配对观测生成、研究用协方差／算子检查或模板探针等开销，三个观测共享模板计算；不是单条生产检测器或三个统计量各自的端到端基准。本轮同时运行两个单线程工作进程。']
    lines+=['','继承的生成器每帧新建P码对象，未设置累计码时刻。本批结果条件于该匹配块模型；连续长码源／副本接口须另行门控。实际对象核验见code_continuity_audit.json及docs/2026-10-13_signal_continuity_boundary.md。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'completion_status.json',dict(complete=True,observations=36,independent_geometries=12,
        independent_H0_records=12,numerical_gates_passed=True,grid_convergence_passed=convergence_pass,
        paired_seed_ledger_verified=True,role=p['role']))
    print('\n'.join(lines[:12]))


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','progress','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    {'prepare':prepare,'progress':progress,'report':report}[args.action](args.root.resolve())


if __name__=='__main__':
    main()
