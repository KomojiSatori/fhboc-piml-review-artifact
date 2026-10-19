"""Paired warm-bank timing of existing 58/30/14-input controllers.

Reuses physical banks and historical-development fits. Selects captures by
metadata order, role and strength only; no waveform generation or model fitting.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits, threadpool_info

from benchmark_fhboc_field_stop_20261016 import OnlineBank

BASE = Path(__file__).resolve().parents[2]
DATA = BASE/'results/diagnostics/fhboc_large_validation_20261017'
MODELS = BASE/'results/diagnostics/fhboc_revision_readiness_20261018/models'
BANKS = Path(r'EXTERNAL_USER_HOME\fhboc_research_runs\fhboc_large_validation_20261017')
OUT = BASE/'results/diagnostics/fhboc_compact_controller_timing_20261018'
VARIANTS = ('all58', 'without_probes', 'globals14')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    policies = {}
    for variant in VARIANTS:
        packet = joblib.load(MODELS/f'{variant}.joblib')
        scaler, lr = packet['model'][0], packet['model'][1]
        w = lr.coef_[0]/scaler.scale_
        bias = float(lr.intercept_[0]-scaler.mean_@w-logit(.1))
        policies[variant] = (np.asarray(packet['columns']), w, bias)
    write(OUT/'protocol.json', dict(
        geometries=['c1_g1', 'c1_g2', 'c2_g1', 'c2_g2'],
        selection='first four H1 at each integer C/N0 30..38, first 40 holdout_H0 per geometry',
        captures=304, repeats=2, probability_threshold=.1,
        policy_order='cyclic rotation by observation index plus repeat',
        fitting='existing eight-geometry historical-development models; no new fitting',
        timing='warm cached whitened PMF and physical response bank; projection, feature extraction, decision, continuation, readout included',
        threads=2, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]; checks=[]
    for condition, geometry in ((1,1), (1,2), (2,1), (2,2)):
        name=f'c{condition}_g{geometry}'
        bank=OnlineBank(BANKS/name); meta=list(bank.d['metadata'])
        gate=read(DATA/name/'calibration.json')['gates']['PIML-64']
        reference={r['observation']:np.asarray(r['features']) for r in
                   (json.loads(line) for line in (DATA/name/'controller_inputs.jsonl').open(encoding='utf8'))}
        events={r['observation']:r for r in
                (json.loads(line) for line in (DATA/name/'events.jsonl').open(encoding='utf8'))
                if r['method']=='PIML-64' and not r['calibration']}
        selected=[]
        for strength in range(30,39):
            selected += [i for i,m in enumerate(meta) if m['role']=='H1' and m['cn0_db_hz']==strength][:4]
        selected += [i for i,m in enumerate(meta) if m['role']=='holdout_H0'][:40]
        assert len(selected)==76
        # Warm complete execution and validate every selected online feature vector.
        for o in selected:
            b,inner,p16,latent=bank.projection(o,32,need_probes=True)
            x=bank.cheap(o,b,p16,latent)
            gap=float(np.max(abs(x-reference[o+1])))
            assert np.allclose(x,reference[o+1],rtol=2e-5,atol=2e-5), (name,o,gap)
            for variant,(cols,w,bias) in policies.items():
                assert bool(x[cols]@w+bias<=0)==bool(reference[o+1][cols]@w+bias<=0)
            checks.append(dict(geometry=name,observation=o+1,feature_gap=gap))
        b,*_=bank.projection(selected[0],64);bank.readout(selected[0],b,gate)
        for repeat in range(2):
            for j,o in enumerate(selected):
                for variant in np.roll(VARIANTS,(j+repeat)%3):
                    cols,w,bias=policies[variant]; probes=variant=='all58'
                    started=time.perf_counter()
                    b,inner,p16,latent=bank.projection(o,32,need_probes=probes)
                    t=time.perf_counter();x=bank.cheap(o,b,p16,latent,probe=probes)
                    if variant=='globals14':x=x[:14]
                    feature_seconds=time.perf_counter()-t
                    stop=bool(x@w+bias<=0)
                    if stop:answer=dict(detected=False)
                    else:
                        b,*_=bank.projection(o,64,previous=inner)
                        answer=bank.readout(o,b,gate)
                    elapsed=time.perf_counter()-started
                    assert stop==bool(reference[o+1][cols]@w+bias<=0)
                    assert answer['detected']==(events[o+1]['detected'] and not stop)
                    rows.append(dict(geometry=name,observation=o+1,repeat=repeat,method=str(variant),
                        signal=bool(meta[o]['signal']),seconds=elapsed,feature_seconds=feature_seconds,
                        stopped=stop,detected=answer['detected']))
                if (j+1)%19==0:print(name,'repeat',repeat+1,'captures',j+1,flush=True)
            write(OUT/'records.json',rows)
        del bank;gc.collect()
    stats=[]
    for variant in VARIANTS:
        subset=[r for r in rows if r['method']==variant]
        stats.append(dict(method=variant,n=len(subset),mean_ms=1000*np.mean([r['seconds'] for r in subset]),
            median_ms=1000*np.median([r['seconds'] for r in subset]),
            feature_mean_ms=1000*np.mean([r['feature_seconds'] for r in subset]),
            continued_per_repeat=sum(not r['stopped'] for r in subset)//2))
    write(OUT/'summary.json',dict(methods=stats,captures=304,repeats=2,
        max_feature_gap=max(r['feature_gap'] for r in checks),all_decisions_match=True,
        all_continuation_detection_events_match=True,threadpools=threadpool_info(),
        scope='targeted paired timing subset; retention totals are in feature_block_ablation.csv'))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
