"""Independent waveform/whitening/coordinate/window audit before readout edits."""
import gc
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from campaign_fhboc_large_20261017 import BASE, BULK, ROOT as SOURCE, read, write, sha, verify_main
from benchmark_fhboc_field_stop_20261016 import (
    OnlineBank, path_scores, state_features, angular_log_bf, standardized_features,
)

OUT = BASE/'results/diagnostics/fhboc_receiver_audit_20261018'


def localization_field(bank, o, b):
    d = bank.d; E = d['observation_energy'][:,o]
    ps = path_scores(b, bank.q, E, bank.D)
    packet = {k:d[k] for k in ('cfg','axes','aiding','template_energy','states')}
    packet.update(projected=b[:,:,None], observation_energy=E[:,None], dimensions=bank.D,
                  path_scores=ps[:,None,:])
    x = standardized_features(state_features(packet,angular_log_bf(ps[:,2,None],bank.D.sum()))[0],bank.model)
    ll = bank.prior+bank.checkpoint.bf(ps[:,2])+x@np.asarray(bank.model['beta_localization'])
    return ll, ps


def run(c,g):
    name=f'c{c}_g{g}'; target=OUT/f'{name}_piml_audit.json'
    if target.exists(): return read(target)
    direct=loadmat(OUT/f'{name}_direct.mat',simplify_cells=True)
    bank=OnlineBank(BULK/name);d=bank.d
    raw=loadmat(BULK/name/'frontend.mat',simplify_cells=True)
    metadata=list(d['metadata']);seed=next(int(m['noise_seed']) for m in metadata if m['signal'])
    indices={int(m['cn0_db_hz']):i for i,m in enumerate(metadata) if m['signal'] and int(m['noise_seed'])==seed}
    a,b=indices[30],indices[38];ratio=10**(-8/20)
    raw_errors=[];white_errors=[];affine_errors=[];energy_errors=[]
    for k,z in enumerate(bank.bank['whitened_observations']):
        v=raw['observations'][k]
        clean=(v[:,b]-ratio*v[:,a])/(1-ratio)
        clean_white=(z[:,b]-ratio*z[:,a])/(1-ratio)
        raw_errors.append(float(np.linalg.norm(clean-direct['clean_pmf'][k])/np.linalg.norm(direct['clean_pmf'][k])))
        white_errors.append(float(np.linalg.norm(clean_white-direct['clean_whitened'][k])/np.linalg.norm(direct['clean_whitened'][k])))
        for cn,i in indices.items():
            predicted=clean+10**((30-cn)/20)*(v[:,a]-clean)
            affine_errors.append(float(np.linalg.norm(v[:,i]-predicted)/np.linalg.norm(v[:,i])))
        energy_errors.append(float(abs(np.vdot(z[:,a],z[:,a]).real-d['observation_energy'][k,a])/d['observation_energy'][k,a]))
        z[:,a]=np.asarray(direct['clean_whitened'][k])
        d['observation_energy'][k,a]=np.vdot(z[:,a],z[:,a]).real
    assert max(raw_errors)<1e-8 and max(white_errors)<1e-8
    assert max(affine_errors)<1e-10 and max(energy_errors)<1e-10
    np.testing.assert_allclose(d['truth_reference'],direct['truth_direct'],atol=1e-10,rtol=0)
    s=d['states'];t=.0025;eta=-d['cfg']['signal']['fs']/d['cfg']['signal']['rf_carrier_hz']
    initial_fd=bank.fd-s[:,2]*t
    initial_tau=bank.tau-eta*(initial_fd*t+.5*s[:,2]*t*t)
    np.testing.assert_allclose(initial_fd,s[:,0],atol=1e-8,rtol=0)
    np.testing.assert_allclose(initial_tau,s[:,1],atol=1e-8,rtol=0)
    b=bank.projection(a,64)[0];ll,ps=localization_field(bank,a,b)
    alpha=2.1*np.exp(.73j);rotated=path_scores(b.astype(complex)*np.exp(-.73j),bank.q,
                                          d['observation_energy'][:,a]*abs(alpha)**2,bank.D)
    np.testing.assert_allclose(rotated,ps,atol=1e-6,rtol=1e-10)
    weights=np.exp(ll-ll.max());mass=bank.plan.mass(weights)
    old=int(np.argmax(mass));ties=np.flatnonzero(mass==mass.max());peak=int(np.argmax(ll))
    # Candidate is evaluated only after independent operator/coordinate gates.
    candidate=int(ties[np.argmax(ll[ties])])
    queries=np.unique(np.r_[old,peak,candidate,ties[np.linspace(0,len(ties)-1,min(30,len(ties)),dtype=int)],np.linspace(0,len(s)-1,64,dtype=int)])
    brute=np.asarray([weights[(abs(bank.fd-bank.fd[i])<=250)&(abs(bank.tau-bank.tau[i])<=1)].sum() for i in queries])
    np.testing.assert_allclose(mass[queries],brute,atol=1e-11*weights.sum(),rtol=1e-12)
    frozen=bank.readout(a,b)
    assert frozen['fd_hat_hz']==bank.fd[old] and frozen['tau_hat_samples']==bank.tau[old]
    truth=d['truth_reference']
    def label(i):
        df=float(bank.fd[i]-truth[0]);dt=float(bank.tau[i]-truth[1])
        return dict(fd_error_hz=df,tau_error_samples=dt,localized=abs(df)<=250 and abs(dt)<=1,
                    mass=float(mass[i]),state_index=int(i))
    result=dict(complete=True,geometry=name,raw_clean_max_relative_error=max(raw_errors),
                whitened_clean_max_relative_error=max(white_errors),pairing_affine_max_relative_error=max(affine_errors),
                recorded_energy_max_relative_error=max(energy_errors),
                window_direct_sum_max_absolute_error=float(np.max(abs(mass[queries]-brute))),
                coordinate_forward_inverse_passed=True,scale_phase_invariance_passed=True,
                exact_maximum_mass_ties=len(ties),frozen_window=label(old),MAP=label(peak),
                lexicographic_candidate=label(candidate),all_implementation_gates_passed=True,
                source_sha256=sha(Path(__file__)))
    write(target,result);print('PIML AUDIT',name,'ties',len(ties),'old',label(old),'candidate',label(candidate),flush=True)
    return result


def main():
    verify_main();results=[]
    with threadpool_limits(limits=2):
        for c in (1,2):
            for g in range(1,10):
                results.append(run(c,g));gc.collect()
    write(OUT/'piml_audit_complete.json',dict(complete=True,geometries=18,
          implementation_gates_passed=all(r['all_implementation_gates_passed'] for r in results),
          old_correct=sum(r['frozen_window']['localized'] for r in results),
          MAP_correct=sum(r['MAP']['localized'] for r in results),
          candidate_correct=sum(r['lexicographic_candidate']['localized'] for r in results),
          raw_clean_max_relative_error=max(r['raw_clean_max_relative_error'] for r in results),
          whitened_clean_max_relative_error=max(r['whitened_clean_max_relative_error'] for r in results),
          window_direct_sum_max_absolute_error=max(r['window_direct_sum_max_absolute_error'] for r in results),
          frozen_source_files_unchanged=True,source_sha256=sha(Path(__file__))))
    verify_main()


if __name__=='__main__':main()
