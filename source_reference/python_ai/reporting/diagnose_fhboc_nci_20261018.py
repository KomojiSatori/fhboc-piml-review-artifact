"""Paired NCI mechanism controls on retained captures; no new simulation/fitting.

Four geometries are fixed by index: c1/c2 x g1/g2. All 400 calibration H0,
800 test H0 and all 80 H1 at 32/34/36 dB-Hz are used in each geometry.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftshift, fftfreq
from scipy.io import loadmat
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from campaign_fhboc_large_20261017 import BASE, ROOT as SOURCE, BULK, read, write, sha, quantile_gate, verify_main
from fhboc_fair_baselines_20261015 import TrajectoryNCI

OUT = BASE / 'results/diagnostics/fhboc_receiver_audit_20261018'
METHODS = ['static-original', 'static-maximum'] + [
    f'trajectory-{domain}-{normalization}-{rule}'
    for domain in ('zero', 'full') for normalization in ('global', 'local')
    for rule in ('maximum', 'marginal')]


def glrt(energy, frames=6):
    active = energy > frames
    result = np.zeros_like(energy)
    result[active] = energy[active]-frames-frames*np.log(energy[active]/frames)
    return result


def run(c, g):
    name = f'c{c}_g{g}'
    target = OUT / f'{name}_nci.json'
    if target.exists():
        print('REUSE', name, flush=True)
        return read(target)
    d = loadmat(BULK/name/'geometry.mat', simplify_cells=True)
    raw = loadmat(BULK/name/'frontend.mat', simplify_cells=True)
    f = raw['frontend']; meta = list(d['metadata']); P = int(f['pmf_segments'])
    trajectory = TrajectoryNCI(f, 6, nfft=4096)
    np.testing.assert_allclose(trajectory.states, d['states'], atol=1e-8, rtol=0)
    states = trajectory.states; zero = np.flatnonzero(states[:, 2] == 0)
    prior = np.asarray(d['log_aided'], float)
    eta = -f['sample_rate_hz']/f['rf_carrier_hz']; t = .0025
    fd = states[:, 0]+states[:, 2]*t
    tau = states[:, 1]+eta*(states[:, 0]*t+.5*states[:, 2]*t*t)
    grid = f['aiding_fd_hz']+fftshift(fftfreq(1024, f['pmf_segment_length']/f['sample_rate_hz']))
    mask = abs(grid-f['aiding_fd_hz']) <= 2500
    ar_tau = f['aiding_tau_samples']+eta*f['aiding_fd_hz']*t
    delays = np.floor(ar_tau+.5)+np.arange(-25, 26)
    keep = abs(delays-ar_tau) <= f['tau_half_width_samples']
    sf, st = np.meshgrid(grid[mask], delays[keep], indexing='ij')
    static_prior = -.5*((sf-f['aiding_fd_hz'])/1000)**2-.5*((st-ar_tau)/(.5*f['sample_rate_hz']/f['code_rate_hz']))**2
    static_prior -= logsumexp(static_prior)
    selected = [i for i, m in enumerate(meta) if not m['signal'] or int(m['cn0_db_hz']) in (32, 34, 36)]
    archived = {}
    for line in (SOURCE/name/'events.jsonl').open(encoding='utf8'):
        e = json.loads(line)
        if e['method'] == 'NCI-compensated': archived[e['observation']-1] = e
    calibration = read(SOURCE/name/'calibration.json')
    archived_cal = {e['observation']: e['answer'] for e in calibration['records'] if e['method'] == 'NCI-compensated'}
    rows = []; static_gaps = []; comp_gaps = []; dft_gaps = []; normalized_ratios = []
    for count, o in enumerate(selected):
        accum = np.zeros(sf.shape); total = np.zeros(len(states)); local = total.copy()
        scales = []
        for k in range(6):
            values = np.asarray(raw['observations'][k])[:, o].reshape(P, -1, order='F')
            spectrum = fftshift(fft(values, n=4096, axis=0), axes=0)
            power = abs(spectrum)**2
            pk = power[::4][mask]
            local_scale = pk.mean(); global_scale = power.mean()
            accum += (pk/local_scale)[:, keep]
            left, right = trajectory.bands[k]
            x, y, wx, wy = trajectory.paths[k]
            p = power[left:right]
            interpolation = ((1-wx)*((1-wy)*p[x, y]+wy*p[x, y+1])
                             + wx*((1-wy)*p[x+1, y]+wy*p[x+1, y+1]))
            total += interpolation/global_scale
            local += interpolation/local_scale
            scales.append(local_scale/global_scale)
            if count == 0:
                # Direct DFT checks bin sign, zero padding, normalization and
                # the four power-interpolation corners independently of FFT.
                ids = np.linspace(0, len(states)-1, 12, dtype=int)
                for i in ids:
                    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        j = left+x[i]+dx; col = y[i]+dy
                        direct = np.exp(-2j*np.pi*((j-2048)/4096)*np.arange(P)) @ values[:, col]
                        dft_gaps.append(float(abs(direct-spectrum[j, col])/max(abs(direct), 1)))
        normalized_ratios.append(dict(signal=bool(meta[o]['signal']), cn0=float(meta[o]['cn0_db_hz']) if meta[o]['signal'] else None,
                                      mean_local_to_global_scale=float(np.mean(scales))))
        answer = {}
        post = glrt(accum)+static_prior
        i = np.unravel_index(np.argmax(post), post.shape)
        answer['static-original'] = (float(logsumexp(post)), float(sf[i]), float(st[i]), 0.)
        i = np.unravel_index(np.argmax(accum), accum.shape)
        answer['static-maximum'] = (float(accum[i]), float(sf[i]), float(st[i]), 0.)
        static_gaps.append(abs(answer['static-original'][0]-raw['nci_scores'][o]))
        np.testing.assert_allclose(answer['static-original'][1:3], raw['nci_estimates'][o], atol=1e-8, rtol=0)
        for domain, ids in (('zero', zero), ('full', np.arange(len(states)))):
            lp = prior[ids]-logsumexp(prior[ids])
            for normalization, energy in (('global', total), ('local', local)):
                e = energy[ids]; posterior = glrt(e)+lp
                for rule, score, j in (('maximum', float(e.max()), int(np.argmax(e))),
                                       ('marginal', float(logsumexp(posterior)), int(np.argmax(posterior)))):
                    index = ids[j]
                    answer[f'trajectory-{domain}-{normalization}-{rule}'] = (score, float(fd[index]), float(tau[index]), float(states[index, 2]))
        frozen = archived[o] if o in archived else archived_cal[o]
        comp_gaps.append(abs(answer['trajectory-full-global-maximum'][0]-frozen['score']))
        if o in archived:
            truth = d['truth_reference']
            np.testing.assert_allclose([answer['trajectory-full-global-maximum'][1]-truth[0], answer['trajectory-full-global-maximum'][2]-truth[1]],
                                       [frozen['fd_error_hz'], frozen['tau_error_samples']] if meta[o]['signal'] else [answer['trajectory-full-global-maximum'][1]-truth[0], answer['trajectory-full-global-maximum'][2]-truth[1]], atol=1e-8, rtol=0)
        for m, (score, fh, th, rate) in answer.items():
            rows.append(dict(condition=c, geometry=g, observation=o+1, method=m, role=meta[o]['role'],
                             cn0=int(meta[o]['cn0_db_hz']) if meta[o]['signal'] else None,
                             score=score, fd_error_hz=float(fh-d['truth_reference'][0]),
                             tau_error_samples=float(th-d['truth_reference'][1]), rate_hat_hz_s=rate))
        if (count+1) % 240 == 0: print('NCI AUDIT', name, count+1, '/', len(selected), flush=True)
    assert max(static_gaps) < 1e-9 and max(comp_gaps) < 1e-9 and max(dft_gaps) < 1e-9
    summary = []
    for m in METHODS:
        rr = [r for r in rows if r['method'] == m]
        gate = quantile_gate([r['score'] for r in rr if r['role'] == 'calibration_H0'])
        for r in rr:
            r['threshold'] = gate; r['detected'] = bool(r['score'] > gate)
            r['handoff'] = bool(r['role'] == 'H1' and r['detected'] and abs(r['fd_error_hz']) <= 250 and abs(r['tau_error_samples']) <= 1)
        for cn in (None, 32, 34, 36):
            rr2 = [r for r in rr if (r['role'] == 'holdout_H0' if cn is None else r['role'] == 'H1' and r['cn0'] == cn)]
            summary.append(dict(condition=c, geometry=g, method=m, cn0=cn, n=len(rr2),
                                detected=sum(r['detected'] for r in rr2), handoff=sum(r['handoff'] for r in rr2), threshold=gate))
    with (OUT/f'{name}_nci_events.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(complete=True, geometry=name, observations=len(selected), summary=summary,
                  static_score_max_gap=max(static_gaps), compensated_score_max_gap=max(comp_gaps),
                  independent_DFT_max_relative_gap=max(dft_gaps),
                  mean_normalization_ratio_by_cn0={str(cn):float(np.mean([r['mean_local_to_global_scale'] for r in normalized_ratios if r['cn0']==cn])) for cn in (None, 32, 34, 36)},
                  source_sha256=sha(Path(__file__)))
    write(target, result); print('NCI COMPLETE', name, flush=True)
    return result


def main():
    verify_main(); OUT.mkdir(exist_ok=True)
    write(OUT/'nci_protocol.json', dict(geometries=['c1_g1','c1_g2','c2_g1','c2_g2'], strengths=[32,34,36],
          calibration_H0_per_geometry=400, test_H0_per_geometry=800, H1_per_strength_per_geometry=80,
          methods=METHODS, new_observations=0, diagnostic_only=True,
          selection='First two nominal geometries per dynamic; all records at preselected strengths.',
          zero_rate='A zero-rate trajectory still propagates code delay with candidate initial Doppler.',
          global_normalization='Full 4096-bin by 51-delay FFT power mean, matching frozen compensated NCI.',
          local_normalization='Original 1024 FFT bins within aided +/-2500 Hz, all 51 delays, matching frozen static NCI.',
          marginal='Same six-frame scale-GLRT transform and normalized aided state prior in both rate domains.',
          source_sha256=sha(Path(__file__))))
    with threadpool_limits(limits=2):
        results = [run(c,g) for c,g in ((1,1),(1,2),(2,1),(2,2))]
    rows = [r for result in results for r in result['summary']]
    with (OUT/'nci_summary.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'nci_complete.json',dict(complete=True, geometries=4, observations=5760,
          new_observations=0, source_sha256=sha(Path(__file__))))
    verify_main()


if __name__ == '__main__': main()
