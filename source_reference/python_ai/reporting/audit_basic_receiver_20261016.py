"""Independent BASIC algebra checks and controlled PMF aperture replay.

The frozen 15 October implementation and all of its results remain intact.
Short apertures are a receiver-adapter sensitivity study of the same records.
Truth is used only for diagnostics and output error labels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftshift, fftfreq
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from fhboc_fair_baselines_20261015 import axis, complex_delay_interpolate, basic_from_input
from basic_yang2008 import estimate

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE/'results/diagnostics/fhboc_fair_baseline_comparison_20261015'
OUT = BASE/'results/diagnostics/fhboc_basic_audit_20261016'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')


class BlockAdapter:
    """Eq. 8 equal-position block products; exact known-time frequency DFT.

    One frame is one block. J=4 retains four 16-segment subcorrelations;
    J=1 is a matched coarse-grid control. Both use the same 500 Hz coarse
    grid, rate FFT, residual-frequency grid, initial bounds and calibration.
    The 32 missing raw samples per frame are retained in the time coordinates.
    """
    def __init__(self, frontend, frames, pieces=4):
        self.f=frontend; self.M=frames; self.J=pieces
        f=frontend; fs=f['sample_rate_hz']; L=f['pmf_segment_length']; P=int(f['pmf_segments'])
        assert P%pieces==0
        self.tref=(frames-1)*.001/2; self.eta=-fs/f['rf_carrier_hz']
        self.tau=axis(f['aiding_tau_samples']-f['tau_half_width_samples'],f['aiding_tau_samples']+f['tau_half_width_samples'],.5)
        envelope=2500+250000*(self.tref+.0005)+250
        self.fc=f['aiding_fd_hz']+np.arange(-np.floor(envelope/500)*500,np.floor(envelope/500)*500+1,500)
        self.mid=(np.arange(pieces)+.5)*(P//pieces)*L/fs-.5/fs
        self.t=np.arange(frames)[:,None]*.001+self.mid[None,:]
        self.ra=fftshift(fftfreq(2048,.001))/.001
        self.allowed=abs(self.ra)<=250000
        self.fr=fftshift(fftfreq(2048,.001)); self.fi=np.where(abs(self.fr)<=250)[0]
        self.basis=[]
        for j in range(pieces):
            ids=np.arange(j*(P//pieces),(j+1)*(P//pieces))
            self.basis.append(np.exp(-2j*np.pi*(self.fc-f['aiding_fd_hz'])[:,None]*(ids*L/fs+(L-1)/(2*fs))[None,:]))

    def samples(self, pmf):
        f=self.f; P=int(f['pmf_segments']); out=[]
        for k in range(self.M):
            v=np.asarray(pmf[k]).reshape(P,-1,order='F')
            v=complex_delay_interpolate(v,f['frame_tau_axes'][:,k],self.tau+self.eta*f['aiding_fd_hz']*k*.001)
            a=np.asarray([b@vv for b,vv in zip(self.basis,np.array_split(v,self.J))])
            a*=np.exp(-2j*np.pi*self.fc*k*.001)[None,:,None]
            out.append(a)
        return np.asarray(out).reshape(self.M,self.J,-1)

    def infer(self, pmf, fixed_rate=None):
        x=self.samples(pmf); n=x.shape[-1]; nt=len(self.tau)
        coarse=np.repeat(self.fc,nt); delay=np.tile(self.tau+self.eta*self.f['aiding_fd_hz']*self.tref,len(self.fc))
        powers=np.empty(n); rates=np.empty(n); residual=np.empty(n)
        for first in range(0,n,128):
            last=min(first+128,n); xx=x[:,:,first:last]
            d=np.sum(np.conj(xx[:-1])*xx[1:],axis=1)
            a=fftshift(fft(d,n=2048,axis=0),axes=0)
            r=self.ra[np.argmax(np.where(self.allowed[:,None],abs(a),-np.inf),axis=0)]
            if fixed_rate is not None:r=np.full(last-first,fixed_rate)
            corrected=xx*np.exp(-1j*np.pi*(self.t[:,:,None]-self.tref)**2*r[None,None,:])
            spectrum=fftshift(fft(corrected,n=2048,axis=0),axes=0)[self.fi]
            # Each subcorrelation has its own aperture midpoint. This is the
            # exact DFT at the actual sample times, including interframe gaps.
            spectrum=np.sum(spectrum*np.exp(-2j*np.pi*self.fr[self.fi,None,None]*self.mid[None,:,None]),axis=1)
            freq=self.fr[self.fi,None]; f0=coarse[None,first:last]+freq-r[None,:]*self.tref
            tau0=delay[None,first:last]-self.eta*(f0*self.tref+.5*r[None,:]*self.tref**2)
            mask=(abs(f0-self.f['aiding_fd_hz'])<=2500)&(abs(tau0-self.f['aiding_tau_samples'])<=self.f['tau_half_width_samples']+1e-10)
            idx=np.argmax(np.where(mask,abs(spectrum),-np.inf),axis=0); ii=np.arange(last-first)
            powers[first:last]=np.where(mask.any(axis=0),abs(spectrum[idx,ii])**2,-np.inf)
            rates[first:last]=r; residual[first:last]=self.fr[self.fi[idx]]
        winner=int(np.argmax(powers)); dc=winner%nt; code=powers.reshape(len(self.fc),nt).max(axis=0); fraction=0.
        if 0<dc<nt-1:
            left,center,right=code[dc-1:dc+2]; curve=left-2*center+right
            if np.isfinite(curve) and curve<0:fraction=float(np.clip(.5*(left-right)/curve,-.5,.5))
        fd=float(coarse[winner]+residual[winner]); rate=float(rates[winner]); tau=float(delay[winner]+fraction*.5)
        f0=fd-rate*self.tref; initial=tau-self.eta*(f0*self.tref+.5*rate*self.tref**2)
        initial=np.clip(initial,self.f['aiding_tau_samples']-self.f['tau_half_width_samples'],self.f['aiding_tau_samples']+self.f['tau_half_width_samples'])
        tau=float(initial+self.eta*(f0*self.tref+.5*rate*self.tref**2))
        return dict(score=float(powers[winner]/np.mean(np.sum(abs(x)**2,axis=(0,1)))),fd_hat_hz=fd,tau_hat_samples=tau,rate_hat_hz_s=rate,state_index=winner)


def algebra():
    checks=[]
    # Chosen exactly on the algorithm's FFT grids, independent direct sums.
    for M in (6,10):
        for rate in (-200195.3125,0.,200195.3125):
            dt=.001; t=np.arange(M)*dt; freq=125.
            x=np.exp(2j*np.pi*(freq*t+.5*rate*t*t)+.37j)
            pred=estimate(x,dt,1,rate_nfft=2048,frequency_nfft=2048,rate_bounds_hz_s=(-250000,250000),known_constant_bits=True)
            rate_err=abs(pred['rate_hat_hz_s']-rate); frequency_err=abs(pred['f_hat_hz']-freq)
            assert rate_err<1e-8 and frequency_err<1e-8
            checks.append(dict(frames=M,rate=rate,rate_error=rate_err,frequency_error=frequency_err))
    # Independent explicit summations verify FFT indexing and conjugation.
    x=np.array([1+.2j,-.4+.8j,.7-.1j,-.3-.6j,1.1j,.5+.4j]); dt=.001
    rates=fftshift(fftfreq(2048,dt))/dt; d=np.conj(x[:-1])*x[1:]
    direct=np.asarray([sum(d[k]*np.exp(-2j*np.pi*r*dt*dt*k) for k in range(len(d))) for r in rates])
    fast=fftshift(fft(d,n=2048)); gap=float(np.max(abs(direct-fast)));assert gap<1e-11
    # Independently sum at all actual subaperture timestamps. This checks the
    # frequency stage, especially the nonuniform gap at each frame boundary.
    t=np.arange(6)[:,None]*.001+(np.arange(4)[None,:]+.5)*.00024992-.5e-8
    xx=np.exp(1j*np.arange(24).reshape(6,4)*.71)+.2j
    r=200230.; u=xx*np.exp(-1j*np.pi*r*(t-.0025)**2)
    frequencies=fftshift(fftfreq(2048,.001)); ids=np.array([512,800,1024,1500])
    fast_frequency=np.sum(fftshift(fft(u,n=2048,axis=0),axes=0)[ids]*np.exp(-2j*np.pi*frequencies[ids,None]*t[0,None,:]),axis=1)
    direct_frequency=np.array([np.sum(u*np.exp(-2j*np.pi*f*t)) for f in frequencies[ids]])
    frequency_gap=float(np.max(abs(fast_frequency-direct_frequency)));assert frequency_gap<1e-10
    return dict(ideal_chirp_checks=checks,direct_rate_DFT_max_gap=gap,direct_actual_time_frequency_DFT_max_gap=frequency_gap,passed=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol.json',dict(date='2026-10-16',conditions=[2],geometries=[1,2],frames=[6,10],cn0_db_hz=32,
        adapters={'BASIC-1ms-500Hz':1,'BASIC-quarter-ms-500Hz':4},coarse_step_hz=500,rate_fft=2048,frequency_fft=2048,
        calibration='maximum of 50 calibration H0 per geometry, strict greater than',
        purpose='Controlled retrospective adapter sensitivity, identical saved PMF; existing holdout records reused',
        selection='Apertures fixed from the 1ms sinc model; no coefficient fitting',new_raw_observations=0))
    result=algebra(); clean=[]; all_rows=[]
    for gi in (1,2):
        d=loadmat(SOURCE/f'condition_2_geometry_{gi}'/'shared_frontend.mat',simplify_cells=True); f=d['frontend']
        for M in (6,10):
            truth=np.asarray(d['evaluation']['truth_six' if M==6 else 'truth_ten']); true0=np.asarray(d['evaluation']['truth_initial'])
            for name,J in [('BASIC-1ms-500Hz',1),('BASIC-quarter-ms-500Hz',4)]:
                plan=BlockAdapter(f,M,J); pred=plan.infer(list(d['clean_observations']))
                x=plan.samples(list(d['clean_observations']))[:,:,pred['state_index']]
                fc=np.repeat(plan.fc,len(plan.tau))[pred['state_index']]
                residual=x*np.exp(-2j*np.pi*((true0[0]-fc)*plan.t+.5*true0[2]*plan.t**2))
                energy=abs(x)**2
                clean.append(dict(geometry=gi,frames=M,adapter=name,fd_error_hz=float(pred['fd_hat_hz']-truth[0]),
                    tau_error_samples=float(pred['tau_hat_samples']-truth[1]),rate_error_hz_s=float(pred['rate_hat_hz_s']-true0[2]),
                    energy_participation=float(energy.sum()**2/(energy**2).sum()),coherence=float(abs(residual.sum())**2/(M*J*energy.sum())),
                    relative_amplitudes=(abs(x)/abs(x).max()).tolist()))
                # J=1 generated time-domain branch equals the legacy adapter
                # fed the identical exact coarse bank, including phase epochs.
                if J==1:
                    P=int(f['pmf_segments']); L=f['pmf_segment_length']; fs=f['sample_rate_hz']
                    phase=plan.fc[None,:]*np.arange(M)[:,None]*.001+(plan.fc-f['aiding_fd_hz'])[None,:]*(L-1)/(2*fs)
                    o=dict(complex_surface=plan.samples(list(d['clean_observations']))[:,0].reshape(M,len(plan.fc),-1)*np.exp(2j*np.pi*phase[:,:,None]),
                        frame_start_s=np.arange(M)*.001,coarse_frequency_hz=plan.fc,reference_delay_axis_samples=plan.tau+plan.eta*f['aiding_fd_hz']*plan.tref,
                        sample_rate_hz=fs,pmf_segment_length=L,pmf_segments=P,frame_duration_s=.001,reference_time_s=plan.tref,known_constant_bits=True,
                        rate_half_width_hz_s=250000,post_correlation_frequency_half_width_hz=250,frequency_half_width_hz=2500,
                        aiding_fd_hz=f['aiding_fd_hz'],aiding_tau_samples=f['aiding_tau_samples'],tau_half_width_samples=f['tau_half_width_samples'],rf_carrier_hz=f['rf_carrier_hz'])
                    old=basic_from_input(o)
                    assert abs(old['score']-pred['score'])<1e-8 and abs(old['fd_hat_hz']-pred['fd_hat_hz'])<1e-8
                path=OUT/f'geometry_{gi}_{M}_{J}.json'
                if path.exists():all_rows+=json.loads(path.read_text(encoding='utf8'));continue
                rows=[]; start=time.perf_counter()
                for oi,meta in enumerate(d['evaluation']['metadata']):
                    if meta['role']=='development_H0':continue
                    pmf=[d['observations'][k,oi] for k in range(M)]
                    p=plan.infer(pmf); signal=bool(meta['signal'])
                    rows.append(dict(geometry=gi,frames=M,adapter=name,role=meta['role'],record_index=int(meta['record_index']),**p,
                        signal=signal,fd_error_hz=float(p['fd_hat_hz']-truth[0]) if signal else None,
                        tau_error_samples=float(p['tau_hat_samples']-truth[1]) if signal else None,
                        rate_error_hz_s=float(p['rate_hat_hz_s']-true0[2]) if signal else None,
                        localized=bool(signal and abs(p['fd_hat_hz']-truth[0])<=250 and abs(p['tau_hat_samples']-truth[1])<=1)))
                    if len(rows)%50==0:print(gi,M,J,len(rows),round(time.perf_counter()-start,1),flush=True)
                save(path,rows);all_rows+=rows
    summaries=[]
    for M in (6,10):
        for name in ('BASIC-1ms-500Hz','BASIC-quarter-ms-500Hz'):
            counts=dict(H1=0,detected=0,handoff=0,localized=0,H0=0,false_alarms=0)
            for gi in (1,2):
                rows=[r for r in all_rows if r['geometry']==gi and r['frames']==M and r['adapter']==name]
                gate=max(r['score'] for r in rows if r['role']=='calibration_H0')
                for r in rows:
                    if r['signal']:
                        counts['H1']+=1;counts['detected']+=r['score']>gate;counts['handoff']+=r['score']>gate and r['localized'];counts['localized']+=r['localized']
                    elif r['role']=='holdout_H0':counts['H0']+=1;counts['false_alarms']+=r['score']>gate
            summaries.append(dict(frames=M,adapter=name,**counts,Pd=counts['detected']/counts['H1'],Ph=counts['handoff']/counts['H1'],Pfa=counts['false_alarms']/counts['H0']))
    save(OUT/'audit_results.json',dict(algebra=result,clean_checks=clean,summary=summaries));print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
