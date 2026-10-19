"""Observation-only trajectory NCI and Yang BASIC receiver adapter.

Both consume the same complex PMF segments, aided initial-state bounds and pilot.
No signal labels, truth, strength or noise seeds enter either detector.
"""
import numpy as np
from scipy.fft import fft,fftshift,fftfreq
from basic_yang2008 import pilot_pmf_bank_estimate


def axis(low,high,step):
    return np.unique(np.r_[low,np.arange(np.ceil(low/step)*step,high,step),high])


def state_grid(frontend,frames):
    f=frontend;a=float(f['aiding_fd_hz']);half=float(f['frequency_half_width_hz'])
    if frames==6:
        fd=a+np.arange(-half,half+1,62.5);rate=np.linspace(-250000,250000,21)
    else:
        fd=a+np.unique(np.r_[-half,np.arange(-2475,2475+1,37.5),half]);rate=np.linspace(-250000,250000,61)
    tau=axis(f['aiding_tau_samples']-f['tau_half_width_samples'],f['aiding_tau_samples']+f['tau_half_width_samples'],.5)
    # Match MATLAB ndgrid(:) order, including its deterministic tie convention.
    F,T,R=np.meshgrid(fd,tau,rate,indexing='ij')
    return np.column_stack([x.ravel(order='F') for x in (F,T,R)]),tau


def complex_delay_interpolate(values,source,target):
    """Linear interpolation of raw complex PMF columns, retaining the PMF data."""
    index=(np.asarray(target)-source[0])/(source[1]-source[0]);lo=np.floor(index).astype(int);w=index-lo
    if np.any(lo<0) or np.any(lo+1>=len(source)):raise ValueError('Insufficient shared delay halo')
    return values[:,lo]*(1-w)+values[:,lo+1]*w


class TrajectoryNCI:
    """Sum of per-frame normalized FFT power along each complete aided path.

    Frequency is the instantaneous value at the PMF aperture midpoint. Code is
    propagated to frame start, matching the shared PMF/dewarp convention.
    FFT zero padding reduces interpolation loss; no cross-frame phase is used.
    """
    def __init__(self,frontend,frames,*,nfft=4096):
        self.f=frontend;self.M=frames;self.nfft=nfft;f=frontend
        self.states,_=state_grid(f,frames);s=self.states;self.tref=(frames-1)*.001/2
        eta=-f['sample_rate_hz']/f['rf_carrier_hz'];fs=f['sample_rate_hz'];L=f['pmf_segment_length'];P=int(f['pmf_segments'])
        mid=(P*L-1)/(2*fs);grid=fftshift(fftfreq(nfft,L/fs));self.paths=[];self.bands=[]
        for k in range(frames):
            t=k*.001;fd=s[:,0]+s[:,2]*(t+mid);tau=s[:,1]+eta*(s[:,0]*t+.5*s[:,2]*t*t)
            absolute=f['frame_centers'][k]+grid
            left=max(0,np.searchsorted(absolute,fd.min())-1);right=min(nfft,np.searchsorted(absolute,fd.max())+2)
            gf=absolute[left:right];gt=f['frame_tau_axes'][:,k]
            x=(fd-gf[0])/(gf[1]-gf[0]);y=tau-gt[0];ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
            assert np.all((ix>=0)&(ix+1<len(gf))&(iy>=0)&(iy+1<len(gt)))
            self.paths.append((ix,iy,x-ix,y-iy));self.bands.append((left,right))
    def infer(self,pmf):
        scores=np.zeros(len(self.states));P=int(self.f['pmf_segments'])
        for k in range(self.M):
            values=np.asarray(pmf[k]).reshape(P,-1,order='F');power=abs(fftshift(fft(values,n=self.nfft,axis=0),axes=0))**2
            scale=float(power.mean());assert scale>0
            left,right=self.bands[k];p=power[left:right]/scale;x,y,wx,wy=self.paths[k]
            scores+=(1-wx)*((1-wy)*p[x,y]+wy*p[x,y+1])+wx*((1-wy)*p[x+1,y]+wy*p[x+1,y+1])
        i=int(np.argmax(scores));s=self.states[i];t=self.tref;eta=-self.f['sample_rate_hz']/self.f['rf_carrier_hz']
        return dict(score=float(scores[i]),fd_hat_hz=float(s[0]+s[2]*t),
            tau_hat_samples=float(s[1]+eta*(s[0]*t+.5*s[2]*t*t)),rate_hat_hz_s=float(s[2]),state_index=i)


def basic_input(pmf,frontend,frames):
    f=frontend;P=int(f['pmf_segments']);fs=f['sample_rate_hz'];L=f['pmf_segment_length'];tref=(frames-1)*.001/2
    tau0=axis(f['aiding_tau_samples']-f['tau_half_width_samples'],f['aiding_tau_samples']+f['tau_half_width_samples'],.5)
    eta=-fs/f['rf_carrier_hz'];aid=f['aiding_fd_hz']
    centers=aid+fftshift(fftfreq(1024,L/fs));envelope=2500+250000*(tref+.0005)+250
    mask=abs(centers-aid)<=envelope;surfaces=[]
    for k in range(frames):
        v=np.asarray(pmf[k]).reshape(P,-1,order='F')
        aligned=complex_delay_interpolate(v,f['frame_tau_axes'][:,k],tau0+eta*aid*k*.001)
        surfaces.append(fftshift(fft(aligned,n=1024,axis=0),axes=0)[mask])
    return dict(complex_surface=np.asarray(surfaces),frame_start_s=np.arange(frames)*.001,
        coarse_frequency_hz=centers[mask],reference_delay_axis_samples=tau0+eta*aid*tref,
        initial_delay_axis_samples=tau0,aiding_tau_samples=f['aiding_tau_samples'],tau_half_width_samples=f['tau_half_width_samples'],
        rf_carrier_hz=f['rf_carrier_hz'],aiding_fd_hz=aid,pmf_segment_length=L,sample_rate_hz=fs,pmf_segments=P,
        frame_duration_s=.001,known_constant_bits=True,rate_half_width_hz_s=250000,reference_time_s=tref,
        post_correlation_frequency_half_width_hz=250,frequency_half_width_hz=2500)


def basic_estimate(pmf,frontend,frames):
    return basic_from_input(basic_input(pmf,frontend,frames))


def basic_from_input(o,*,rate_nfft=2048,frequency_nfft=2048):
    """Same BASIC equations as frozen kernel; apply the initial code-domain mask.

    Intermediate FFTs are evaluated in column chunks, preserving every branch.
    Delay interpolation belongs to the shared receiver adapter, not BASIC core.
    """
    surf=np.asarray(o['complex_surface'],complex);M,nf,nt=surf.shape
    starts=np.asarray(o['frame_start_s']);centers=np.asarray(o['coarse_frequency_hz']);delays=np.asarray(o['reference_delay_axis_samples'])
    fs=o['sample_rate_hz'];L=o['pmf_segment_length'];dt=o['frame_duration_s'];tref=o['reference_time_s']
    assert M>=3 and o['known_constant_bits'] and np.allclose(np.diff(starts),dt)
    phase=centers[None,:]*starts[:,None]+(centers-o['aiding_fd_hz'])[None,:]*(L-1)/(2*fs)
    x=(surf*np.exp(-2j*np.pi*phase[:,:,None])).reshape(M,-1)
    ra=fftshift(fftfreq(rate_nfft,dt))/dt;allowed=abs(ra)<=o['rate_half_width_hz_s']
    freq=fftshift(fftfreq(frequency_nfft,dt));cols=np.repeat(centers,nt);td=np.tile(delays,nf)
    mid=(o['pmf_segments']*L-1)/(2*fs);times=starts+mid-tref;eta=-fs/o.get('rf_carrier_hz',np.inf)
    rates=np.empty(x.shape[1]);residual=np.empty(x.shape[1]);powers=np.empty(x.shape[1]);valid_tau=np.ones(x.shape[1],bool)
    for first in range(0,x.shape[1],256):
        last=min(first+256,x.shape[1]);xx=x[:,first:last];d=np.conj(xx[:-1])*xx[1:]
        a=fftshift(fft(d,n=rate_nfft,axis=0),axes=0);ri=np.argmax(np.where(allowed[:,None],abs(a),-np.inf),axis=0);r=ra[ri]
        corrected=xx*np.exp(-1j*np.pi*times[:,None]**2*r[None,:]);s=fftshift(fft(corrected,n=frequency_nfft,axis=0),axes=0)
        f0=cols[None,first:last]+freq[:,None]-r[None,:]*tref
        mask=(abs(freq[:,None])<=o['post_correlation_frequency_half_width_hz']) & (abs(f0-o['aiding_fd_hz'])<=o['frequency_half_width_hz'])
        if 'aiding_tau_samples' in o:
            tau_initial=td[None,first:last]-eta*(f0*tref+.5*r[None,:]*tref*tref)
            mask&=abs(tau_initial-o['aiding_tau_samples'])<=o['tau_half_width_samples']+1e-10
        fi=np.argmax(np.where(mask,abs(s),-np.inf),axis=0);j=np.arange(last-first)
        powers[first:last]=np.where(mask.any(axis=0),abs(s[fi,j])**2,-np.inf);rates[first:last]=r;residual[first:last]=freq[fi]
    winner=int(np.argmax(powers));assert np.isfinite(powers[winner]);fc,dc=np.unravel_index(winner,(nf,nt))
    code=powers.reshape(nf,nt).max(axis=0);fraction=0.
    if 0<dc<nt-1:
        a,b,c=code[dc-1:dc+2];curve=a-2*b+c
        if np.isfinite(curve) and curve<0:fraction=float(np.clip(.5*(a-c)/curve,-.5,.5))
    r=float(rates[winner]);fd=float(centers[fc]+residual[winner]);tau=float(delays[dc]+fraction*(delays[min(dc+1,nt-1)]-delays[max(dc-1,0)])/2)
    f0=fd-r*tref
    if 'aiding_tau_samples' in o:
        initial=tau-eta*(f0*tref+.5*r*tref*tref)
        initial=np.clip(initial,o['aiding_tau_samples']-o['tau_half_width_samples'],o['aiding_tau_samples']+o['tau_half_width_samples'])
        tau=float(initial+eta*(f0*tref+.5*r*tref*tref))
    scale=float(np.mean(np.sum(abs(x)**2,axis=0)));assert scale>0
    return dict(score=float(powers[winner]/scale),fd_hat_hz=fd,tau_hat_samples=tau,rate_hat_hz_s=r,
        state_index=winner,rate_step_hz_s=float(ra[1]-ra[0]),frequency_step_hz=float(freq[1]-freq[0]))
