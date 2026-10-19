"""Full-grid coherent field and sparse posterior diagnostics, with no pruning."""
import numpy as np
from scipy.special import logsumexp
from fhboc_allgrid_inference_20261013 import angular_log_bf
from fhboc_amplitude_prior_20261013 import model_log_bf

GLOBAL_NAMES=['field_gmax','field_evidence','field_entropy','field_g90','field_g99',
 'field_peak_mass','field_competitor_mass','field_fd_spread','field_tau_spread',
 'surrogate_score','residual_mean','residual_std','residual_max','effective_states']+['phi_'+str(i) for i in range(8)]+['raw_'+str(i) for i in range(8)]

def neighbours(ids,shape):
    ids=np.asarray(ids,int);coord=np.array(np.unravel_index(ids,shape,order='F')).T
    strides=np.r_[1,np.cumprod(shape)[:-1]];result=[]
    for axis,step in enumerate(strides):
        result.append((np.maximum(ids-step,0),np.minimum(ids+step,np.prod(shape)-1),coord[:,axis]>0,coord[:,axis]<shape[axis]-1))
    return result

def sparse_complex(b,q,E,D,states,shape,ids):
    """Exactly the existing eight complex features, at requested states only."""
    ids=np.asarray(ids,int);nb=neighbours(ids,shape)
    union=np.unique(np.r_[ids,*[n for pair in nb for n in pair[:2]]])
    z=np.asarray(b[union],complex);qq=q[union];sigma=E.sum()/D.sum()
    dots=z*np.sqrt(qq*(E/D)[None,:]);total=dots.sum(axis=1);Q=qq.sum(axis=1)
    ii=np.searchsorted(union,ids);a=dots[ii]/np.sqrt(qq[ii]*sigma);dq=dots[ii];sq=qq[ii];rest=Q[ii,None]-sq
    prediction=sq*(total[ii,None]-dq)/rest
    innovation=abs(dq-prediction)**2/(sq*sigma*(1+sq/rest))
    adjacent=np.real(np.sum(a[:,:-1]*a[:,1:].conj(),axis=1))/np.maximum(np.sqrt(np.sum(abs(a[:,:-1])**2,axis=1)*np.sum(abs(a[:,1:])**2,axis=1)),1e-30)
    odd=dq[:,::2].sum(axis=1);even=dq[:,1::2].sum(axis=1)
    split=np.real(odd*even.conj())/np.maximum(abs(odd)*abs(even),1e-30)
    leave=total[ii,None]-dq
    cosine=np.real(dq*leave.conj())/np.maximum(abs(dq)*abs(leave),1e-30)
    value=[np.log1p(innovation.mean(axis=1)),np.log1p(innovation.max(axis=1)),adjacent,split,cosine.min(axis=1)]
    phase=2*np.pi*(states[union,0]*.003+.5*states[union,2]*.003**2)
    amplitude=total/np.sqrt(Q*sigma)*np.exp(-1j*phase);center=amplitude[ii]
    for left,right,vl,vr in nb:
        l=amplitude[np.searchsorted(union,left)];r=amplitude[np.searchsorted(union,right)]
        count=np.maximum(vl.astype(int)+vr.astype(int),1)
        near=(l*vl+r*vr)/count;power=(abs(l)**2*vl+abs(r)**2*vr)/count
        value.append(np.clip(2*np.real(near*center.conj())/np.maximum(power+abs(center)**2,1e-30),-1,1))
    return np.stack(value,axis=1)

def sparse_raw(b,q,E,D,g,shape,rate_mass,ids):
    """Exact original eight state features at the posterior diagnostic nodes."""
    ids=np.asarray(ids,int);dots=np.asarray(b[ids],complex)*np.sqrt(q[ids]*(E/D)[None,:])
    def grouped(groups):
        result=np.zeros(len(ids))
        for ix in groups:
            c=abs(dots[:,ix].sum(axis=1))**2/(q[ids][:,ix].sum(axis=1)*E[ix].sum())
            result-=D[ix].sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps))
        return result
    frames=grouped([[k] for k in range(6)]);pairs=grouped([range(k,k+2) for k in (0,2,4)])
    halves=grouped([range(3),range(3,6)]);power=abs(b[ids])**2
    values=[g[ids]-frames+5,g[ids]-pairs+2,power.sum(axis=1)**2/np.maximum((power**2).sum(axis=1),1e-30),g[ids]-halves+1]
    center=angular_log_bf(g[ids],D.sum())
    for left,right,vl,vr in neighbours(ids,shape):
        l=angular_log_bf(g[left],D.sum());r=angular_log_bf(g[right],D.sum())
        terms=np.stack([center+np.log(.5),np.where(vl,l+np.log(.25),-np.inf),np.where(vr,r+np.log(.25),-np.inf)],axis=1)
        values.append(logsumexp(terms,axis=1)-np.log(.5+.25*vl+.25*vr)-center)
    stride=shape[0]*shape[1];base=ids%stride
    profile=angular_log_bf(g[base[:,None]+stride*np.arange(shape[2])],D.sum())
    values.append(logsumexp(profile+np.log(rate_mass),axis=1)-center)
    return np.stack(values,axis=1).astype(np.float32)

class FieldCheckpoint:
    def __init__(self,d,model,samples=128):
        from fhboc_allgrid_inference_20261013 import prior_arrays,reference_coordinates
        self.d=d;self.model=model;self.samples=samples
        self.shape=tuple(len(np.atleast_1d(d['axes'][k])) for k in ('fd_hz','tau_samples','rate_hz_s'))
        arrays=prior_arrays(d['axes'],d['aiding'],d['cfg']);self.rate_mass=arrays['rate_mass']
        self.prior=np.logaddexp(d['log_aided']+np.log(.9),d['log_uniform']+np.log(.1))
        self.fd,self.tau=reference_coordinates(d['states'],d['cfg'],.0025)
        self.axis=np.arange(0,256.0001,.01);self.lut=model_log_bf(self.axis,np.sum(d['dimensions']),model['amplitude_model'])
    def __call__(self,b,E):
        from fhboc_allgrid_inference_20261013 import standardized_features
        d=self.d;D=np.asarray(d['dimensions'],float);q=d['template_energy'];H=len(q)
        dots=np.asarray(b,complex)*np.sqrt(q*(E/D)[None,:]);total=dots.sum(axis=1)
        c=abs(total)**2/(q.sum(axis=1)*E.sum());g=-D.sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps))
        bf=np.interp(g,self.axis,self.lut);outside=g>self.axis[-1]
        if outside.any():bf[outside]=model_log_bf(g[outside],D.sum(),self.model['amplitude_model'])
        base=self.prior+bf;T=logsumexp(base);p=np.exp(base-T)
        cumulative=np.cumsum(p);cumulative[-1]=1
        ids=np.searchsorted(cumulative,(np.arange(self.samples)+.5)/self.samples)
        raw=sparse_raw(b,q,E,D,g,self.shape,self.rate_mass,ids)
        x=standardized_features(raw,self.model);residual=x@np.asarray(self.model['beta_detection'])
        w=np.exp(residual-logsumexp(residual));cf=sparse_complex(b,q,E,D,d['states'],self.shape,ids)
        phi=w@cf;correction=np.clip((phi-self.model['complex_feature_mean'])/self.model['complex_feature_scale'],-4,4)@np.asarray(self.model['complex_feature_beta'])
        surrogate=float(T+logsumexp(residual)-np.log(self.samples)+correction)
        best=int(np.argmax(base));mask=(abs(self.fd-self.fd[best])<=250)&(abs(self.tau-self.tau[best])<=1)
        comp=(abs(self.fd-self.fd[best])>500)|(abs(self.tau-self.tau[best])>2)
        values=[g.max(),T,-np.sum(p*np.log(np.maximum(p,1e-300)))/np.log(H),*np.quantile(g,[.9,.99]),
            p[mask].sum(),p[comp].max(),np.sqrt(p@(self.fd-p@self.fd)**2)/2500,
            np.sqrt(p@(self.tau-p@self.tau)**2),surrogate,residual.mean(),residual.std(),np.max(abs(residual)),1/(p@p)/H,*phi,*raw.mean(axis=0)]
        result=np.asarray(values,float);assert len(result)==len(GLOBAL_NAMES) and np.isfinite(result).all()
        return result
