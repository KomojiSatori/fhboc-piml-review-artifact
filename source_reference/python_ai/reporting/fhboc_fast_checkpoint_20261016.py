"""Algebraically equivalent checkpoint; fused projection and uniform LUT lookup."""
import numpy as np
from scipy.special import logsumexp
from fhboc_allgrid_inference_20261013 import standardized_features
from fhboc_amplitude_prior_20261013 import model_log_bf
from fhboc_sparse_checkpoint_20261016 import FieldCheckpoint,sparse_raw,sparse_complex,GLOBAL_NAMES

class FastFieldCheckpoint(FieldCheckpoint):
    def __init__(self,d,model,samples=128):
        super().__init__(d,model,samples);self.sqrtq=np.sqrt(d['template_energy']);self.Q=d['template_energy'].sum(axis=1)
    def bf(self,g):
        position=np.maximum(g,0)*100.;left=np.minimum(position.astype(np.int64),len(self.lut)-2)
        fraction=position-left;value=self.lut[left]+fraction*(self.lut[left+1]-self.lut[left])
        outside=g>self.axis[-1]
        if outside.any():value[outside]=model_log_bf(g[outside],np.sum(self.d['dimensions']),self.model['amplitude_model'])
        return value
    def __call__(self,b,E):
        d=self.d;D=np.asarray(d['dimensions'],float);q=d['template_energy'];H=len(q)
        total=np.einsum('ij,ij,j->i',b,self.sqrtq,np.sqrt(E/D),optimize=False)
        c=abs(total)**2/(self.Q*E.sum());g=-D.sum()*np.log1p(-np.clip(c,0,1-np.finfo(float).eps))
        base=self.prior+self.bf(g);shift=base.max();p=np.exp(base-shift);normalizer=p.sum();p/=normalizer;T=shift+np.log(normalizer)
        cumulative=np.cumsum(p);cumulative[-1]=1
        ids=np.searchsorted(cumulative,(np.arange(self.samples)+.5)/self.samples)
        raw=sparse_raw(b,q,E,D,g,self.shape,self.rate_mass,ids)
        x=standardized_features(raw,self.model);residual=x@np.asarray(self.model['beta_detection'])
        w=np.exp(residual-logsumexp(residual));cf=sparse_complex(b,q,E,D,d['states'],self.shape,ids)
        phi=w@cf;correction=np.clip((phi-self.model['complex_feature_mean'])/self.model['complex_feature_scale'],-4,4)@np.asarray(self.model['complex_feature_beta'])
        surrogate=float(T+logsumexp(residual)-np.log(self.samples)+correction)
        best=int(np.argmax(base));mask=(abs(self.fd-self.fd[best])<=250)&(abs(self.tau-self.tau[best])<=1)
        comp=(abs(self.fd-self.fd[best])>500)|(abs(self.tau-self.tau[best])>2)
        values=[g.max(),T,(T-p@base)/np.log(H),*np.quantile(g,[.9,.99]),
            p[mask].sum(),p[comp].max(),np.sqrt(p@(self.fd-p@self.fd)**2)/2500,
            np.sqrt(p@(self.tau-p@self.tau)**2),surrogate,residual.mean(),residual.std(),np.max(abs(residual)),1/(p@p)/H,*phi,*raw.mean(axis=0)]
        result=np.asarray(values,float);assert len(result)==len(GLOBAL_NAMES) and np.isfinite(result).all()
        return result
