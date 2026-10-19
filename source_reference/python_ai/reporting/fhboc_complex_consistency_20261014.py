"""Eight scale/phase invariant state features from all six complex projections."""
import numpy as np
from scipy.ndimage import convolve1d

NAMES=['loo_innovation_mean_log','loo_innovation_max_log','adjacent_phase_agreement',
       'odd_even_phase_agreement','minimum_loo_phase_agreement',
       'frequency_complex_support','delay_complex_support','rate_complex_support']


def state_features(projected,q,energy,dimensions,states,shape,midpoint_s=.003):
    b=np.asarray(projected,complex);q=np.asarray(q,float);E=np.asarray(energy,float);D=np.asarray(dimensions,float)
    assert b.shape==q.shape and b.shape[1]==6 and E.shape==D.shape==(6,)
    sigma=E.sum()/D.sum();assert sigma>0 and np.all(q>0)
    dots=b*np.sqrt(q*(E/D)[None,:]);total=dots.sum(axis=1);Q=q.sum(axis=1);restQ=Q[:,None]-q
    prediction=q*(total[:,None]-dots)/restQ
    innovation=abs(dots-prediction)**2/(q*sigma*(1+q/restQ))
    a=dots/np.sqrt(q*sigma)
    adjacent=np.real((a[:,:-1]*a[:,1:].conj()).sum(axis=1))/np.maximum(
        np.sqrt((abs(a[:,:-1])**2).sum(axis=1)*(abs(a[:,1:])**2).sum(axis=1)),1e-30)
    odd=dots[:,::2].sum(axis=1);even=dots[:,1::2].sum(axis=1)
    split=np.real(odd*even.conj())/np.maximum(abs(odd)*abs(even),1e-30)
    leave=total[:,None]-dots
    cosine=np.real(dots*leave.conj())/np.maximum(abs(dots)*abs(leave),1e-30)
    values=[np.log1p(innovation.mean(axis=1)),np.log1p(innovation.max(axis=1)),adjacent,split,cosine.min(axis=1)]
    phase=2*np.pi*(np.asarray(states)[:,0]*midpoint_s+.5*np.asarray(states)[:,2]*midpoint_s**2)
    amplitude=total/np.sqrt(Q*sigma)*np.exp(-1j*phase)
    grid=amplitude.reshape(shape,order='F');power=abs(grid)**2
    for axis in range(3):
        neighbors=convolve1d(grid,[.5,0,.5],axis=axis,mode='constant')
        norm=convolve1d(np.ones(shape),[.5,0,.5],axis=axis,mode='constant')
        local_power=convolve1d(power,[.5,0,.5],axis=axis,mode='constant')
        # Edge cells average available neighbors, maintaining bounded support.
        neighbors/=np.maximum(norm,1e-30);local_power/=np.maximum(norm,1e-30)
        support=2*np.real(neighbors*grid.conj())/np.maximum(local_power+power,1e-30)
        values.append(np.clip(support,-1,1).reshape(-1,order='F'))
    result=np.stack(values,axis=1)
    assert np.isfinite(result).all()
    return result


def aggregate_features(features,log_detection):
    from scipy.special import logsumexp
    p=np.exp(log_detection-logsumexp(log_detection))
    return p@np.asarray(features)
