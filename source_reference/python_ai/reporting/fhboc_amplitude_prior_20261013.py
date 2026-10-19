"""Proper scalar amplitude priors for coherent FULL-grid FHBOC inference.

Uses deterministic special-function evaluation; creates no signal/noise samples.
"""
from __future__ import annotations
import numpy as np
from scipy.special import hyp1f1, logsumexp


def log_kummer_positive_series(dimension, x, tolerance=1e-13):
    """Log 1F1(D;1;x) positive series, used if the fast evaluator overflows."""
    assert dimension >= 1 and x >= 0
    if x == 0: return 0.
    term = 0.; total = 0.; logx = np.log(x)
    for n in range(1, 100000):
        term += np.log(dimension+n-1)+logx-2*np.log(n)
        total = np.logaddexp(total, term)
        next_ratio = (dimension+n)*x/(n+1)**2
        if next_ratio < 1:
            # Ratios decrease for D>=1: geometric remainder bounds the full tail.
            remainder = np.exp(term-total)*next_ratio/max(1-next_ratio, 1e-300)
            if remainder < tolerance: return float(total)
    raise ArithmeticError('Positive Kummer series failed to converge.')


def gaussian_log_bf(score, dimension, rho):
    score = np.asarray(score, float); rho = np.asarray(rho, float)
    assert dimension > 1 and np.all(score >= 0) and np.all(rho >= 0)
    c = -np.expm1(-score/float(dimension))
    return -np.log1p(rho)-dimension*np.log1p(-c[..., None]*rho/(1+rho))


def fixed_magnitude_log_bf(score, dimension, rho):
    """Uniform unknown global phase, fixed normalized squared magnitude rho.

    B = exp(-rho) * 1F1(D;1;rho*c), c=|z^Hu|^2/||z||^2.
    """
    score = np.asarray(score, float); rho = np.asarray(rho, float)
    assert dimension > 1 and np.all(score >= 0) and np.all(rho >= 0)
    c = -np.expm1(-score/float(dimension)); x = c[..., None]*rho
    with np.errstate(over='ignore', invalid='ignore'):
        raw = hyp1f1(float(dimension), 1., x)
        value = np.log(raw)
    invalid = ~np.isfinite(value)
    if invalid.any():
        value = np.array(value, copy=True)
        for index in zip(*np.where(invalid)):
            value[index] = log_kummer_positive_series(float(dimension), float(x[index]))
    return value-rho


def basis_log_bf(score, dimension, rho, families):
    return np.concatenate([(gaussian_log_bf if f=='gaussian' else fixed_magnitude_log_bf)(score, dimension, rho)
                           for f in families], axis=-1)


def model_log_bf(score, dimension, model):
    terms = basis_log_bf(score, dimension, np.asarray(model['rho'], float), model['families'])
    weights = np.asarray(model['weights'], float)
    assert len(weights)==terms.shape[-1] and np.all(weights>0) and abs(weights.sum()-1)<1e-10
    return logsumexp(terms+np.log(weights), axis=-1)


def broad_weights(rho, families):
    axis=np.log(np.asarray(rho, float));edges=np.r_[axis[0],(axis[:-1]+axis[1:])/2,axis[-1]]
    weights=np.diff(edges);weights/=weights.sum()
    return np.tile(weights/len(families),len(families))


def robust_path_prior(log_aided, log_uniform, uniform_fraction):
    assert 0 < uniform_fraction < 1
    return np.logaddexp(np.asarray(log_aided)+np.log1p(-uniform_fraction),
                        np.asarray(log_uniform)+np.log(uniform_fraction))


def piecewise_lut_axis():
    return np.r_[np.arange(8193)/512,
                 16+np.arange(1,12289)/256,
                 64+np.arange(1,24577)/128]


def lut_log_bf(score, dimension, model, axis, values):
    """Linear interpolation in range; exact scalar CPU evaluation for tails."""
    score=np.asarray(score,float);result=np.interp(score,axis,values)
    outside=(score<axis[0])|(score>axis[-1])
    if np.any(outside):
        result=np.asarray(result);result[outside]=model_log_bf(score[outside],dimension,model)
    return result


def scalar_slope_bound(left, dimension, rho_max):
    if left==0: return max(1.,rho_max)
    c=-np.expm1(-left/dimension)
    fixed=np.exp(-left/dimension)*(rho_max/(2*dimension)+
            np.sqrt(rho_max**2/(4*dimension**2)+rho_max/(dimension*c)))
    return max(1.,min(rho_max,fixed))


def lut_interpolation_bound(dimension, rho_max):
    """Uniform absolute log-BF error for the exported piecewise linear table.

    Integer D: fixed-magnitude log BF is concave via Laguerre positive roots.
    The mixture curvature is E[g'']+Var(g'); low-range bound is global.
    Mid/high intervals use decreasing slope envelopes and Lipschitz error.
    """
    assert dimension > 1 and dimension == int(dimension)
    curvature=max(rho_max**2*(dimension-1)/(2*dimension)+rho_max/dimension,
                  1/(4*dimension))+max(1.,rho_max)**2/4
    bounds=[curvature*(1/512)**2/8,
            scalar_slope_bound(16.,dimension,rho_max)*(1/256)/2,
            scalar_slope_bound(64.,dimension,rho_max)*(1/128)/2]
    return max(bounds),bounds
