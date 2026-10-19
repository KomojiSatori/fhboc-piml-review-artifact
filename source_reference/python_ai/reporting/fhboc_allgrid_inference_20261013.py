"""Full-domain coherent evidence, rectangular posterior actions, and state features."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import logsumexp, ndtr

RHO = np.array([1., 4., 16., 64.])
FEATURE_NAMES = ['six_minus_frame_plus_5', 'six_minus_two_plus_2',
                 'frame_power_participation', 'six_minus_halves_plus_1',
                 'frequency_neighbour_log_ratio', 'delay_neighbour_log_ratio',
                 'rate_neighbour_log_ratio', 'rate_profile_log_ratio']


def angular_log_bf(score, dimension, rho=RHO, weights=None):
    """Exact common-amplitude directional likelihood, from the coherent GLRT."""
    score = np.asarray(score, float)
    rho = np.asarray(rho, float)
    weights = np.ones(len(rho)) if weights is None else np.asarray(weights, float)
    assert dimension > 1 and np.all(score >= 0) and np.all(rho >= 0)
    assert len(weights) == len(rho) and np.all(weights >= 0) and weights.sum() > 0
    c = -np.expm1(-score / float(dimension))
    terms = (-np.log1p(rho) - dimension*np.log1p(-c[..., None]*(rho/(1+rho)))
             + np.log(weights/weights.sum()))
    return logsumexp(terms, axis=-1)


def prior_arrays(axes, aiding, cfg):
    """Same quadrature cell masses and continuous densities as MATLAB v1."""
    keys = ('fd_hz', 'tau_samples', 'rate_hz_s')
    axis = [np.asarray(axes[k], float).reshape(-1) for k in keys]
    means = [float(aiding['fd_hz']), float(aiding['tau_samples']), 0.]
    sigmas = [1000., .5*cfg['signal']['fs']/cfg['signal']['fc'], np.inf]
    mass_u, mass_a, density_a = [], [], []
    for x, mu, sigma in zip(axis, means, sigmas):
        edges = np.r_[x[0], (x[:-1]+x[1:])/2, x[-1]]
        u = np.diff(edges)/(x[-1]-x[0])
        if np.isinf(sigma):
            a = u
            density = np.full(len(x), -np.log(x[-1]-x[0]))
        else:
            a = np.diff(ndtr((edges-mu)/sigma)); a /= a.sum()
            z = ndtr((x[-1]-mu)/sigma)-ndtr((x[0]-mu)/sigma)
            density = -.5*((x-mu)/sigma)**2-np.log(np.sqrt(2*np.pi)*sigma*z)
        assert np.all(u > 0) and np.all(a > 0)
        mass_u.append(u); mass_a.append(a); density_a.append(density)
    def combine(vectors, operation):
        grid = np.meshgrid(*vectors, indexing='ij')
        value = operation(grid)
        return value.reshape(-1, order='F')
    uniform = combine(mass_u, lambda g: g[0]*g[1]*g[2])
    aided = combine(mass_a, lambda g: g[0]*g[1]*g[2])
    density = combine(density_a, lambda g: g[0]+g[1]+g[2])
    return dict(log_uniform=np.log(uniform/uniform.sum()),
                log_aided=np.log(aided/aided.sum()), log_aided_density=density,
                rate_mass=mass_u[2])


def reference_coordinates(states, cfg, reference_time):
    states = np.asarray(states, float); t = float(reference_time)
    eta = -cfg['signal']['fs']/cfg['signal']['rf_carrier_hz']
    return (states[:, 0]+states[:, 2]*t,
            states[:, 1]+eta*(states[:, 0]*t+.5*states[:, 2]*t*t))


class WindowPlan:
    """Exact closed rectangle masses at EVERY input action, grouped by frequency.

    Sorting and range queries are precomputed; no state pruning or discretized
    histogram. The aided grids have about 100 distinct reference frequencies.
    """
    def __init__(self, fd, tau, fd_tolerance=250., tau_tolerance=1.):
        self.n = len(fd); self.groups = []
        fd = np.asarray(fd, float); tau = np.asarray(tau, float)
        for value in np.unique(fd):
            query = np.flatnonzero(fd == value)
            support = np.flatnonzero((fd >= value-fd_tolerance) &
                                     (fd <= value+fd_tolerance))
            support = support[np.argsort(tau[support], kind='stable')]
            lower = np.searchsorted(tau[support], tau[query]-tau_tolerance, side='left')
            upper = np.searchsorted(tau[support], tau[query]+tau_tolerance, side='right')
            self.groups.append((query, support, lower, upper))

    def mass(self, weights):
        weights = np.asarray(weights, float)
        assert weights.shape[0] == self.n and np.all(weights >= 0)
        result = np.zeros_like(weights)
        for query, support, lower, upper in self.groups:
            cumulative = np.concatenate((np.zeros((1,)+weights.shape[1:]),
                                         np.cumsum(weights[support], axis=0)), axis=0)
            result[query] = np.maximum(0., cumulative[upper]-cumulative[lower])
        return result


def state_features(data, log_bf):
    """Eight bounded, scale/phase invariant FULL-grid learning inputs (O,H,F)."""
    scores = np.asarray(data['path_scores'], float)  # H,O,(frame,2ms,6ms)
    b = np.asarray(data['projected'], complex)  # H,6,O, candidate phase absorbed
    power = abs(b)**2
    participation = power.sum(axis=1)**2 / np.maximum((power**2).sum(axis=1), 1e-30)
    q = np.asarray(data['template_energy'], float)
    E = np.asarray(data['observation_energy'], float)
    dims = np.asarray(data['dimensions'], float).reshape(-1)
    dots = b*np.sqrt(q[:, :, None]*(E/dims[:, None])[None, :, :])
    halves = np.zeros_like(log_bf)
    for ids in (np.arange(3), np.arange(3, 6)):
        c = abs(dots[:, ids].sum(axis=1))**2 / (q[:, ids].sum(axis=1)[:, None]*E[ids].sum(axis=0))
        halves -= dims[ids].sum()*np.log1p(-np.clip(c, 0, 1-np.finfo(float).eps))
    values = [scores[:, :, 2]-scores[:, :, 0]+5,
              scores[:, :, 2]-scores[:, :, 1]+2, participation,
              scores[:, :, 2]-halves+1]
    shape = tuple(len(np.atleast_1d(data['axes'][k]))
                  for k in ('fd_hz', 'tau_samples', 'rate_hz_s'))
    grid = log_bf.reshape(shape+(scores.shape[1],), order='F')
    # Stable globally shifted likelihoods; the shift cancels from all ratios.
    shift = np.max(grid, axis=(0, 1, 2), keepdims=True)
    evidence = np.exp(grid-shift)
    for axis in range(3):
        nearby = convolve1d(evidence, [.25, .5, .25], axis=axis, mode='constant')
        normalizer = convolve1d(np.ones(shape+(1,)), [.25, .5, .25], axis=axis, mode='constant')
        ratio = np.log(np.maximum(nearby/normalizer, 1e-300))+shift-grid
        values.append(ratio.reshape(log_bf.shape, order='F'))
    rate_mass = prior_arrays(data['axes'], data['aiding'], data['cfg'])['rate_mass']
    profile = logsumexp(grid+np.log(rate_mass)[None, None, :, None], axis=2, keepdims=True)
    values.append(np.broadcast_to(profile-grid, grid.shape).reshape(log_bf.shape, order='F'))
    features = np.stack(values, axis=-1).transpose(1, 0, 2).astype(np.float32)
    assert np.isfinite(features).all()
    return features


def standardized_features(features, model):
    return np.clip((np.asarray(features, float)-model['mean'])/model['scale'],
                   -model['clip'], model['clip'])
