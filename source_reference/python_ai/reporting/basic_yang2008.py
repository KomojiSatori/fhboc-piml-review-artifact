"""Yang et al. 2008 BASIC, DTIC ADA484228, section III equations 6--22.

Physical convention: phase = 2*pi*(f*t + rate*t*t/2), so rate = 2*alpha
in the paper's equation (2). No adjacent-product squaring or CSP estimator.
"""
from __future__ import annotations

import numpy as np


def _pinning(products, frequencies):
    """Eqs. 15--17: FFT of singleton samples, exactly evaluated as phasors."""
    h = products[:, None] * np.exp(-2j*np.pi*np.arange(len(products))[:, None]*frequencies)
    total = h[0].copy()
    signs = np.ones(h.shape, dtype=np.int8)
    for k in range(1, len(products)):
        signs[k] = np.where(np.real(np.conj(total)*h[k]) >= 0, 1, -1)
        total += signs[k]*h[k]
    return total, signs


def estimate(x, sample_interval_s, samples_per_block, *, rate_nfft=2048,
             frequency_nfft=4096, rate_bounds_hz_s=None,
             frequency_bounds_hz=None, known_constant_bits=False):
    """Original unknown-bit block/offset/sign branches, or its no-bit Eq. 9.

    x is a uniform scalar post-correlation sequence. Returned frequency is at
    the first input sample, irrespective of the selected bit alignment.
    Offsets enumerate complete equal-length blocks, without padding data.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    ts, m = float(sample_interval_s), int(samples_per_block)
    if ts <= 0 or m < 1 or not np.isfinite(x).all():
        raise ValueError('Finite samples, positive Ts and integer M required')
    if m != samples_per_block or rate_nfft % 2 or rate_nfft < 2:
        raise ValueError('Integer M and even chirp FFT size required')
    offsets = [0] if known_constant_bits else range(m)
    kblocks = (len(x) if known_constant_bits else len(x)-m+1)//m
    if kblocks < 3:
        raise ValueError('At least three complete blocks per alignment required')
    block_time = m*ts
    nu = np.fft.fftshift(np.fft.fftfreq(rate_nfft))
    rates = nu/block_time**2  # Eq. 11, with physical rate=2*alpha
    best_a = None
    for offset in offsets:
        blocks = x[offset:offset+kblocks*m].reshape(kblocks, m)
        products = np.sum(np.conj(blocks[:-1])*blocks[1:], axis=1)  # Eqs. 6,8
        if known_constant_bits:
            totals = np.fft.fftshift(np.fft.fft(products, n=rate_nfft))
            signs = np.ones((kblocks-1, rate_nfft), dtype=np.int8)
        else:
            totals, signs = _pinning(products, nu)
        # Eqs. 18--20 select the strongest bit alignment and normalized bin.
        ai = int(np.argmax(abs(totals)))
        if best_a is None or abs(totals[ai]) > best_a[0]:
            best_a = (float(abs(totals[ai])), offset, ai, signs)
    a_score, offset, ai, signs = best_a
    # Eq. 21 companion peak and both differential-sign reconstruction logics.
    peaks = [ai] if known_constant_bits else [ai, (ai+rate_nfft//2)%rate_nfft]
    times = np.arange(kblocks*m)*ts + offset*ts
    segment = x[offset:offset+kblocks*m]
    freq = np.fft.fftshift(np.fft.fftfreq(frequency_nfft, ts))
    allowed = np.ones(freq.shape, dtype=bool)
    if frequency_bounds_hz is not None:
        allowed &= (freq >= frequency_bounds_hz[0]) & (freq <= frequency_bounds_hz[1])
    if not allowed.any():
        raise ValueError('Empty initial-frequency grid')
    best = None
    for peak in peaks:
        rate = float(rates[peak])
        if rate_bounds_hz_s is not None and not rate_bounds_hz_s[0] <= rate <= rate_bounds_hz_s[1]:
            continue
        for logic in ([1] if known_constant_bits else [1, -1]):
            bits = np.r_[1, np.cumprod(logic*signs[:, peak])]
            samples = segment*np.repeat(bits, m)*np.exp(-1j*np.pi*rate*times**2)
            spectrum = np.fft.fftshift(np.fft.fft(samples, n=frequency_nfft))
            fi = int(np.argmax(np.where(allowed, abs(spectrum), -np.inf)))
            amplitude = float(abs(spectrum[fi]))
            if best is None or amplitude > best['amplitude']:
                best = dict(f_hat_hz=float(freq[fi]), rate_hat_hz_s=rate,
                            amplitude=amplitude, power=amplitude**2,
                            offset_hat=offset, block_bits=bits, logic=logic,
                            a_score=a_score, blocks=kblocks, samples_used=len(segment),
                            chirp_fft_bin=peak, rate_step_hz_s=1/(rate_nfft*block_time**2),
                            frequency_step_hz=1/(frequency_nfft*ts))
    if best is None:
        raise ValueError('BASIC selected peaks outside declared rate domain')
    return best


def pilot_pmf_bank_estimate(observations, *, rate_nfft=2048, frequency_nfft=2048):
    """No-navigation-bit BASIC on actual six-frame PMF-FFT coarse banks.

    M=1 is the known-pilot specialization: each frame FFT cell supplies one
    complex 1 ms correlation. Eq. 9 estimates rate first; Eq. 22 then estimates
    initial residual frequency. All frequency/code bank columns participate.
    The input carries no truth, C/N0, noise realization or true parameters.
    """
    o = observations
    surface = np.asarray(o['complex_surface'], dtype=np.complex128)
    # Export shape: frames x frequency x delay, common residual grid indices.
    starts = np.asarray(o['frame_start_s'], dtype=float).reshape(-1)
    centers = np.asarray(o['coarse_frequency_hz'], dtype=float).reshape(-1)
    delays = np.asarray(o['reference_delay_axis_samples'], dtype=float).reshape(-1)
    aid_fd = float(o['aiding_fd_hz'])
    l = int(o['pmf_segment_length'])
    fs = float(o['sample_rate_hz'])
    seg_mid = (l-1)/(2*fs)
    frame_mid = (int(o['pmf_segments'])*l-1)/(2*fs)
    dt = float(o['frame_duration_s'])
    if surface.shape != (len(starts), len(centers), len(delays)) or len(starts) < 3:
        raise ValueError('Incomplete frame/frequency/code bank')
    if not np.allclose(np.diff(starts), dt, atol=1e-12, rtol=0):
        raise ValueError('BASIC slow-time FFT requires uniform frame spacing')
    if not bool(o['known_constant_bits']):
        raise ValueError('Use full bit-enumerating BASIC for modulated data')
    # PMF FFT indexes segment starts; undo its segment-center phase, and the
    # per-frame coarse wipe-off reset, using known frequencies/times only.
    phase = centers[None, :]*starts[:, None] + (centers-aid_fd)[None, :]*seg_mid
    x = (surface*np.exp(-2j*np.pi*phase[:, :, None])).reshape(len(starts), -1)
    d = np.conj(x[:-1])*x[1:]  # Eq. 6 at M=1; NO squaring
    rate_axis = np.fft.fftshift(np.fft.fftfreq(rate_nfft, dt))/dt
    rate_allowed = abs(rate_axis) <= float(o['rate_half_width_hz_s'])
    a = np.fft.fftshift(np.fft.fft(d, n=rate_nfft, axis=0), axes=0)
    ai = np.argmax(np.where(rate_allowed[:, None], abs(a), -np.inf), axis=0)
    rates = rate_axis[ai]
    readout_time = float(o['reference_time_s'])
    # Express the chirp at the common readout epoch. Otherwise a coarse cell
    # near the middle of a high-rate record has an initial residual >250 Hz,
    # and restricting Eq. 22 there selects a 1 kHz slow-time alias.
    times = starts+frame_mid-readout_time
    dechirped = x*np.exp(-1j*np.pi*times[:, None]**2*rates[None, :])
    f = np.fft.fftshift(np.fft.fftfreq(frequency_nfft, dt))
    # The paper's scalar-correlator residual-frequency limit is ±250 Hz.
    coarse_columns = np.repeat(centers, len(delays))
    fa = ((abs(f[:,None]) <= float(o['post_correlation_frequency_half_width_hz'])) &
          (abs(f[:,None]+coarse_columns[None,:]-rates[None,:]*readout_time-aid_fd)
           <= float(o['frequency_half_width_hz'])))
    spec = np.fft.fftshift(np.fft.fft(dechirped, n=frequency_nfft, axis=0), axes=0)
    fi = np.argmax(np.where(fa, abs(spec), -np.inf), axis=0)
    columns = np.arange(x.shape[1])
    powers = np.where(fa.any(axis=0), abs(spec[fi, columns])**2, -np.inf)
    winner = int(np.argmax(powers))
    fcol, dcol = np.unravel_index(winner, (len(centers), len(delays)))
    rate = float(rates[winner])
    fd = float(centers[fcol]+f[fi[winner]])
    assert abs(fd-rate*readout_time-aid_fd) <= float(o['frequency_half_width_hz'])+1e-8
    code_scores = powers.reshape(len(centers), len(delays)).max(axis=0)
    fraction = 0.0
    if 0 < dcol < len(delays)-1:
        left, middle, right = code_scores[dcol-1:dcol+2]
        curvature = left-2*middle+right
        if curvature < 0:
            fraction = float(np.clip(.5*(left-right)/curvature, -.5, .5))
    delay = float(delays[dcol]+fraction)
    # Wrapper normalization uses one observed scale for the whole bank.
    # It preserves all original amplitude rankings and needs no true N0.
    scale = float(np.mean(np.sum(abs(x)**2, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Nonpositive observed bank scale')
    return dict(score=float(powers[winner]/scale), doppler_hz=fd,
                delay_samples=delay, rate_hz_s=rate,
                coarse_frequency_hz=float(centers[fcol]),
                residual_frequency_hz=float(f[fi[winner]]),
                code_grid_index=int(dcol), subgrid_code_offset=fraction,
                raw_peak_power=float(powers[winner]), observed_scale=scale,
                rate_step_hz_s=float(rate_axis[1]-rate_axis[0]),
                residual_frequency_step_hz=float(f[1]-f[0]),
                frames=len(starts), input_branches=x.shape[1],
                specialization='known pilot, M=1, equations 6/8/9/11/22')
