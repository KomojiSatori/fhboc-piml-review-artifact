function wave=fhboc_continuous_waveform_batch_20261013(reference,cfg,start_s,tau,fd,rate,phase_cycles,sample_count)
% Local physical state at start_s; reference queries use global transmitter time.
% Called with candidate states by detectors; no trial/truth structure accepted.
if nargin<7,phase_cycles=0;end
if nargin<8,sample_count=reference.frame_samples;end
n=(0:sample_count-1)';t=n/cfg.signal.fs;
tau=double(tau(:)');fd=double(fd(:)');rate=double(rate(:)');
phase=t*fd+.5*t.^2*rate;
warp=cfg.signal.fs/cfg.signal.rf_carrier_hz*phase;
query=round(start_s*cfg.signal.fs)+n-tau+warp-reference.sample_origin+1;
assert(min(query,[],'all')>=1&&max(query,[],'all')<numel(reference.samples),'Reference halo exhausted.');
left=floor(query);fraction=query-left;s=reference.samples;
base=(1-fraction).*s(left)+fraction.*s(left+1);
wave=base.*exp(1j*2*pi*(phase+phase_cycles));
end
