function pmf=fhboc_continuous_pmf_direct_20261013(rx,reference,cfg,start_s,tau_axis,center_fd)
% Independent sample-domain sum used to gate the sparse operator.
rx=rx(:);N=numel(rx);P=cfg.acq.P_segments;L=floor(N/P);
if cfg.acq.code_doppler_dewarp_enable
 rx=compensate_code_doppler(rx,cfg.signal.fs,cfg.signal.rf_carrier_hz, ...
  cfg.acq.code_doppler_reference_fd_hz,cfg.acq.code_doppler_reference_fdot_hz_s);
end
rx=rx.*exp(-1j*2*pi*center_fd*(0:N-1)'/cfg.signal.fs);
pmf=complex(zeros(P,numel(tau_axis)));start=round(start_s*cfg.signal.fs);
for j=1:numel(tau_axis)
 index=start+(0:P*L-1)'-tau_axis(j)-reference.sample_origin+1;
 product=rx(1:P*L).*conj(reference.samples(index));
 pmf(:,j)=sum(reshape(product,L,P),1).';
end
end
