function response = fhboc_pmf_bank_response(bank,cfg,tau_start,fd_start,rate)
% Local residual response after SAME aiding dewarp as exact anchor bank.
% Rate and Doppler influence every segment; no single energy path is fixed.
tau_start=tau_start(:)';fd_start=fd_start(:)';rate=rate(:)';P=cfg.acq.P_segments;
ts=bank.segment_times_s;
% Source time of known constant-rate-zero dewarp.
source_t=ts/(1+bank.center_fd_hz/cfg.signal.rf_carrier_hz);
phase=source_t*(fd_start-bank.center_fd_hz)+.5*source_t.^2*rate;
tau=tau_start-cfg.signal.fs/cfg.signal.rf_carrier_hz*phase;
response=complex(zeros(size(bank.response,1),numel(tau_start)));
% A uniform anchor grid permits exact linear interpolation by indexing.
% This only removes interp1 setup overhead; it does not change the response.
step=bank.tau_grid(2)-bank.tau_grid(1);
if max(abs(diff(bank.tau_grid)-step))<1e-10*step
 position=(tau-bank.tau_grid(1))/step+1;
 valid=position>=1 & position<=numel(bank.tau_grid);
 left=min(max(floor(position),1),numel(bank.tau_grid)-1);
 fraction=position-left;
 rows=(1:size(bank.response,1))';segments=mod(rows-1,P)+1;
 columns=left(segments,:);fraction=fraction(segments,:);
 linear=rows+size(bank.response,1)*(columns-1);
 response=(1-fraction).*bank.response(linear)+fraction.*bank.response(linear+size(bank.response,1));
 response=response.*exp(1j*2*pi*phase(segments,:));
 response(~valid(segments,:))=0;
 return;
end
for p=1:P
 rows=p:P:size(bank.response,1);
 response(rows,:)=interp1(bank.tau_grid,bank.response(rows,:).',tau(p,:),'linear',0).' .*exp(1j*2*pi*phase(p,:));
end
end
