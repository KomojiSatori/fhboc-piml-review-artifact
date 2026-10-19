function model=fhboc_continuous_pmf_operator_20261013(reference,cfg,start_s,tau_axis,center_fd)
% Exact raw frame -> complex PMF. Reference history is real data, not zero-filled.
N=reference.frame_samples;P=cfg.acq.P_segments;L=floor(N/P);tau_axis=double(tau_axis(:));D=numel(tau_axis);
n=(0:N-1)';t=n/cfg.signal.fs;
if cfg.acq.code_doppler_dewarp_enable
 fd=cfg.acq.code_doppler_reference_fd_hz;rate=cfg.acq.code_doppler_reference_fdot_hz_s;
 clock=n+cfg.signal.fs/cfg.signal.rf_carrier_hz*(fd*t+.5*rate*t.^2);
 src=interp1(clock,n,n,'linear',NaN);valid=isfinite(src);src(~valid)=n(~valid);
 rotation=exp(1j*2*pi*(fd*t+.5*rate*t.^2-fd*src/cfg.signal.fs-.5*rate*(src/cfg.signal.fs).^2));
else,src=n;valid=true(N,1);rotation=ones(N,1);end
lo=floor(src);hi=min(lo+1,N-1);w=src-lo;
rotation=rotation.*exp(-1j*2*pi*center_fd*t);
rows=cell(D,1);cols=cell(D,1);values=cell(D,1);start=round(start_s*cfg.signal.fs);
for j=1:D
 indices=find(valid & n<P*L);r=floor(n(indices)/L)+1+(j-1)*P;
 ri=start+n(indices)-tau_axis(j)-reference.sample_origin+1;
 assert(all(ri>=1&ri<=numel(reference.samples))&&all(ri==floor(ri)));
 a=conj(reference.samples(ri)).*rotation(indices);
 rows{j}=[r;r];cols{j}=[lo(indices)+1;hi(indices)+1];values{j}=[a.*(1-w(indices));a.*w(indices)];
end
B=sparse(vertcat(rows{:}),vertcat(cols{:}),vertcat(values{:}),P*D,N);C=B*B';C=(C+C')/2;
permutation=reshape(1:P*D,P,D)';permutation=permutation(:);
[lower,flag]=chol(C(permutation,permutation),'lower');whitening=[];
if flag~=0||min(abs(diag(lower)))<1e-4*max(abs(diag(lower)))
 [U,E]=eig(full(C),'vector');keep=real(E)>1e-8*max(real(E));
 whitening=diag(1./sqrt(real(E(keep))))*U(:,keep)';lower=[];
end
retained=size(C,1);if ~isempty(whitening),retained=size(whitening,1);end
model=struct('operator',B,'covariance',C,'lower',lower,'whitening',whitening, ...
 'permutation',permutation,'retained_dimension',retained,'tau_axis',tau_axis, ...
 'start_s',start_s,'center_fd_hz',center_fd,'dimension',P*D,'ridge',0, ...
 'noise_model','independent raw frames; proper AWGN and exact known PMF propagation');
end
