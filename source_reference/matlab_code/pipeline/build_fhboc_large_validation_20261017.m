function build_fhboc_large_validation_20261017(plan_path,condition,geometry,pilot)
% Fresh six-frame captures, all states, full templates and nested POD library.
p=jsondecode(fileread(plan_path));
if pilot,tag=sprintf('profile_c%d_g%d',condition,geometry);else,tag=sprintf('c%d_g%d',condition,geometry);end
out=fullfile(p.bulk_root,tag);if ~exist(out,'dir'),mkdir(out);end
if exist(fullfile(out,'export_complete.json'),'file'),fprintf('REUSE %s\n',tag);return;end
maxNumCompThreads(4);started=tic;
K=load(sprintf('results/diagnostics/fhboc_cn034_validation_20261014/condition_%d_geometry_1/pilot.mat',condition),'cfg');cfg=K.cfg;
cfg.sweep.local_search_reference_mode='aiding';cfg.sweep.local_search_delay_reference_mode='aiding_propagated';
offset=condition*1000000+geometry*10000;
truth_stream=RandStream('mt19937ar','Seed',p.seeds.truth+offset);
transfer=mod(geometry,3)==0 && ~pilot;
fd=[50000 1500000];rate=[3230 200230];
if transfer
 fd=fd(condition)*(1+.10*(2*rand(truth_stream)-1));
 rate=rate(condition)*(1+.15*(2*rand(truth_stream)-1));
 if mod(geometry/3,2)==1,rate=-rate;end
else,fd=fd(condition);rate=rate(condition);end
trial=struct('fd_hz',fd,'fdot_hz_s',rate,'tau_samples',650+100*rand(truth_stream), ...
 'carrier_phase_cycles',rand(truth_stream),'hop_seed',p.seeds.hop+offset);
aid_stream=RandStream('mt19937ar','Seed',p.seeds.aiding+offset);draw=max(-2,min(2,randn(aid_stream,2,1)));
trial.aiding_fd_hz=trial.fd_hz+1000*draw(1);
trial.aiding_tau_samples=trial.tau_samples+.5*cfg.signal.fs/cfg.signal.fc*draw(2);
aiding=struct('fd_hz',trial.aiding_fd_hz,'tau_samples',trial.aiding_tau_samples);
ro=struct('code_epoch_chips',20000000+offset,'hop_origin_chip',-8192,'reference_halo_samples',4096);
reference=fhboc_continuous_reference_20261013(cfg,trial.hop_seed,6,ro);
clean=fhboc_continuous_waveform_batch_20261013(reference,cfg,0,trial.tau_samples,trial.fd_hz,trial.fdot_hz_s,trial.carrier_phase_cycles,6*reference.frame_samples);
signal_power=mean(abs(clean).^2);
[states,log_uniform,log_aided,axes]=physical_path_grid_prior(cfg,aiding,p.grid);H=size(states,1);
if pilot,n=2;nc=10;nt=20;else,n=p.H1_per_strength;nc=p.calibration_H0;nt=p.test_H0;end
levels=p.cn0_db_hz(:)';O=numel(levels)*n+nc+nt;base_count=n+nc+nt;
base_seed=p.seeds.noise+offset;
base_records=[1:n 2001:2000+nc 4001:4000+nt];
% One complete RNG stream per six-frame noise realization.
noise=complex(zeros(numel(clean),base_count));
for o=1:base_count
 stream=RandStream('mt19937ar','Seed',base_seed+base_records(o));
 noise(:,o)=(randn(stream,size(clean))+1j*randn(stream,size(clean)))/sqrt(2);
end
metadata=repmat(struct('observation',0,'record_index',0,'role','','signal',false,'cn0_db_hz',NaN,'noise_seed',0),1,O);
o=0;
for cn=levels
 for j=1:n,o=o+1;metadata(o)=struct('observation',o,'record_index',j,'role','H1','signal',true,'cn0_db_hz',cn,'noise_seed',base_seed+j);end
end
for j=1:nc+nt
 o=o+1;if j<=nc,role='calibration_H0';else,role='holdout_H0';end
 metadata(o)=struct('observation',o,'record_index',base_records(n+j),'role',role,'signal',false,'cn0_db_hz',NaN,'noise_seed',base_seed+base_records(n+j));
end
G=8;region=discretize(states(:,2),linspace(min(states(:,2)),max(states(:,2)),G+1));
train_ids=cell(G,1);region_ids=cell(G,1);stream=RandStream('mt19937ar','Seed',p.seeds.pod+offset);
for g=1:G,region_ids{g}=find(region==g);ii=region_ids{g};train_ids{g}=ii(randperm(stream,numel(ii),128));end
weights=.5*exp(log_uniform)+.5*exp(log_aided);stream=RandStream('mt19937ar','Seed',p.seeds.probe+offset);
[~,order]=sort(-log(rand(stream,H,1))./weights);probe_ids=sort(order(1:512));
probe_retention=zeros(512,6,2,'single');lookup=zeros(H,1);lookup(probe_ids)=1:512;
basis=cell(6,G);coefficients=cell(6,G);whitened_observations=cell(6,1);observations=cell(6,1);
template_energy=zeros(H,6);observation_energy=zeros(6,O);dimensions=zeros(1,6);
checks=zeros(6,4);timing=zeros(6,4);frame_tau_axes=zeros(51,6);frame_centers=zeros(1,6);
full_precision_probe=cell(6,1);
eta=-cfg.signal.fs/cfg.signal.rf_carrier_hz;
for k=1:6
 clock=tic;t=(k-1)*.001;ref=candidate_local_search_reference(cfg,trial,struct('frame_start_time_s',t));
 cc=candidate_capture_config_for_reference(cfg,ref);tau_axis=round(ref.tau_samples)+(-cfg.acq.search_half_width:cfg.acq.search_half_width);
 model=fhboc_continuous_pmf_operator_20261013(reference,cc,t,tau_axis,ref.fd_hz);
 frame=struct('reference',reference,'operator_model',model,'center_fd_hz',ref.fd_hz);
 bank=fhboc_continuous_response_bank_20261013(frame,cfg,p.response);timing(k,1)=toc(clock);
 ids=(k-1)*reference.frame_samples+(1:reference.frame_samples);
 rawnoise=model.operator*noise(ids,:);rawclean=model.operator*clean(ids);raw=complex(zeros(size(rawnoise,1),O));
 for ci=1:numel(levels)
  ii=(ci-1)*n+(1:n);sigma=sqrt(signal_power*cfg.signal.fs/10^(levels(ci)/10));raw(:,ii)=rawclean+sigma*rawnoise(:,1:n);
 end
 raw(:,numel(levels)*n+1:end)=rawnoise(:,n+1:end);observations{k}=raw;
 Z=fhboc_pmf_whiten(model,raw);whitened_observations{k}=Z;observation_energy(k,:)=sum(abs(Z).^2,1);dimensions(k)=size(Z,1);
 frame_tau_axes(:,k)=tau_axis(:);frame_centers(k)=ref.fd_hz;
 sigma=sqrt(signal_power*cfg.signal.fs/10^(levels(1)/10));rx=clean(ids)+sigma*noise(ids,1);
 direct=fhboc_continuous_pmf_direct_20261013(rx,reference,cc,t,tau_axis,ref.fd_hz);
 checks(k,1)=norm(raw(:,1)-direct(:))/norm(direct(:));assert(checks(k,1)<1e-8);
 ss=states;ss(:,2)=states(:,2)+eta*(states(:,1)*t+.5*states(:,3)*t*t);ss(:,1)=states(:,1)+states(:,3)*t;
 phase=exp(2j*pi*(states(:,1)*t+.5*states(:,3)*t*t));clock=tic;
 for g=1:G
  s=ss(train_ids{g},:);V=fhboc_pmf_whiten(model,fhboc_pmf_bank_response(bank,cfg,s(:,2),s(:,1),s(:,3)));
  [U,~,~]=svd(V./sqrt(sum(abs(V).^2,1)),'econ');basis{k,g}=U(:,1:64);
  coefficients{k,g}=complex(zeros(64,numel(region_ids{g})));
 end
 timing(k,2)=toc(clock);clock=tic;
 filename=fullfile(out,sprintf('full_frame_%d.c64',k));fid=fopen(filename,'w');assert(fid>=0);
 cleanup=onCleanup(@()fclose(fid));
 for first=1:256:H
  ii=first:min(first+255,H);s=ss(ii,:);
  V=fhboc_pmf_whiten(model,fhboc_pmf_bank_response(bank,cfg,s(:,2),s(:,1),s(:,3)));
  q=sum(abs(V).^2,1);template_energy(ii,k)=q';
  packed=[real(V(:)).';imag(V(:)).'];fwrite(fid,packed,'single');
  for g=1:G
   jj=find(region(ii)==g);if isempty(jj),continue;end
   global_ids=ii(jj);[~,local]=ismember(global_ids,region_ids{g});
   C=(basis{k,g}'*V(:,jj)).*(phase(global_ids).'./sqrt(q(jj)));
   coefficients{k,g}(:,local)=C;
   selected=find(lookup(global_ids)>0);pp=lookup(global_ids(selected));
   for r=1:2,probe_retention(pp,k,r)=single(sum(abs(C(1:16*r,selected)).^2,1)');end
  end
  if first==1
   full_precision_probe{k}=V(:,1:8);
   z=Z(:,1:min(3,O));a=z'*V(:,1:8);b=double(single(z))'*double(single(V(:,1:8)));
   checks(k,2)=norm(a-b,'fro')/norm(a,'fro');assert(checks(k,2)<2e-6);
   probe=ss(ii(1:3),:);exact=fhboc_pmf_whiten(model,model.operator*fhboc_continuous_waveform_batch_20261013(reference,cfg,t,probe(:,2),probe(:,1),probe(:,3)));
   checks(k,3)=norm(exact-V(:,1:3),'fro')/norm(exact,'fro');assert(checks(k,3)<.10);
  end
 end
 clear cleanup;timing(k,3)=toc(clock);
 file=dir(filename);assert(file.bytes==8*H*dimensions(k));
 fprintf('LARGE BANK %s frame%d H%d O%d setup%.1f full%.1f checks%.3g %.3g %.3g\n',tag,k,H,O,timing(k,1),timing(k,3),checks(k,1:3));
end
clear noise rawnoise raw Z;
clock=tic;L=floor(reference.frame_samples/cfg.acq.P_segments);
fd=trial.aiding_fd_hz+(-cfg.acq.N_fft/2:cfg.acq.N_fft/2-1)'*cfg.signal.fs/(L*cfg.acq.N_fft);mask=abs(fd-trial.aiding_fd_hz)<=2500;
ar=aided_static_nci_reference_state(cfg,trial,.0025);delay=round(ar.tau_samples)+(-cfg.acq.search_half_width:cfg.acq.search_half_width);
keep=abs(delay-ar.tau_samples)<=1.5*cfg.signal.fs/cfg.signal.fc;nci_scores=zeros(O,1);nci_estimates=zeros(O,2);nci_seconds=zeros(O,1);
for o=1:O
 tick=tic;accum=zeros(nnz(mask),numel(delay));
 for k=1:6
  f=fftshift(fft(reshape(observations{k}(:,o),cfg.acq.P_segments,[]),cfg.acq.N_fft,1),1);pk=abs(f(mask,:)).^2;accum=accum+pk/mean(pk(:));
 end
 a=aided_static_nci_sufficient_statistics(accum(:,keep),fd(mask)-trial.aiding_fd_hz,trial.aiding_fd_hz,delay(keep),true(nnz(mask),1),cfg,trial,6,[.5 1000]);
 nci_scores(o)=a.marginal_score;nci_estimates(o,:)=[a.est_fd_hz a.est_tau_samples];nci_seconds(o)=toc(tick);
end
frontend=struct('sample_rate_hz',cfg.signal.fs,'code_rate_hz',cfg.signal.fc,'rf_carrier_hz',cfg.signal.rf_carrier_hz, ...
 'pmf_segments',cfg.acq.P_segments,'pmf_segment_length',L,'frame_duration_s',.001,'frame_samples',reference.frame_samples, ...
 'frame_tau_axes',frame_tau_axes,'frame_centers',frame_centers,'aiding_fd_hz',trial.aiding_fd_hz, ...
 'aiding_tau_samples',trial.aiding_tau_samples,'frequency_half_width_hz',2500,'rate_half_width_hz_s',250000,'tau_half_width_samples',1.5*cfg.signal.fs/cfg.signal.fc);
truth_reference=[trial.fd_hz+trial.fdot_hz_s*.0025,trial.tau_samples+eta*(trial.fd_hz*.0025+.5*trial.fdot_hz_s*.0025^2)];
information=struct('condition',condition,'geometry',geometry,'pilot',pilot,'transfer',transfer,'code_epoch_chips',ro.code_epoch_chips, ...
 'hop_seed',trial.hop_seed,'aiding_seed',p.seeds.aiding+offset,'source','fresh raw AWGN streams; C/N0-paired noise', ...
 'full_dictionary_precision','complex single; q/POD/whitening double; independently checked inner products','setup_seconds',toc(started));
save(fullfile(out,'geometry.mat'),'cfg','states','axes','metadata','aiding','truth_reference','template_energy','observation_energy','dimensions','log_uniform','log_aided','information','timing','checks','trial','full_precision_probe','-v7');
save(fullfile(out,'online_bank.mat'),'basis','coefficients','whitened_observations','region_ids','probe_ids','-v7');
save(fullfile(out,'probes.mat'),'probe_ids','probe_retention','weights','-v7');
save(fullfile(out,'frontend.mat'),'observations','frontend','nci_scores','nci_estimates','nci_seconds','-v7');
value=struct('complete',true,'observations',O,'hypotheses',H,'setup_seconds',toc(started),'static_nci_seconds',toc(clock),'checks',checks);
fid=fopen(fullfile(out,'export_complete.json'),'w');fwrite(fid,jsonencode(value));fclose(fid);
fprintf('COMPLETE LARGE BANK %s O%d %.1fs\n',tag,O,toc(started));
end
