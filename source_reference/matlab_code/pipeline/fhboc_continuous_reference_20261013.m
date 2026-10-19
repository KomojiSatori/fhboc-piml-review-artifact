function reference=fhboc_continuous_reference_20261013(cfg,hop_seed,frames,options)
% Known transmitter reference with an absolute P-code epoch and sample halo.
% Hop sequence and subcarrier phase start at a FIXED chip origin, not each frame.
assert(exist('gpsPCode','file')==2&&cfg.signal.use_gps_pcode,'Actual GPS P-code toolbox required.');
assert(strcmp(cfg.signal.hop_pattern_mode,'uniform'),'First continuous pilot supports uniform per-chip hops.');
fs=cfg.signal.fs;fc=cfg.signal.fc;N=round(frames*cfg.signal.T_coh*fs);
halo=options.reference_halo_samples;origin=-halo;axis=(origin:N+halo-1)';
position=axis*fc/fs;chips=floor(position);first=min(chips);last=max(chips);
assert(options.code_epoch_chips+first>=0&&first>=options.hop_origin_chip);
p=gpsPCode();p.PRNID=cfg.signal.prn_id;p.InitialStateFormat='chips';
p.InitialNumChipsElapsed=options.code_epoch_chips+first;p.OutputCodeLength=last-first+1;
bits=1-2*double(p());bits=bits(:);
levels=(cfg.signal.alpha_min:cfg.signal.alpha_step:cfg.signal.alpha_max)*cfg.signal.f0;
stream=RandStream('mt19937ar','Seed',double(hop_seed));
hop=randi(stream,numel(levels),last-options.hop_origin_chip+1,1);
frequencies=levels(hop);frequencies=frequencies(:);
phase_before=[0;cumsum(frequencies/fc)];index=chips-options.hop_origin_chip+1;
cycles=phase_before(index)+(frequencies(index)/fc).*(position-chips);
subcarrier=1-2*(mod(cycles,1)>=.5);
samples=bits(chips-first+1).*subcarrier;
reference=struct('samples',samples,'sample_origin',origin,'sample_rate_hz',fs, ...
 'frame_samples',round(cfg.signal.T_coh*fs),'frames',frames,'code_epoch_chips',options.code_epoch_chips, ...
 'hop_seed',hop_seed,'hop_origin_chip',options.hop_origin_chip,'first_chip',first, ...
 'code_bits',bits,'chip_axis',(first:last)','model_version','continuous_pcode_phase_v1_20261013');
end
