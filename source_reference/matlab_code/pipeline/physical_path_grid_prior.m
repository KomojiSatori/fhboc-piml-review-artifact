function [states,log_uniform,log_aided,axes,density] = physical_path_grid_prior(cfg,aiding,options)
% Fixed quadrature over the entire aided domain, with normalized cell masses.
% No observation-dependent pruning or true motion/CN0 inputs.
get=@(name,value) local_option(options,name,value);
f=aiding.fd_hz+(-get('fd_half_width_hz',2500):get('fd_step_hz',500):get('fd_half_width_hz',2500));
half=get('tau_half_width_chips',1.5)*cfg.signal.fs/cfg.signal.fc;
low=aiding.tau_samples-half;high=aiding.tau_samples+half;
tau=unique([low,ceil(low/get('tau_step_samples',1))*get('tau_step_samples',1):get('tau_step_samples',1):high,high]);
bounds=get('rate_bounds_hz_s',[-250000 250000]);
rate=unique([bounds(1):get('rate_step_hz_s',50000):bounds(2),bounds(2)]);
assert(numel(f)>1&&numel(tau)>1&&numel(rate)>1);
[fu,fg]=masses(f,f(1),f(end),aiding.fd_hz,get('aiding_fd_sigma_hz',1000));
[tu,tg]=masses(tau,low,high,aiding.tau_samples,get('aiding_tau_sigma_chips',.5)*cfg.signal.fs/cfg.signal.fc);
[ru,~]=masses(rate,bounds(1),bounds(2),0,Inf);
[F,T,R]=ndgrid(f,tau,rate);states=[F(:),T(:),R(:)];
[U,V,W]=ndgrid(fu,tu,ru);uniform=U(:).*V(:).*W(:);
[U,V,W]=ndgrid(fg,tg,ru);aided=U(:).*V(:).*W(:);
log_uniform=log(uniform/sum(uniform));log_aided=log(aided/sum(aided));
axes=struct('fd_hz',f,'tau_samples',tau,'rate_hz_s',rate);
% Quadrature cell masses above are for integration, never MAP readout.
% A boundary cell is smaller but its continuous prior density is not lower.
volume=(f(end)-f(1))*(high-low)*(bounds(2)-bounds(1));
uniform_density=repmat(-log(volume),size(states,1),1);
sf=get('aiding_fd_sigma_hz',1000);st=get('aiding_tau_sigma_chips',.5)*cfg.signal.fs/cfg.signal.fc;
aided_density=log_density(states(:,1),f(1),f(end),aiding.fd_hz,sf) ...
 +log_density(states(:,2),low,high,aiding.tau_samples,st)-log(bounds(2)-bounds(1));
density=struct('log_uniform',uniform_density,'log_aided',aided_density);
end
function value=log_density(axis,low,high,mean_value,sigma)
if isinf(sigma)
 value=repmat(-log(high-low),size(axis));
else
 z=.5*(erf((high-mean_value)/(sqrt(2)*sigma))-erf((low-mean_value)/(sqrt(2)*sigma)));
 value=-.5*((axis-mean_value)/sigma).^2-log(sqrt(2*pi)*sigma*z);
end
end
function [uniform,gaussian]=masses(axis,low,high,mean_value,sigma)
edges=[low,(axis(1:end-1)+axis(2:end))/2,high];
uniform=diff(edges)/(high-low);
if isinf(sigma),gaussian=uniform;else
 validateattributes(sigma,{'numeric'},{'scalar','finite','positive'});
 cdf=.5*(1+erf((edges-mean_value)/(sqrt(2)*sigma)));
 gaussian=diff(cdf);gaussian=gaussian/sum(gaussian);
end
assert(all(uniform>0)&&all(gaussian>0));
end
function value=local_option(s,name,value)
if isfield(s,name),value=s.(name);end
end
