function truth = eci_los_trajectory_truth(cfg, time_s, base_tau_samples)
%ECI_LOS_TRAJECTORY_TRUTH Interpolate the configured ECI/LOS trajectory.
% The CSV is sampled in absolute time.  Phase is returned in cycles relative
% to t=0; callers can subtract the frame-start value for local synthesis.

if nargin < 3 || isempty(base_tau_samples)
    base_tau_samples = 0;
end
if ~isfield(cfg, 'trajectory') || ...
        ~isfield(cfg.trajectory, 'enable') || ~cfg.trajectory.enable
    error('eci_los_trajectory_truth:Disabled', ...
        'ECI/LOS trajectory mode is not enabled.');
end
path = char(cfg.trajectory.path);
persistent cached_path cached_data
if isempty(cached_path) || ~strcmp(cached_path, path)
    if exist(path, 'file') ~= 2
        error('eci_los_trajectory_truth:MissingFile', ...
            'Trajectory CSV does not exist: %s', path);
    end
    data = readtable(path, 'VariableNamingRule', 'preserve');
    required = {'time_s', 'doppler_hz', 'doppler_rate_hz_s', ...
        'carrier_phase_rad'};
    if ~all(ismember(required, data.Properties.VariableNames))
        error('eci_los_trajectory_truth:Columns', ...
            'Trajectory CSV is missing required columns.');
    end
    cached_path = path;
    cached_data = struct();
    cached_data.time_s = double(data.time_s(:));
    cached_data.fd_hz = double(data.doppler_hz(:));
    cached_data.fdot_hz_s = double(data.doppler_rate_hz_s(:));
    cached_data.phase_cycles = double(data.carrier_phase_rad(:)) / (2*pi);
    [cached_data.time_s, order] = sort(cached_data.time_s);
    cached_data.fd_hz = cached_data.fd_hz(order);
    cached_data.fdot_hz_s = cached_data.fdot_hz_s(order);
    cached_data.phase_cycles = cached_data.phase_cycles(order);
end

output_size = size(time_s);
t = double(time_s(:));
range_t = [cached_data.time_s(1), cached_data.time_s(end)];
if any(t < range_t(1) - 1e-12 | t > range_t(2) + 1e-12)
    error('eci_los_trajectory_truth:TimeRange', ...
        'Requested time %.9g--%.9g s is outside trajectory %.9g--%.9g s.', ...
        min(t), max(t), range_t(1), range_t(2));
end
truth = struct();
truth.time_s = t;
truth.fd_hz = interp1(cached_data.time_s, cached_data.fd_hz, t, 'pchip');
truth.fdot_hz_s = interp1(cached_data.time_s, cached_data.fdot_hz_s, t, 'pchip');
truth.phase_cycles = interp1(cached_data.time_s, cached_data.phase_cycles, t, 'pchip');
fs = get_required_signal_scalar(cfg, 'fs');
rf = get_required_signal_scalar(cfg, 'rf_carrier_hz');
truth.tau_samples = double(base_tau_samples) - fs / rf * ...
    truth.phase_cycles;
truth.tau_rate_samples_s = -fs / rf .* truth.fd_hz;
truth.fractional_doppler = truth.fd_hz / rf;
truth.time_s = reshape(truth.time_s, output_size);
truth.fd_hz = reshape(truth.fd_hz, output_size);
truth.fdot_hz_s = reshape(truth.fdot_hz_s, output_size);
truth.phase_cycles = reshape(truth.phase_cycles, output_size);
truth.tau_samples = reshape(truth.tau_samples, output_size);
truth.tau_rate_samples_s = reshape(truth.tau_rate_samples_s, output_size);
truth.fractional_doppler = reshape(truth.fractional_doppler, output_size);
end

function value = get_required_signal_scalar(cfg, name)
if ~isfield(cfg, 'signal') || ~isfield(cfg.signal, name) || ...
        ~isfinite(cfg.signal.(name)) || cfg.signal.(name) <= 0
    error('eci_los_trajectory_truth:SignalConfig', ...
        'cfg.signal.%s must be a positive finite scalar for ECI mode.', name);
end
value = double(cfg.signal.(name));
end
