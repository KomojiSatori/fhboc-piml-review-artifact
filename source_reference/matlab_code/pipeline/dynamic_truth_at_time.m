function truth = dynamic_truth_at_time(cfg, trial, time_s)
%DYNAMIC_TRUTH_AT_TIME Evaluate carrier and code truth on one time axis.

truth = struct();
truth.time_s = time_s;
if isfield(cfg, 'trajectory') && isfield(cfg.trajectory, 'enable') && ...
        cfg.trajectory.enable
    base_tau = get_optional_field(trial, 'initial_tau_samples', ...
        get_optional_field(trial, 'tau_samples', 0));
    traj = eci_los_trajectory_truth(cfg, time_s, base_tau);
    truth.fd_hz = traj.fd_hz;
    truth.fdot_hz_s = traj.fdot_hz_s;
    truth.carrier_phase_cycles = traj.phase_cycles;
    truth.tau_samples = traj.tau_samples;
    truth.tau_rate_samples_s = traj.tau_rate_samples_s;
    truth.fractional_doppler = traj.fractional_doppler;
    return;
end
truth.fd_hz = trial.fd_hz + trial.fdot_hz_s * time_s;
truth.carrier_phase_cycles = trial.fd_hz * time_s + ...
    0.5 * trial.fdot_hz_s * time_s.^2;
truth.tau_samples = trial.tau_samples;
truth.tau_rate_samples_s = 0;
truth.fractional_doppler = 0;

enabled = isfield(cfg.signal, 'code_doppler_enable') && ...
    cfg.signal.code_doppler_enable;
has_rf = isfield(cfg.signal, 'rf_carrier_hz') && ...
    isfinite(cfg.signal.rf_carrier_hz) && cfg.signal.rf_carrier_hz > 0;
if ~enabled || ~has_rf
    return;
end

rf_hz = cfg.signal.rf_carrier_hz;
integrated_fractional_doppler_s = truth.carrier_phase_cycles / rf_hz;
truth.fractional_doppler = truth.fd_hz / rf_hz;
truth.tau_samples = trial.tau_samples - ...
    cfg.signal.fs * integrated_fractional_doppler_s;
truth.tau_rate_samples_s = -cfg.signal.fs * truth.fractional_doppler;
end

function value = get_optional_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
