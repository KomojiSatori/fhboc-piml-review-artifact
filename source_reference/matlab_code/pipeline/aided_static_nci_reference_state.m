function state = aided_static_nci_reference_state(cfg, trial, time_s)
%AIDED_STATIC_NCI_REFERENCE_STATE Propagate aiding under the static model.

if isfield(cfg, 'acq')
    assumed_fdot_hz_s = double(get_optional_field(cfg.acq, ...
        'code_doppler_aided_fdot_hz_s', 0));
else
    assumed_fdot_hz_s = 0;
end
state = struct();
state.time_s = double(time_s);
state.assumed_fdot_hz_s = assumed_fdot_hz_s;
state.fd_hz = double(trial.aiding_fd_hz) + ...
    assumed_fdot_hz_s * state.time_s;
state.tau_samples = double(trial.aiding_tau_samples);

enabled = isfield(cfg.signal, 'code_doppler_enable') && ...
    cfg.signal.code_doppler_enable;
has_rf = isfield(cfg.signal, 'rf_carrier_hz') && ...
    isfinite(cfg.signal.rf_carrier_hz) && cfg.signal.rf_carrier_hz > 0;
if ~enabled || ~has_rf
    return;
end

integrated_doppler_cycles = double(trial.aiding_fd_hz) * state.time_s + ...
    0.5 * assumed_fdot_hz_s * state.time_s^2;
state.tau_samples = state.tau_samples - double(cfg.signal.fs) / ...
    double(cfg.signal.rf_carrier_hz) * integrated_doppler_cycles;
end

function value = get_optional_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
