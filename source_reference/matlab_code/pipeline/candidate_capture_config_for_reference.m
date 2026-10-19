function capture_cfg = candidate_capture_config_for_reference(cfg, reference)
%CANDIDATE_CAPTURE_CONFIG_FOR_REFERENCE Apply per-trial aided search bounds.

capture_cfg = cfg;
if get_optional_field(cfg.acq, 'code_doppler_dewarp_enable', false)
    capture_cfg.acq.code_doppler_reference_fd_hz = ...
        double(reference.fd_hz);
    capture_cfg.acq.code_doppler_reference_fdot_hz_s = ...
        double(get_optional_field(cfg.acq, ...
        'code_doppler_aided_fdot_hz_s', 0));
end
if ~get_optional_field(cfg.sweep, 'local_fd_search_enable', false)
    return;
end
reference_mode = lower(char(get_optional_field(cfg.sweep, ...
    'local_search_reference_mode', 'instantaneous')));
if ~strcmp(reference_mode, 'aiding')
    return;
end

% Coarse wipe-off centres and the residual Doppler output window are
% independent controls. In particular, a zero coarse-search half-width means
% one wipe-off centre; it must not collapse the residual FFT to one bin.
output_half_hz = get_optional_field(cfg.sweep, ...
    'local_fd_output_half_width_hz', NaN);
if ~isscalar(output_half_hz) || ~isfinite(output_half_hz)
    output_half_hz = get_optional_field(cfg.sweep, ...
        'local_fd_search_half_hz', NaN);
end
if ~isscalar(output_half_hz) || ~isfinite(output_half_hz) || ...
        output_half_hz <= 0
    output_half_hz = get_optional_field(cfg.acq, ...
        'doppler_output_half_width_hz', 10000);
end
if ~isscalar(output_half_hz) || ~isfinite(output_half_hz) || output_half_hz < 0
    return;
end
center_offset_hz = double(get_optional_field(cfg.sweep, ...
    'local_fd_search_center_offset_hz', 0));
center_hz = double(reference.fd_hz) + center_offset_hz;
capture_cfg.acq.output_fd_min_hz = center_hz - output_half_hz;
capture_cfg.acq.output_fd_max_hz = center_hz + output_half_hz;
end

function value = get_optional_field(s, field_name, default_value)
if isfield(s, field_name)
    value = s.(field_name);
else
    value = default_value;
end
end
