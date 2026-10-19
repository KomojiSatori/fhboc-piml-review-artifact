function reference = candidate_local_search_reference( ...
        cfg, initial_trial, frame_trial)
%CANDIDATE_LOCAL_SEARCH_REFERENCE Resolve the frozen local-search center.

mode = lower(char(get_optional_field(cfg.sweep, ...
    'local_search_reference_mode', 'instantaneous')));
switch mode
    case 'instantaneous'
        reference = frame_trial;
    case 'initial'
        reference = frame_trial;
        reference.fd_hz = initial_trial.fd_hz;
        reference.tau_samples = initial_trial.tau_samples;
    case 'aiding'
        required = {'aiding_fd_hz', 'aiding_tau_samples'};
        for idx = 1:numel(required)
            if ~isfield(initial_trial, required{idx}) || ...
                    ~isfinite(initial_trial.(required{idx}))
                error('candidate_local_search_reference:MissingAiding', ...
                    'Initial trial lacks finite %s.', required{idx});
            end
        end
        reference = frame_trial;
        reference.fd_hz = initial_trial.aiding_fd_hz;
        reference.tau_samples = initial_trial.aiding_tau_samples;
        delay_mode = lower(char(get_optional_field(cfg.sweep, ...
            'local_search_delay_reference_mode', 'initial_aiding')));
        if strcmp(delay_mode, 'aiding_propagated')
            time_s = double(get_optional_field(frame_trial, ...
                'frame_start_time_s', 0));
            predicted = aided_static_nci_reference_state(cfg, ...
                initial_trial, time_s);
            reference.tau_samples = predicted.tau_samples;
        elseif ~strcmp(delay_mode, 'initial_aiding')
            error('candidate_local_search_reference:UnknownDelayMode', ...
                'Unknown local delay reference mode: %s.', delay_mode);
        end
    otherwise
        error('candidate_local_search_reference:UnknownMode', ...
            'Unknown local_search_reference_mode: %s.', mode);
end
end

function value = get_optional_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
