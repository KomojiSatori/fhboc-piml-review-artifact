function summary = aided_static_nci_sufficient_statistics( ...
        normalized_accumulator, residual_freq_axis_hz, coarse_bins_hz, ...
        tau_axis_samples, valid_freq_mask, cfg, trial, frame_count, ...
        prior_std, observation_count)
%AIDED_STATIC_NCI_SUFFICIENT_STATISTICS Static-state aided energy detector.
%   The cell statistic is the GLRT for a scale increase in independent
%   complex-Gaussian matched-filter outputs. Detection marginalizes the
%   static cell over the same Doppler/delay aiding prior used by V5.4;
%   localization uses the posterior MAP cell.

summary = empty_summary();
if isempty(normalized_accumulator)
    return;
end
if nargin < 10 || isempty(observation_count)
    observation_count = frame_count * ones(size(normalized_accumulator));
end
validateattributes(frame_count, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(prior_std, {'numeric'}, ...
    {'vector', 'numel', 2, 'positive', 'finite'});

accumulator = double(normalized_accumulator);
[freq_count, tau_count, coarse_count] = size(accumulator);
if ~isequal(size(valid_freq_mask), [freq_count, coarse_count]) || ...
        numel(residual_freq_axis_hz) ~= freq_count || ...
        numel(coarse_bins_hz) ~= coarse_count || ...
        numel(tau_axis_samples) ~= tau_count
    error('aided_static_nci_sufficient_statistics:Shape', ...
        'Accumulator dimensions and coordinate axes are inconsistent.');
end
valid_cells = repmat(reshape(logical(valid_freq_mask), ...
    [freq_count, 1, coarse_count]), [1, tau_count, 1]);
if ~isequal(size(observation_count), size(accumulator))
    error('aided_static_nci_sufficient_statistics:ObservationCount', ...
        'Observation-count map must match the accumulator.');
end
observation_count = double(observation_count);
valid_cells = valid_cells & observation_count == double(frame_count);

energy_ratio = accumulator ./ max(observation_count, 1);
cell_log_lr = zeros(size(accumulator));
active = valid_cells & energy_ratio > 1;
cell_log_lr(active) = accumulator(active) - observation_count(active) - ...
    observation_count(active) .* log(energy_ratio(active));
cell_log_lr(~valid_cells) = -inf;

fd_grid_hz = reshape(double(residual_freq_axis_hz(:)), [], 1, 1) + ...
    reshape(double(coarse_bins_hz(:)), 1, 1, []);
tau_grid_chips = reshape(double(tau_axis_samples(:)) * ...
    double(cfg.signal.fc) / double(cfg.signal.fs), 1, [], 1);
reference_time_s = max(frame_count - 1, 0) * ...
    double(cfg.signal.T_coh) / 2;
aiding_state = aided_static_nci_reference_state( ...
    cfg, trial, reference_time_s);
aiding_fd_hz = aiding_state.fd_hz;
aiding_tau_chips = aiding_state.tau_samples * ...
    double(cfg.signal.fc) / double(cfg.signal.fs);
log_prior = -0.5 * ((fd_grid_hz - aiding_fd_hz) / prior_std(2)).^2 - ...
    0.5 * ((tau_grid_chips - aiding_tau_chips) / prior_std(1)).^2;
log_prior = log_prior + zeros(size(accumulator));
log_prior(~valid_cells) = -inf;
log_prior = log_prior - logsumexp_finite(log_prior(valid_cells));

posterior_cell_score = cell_log_lr + log_prior;
[map_score, linear_idx] = max(posterior_cell_score(:));
[freq_idx, tau_idx, coarse_idx] = ind2sub( ...
    size(accumulator), linear_idx);
marginal_score = logsumexp_finite(posterior_cell_score(valid_cells));
estimated_fd_hz = double(coarse_bins_hz(coarse_idx)) + ...
    double(residual_freq_axis_hz(freq_idx));
estimated_tau_samples = double(tau_axis_samples(tau_idx));

truth = dynamic_truth_at_time(cfg, trial, reference_time_s);
summary.enabled = 1;
summary.frame_count = frame_count;
summary.marginal_score = marginal_score;
summary.map_score = map_score;
summary.cell_log_lr_at_map = cell_log_lr(linear_idx);
summary.log_prior_at_map = log_prior(linear_idx);
summary.est_fd_hz = estimated_fd_hz;
summary.est_tau_samples = estimated_tau_samples;
summary.reference_time_s = reference_time_s;
summary.fd_error_hz = abs(estimated_fd_hz - truth.fd_hz);
summary.tau_error_samples = abs(estimated_tau_samples - truth.tau_samples);
summary.normalized_accumulator_bytes_float32 = 4 * numel(accumulator);
summary.valid_map_cells = nnz(valid_cells);
summary.aiding_doppler_std_hz = double(prior_std(2));
summary.aiding_delay_std_chips = double(prior_std(1));
summary.aiding_reference_fd_hz = aiding_state.fd_hz;
summary.aiding_reference_tau_samples = aiding_state.tau_samples;
summary.full_frame_valid_cells = nnz(valid_cells);
end

function value = logsumexp_finite(values)
values = double(values(:));
values = values(isfinite(values));
if isempty(values)
    error('aided_static_nci_sufficient_statistics:NoFiniteCells', ...
        'No finite valid cell is available.');
end
maximum = max(values);
value = maximum + log(sum(exp(values - maximum)));
end

function summary = empty_summary()
fields = {'frame_count', 'marginal_score', 'map_score', ...
    'cell_log_lr_at_map', 'log_prior_at_map', 'est_fd_hz', ...
    'est_tau_samples', 'reference_time_s', 'fd_error_hz', ...
    'tau_error_samples', 'normalized_accumulator_bytes_float32', ...
    'valid_map_cells', 'aiding_doppler_std_hz', ...
    'aiding_delay_std_chips', 'aiding_reference_fd_hz', ...
    'aiding_reference_tau_samples', 'full_frame_valid_cells'};
summary = struct('enabled', 0);
for idx = 1:numel(fields)
    summary.(fields{idx}) = NaN;
end
end
