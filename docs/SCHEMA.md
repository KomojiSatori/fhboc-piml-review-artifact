# Coordinate and file schema

## Main JSONL

condition: 1 low dynamics, 2 high dynamics. geometry: 1..9 per condition.
observation is a one-based index local to a geometry. role is H1 or holdout_H0;
calibration records are supplied separately in calibration.json.
cn0 is dB-Hz for H1 and null for H0. noise_seed labels the standardized
six-frame stream. Paired strengths share that seed. All methods share the
geometry/observation key. stratum is nominal or transfer (perturbed).

score and threshold define strict score > threshold detection. stopped captures
return H0 and may have a null score. handoff is detection plus absolute
frequency error <=250 Hz and absolute delay error <=1 raw sample at 2.5 ms.
fd_error_hz and tau_error_samples are estimated minus true values; unavailable
readouts are null. seconds records the original per-capture method timing.
Timing_scope specifies the prepared-observation/prepared-library backend range.
The six observed frames span 0..6 ms; 2.5 ms is the state reference epoch.

calibration.json contains 400 scores for each of five reference methods and
their gates. The ascending rank 393 score sets each gate. Stopping policies
inherit the fixed-rank64 gate. geometry.json contains actual signal parameters,
truth, aiding, axes, metadata, supported dimensions and effective evaluation
tolerances. Its profiling_case describes campaign profiling, not pilot symbols.

controller_inputs.jsonl has observation and features. Feature order is in
models/feature_catalog.json. models/stop_variants.json supplies selected columns,
raw_weights and raw_bias: probability=sigmoid(X[:,columns]@raw_weights+raw_bias).
The fixed-score ablation continues when probability >0.1. Original standardized
coefficients and means/scales are also supplied. compiled_controller.json has
the original affine stop rule with its chosen threshold already absorbed.

## CSV and additional evidence

CSV Boolean values can be True/False or 1/0. Main component tables use
cn0_db_hz; recorded tables use sbr_db and source. recorded geometry IDs are
201/202 per source. Component and recorded CSVs retain calibration_H0 and
holdout_H0 roles and all method variants. The same background can be shared
across paired strength rows; use roles and source/geometry/record keys.

compact_timing_records.json contains the recorded paired replay, including
repeat and policy order. Its timing mixture is 304 captures and two repeats.
Controller input removal and upstream operator removal are distinct operations.
Reference thresholds, interval conventions and branch sampling parameters are
provided in config/ and evidence/. No hidden test labels are required by the
released inference weights; truth fields serve only numerical evaluation.
