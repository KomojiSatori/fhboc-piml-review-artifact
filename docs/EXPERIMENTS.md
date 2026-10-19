# Experiment identifiers

| ID | Observations and conditions | Role in manuscript |
|---|---|---|
| MAIN_GAUSSIAN | 18 geometries; 6 frames; 100 MS/s; 30..38 dB-Hz; 80 H1/geometry/strength, 400 calibration H0 and 800 test H0/geometry | Section 5.1, Figs. 4/7/8; full, rank64, stopping and baseline comparisons |
| COMPONENT_GAUSSIAN_6F | 4 geometries; 32/36 dB-Hz; 400 H1 + 200 calibration H0 + 400 test H0 | Section 5.3; G_ind/G_coh and P/A/S/F mechanism comparisons |
| RECORDED_BACKGROUND | CTTC/TEXBAT; 4 MS/s scaled target; -34/-30 dB SBR; 2 geometries/source | Section 5.2 and recorded components in 5.3; Figs. 5/6 |
| STOPPING_DEVELOPMENT | 8 historical geometries; 600 observations, of which 370 non-calibration observations fit stopping | Section 3.6; model/threshold selection |

MAIN_GAUSSIAN contains 34,560 observations in total: 12,960 H1, 7,200
calibration H0 and 14,400 test H0. Each non-calibration observation has seven
method events (191,520 rows). The nine signal strengths reuse a standardized
noise stream within a geometry: 1,440 distinct H1 noise streams, not 12,960.

Geometries with g modulo 3 equal to zero form the perturbed stratum (legacy
label transfer); the other six per dynamic regime form the nominal stratum.
The high-dynamic nominal 34 dB-Hz cell contains 480 H1: PIML 329, BASIC 243,
compensated NCI 180 correct initializations (68.54%, 50.625%, 37.50%).
These three numbers all belong to MAIN_GAUSSIAN. The component experiment
contains no 34 dB-Hz condition.

Recorded-background tables report two operating points: thresholds from
independent calibration backgrounds, and descriptive empirical 4/200 test-H0
false alarms. Their output names keep these operating points separate.
Input-block variants and matched-retention ranking are retrospective analyses
of completed evaluation records; their fit and threshold-selection protocols
are retained in evidence/controller_feature_ablation_protocol.json.
