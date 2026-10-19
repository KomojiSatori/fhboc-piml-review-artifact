# FHBOC acquisition: numerical reviewer archive

Archive ID: fhboc-review-v1-20261019. This package contains configurations,
physical/learned weights, per-observation event ledgers and portable analysis.

## Run

Python 3.10 or newer:

    python -m pip install -r requirements.txt
    python analysis/reproduce.py --output reproduced

The program verifies SHA-256 hashes before reading inputs, recomputes the main
performance cells and bootstrap interval, verifies calibration thresholds and
stopping identities, evaluates the supplied logistic weights, and generates
tables and figures. A successful run writes reproduced/verification.json.
Use --no-figures for numerical verification alone. No MATLAB, project checkout,
network access during analysis, or executable model pickle is required.

## Find a result

- docs/EXPERIMENTS.md distinguishes the main experiment, component experiment,
  recorded-background experiment and stopping development.
- docs/CLAIM_MAP.csv maps headline numbers to data and generated outputs.
- docs/SCHEMA.md defines coordinates, labels, seeds and timing fields.
- models/ contains physical inference parameters and portable logistic weights.
- data/main/ contains 18 complete non-calibration event ledgers, calibration
  scores, actual geometry metadata and 58-input vectors.
- data/component/ and data/recorded/ contain the separate component and
  recorded-background event tables.
- data/stopping_development/ contains original development features and records.
- evidence/ retains component controls, input ablations and recorded timing.
- source_reference/ contains the implementation and its local source dependency
  closure, with machine/user paths scrubbed. The runnable numerical entry is
  analysis/reproduce.py; historical simulation drivers retain their original
  experiment layout and require raw observations/response libraries separately.
- figures_reference/ contains the supplied manuscript figure assets; regenerated
  numerical plots use the same ledgers with a compact reviewer layout.

Elapsed-time fields reproduce recorded timing summaries. Running this analysis
measures analysis time; it does not remeasure acquisition-backend latency.
Full raw I/Q recordings and large PMF/template banks are not part of this
numerical archive. Public recording sources are listed in docs/SOURCE_AND_REUSE.md.
MANIFEST.json covers every supplied input; PROVENANCE.json records source hashes
and transformations. The ZIP has fixed timestamps and no local owner metadata.
