# Changelog

All notable changes to Q-TRACE are documented here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. This
project does not yet use formal semantic version tags (pre-1.0,
research-code stage); entries are dated instead.

Documenting these fixes explicitly, including ones that changed
reported numbers, is intentional: several affect results that appear
in the accompanying manuscript, and a reviewer or reproducer should be
able to see exactly what changed and why rather than discover a
discrepancy between an old cached run and the current code silently.

## [Unreleased]

### Added
- `evaluation.run_multiseed`: repeats the full dataset-generation +
  training + evaluation pipeline across N independent seeds and
  aggregates mean ± std per (split, model, metric). Supports resuming
  an interrupted multi-seed sweep.
- `evaluation.run_sweep_experiment`: zero-day detection recall/AUROC
  swept across a channel-loss × detector-efficiency grid.
- `evaluation.run_adversarial_evasion`: physically-constrained
  evolutionary search for minimum-detectable attack instances, with
  stagnation-triggered search-radius re-inflation.
- `quantum.ibm_runtime_job`: real-hardware execution script for IBM
  Quantum backends (kernel and VQC-inference modes), budgeted to a
  configurable QPU-time ceiling with a pre-flight static estimate.
- `quantum.decode_hardware_counts`: decodes raw hardware measurement
  counts into class-1 probabilities using VQC's parity interpretation.
- `evaluation.metrics.PhysicsConsistencyModel`: a properly fitted
  physics-consistency term for the Q-TRACE risk score (see Fixed,
  below).
- `evaluation.metrics.gating_effectiveness` and
  `find_threshold_for_recall`: companion metrics to Secret-Key
  Retention Rate (SKRR) that expose what fraction of actual attacks a
  given gating threshold catches, rather than reporting SKRR alone
  (see Fixed, below).

### Fixed
- **`evaluation.metrics.physics_inconsistency_score` (now removed,
  replaced by `PhysicsConsistencyModel`).** The QBER term in the
  original implementation was multiplied by a coefficient of `0.0`
  (a dead stub), and the "expected healthy" baseline was recomputed
  from whichever split was currently being scored -- including that
  split's own attack rows, which could suppress the very signal the
  term was meant to surface. `PhysicsConsistencyModel` fits a robust
  regression (`HuberRegressor`) of `delta_Y`/`delta_E` on physical
  covariates ONCE on training-split healthy rows only, then scores new
  data as a z-scored residual against that fixed baseline. This
  changed the Q-TRACE ablation result from "removing the physics term
  sometimes improves AUROC" to "the physics term improves AUROC in 4/5
  evaluated seeds" -- results computed before this fix should not be
  reported.
- **`evaluation.run_adversarial_evasion`'s evolutionary search** could
  freeze at a local optimum within ~5 iterations because the elite
  population's standard deviation (search radius) could collapse
  toward its floor with no recovery mechanism. Added a stagnation
  counter that re-inflates the search radius around the current best
  point after 6 stagnant iterations, plus a radius floor set as a
  fraction of each parameter's range rather than a small absolute
  constant.
- **`evaluation.run_experiments`'s VQC evaluation slice** used
  `X_red[:qsub]`, the first `qsub` rows in file order. Because
  `make_windows` groups rows contiguously by run, this could land
  entirely inside one or two runs of the same class (typically
  "normal"), producing a single-class test set and `AUROC = NaN` for
  VQC on every split. Fixed to draw the same random, class-mixed
  subsample already used for the quantum kernel.
- **`quantum.ibm_runtime_job`** defaulted to IBM Runtime `Session` mode,
  which is not permitted on IBM's Open (free) plan (`HTTP 400: You are
  not authorized to run a session when using the open plan`). Now
  defaults to plain job mode (works on every plan tier); `--use-session`
  is available for paid-plan users who want session-mode queue
  priority.
- **SKRR reporting** previously showed only a single, arbitrarily-chosen
  operating point (top-decile risk threshold), at which Q-TRACE
  detected only ~13% of actual zero-day attacks -- a fact invisible
  from the SKRR number alone. `gating_effectiveness` and
  `find_threshold_for_recall` now let SKRR be reported alongside an
  explicit attack-detection rate at multiple named sensitivity targets
  (50%/80%/95% recall), exposing the full security/key-retention
  tradeoff rather than one favorable point on it.

### Changed
- `evaluation.run_experiments`'s SKRR section now writes
  `results/skrr_gating_effectiveness_table.csv` in addition to the
  original `skrr_table.csv`.

## [0.1.0] -- initial internal release

- Initial QKD digital twin (`qkd_simulator/`), six attack models
  (`attacks/`), dataset generator (`dataset/generate.py`), classical
  and anomaly baselines (`models/`), quantum kernel + VQC
  (`quantum/`), and the Q-TRACE composite detector
  (`models/qtrace_detector.py`).
