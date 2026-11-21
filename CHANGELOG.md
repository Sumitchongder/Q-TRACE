# Changelog

## [2.0.0] - 2026-11-21

Complete redesign. Results of v0.1 should not be compared with or cited alongside v2.0 results; the
corrections below explain why.

### Corrections to v0.1
- **Window labels.** v0.1 labelled every window of an attack run as attacked, including the 20–60 % of
  windows before the attack onset, which are physically honest. This capped attainable AUROC and
  corrupted supervised training. v2.0 labels windows by whether an attack is active.
- **Hardware inputs.** v0.1's IBM jobs (`quantum/ibm_runtime_job.py`) evaluated kernel and VQC circuits on
  uniformly random angles, not on QKD telemetry, so they did not validate detection. v2.0 replaces them
  with a hardware experiment in which BB84 and the attacks themselves run on the processor.
- **Key rate.** The single-photon gain omitted the factor `μ e^{−μ}`; fixed.
- **Vacuum yield.** v0.1 used the expected rather than a sampled vacuum yield; v2.0 samples every cell.
- **Normalisation leakage.** v0.1 min-max normalised scores on the evaluated split; v2.0 fits every
  normalisation on training or commissioning data only.
- **Baselines.** Logistic regression and SVM received unscaled features; all baselines now receive
  per-link standardised telemetry with the same temporal context as Q-TRACE, and thresholds matched in
  false-alarm rate.

### Added
- Count-level decoy-state twin with per-detector outcome tables and photon-number-resolved attacks.
- Honest-manifold detector (calibration, profiling, omnibus/score/setting channels, moving reference).
- Detectability exponent and Lorden delay bound; telemetry-blind attack identification.
- Multiplexed BB84 hardware experiment with physical probes, pilot-based budget control, rehearsal mode.
- Held-out multi-seed benchmark, unit tests, CI, Docker, publication figures.

### Removed
- VQC and quantum-kernel *detectors* as proposed methods (v0.1 ablation already showed no benefit); the
  quantum-kernel one-class SVM is retained as a baseline.

## [0.1.0] - 2026-09-10
Initial internal release.
