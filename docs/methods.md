# Methods

## 1. Telemetry

For every telemetry window of `N` pulses, Alice and Bob share the count table
`n[k, a, α, β, o]`: intensity class `k ∈ {signal, decoy, vacuum}`, Alice's bit `a`, Alice's basis `α`,
Bob's basis `β`, and receiver outcome `o ∈ {none, D0, D1, double}` (96 cells). After error correction
the error count of the sifted key is known exactly, and unsifted rounds and decoy rounds are announced
anyway, so the full table is available at no key cost.

## 2. Receiver model

With silence probabilities `S_j` of detector `j` and joint silence `S_01`:
`P(none)=S_01, P(D0)=S_1−S_01, P(D1)=S_0−S_01, P(double)=1−S_0−S_1+S_01`.
For a photon-number distribution `P(n)` and arrival generating function `G_n(z)=E[z^{n'}|n]`,
`S_j=(1−p_dj) Σ_n P(n) G_n(1−π_j)` and `S_01=(1−p_d0)(1−p_d1) Σ_n P(n) G_n(1−π_0−π_1)`, where `π_j`
is the per-photon probability of reaching and firing detector `j` (routing with optical error `e_β`
for matching bases, 1/2 otherwise, times relative efficiency `1` or `κ`). For a Poissonian source and a
lossy channel this reduces to `exp(−μ η π_j)`. Afterpulsing adds a background proportional to each
detector's mean click rate.

## 3. Honest manifold and calibration

`P_{ψ,η}` with static parameters `ψ = (e_Z, e_X, p_d0, p_d1, κ, ρ)` and per-window transmittance `η_t`.
The signal intensity is read from the transmitter monitor and the afterpulse probability from the
detector datasheet (±10 % error in the benchmark). `ψ` is fitted by maximum likelihood on `W_cal = 60`
commissioning windows with `η_t` profiled inside the objective (vectorised Newton on `log η`).

## 4. Alarm channels

Let `η̂_t` be the profiled transmittance and `E` the expected counts.

* **Omnibus.** Freeman–Tukey residuals `r = √n + √(n+1) − √(4E+1)` for the 72 click cells and 24
  setting-allocation cells, standardised per component by commissioning mean and spread (shrunk toward
  1). Persistent-shift statistic `‖Σ_M z‖² / (M(1+M/W_cal))` and energy statistic `Σ_M ‖z‖²`, each
  centred and scaled by its χ² reference.
* **Physics score test.** Efficient score `u_t` and information `I_t` for the nine directions
  `(e_Z, e_X, p_d0, p_d1, κ, ρ, Δe_decoy, double-click excess, ε_X)` with `η` removed by Schur complement.
  Over a horizon, `S_M = v^T Φ^{-1} v` with `v = I_M^{-1/2}(U_M − M b)`, where the bias `b` and dispersion
  matrix `Φ` (shrunk, floored at the identity) are estimated on commissioning windows. Locally most
  powerful against deviations in the span of the probed directions.
* **Setting test.** The same construction for the five free setting probabilities.
* **Moving reference.** For each channel, a window-limited GLR compares the recent block `(t−M, t]`
  with the immediately preceding `R = 48` windows, `d = U_recent − (M/R) U_ref`, cancelling drift that is
  slow on the scale `M + R`.

Horizons: `M ∈ {1, 4, 16}` (commissioning reference) and `{4, 16, 32}` (moving reference). Each channel is
standardised by its median and 99.9 % quantile on attack-free *training* links; the alarm statistic is
the maximum. The threshold is the `1−α` quantile on attack-free *validation* links (half with benign
loss faults).

## 5. Physical attribution

At an alarm, `δ = I_M^{-1} U_M` (moving reference, `M = 16`) standardised by its sampling error names the
device parameter whose apparent displacement is largest, i.e. the physical constraint the attack violates.

## 6. Detectability

`D* = min_η KL(Q ‖ P_η)` nats per pulse, evaluated at the run's mean physical state (time-averaged over
one period for modulated attacks). By Lorden (1971) and Lai (1998), any detector with mean time to false
alarm `≥ 1/α` windows has worst-case mean delay `≳ log(1/α) / (N D*)` windows. `D* = 0` means the attacked
statistics are reproduced exactly by some honest link.

## 7. Benchmark protocol

Per seed: 60 honest training links (+30 per known attack for the supervised baseline only), 75
validation links, and test links comprising 50 honest, 25 loss-fault, 15 dark-count-fault, 35 per attack
family, 20 per composition, and a physical-shift regime (60–120 km, receiver efficiency 0.05–0.20) with 25
honest and 15 per known attack. 200 windows of 10⁷ pulses per link (1 s at 10 MHz), 60 commissioning
windows, onset uniformly 20–100 windows after commissioning, window-level labels. Attack strengths are
drawn log-uniformly over physically motivated ranges (see `evaluation/benchmark.py:sample_attack`).

## 8. Baselines

All baselines receive the 19 standard telemetry scalars standardised per link against commissioning,
plus their trailing 16-window mean: QBER static (11 %), one-sided QBER CUSUM, two-sided decoy-state
CUSUM (Y1 lower bound, e1 upper bound, decoy/signal gain ratio), Isolation Forest, one-class SVM, MLP
autoencoder, one-class SVM on a 6-qubit ZZ-feature-map fidelity kernel (exact statevectors, verified
against Qiskit), and XGBoost trained on the known attack families.
