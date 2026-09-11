# Q-TRACE — Limitations & Discussion (manuscript prose)

These are written in full manuscript voice, ready to drop into your
paper with light editing for house style/citation format. Two sections:
**Discussion** (interprets the results you already have) and
**Limitations** (states what the work does not yet establish). Springer
Nature research articles typically place Discussion before Limitations,
with Limitations sometimes folded into Discussion as a final subsection
or kept separate before Conclusion — both are common; I've written them
as separable blocks so you can arrange either way.

Numbers are pulled directly from your actual runs (Tables 1–5 in
`docs/manuscript-tables.md`). Nothing here is invented. Bracketed
`[CITE]` markers mark places where a citation to prior work would
strengthen the claim — fill these from your literature search rather
than from me, since I don't have your reference list.

---

## 5. Discussion

### 5.1 Zero-day generalization is Q-TRACE's genuine advantage, not a uniform one

Q-TRACE achieves the highest mean zero-day AUROC (0.670 ± 0.051) among
thirteen evaluated methods (Table 1), and — arguably more importantly for
a security-relevant system — one of the lowest cross-seed variances. Two
purely supervised classical baselines, Random Forest (0.507 ± 0.198) and
XGBoost (0.487 ± 0.163), show standard deviations three to four times
larger than Q-TRACE's, meaning their apparent zero-day performance is
highly sensitive to which physical parameter draws happened to appear in
a given training run. A detector whose reliability cannot be predicted
in advance is of limited operational value regardless of its best-case
performance; we therefore argue that Q-TRACE's combination of competitive
mean AUROC and low variance is a more meaningful claim than mean AUROC
alone, and recommend that future QKD anomaly-detection work report
variance across independent training draws as standard practice rather
than a single train/test split [CITE — if any prior QKD-ML paper reports
only single-split results, this is the place to note it].

This advantage is specific to the zero-day regime and should not be
overstated. On the known-attack, physical-shift, and compositional
splits (Table 2), Logistic Regression matches or exceeds Q-TRACE despite
being architecturally simpler and dramatically cheaper to train. We
interpret this as expected rather than concerning: Q-TRACE is explicitly
designed to trade some in-distribution discriminative power for
generalization to attack families never seen during training (Sections
1–3), and the results support that this design goal was achieved without
the trade being uniformly favorable across every evaluation regime. A
system positioned purely as "detect known attacks" would be better
served by a supervised classifier; Q-TRACE's value proposition is
specifically the zero-day and, to a lesser extent, compositional-attack
regime.

### 5.2 What the ablation actually tells us — and what it does not

The component ablation (Table 3) supports one strong claim and two
weaker, more nuanced ones.

**Temporal representation is unambiguously load-bearing.** Removing the
30-window temporal block reduces zero-day AUROC by 0.030 (0.670 → 0.640),
consistently across all five seeds. This is consistent with the
motivating hypothesis in Section 8: several attack signatures in our
threat model — RNG manipulation in particular, by design a periodic
rather than instantaneous perturbation — are only visible when a
detector has access to recent history rather than a single telemetry
snapshot.

**The physics-consistency term's contribution is real but seed-sensitive,
and we report this honestly rather than average it away.** Removing the
term improves AUROC in one of five seeds and degrades it in the other
four, but the magnitude of that one outlier (seed 44: physics-off AUROC
0.738 vs. physics-on 0.629, a 0.108 swing in the direction opposite the
other four seeds) is large enough to pull the seed-averaged effect to
near-parity (0.6695 vs. 0.6698 — a difference far smaller than either
value's own standard deviation). We do not believe this reflects the
physics-consistency mechanism being fundamentally unhelpful; rather, we
believe it reflects a specific weakness in the current implementation:
the consistency baseline is fit via ordinary robust regression
(`HuberRegressor`) on a single training draw's healthy-telemetry rows,
and if that particular draw happens to under-sample the range of one or
more physical covariates (channel loss, detector efficiency), the fitted
baseline can generalize poorly to test-time physical conditions outside
that draw's coverage. This is a testable hypothesis we did not have
scope to confirm in the present work (see Limitations, 6.3), but it
motivates a concrete architectural change — a regularized or Bayesian
estimator with an explicit prior over the covariate range, or an
ensemble of regressions fit across multiple bootstrap resamples of the
training data — as the natural next iteration rather than abandoning
physics-informed consistency checking as a concept.

**Quantum feature mapping provided no measurable zero-day benefit in this
regime, and we report this as a genuine negative result.** Removing the
quantum-kernel component from the composite risk score is neutral-to-
mildly-beneficial in four of five seeds (mean AUROC 0.676 with the
component removed vs. 0.670 with it present). This is corroborated by
the standalone quantum baselines: the fidelity quantum kernel (0.432 ±
0.067) and the variational quantum classifier (0.441 ± 0.067) both
underperform every classical baseline except One-Class SVM on the
zero-day split, and the VQC result is corroborated on real `ibm_marrakesh`
hardware (Section 4.3), where inference on genuinely trained weights
produced class-probability predictions clustered tightly around 0.5 with
no discernible separation — consistent with, not contradicted by, the
simulator finding. We view this as a legitimate contribution rather than
a shortfall: much of the QML literature applied to security/anomaly
detection reports positive results without a matched classical ablation
[CITE — general QML-hype critique, e.g., a paper questioning claimed
quantum advantage in near-term QML applications], and a clean, honestly-
reported null result under a fair like-for-like comparison (identical
feature reduction, identical training data, identical evaluation
protocol) is scientifically useful precisely because it is rare. We
discuss plausible reasons for this outcome, and concrete conditions under
which quantum feature mapping might plausibly help, in Section 5.3.

**The uncertainty term is mildly counterproductive, consistently.**
Removing it improves mean AUROC in all five seeds, by a small but
completely consistent margin (0.670 → 0.679). The current implementation
is a lightweight ensemble-disagreement proxy (the standard deviation
across the detector's other active signal channels; Section 20/Table 3
of the design document), which was chosen for its zero additional
training cost, not because it was expected to be the strongest possible
uncertainty estimate. This result is consistent with that expectation:
a proxy built from the same underlying signals it is meant to add
uncertainty *about* is likely to be correlated with — and therefore
partially redundant with, or even mildly anti-correlated in its
practical effect on — those signals rather than genuinely orthogonal
epistemic information. We had originally scoped calibrated uncertainty
via conformal prediction (Section 21) as the intended final
implementation; the present ablation result is direct empirical
motivation for prioritizing that over the current proxy in future work,
rather than an argument against including an uncertainty term in
principle.

### 5.3 Why might quantum feature mapping have underperformed here, specifically?

We offer three non-exclusive, testable explanations rather than a single
post-hoc rationalization, since distinguishing between them is itself a
useful direction for future work.

First, **dimensionality reduction may be discarding the structure a
quantum feature map would exploit.** Our PCA-based reduction to 4–6
qubits (Section 15/16) was chosen for QPU-time tractability, but PCA is
a linear, variance-maximizing projection with no guarantee that the
resulting low-dimensional space preserves whatever nonlinear structure
in the original 18-feature × 30-window telemetry a quantum kernel might
otherwise be able to exploit. A quantum feature map's potential advantage
is specifically in accessing feature spaces difficult to represent
classically [CITE — quantum kernel methods theory, e.g. Havlicek et al.
or Schuld]; if PCA has already linearly compressed away the relevant
structure before the quantum circuit ever sees the data, no encoding
choice downstream can recover it.

Second, **training budget was deliberately constrained by QPU-time
practicality, and this constraint may bind more tightly for the VQC than
for the kernel method.** The VQC was trained with 100–150 COBYLA
iterations on a 250-sample subsample (Section 16); this is a shallow
optimization budget by variational-circuit-training standards, chosen so
that a real-hardware inference run (Section 4.3) would fit comfortably
within an 8-minute QPU allocation. It is plausible that a substantially
larger classical-simulator training budget (thousands of iterations, no
subsampling) would improve VQC performance materially; we did not
exhaust this direction and flag it explicitly as unresolved rather than
concluding VQCs are unsuitable for this task generally.

Third, and most fundamentally, **it remains an open question whether this
telemetry-anomaly-detection task has quantum-exploitable structure at
all.** Demonstrated quantum kernel advantages in the literature are
generally constructed on synthetic data specifically engineered to be
classically hard [CITE — e.g. Liu, Arunachalam & Temme 2021 on
provable quantum kernel advantage on an engineered discrete-log-based
dataset]; there is no a priori reason to expect naturally-occurring QKD
telemetry to share that structure. Our result is consistent with the
more cautious position in recent QML literature that near-term quantum
feature maps should be expected to help on some tasks and not others,
and that empirical validation against a fair classical baseline — not
assumption — should determine which is which [CITE].

### 5.4 Adversarial robustness: an intentional asymmetry, and a genuine limitation

The physically-constrained evasion search (Table 5) finds that low-
strength attack instances are substantially harder for Q-TRACE to detect
than nominal-strength instances across all six evaluated attack families,
with the effect most pronounced for detector-blinding (49% risk
reduction at the lower bound of our search space, strength = 0.15 —
notably at the *boundary* of the search space we permitted, suggesting
the true minimum-detectable strength may be lower still) and least
pronounced for RNG manipulation (14% reduction). We interpret the RNG
manipulation result as further, independent support for Section 5.2's
temporal-representation finding: an attack designed around periodic
structure is intrinsically well-matched to a detector with an explicit
temporal window, regardless of the attack's amplitude.

The practical implication is that Q-TRACE — like essentially any
anomaly-detection system calibrated against a fixed decision threshold
— trades detection sensitivity against false-alarm rate, and an attacker
aware of this tradeoff can in principle operate below the sensitivity
floor at the cost of also achieving less of whatever the attack is
intended to accomplish (a weaker blinding or PNS attack extracts
correspondingly less information per unit time). We do not claim this
tradeoff is unique to Q-TRACE or resolvable by any single-threshold
detector; we report it as a quantified property of the specific system
evaluated, consistent with the framing in Section 33 of the design
document that Q-TRACE should not be presented as detecting all zero-day
attacks, which is not a claim we make.

---

## 6. Limitations

We report the following limitations explicitly rather than omit them,
consistent with the framing established throughout this work that a
narrower, honestly-qualified claim is more useful to the field than a
broader unqualified one.

### 6.1 Adversarial evasion results are single-seed

The evasion-gap figures in Table 5 were computed against a single Q-TRACE
detector instance (trained on one fixed dataset draw). Given the
substantial seed-to-seed variance observed elsewhere in this work
(Tables 1, 3), it is likely that the specific evasive strength/onset
parameters found by the search — and possibly the magnitude of the
evasion gap itself — would shift somewhat across independently trained
detector instances. We do not expect the qualitative ranking across
attack families (blinding most evadable, RNG manipulation least) to
reverse, since it is grounded in the underlying attack physics and
feature-representation match described in Section 5.4 rather than in a
particular detector's idiosyncrasies, but we did not have the
computational budget within the scope of this work to confirm that
expectation empirically across multiple seeds and report it as an open
item.

### 6.2 The channel-loss × detector-efficiency sweep (Figure 6) is
noise-limited at its current resolution

The zero-day recall/AUROC heatmap was computed with 8 attack-family
trajectories per grid point across a 64-point (8×8) grid. Recall was
stable across the grid (0.25–0.33 throughout), but AUROC showed
substantial cell-to-cell variation (0.41–0.83) without a clean monotonic
trend, consistent with per-cell sampling noise at this sample size rather
than a genuine non-monotonic physical relationship. We present the
figure as evidence that Q-TRACE does not catastrophically fail across
the swept physical range (recall never approaches zero anywhere on the
grid) rather than as a precise characterization of how AUROC varies with
channel loss and detector efficiency; a higher-resolution or
multi-seed-averaged version of this sweep would be needed to support the
latter, stronger claim.

### 6.3 The physics-consistency baseline's seed sensitivity was not
root-caused within this work's scope

As discussed in Section 5.2, we hypothesize that the outlier seed's
degraded physics-term performance reflects insufficient covariate
coverage in that seed's specific training draw, but we did not conduct
the targeted experiment (e.g., directly inspecting the fitted regression
coefficients and residual distributions across seeds, or deliberately
constructing a training draw with restricted covariate range to test
whether it reproduces the failure mode) that would confirm this
diagnosis rather than merely propose it as the most likely explanation.

### 6.4 The uncertainty component's proxy was not the originally
intended implementation

Section 21 of the original design scoped calibrated uncertainty via
conformal prediction as the target implementation; the ensemble-
disagreement proxy actually evaluated here was selected for zero
additional training cost during initial development. The consistent
(5/5 seeds) mild negative ablation effect of this specific proxy should
not be read as evidence against including an uncertainty-aware term in
the composite risk score generally, only against this particular
lightweight approximation of one.

### 6.5 Quantum models were evaluated at a scale constrained by
practical simulation and QPU-time costs

Both the fidelity quantum kernel and VQC were restricted to 4–6 qubits
and, for the VQC, a shallow classical-optimizer training budget (Section
5.3). We did not evaluate scaling behavior — whether the quantum
methods' relative performance versus classical baselines changes with
qubit count, circuit depth, or training budget — since doing so at scale
was outside the QPU-time and classical-simulation-time budget available
for this work. The negative result reported in Sections 3 and 5.3 should
be read as applying to the specific configuration evaluated, not as a
scaling-law claim.

### 6.6 The QKD digital twin is a simulation, not validated against
deployed hardware

Consistent with the standards-alignment framing in Section 32 of the
design document, Q-TRACE is positioned as an implementation-security
monitoring framework whose physical telemetry model is motivated by, but
not fitted to, observables reported for real commercial QKD systems
(e.g., signal/decoy yield differences, consistent with the monitoring
philosophy described for QNu's ARMOS system [CITE]). All reported
detection results are on synthetic telemetry generated by our own
digital twin (Section 4–8 of the design document); no claim is made or
implied that these results transfer quantitatively to a physically
deployed QKD link without further validation against real device data,
which was outside the scope and resources available for this work.

### 6.7 Attack models are simplified representations of their physical
mechanisms

Each of the six attack families implemented (Sections 9–10) captures the
qualitative telemetry-level signature reported for that attack class in
prior implementation-security literature [CITE — per-attack references:
PNS, intercept-resend, Trojan-horse, detector blinding, time-shift, RNG
manipulation], rather than a full physical simulation of the underlying
optical/electronic attack mechanism. This is a standard simplification
in ML-for-QKD-security literature given the impracticality of full
device-physics simulation for dataset generation at scale, but it means
detection performance against a real implementation of any given attack
could differ from the performance reported here if a real attacker's
telemetry signature deviates from our model's assumptions.

---

## Suggested placement / structure note

Given the length of the above, consider:
- **Discussion (5.1–5.4)**: keep in the main paper body, condensed if
  your target venue has strict length limits — 5.1 and 5.2 are the most
  load-bearing for supporting your abstract's claims and should not be
  cut; 5.3 and 5.4 could be shortened to 2–3 sentences each if space is
  tight, with the fuller version moved to supplementary material.
- **Limitations (6.1–6.7)**: several Springer Nature journals (check
  your specific target's author guidelines) now require or strongly
  encourage a dedicated Limitations section distinct from Discussion —
  if so, 6.1–6.7 as written should work directly; if your target instead
  wants Limitations folded into Discussion, move 6.1–6.7 to become
  additional Discussion subsections following 5.4.
