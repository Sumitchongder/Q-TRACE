"""
Q-TRACE :: evaluation.run_adversarial_evasion
=================================================
Section 24: can an attacker perturb observable channel statistics just
enough to evade Q-TRACE without violating physical feasibility?

We do not have gradient access through the full Q-TRACE risk pipeline
(it includes non-differentiable components: IsolationForest-style
autoencoder reconstruction via sklearn, PCA, etc.), so this uses a
physically-constrained black-box search (CMA-ES-style covariance-free
random search / simple evolutionary strategy) rather than a projected-
gradient attack. This is an appropriate and defensible choice for a
mixed classical/quantum, partly non-differentiable pipeline, and is
explicitly noted as such in the methods section this script's docstring
doubles as.

The attacker's perturbation budget is applied directly to attack
strength/onset parameters (NOT to arbitrary telemetry values), because
Section 24 requires perturbations to remain physically feasible --
searching in "attack parameter space" guarantees every candidate
corresponds to a physically realizable attack instance, which searching
in raw feature space would not.

Usage:
    python -m evaluation.run_adversarial_evasion --attack pns \
        --n-iterations 40 --population 12 --out results
"""

import argparse
import os
import numpy as np
import pandas as pd

from qkd_simulator.config import DecoyBB84Config
from qkd_simulator.bb84 import DecoyBB84Simulator
from attacks import make_attack
from evaluation.data_utils import make_windows, align_raw_df
from evaluation.metrics import minmax
from models.qtrace_detector import QTraceDetector


def build_normal_training(seed, n_runs, n_windows, block_size):
    rng = np.random.default_rng(seed)
    cfg = DecoyBB84Config(seed=seed)
    sim = DecoyBB84Simulator(cfg, rng=rng)
    frames = [sim.simulate_run(n_windows=n_windows, run_id=f"normal_{i}")
              for i in range(n_runs)]
    df = pd.concat(frames, ignore_index=True)
    X, y, meta = make_windows(df, block_size=block_size)
    aligned_df = align_raw_df(df, meta)  # every row here is healthy by construction
    return X, aligned_df, sim, rng


def risk_for_attack_instance(detector, sim, attack_name, strength, onset_frac,
                              n_windows, block_size, seed):
    rng = np.random.default_rng(seed)
    attack = make_attack(attack_name, onset_frac=onset_frac, strength=strength, rng=rng)
    df = sim.simulate_run(n_windows=n_windows, attack=attack, run_id="adv_probe",
                           attack_family=attack_name, zero_day=True)
    X, y, meta = make_windows(df, block_size=block_size)

    aligned_df = df.set_index(["run_id", "t"]).loc[list(zip(meta.run_id, meta.t))].reset_index()
    risk = detector.score(X, aligned_df)

    # only score windows AFTER attack onset (where the attack is actually live)
    onset_t = int(onset_frac * n_windows)
    post_mask = meta.t.to_numpy() >= onset_t
    if post_mask.sum() == 0:
        return 1.0  # degenerate: no post-onset windows, treat as fully detected
    return float(np.mean(risk[post_mask]))


def evolutionary_search(detector, sim, attack_name, n_windows, block_size,
                         n_iterations, population, seed,
                         strength_bounds=(0.15, 1.5), onset_bounds=(0.1, 0.7),
                         stagnation_patience=6, min_sigma_frac=0.08):
    """
    (mu, lambda)-style evolution strategy over the 2-D physically-
    constrained parameter box (strength, onset_frac). Minimizes mean
    post-onset Q-TRACE risk score, i.e. searches for the least-detectable
    physically-valid attack instance.

    Two safeguards against premature convergence (a known failure mode of
    naive (mu,lambda)-ES with a small elite count, which was observed to
    freeze exploration within ~5 iterations in earlier runs of this
    script -- see repository issue notes):

      1. `min_sigma_frac` floors sigma at a fixed FRACTION of each
         parameter's range (not a tiny absolute constant), so the search
         never narrows to a single point regardless of how tight the
         elite population happens to be.
      2. Stagnation-triggered re-inflation: if the best-so-far risk hasn't
         improved for `stagnation_patience` consecutive iterations, sigma
         is reset to its initial (wide) value and the search mean is
         re-centered on the current best-known point, giving the search a
         fresh wide-radius look around the best point found instead of
         permanently collapsing there.
    """
    rng = np.random.default_rng(seed)
    init_sigma = np.array([0.3 * (strength_bounds[1] - strength_bounds[0]),
                            0.3 * (onset_bounds[1] - onset_bounds[0])])
    min_sigma = np.array([min_sigma_frac * (strength_bounds[1] - strength_bounds[0]),
                           min_sigma_frac * (onset_bounds[1] - onset_bounds[0])])
    mean = np.array([np.mean(strength_bounds), np.mean(onset_bounds)])
    sigma = init_sigma.copy()

    history = []
    best = (None, np.inf)
    stagnant_count = 0

    for it in range(n_iterations):
        candidates = mean + sigma * rng.standard_normal((population, 2))
        candidates[:, 0] = np.clip(candidates[:, 0], *strength_bounds)
        candidates[:, 1] = np.clip(candidates[:, 1], *onset_bounds)

        scores = np.array([
            risk_for_attack_instance(detector, sim, attack_name, s, o,
                                      n_windows, block_size, seed=seed + it * population + k)
            for k, (s, o) in enumerate(candidates)
        ])

        elite_frac = 0.3
        n_elite = max(2, int(population * elite_frac))
        elite_idx = np.argsort(scores)[:n_elite]
        mean = candidates[elite_idx].mean(axis=0)
        sigma = np.maximum(candidates[elite_idx].std(axis=0), min_sigma)

        it_best_idx = np.argmin(scores)
        improved = scores[it_best_idx] < best[1] - 1e-4
        if improved:
            best = (candidates[it_best_idx].copy(), scores[it_best_idx])
            stagnant_count = 0
        else:
            stagnant_count += 1

        reinflated = False
        if stagnant_count >= stagnation_patience:
            mean = best[0].copy()
            sigma = init_sigma.copy()
            stagnant_count = 0
            reinflated = True

        history.append({"iteration": it, "best_risk_so_far": best[1],
                         "mean_strength": mean[0], "mean_onset": mean[1],
                         "sigma_strength": sigma[0], "sigma_onset": sigma[1],
                         "reinflated": reinflated})
        flag = "  [reinflated search radius -- stagnation detected]" if reinflated else ""
        print(f"  iter {it:3d}  best_risk={best[1]:.4f}  "
              f"(strength={best[0][0]:.3f}, onset={best[0][1]:.3f}){flag}")

    return best, pd.DataFrame(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="pns", choices=[
        "pns", "intercept_resend", "trojan_horse", "blinding", "time_shift", "rng_manipulation"])
    ap.add_argument("--n-iterations", type=int, default=40)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--windows-per-run", type=int, default=60)
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--train-runs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_iterations, args.population = 6, 6
        args.windows_per_run, args.train_runs = 30, 6

    os.makedirs(args.out, exist_ok=True)

    print("[1/2] Training Q-TRACE detector on healthy telemetry...")
    Xn, normal_df, sim, rng = build_normal_training(
        args.seed, args.train_runs, args.windows_per_run, args.block_size)
    detector = QTraceDetector(use_quantum=False).fit(Xn, normal_df)

    # baseline: risk of a "typical" (strength=1.0, onset=0.4) attack instance
    baseline_risk = risk_for_attack_instance(
        detector, sim, args.attack, strength=1.0, onset_frac=0.4,
        n_windows=args.windows_per_run, block_size=args.block_size, seed=args.seed + 999)
    print(f"[baseline] typical '{args.attack}' instance mean post-onset risk = {baseline_risk:.4f}")

    print(f"\n[2/2] Running physically-constrained evolutionary search "
          f"({args.n_iterations} iters x {args.population} candidates)...")
    best, history = evolutionary_search(
        detector, sim, args.attack, args.windows_per_run, args.block_size,
        args.n_iterations, args.population, seed=args.seed)

    (best_params, best_risk) = best
    evasion_gap = baseline_risk - best_risk

    result = {
        "attack": args.attack,
        "baseline_strength": 1.0, "baseline_onset": 0.4, "baseline_risk": baseline_risk,
        "evasive_strength": float(best_params[0]), "evasive_onset": float(best_params[1]),
        "evasive_risk": float(best_risk),
        "evasion_gap": float(evasion_gap),
    }
    print(f"\n[result] {result}")

    out_path = os.path.join(args.out, f"adversarial_evasion_{args.attack}.csv")
    pd.DataFrame([result]).to_csv(out_path, index=False)
    history.to_csv(os.path.join(args.out, f"adversarial_evasion_{args.attack}_history.csv"), index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
