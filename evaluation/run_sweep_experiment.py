"""
Q-TRACE :: evaluation.run_sweep_experiment
=============================================
Section 22 "Zero-Day Stress Test": sweeps channel loss L and detector
efficiency eta_d, retrains/evaluates the Q-TRACE detector's zero-day
recall at each (L, eta_d) grid point, and produces the signature heatmap
figure (Section 22, described as one of the paper's most important
results).

This trains ONE Q-TRACE detector on the nominal training regime (as in
run_experiments.py), then evaluates its zero-day recall separately on
synthetic held-out-attack runs generated at each grid point -- i.e. this
measures GENERALIZATION across the physical parameter space, matching
the research question in Section 23 ("does the detector remain reliable
when the hardware/channel moves outside the training distribution?").

Usage:
    python -m evaluation.run_sweep_experiment --n-loss 6 --n-eta 6 \
        --runs-per-point 6 --windows-per-run 60 --out results --smoke
"""

import argparse
import os
import numpy as np
import pandas as pd

from qkd_simulator.config import DecoyBB84Config, ParamRange
from qkd_simulator.bb84 import DecoyBB84Simulator
from attacks import make_attack
from evaluation.data_utils import make_windows, align_raw_df
from evaluation.metrics import detection_metrics, minmax
from evaluation.plotting import plot_heatmap
from models.qtrace_detector import QTraceDetector

TRAIN_ATTACKS = ["pns", "intercept_resend", "blinding"]
HELDOUT_ATTACKS = ["trojan_horse", "time_shift", "rng_manipulation"]


def build_train_detector(seed, n_runs, n_windows, block_size):
    """Train Q-TRACE once on the nominal regime, exactly as in run_experiments.py
    but without the quantum component (the sweep grid can be large; running
    the quantum kernel at every grid point would be prohibitively expensive
    on a simulator and is out of scope for this sweep -- see README for the
    quantum-inclusive small-scale comparison instead)."""
    rng = np.random.default_rng(seed)
    cfg = DecoyBB84Config(seed=seed)
    sim = DecoyBB84Simulator(cfg, rng=rng)

    frames = []
    for cls in ["normal"] + TRAIN_ATTACKS:
        for i in range(n_runs):
            theta = sim.sample_theta()
            attack = None if cls == "normal" else make_attack(
                cls, onset_frac=rng.uniform(0.2, 0.6), strength=rng.uniform(0.5, 1.3), rng=rng)
            df = sim.simulate_run(n_windows=n_windows, theta=theta, attack=attack,
                                   run_id=f"train_{cls}_{i}", attack_family=cls if attack else "normal")
            frames.append(df)
    train_df = pd.concat(frames, ignore_index=True)
    X_train, y_train, meta_train = make_windows(train_df, block_size=block_size)
    Xn = X_train[y_train == 0]
    aligned_train_df = align_raw_df(train_df, meta_train)
    aligned_train_normal_df = aligned_train_df[y_train == 0].reset_index(drop=True)

    detector = QTraceDetector(use_quantum=False).fit(Xn, aligned_train_normal_df)
    return detector, sim, rng


def eval_at_grid_point(detector, alpha_loss_db_per_km, length_km_fixed, eta_d,
                        n_runs, n_windows, block_size, seed, rng):
    """Generate held-out-attack runs at a fixed (channel_loss, eta_d) point
    and measure Q-TRACE's zero-day recall there."""
    cfg = DecoyBB84Config(seed=seed)
    cfg.fiber_atten_db_per_km = ParamRange(alpha_loss_db_per_km, alpha_loss_db_per_km)
    cfg.link_length_km = ParamRange(length_km_fixed, length_km_fixed)
    cfg.detector_efficiency = ParamRange(eta_d, eta_d)
    sim = DecoyBB84Simulator(cfg, rng=rng)

    frames = []
    for cls in ["normal"] + HELDOUT_ATTACKS:
        for i in range(n_runs):
            theta = sim.sample_theta()
            attack = None if cls == "normal" else make_attack(
                cls, onset_frac=0.3, strength=1.0, rng=rng)
            df = sim.simulate_run(n_windows=n_windows, theta=theta, attack=attack,
                                   run_id=f"grid_{cls}_{i}", attack_family=cls if attack else "normal",
                                   zero_day=bool(attack))
            frames.append(df)
    grid_df = pd.concat(frames, ignore_index=True)
    X, y, meta = make_windows(grid_df, block_size=block_size)

    # PhysicsConsistencyModel (and the detector generally) needs the raw
    # telemetry rows in the SAME order as X/meta -- make_windows groups
    # rows by run_id, so a plain concat order won't match; align explicitly:
    aligned_df = align_raw_df(grid_df, meta)
    score = detector.score(X, aligned_df)

    m = detection_metrics(y, score, threshold=np.percentile(score, 75))
    # recall at a fixed operating point (threshold = 75th percentile risk)
    y_pred = (score >= np.percentile(score, 75)).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    fn = np.sum((y_pred == 0) & (y == 1))
    recall = tp / max(tp + fn, 1)
    return recall, m["AUROC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loss", type=int, default=6)
    ap.add_argument("--n-eta", type=int, default=6)
    ap.add_argument("--loss-range-db", type=float, nargs=2, default=[0.5, 6.0],
                     help="channel_loss_db = alpha*L; swept via link length at fixed alpha")
    ap.add_argument("--eta-range", type=float, nargs=2, default=[0.05, 0.95])
    ap.add_argument("--runs-per-point", type=int, default=4)
    ap.add_argument("--windows-per-run", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=15)
    ap.add_argument("--train-runs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_loss, args.n_eta, args.runs_per_point = 3, 3, 2
        args.windows_per_run, args.train_runs = 20, 5

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("[1/2] Training Q-TRACE detector on nominal regime...")
    detector, _, rng = build_train_detector(args.seed, args.train_runs,
                                             args.windows_per_run, args.block_size)

    loss_grid = np.linspace(*args.loss_range_db, args.n_loss)
    eta_grid = np.linspace(*args.eta_range, args.n_eta)
    fixed_alpha = 0.2  # dB/km; loss varied via link length for a clean "channel loss (dB)" axis

    recall_matrix = np.zeros((len(eta_grid), len(loss_grid)))
    auroc_matrix = np.zeros((len(eta_grid), len(loss_grid)))

    print(f"[2/2] Sweeping {len(loss_grid)}x{len(eta_grid)} grid "
          f"({len(loss_grid)*len(eta_grid)} points)...")
    for i, eta in enumerate(eta_grid):
        for j, loss_db in enumerate(loss_grid):
            length_km = loss_db / fixed_alpha
            recall, auroc = eval_at_grid_point(
                detector, fixed_alpha, length_km, eta,
                args.runs_per_point, args.windows_per_run, args.block_size,
                seed=args.seed + i * len(loss_grid) + j, rng=rng)
            recall_matrix[i, j] = recall
            auroc_matrix[i, j] = auroc if not np.isnan(auroc) else 0.5
            print(f"    L={loss_db:5.2f}dB  eta_d={eta:.2f}  recall={recall:.3f}  AUROC={auroc:.3f}")

    np.savez(os.path.join(args.out, "sweep_grid.npz"),
             loss_grid=loss_grid, eta_grid=eta_grid,
             recall_matrix=recall_matrix, auroc_matrix=auroc_matrix)

    plot_heatmap(recall_matrix, loss_grid, eta_grid,
                 "Channel Loss (dB)", "Detector Efficiency",
                 "Zero-Day Detection Recall vs. Channel Loss and Detector Efficiency",
                 "figures/fig6_zeroday_heatmap")

    print(f"\nSaved sweep results -> {args.out}/sweep_grid.npz")
    print("Saved heatmap -> figures/fig6_zeroday_heatmap.{png,pdf}")


if __name__ == "__main__":
    main()
