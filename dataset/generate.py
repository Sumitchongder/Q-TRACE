"""
Q-TRACE :: dataset.generate
=============================
Builds the QKD-TRACE benchmark dataset described in the research plan
(Sections 9, 10, 11, 26). Produces run-level trajectories that are later
split by RUN, not by row, to avoid leakage across a single trajectory.

Split philosophy (Section 26):
  - train_attacks   : attack families the model is allowed to see in training
  - heldout_attacks : attack families NEVER seen until the zero-day test set
  - composite pairs  : combinations of two attacks, always held out (E4)
  - physical regimes : "shifted" runs use disjoint loss/efficiency ranges (E3)

Usage
-----
    python -m dataset.generate --out dataset/qkd_trace_v1 --runs-per-class 40 \
        --windows-per-run 120 --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd

from qkd_simulator.config import DecoyBB84Config, ParamRange
from qkd_simulator.bb84 import DecoyBB84Simulator
from attacks import make_attack, make_composite, CANONICAL_COMPOSITIONS

TRAIN_ATTACKS = ["pns", "intercept_resend", "blinding"]
HELDOUT_ATTACKS = ["trojan_horse", "time_shift", "rng_manipulation"]


def build_shifted_config(seed):
    """A physical-regime config disjoint from the nominal training regime
    (Section 23 / E3): higher loss, lower detector efficiency."""
    cfg = DecoyBB84Config(seed=seed)
    cfg.fiber_atten_db_per_km = ParamRange(0.20, 0.25)
    cfg.link_length_km = ParamRange(25.0, 40.0)          # nominal was 2-25 km
    cfg.detector_efficiency = ParamRange(0.05, 0.30)      # nominal was 0.10-0.70
    return cfg


def generate_split(sim, attack_names, n_runs, n_windows, split_label,
                    zero_day, physical_regime, rng, strength_range=(0.5, 1.3),
                    composite=False):
    frames = []
    run_counter = 0
    targets = attack_names if not composite else CANONICAL_COMPOSITIONS

    # always include a healthy ("normal") class in every split
    all_classes = ["normal"] + list(targets)

    for cls in all_classes:
        for i in range(n_runs):
            run_id = f"{split_label}_{cls if not composite else '+'.join(cls)}_{i}"
            theta = sim.sample_theta()
            strength = rng.uniform(*strength_range)
            onset = rng.uniform(0.2, 0.6)

            if cls == "normal":
                attack = None
                fam = "normal"
                zd = False
            elif composite:
                attack = make_composite(cls[0], cls[1], onset_frac=onset,
                                         strength=strength, rng=rng)
                fam = attack.name
                zd = zero_day
            else:
                attack = make_attack(cls, onset_frac=onset, strength=strength, rng=rng)
                fam = cls
                zd = zero_day

            df = sim.simulate_run(n_windows=n_windows, theta=theta, attack=attack,
                                   run_id=run_id, attack_family=fam, zero_day=zd,
                                   physical_regime=physical_regime)
            df["split"] = split_label
            frames.append(df)
            run_counter += 1

    return pd.concat(frames, ignore_index=True)


def add_temporal_blocks(df: pd.DataFrame, block_size: int = 30, feature_cols=None):
    """
    Section 8: build temporal blocks X_t = [x_{t-29}, ..., x_t] per run,
    so each sample encodes recent history rather than a single instant.
    Implemented as a dict of numpy arrays keyed by run_id, saved separately
    from the flat CSV (see save()) because fixed-width block columns would
    bloat the CSV; models/*.py reconstruct blocks from the flat table.
    """
    if feature_cols is None:
        feature_cols = [
            "QBER_signal", "QBER_decoy", "Y_signal", "Y_decoy", "delta_Y", "delta_E",
            "detection_rate_signal", "detection_rate_decoy", "entropy",
            "channel_loss_db", "detector_efficiency", "dark_count_rate",
            "timing_jitter_ps", "phase_noise_rad", "Y1_lb", "e1_ub", "Q1_lb", "skr_est",
        ]
    blocks = {}
    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("t")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        padded = np.pad(arr, ((block_size - 1, 0), (0, 0)), mode="edge")
        windows = np.stack([padded[i:i + block_size] for i in range(len(arr))], axis=0)
        blocks[run_id] = windows
    return blocks, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/qkd_trace_v1")
    ap.add_argument("--runs-per-class", type=int, default=40)
    ap.add_argument("--windows-per-run", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nominal_cfg = DecoyBB84Config(seed=args.seed)
    shifted_cfg = build_shifted_config(args.seed + 1)

    sim_nominal = DecoyBB84Simulator(nominal_cfg, rng=rng)
    sim_shifted = DecoyBB84Simulator(shifted_cfg, rng=rng)

    print("[1/5] Generating TRAIN split (known attacks, nominal regime)...")
    train = generate_split(sim_nominal, TRAIN_ATTACKS, args.runs_per_class,
                            args.windows_per_run, "train", zero_day=False,
                            physical_regime="nominal", rng=rng)

    print("[2/5] Generating VALIDATION split (known attacks, nominal regime)...")
    val = generate_split(sim_nominal, TRAIN_ATTACKS, max(5, args.runs_per_class // 4),
                          args.windows_per_run, "val", zero_day=False,
                          physical_regime="nominal", rng=rng)

    print("[3/5] Generating ZERO-DAY TEST split (unseen attack families, nominal regime)...")
    test_zeroday = generate_split(sim_nominal, HELDOUT_ATTACKS, max(5, args.runs_per_class // 4),
                                   args.windows_per_run, "test_zeroday", zero_day=True,
                                   physical_regime="nominal", rng=rng)

    print("[4/5] Generating PHYSICAL-SHIFT TEST split (known attacks, shifted regime)...")
    test_shift = generate_split(sim_shifted, TRAIN_ATTACKS, max(5, args.runs_per_class // 4),
                                 args.windows_per_run, "test_physical_shift", zero_day=False,
                                 physical_regime="shifted", rng=rng)

    print("[5/5] Generating COMPOSITIONAL ZERO-DAY TEST split...")
    test_composite = generate_split(sim_nominal, None, max(5, args.runs_per_class // 4),
                                     args.windows_per_run, "test_composite", zero_day=True,
                                     physical_regime="nominal", rng=rng, composite=True)

    full = pd.concat([train, val, test_zeroday, test_shift, test_composite], ignore_index=True)

    csv_path = os.path.join(args.out, "qkd_trace_telemetry.csv")
    full.to_csv(csv_path, index=False)
    print(f"Saved flat telemetry table -> {csv_path}  ({len(full)} rows, {full.run_id.nunique()} runs)")

    print("Building temporal blocks (this may take a moment)...")
    blocks, feat_cols = add_temporal_blocks(full, block_size=args.block_size)
    block_path = os.path.join(args.out, "qkd_trace_temporal_blocks.npz")
    np.savez_compressed(block_path, feature_cols=np.array(feat_cols),
                         **{k: v for k, v in blocks.items()})
    print(f"Saved temporal blocks -> {block_path}")

    summary = full.groupby(["split", "attack_family"]).size().unstack(fill_value=0)
    print("\n=== Dataset summary (rows per split x attack_family) ===")
    print(summary)
    summary.to_csv(os.path.join(args.out, "dataset_summary.csv"))


if __name__ == "__main__":
    main()
