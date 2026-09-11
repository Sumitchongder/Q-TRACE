"""
Q-TRACE :: evaluation.run_multiseed
======================================
Repeats the ENTIRE pipeline -- dataset generation AND model training/
evaluation -- across N independent seeds, then aggregates each
(split, model, metric) cell into mean +/- std. This is deliberately more
expensive than just re-seeding the models on a single fixed dataset:
re-drawing the physical parameter vectors (theta) fresh per seed is what
makes the resulting error bars honestly reflect Q-TRACE's sensitivity to
which physical regimes/attack-instance draws happened to appear, not just
to model-internal stochasticity. This is the correct level of repetition
for a Q1-journal reproducibility claim (see README.md "Reproducing
paper-scale results").

Implementation note: each seed is run as a SEPARATE subprocess invocation
of the already-validated `dataset.generate` and `evaluation.run_experiments`
CLIs (rather than importing and re-calling their internals in-process).
This is intentional: it reuses exactly the code path you already validated
manually, keeps each seed's run fully isolated (no state leakage between
seeds via global RNG or cached sklearn objects), and makes it trivial to
resume a partially-completed multi-seed sweep if one seed's run fails or
is interrupted (already-completed seeds are skipped on re-run).

Usage:
    python -m evaluation.run_multiseed --seeds 42 43 44 45 46 \
        --runs-per-class 40 --windows-per-run 120 --block-size 30 \
        --n-qubits 6 --out results_multiseed

    # fast correctness check (do not use for reported results):
    python -m evaluation.run_multiseed --seeds 1 2 3 --smoke --out results_multiseed_test
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import pandas as pd

from evaluation.plotting import set_style, save_fig, PALETTE


def run_one_seed(seed, args):
    seed_dataset_dir = os.path.join(args.work_dir, f"dataset_seed{seed}")
    seed_results_dir = os.path.join(args.out, f"seed_{seed}")
    telemetry_csv = os.path.join(seed_dataset_dir, "qkd_trace_telemetry.csv")
    comparison_csv = os.path.join(seed_results_dir, "comparison_table.csv")

    if os.path.exists(comparison_csv) and not args.force:
        print(f"[seed {seed}] already completed -> {comparison_csv} (skipping; use --force to redo)")
        return comparison_csv

    runs_per_class = 3 if args.smoke else args.runs_per_class
    windows_per_run = 20 if args.smoke else args.windows_per_run
    block_size = 10 if args.smoke else args.block_size

    if not os.path.exists(telemetry_csv) or args.force:
        print(f"[seed {seed}] generating dataset -> {seed_dataset_dir}")
        cmd = [sys.executable, "-m", "dataset.generate",
               "--out", seed_dataset_dir,
               "--runs-per-class", str(runs_per_class),
               "--windows-per-run", str(windows_per_run),
               "--block-size", str(block_size),
               "--seed", str(seed)]
        subprocess.run(cmd, check=True)

    print(f"[seed {seed}] running experiments -> {seed_results_dir}")
    cmd = [sys.executable, "-m", "evaluation.run_experiments",
           "--data", telemetry_csv,
           "--block-size", str(block_size),
           "--n-qubits", str(args.n_qubits),
           "--out", seed_results_dir]
    if args.smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)

    return comparison_csv


def aggregate(seed_csv_paths, seeds):
    frames = []
    for seed, path in zip(seeds, seed_csv_paths):
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    agg = (all_df.groupby(["split", "model"])[["AUROC", "AUPRC", "F1", "FAR"]]
           .agg(["mean", "std", "count"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    return all_df, agg


def plot_zeroday_error_bars(agg, out_path):
    set_style()
    import matplotlib.pyplot as plt

    sub = agg[agg.split == "zero_day"].sort_values("AUROC_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = [PALETTE.get(m, "#777777") for m in sub.model]
    ax.bar(sub.model, sub.AUROC_mean, yerr=sub.AUROC_std, capsize=4,
           color=colors, error_kw={"elinewidth": 1.0})
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="Chance")
    ax.set_ylabel("AUROC (mean $\\pm$ std over seeds)")
    ax.set_title("Zero-Day Detection AUROC Across Independent Seeds")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    ax.legend(frameon=False)
    save_fig(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--runs-per-class", type=int, default=40)
    ap.add_argument("--windows-per-run", type=int, default=120)
    ap.add_argument("--block-size", type=int, default=30)
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--work-dir", default="dataset")
    ap.add_argument("--out", default="results_multiseed")
    ap.add_argument("--force", action="store_true",
                     help="Re-run seeds even if their output already exists.")
    ap.add_argument("--smoke", action="store_true",
                     help="Fast correctness check with tiny per-seed datasets; "
                          "do not use for reported results.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Running {len(args.seeds)} independent seeds: {args.seeds}")
    seed_csv_paths = [run_one_seed(seed, args) for seed in args.seeds]

    print("\nAggregating results across seeds...")
    all_df, agg = aggregate(seed_csv_paths, args.seeds)

    all_df.to_csv(os.path.join(args.out, "comparison_table_all_seeds_raw.csv"), index=False)
    agg.to_csv(os.path.join(args.out, "comparison_table_mean_std.csv"), index=False)

    print("\n=== Mean +/- std AUROC by split/model (n_seeds = {}) ===".format(len(args.seeds)))
    display_cols = ["split", "model", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std", "count_AUROC" if "count_AUROC" in agg.columns else "AUROC_count"]
    print_cols = [c for c in ["split", "model", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std"] if c in agg.columns]
    print(agg[print_cols].to_string(index=False))

    os.makedirs("figures", exist_ok=True)
    plot_zeroday_error_bars(agg, "figures/fig5b_zeroday_auroc_error_bars")

    print(f"\nSaved -> {args.out}/comparison_table_mean_std.csv")
    print(f"Saved -> {args.out}/comparison_table_all_seeds_raw.csv")
    print("Saved -> figures/fig5b_zeroday_auroc_error_bars.{png,pdf}")


if __name__ == "__main__":
    main()
