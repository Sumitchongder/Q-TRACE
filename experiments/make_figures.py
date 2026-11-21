"""Aggregate benchmark seeds and hardware data into tables and figures.

    python experiments/make_figures.py

Reads ``results/benchmark/seed_*/`` and ``results/hardware/*.npz`` and writes
``results/summary/*.csv|.tex`` and ``figures/*.pdf|.png``. Hardware files
named ``rehearsal_*`` are simulated dress rehearsals; their outputs go to
``figures/rehearsal/`` and ``results/summary/rehearsal/`` and are never mixed
with hardware results.
"""

from __future__ import annotations

import glob
import json
import pathlib
import warnings

import numpy as np
from scipy.stats import t as t_dist
import pandas as pd

from qtrace.hardware import analysis as A
from qtrace.hardware.runner import load_raw
from qtrace.viz import figures as F

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "benchmark"
SUMMARY = ROOT / "results" / "summary"
FIG = ROOT / "figures"
KNOWN = {"intercept_resend", "pns", "blinding"}
ORDER = ["Q-TRACE", "XGBoost (supervised)", "QBER CUSUM", "Decoy CUSUM", "QBER static",
         "Autoencoder", "Isolation Forest", "One-class SVM", "Quantum-kernel OCSVM"]


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    h = t_dist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean(), h


def fmt(m, h, pct=True):
    if not np.isfinite(m):
        return "--"
    if pct:
        return f"{100 * m:.1f} ± {100 * h:.1f}" if np.isfinite(h) else f"{100 * m:.1f}"
    return f"{m:.3f} ± {h:.3f}" if np.isfinite(h) else f"{m:.3f}"


def to_latex(df, path, caption, label):
    cols = "l" + "c" * (df.shape[1])
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join([df.index.name or ""] + list(map(str, df.columns))) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = [str(v).replace("±", r"$\pm$") for v in row.values]
        name = rf"\textbf{{{idx}}}" if idx == "Q-TRACE" else str(idx)
        lines.append(" & ".join([name] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def load_benchmark():
    seeds = sorted(glob.glob(str(BENCH / "seed_*")))
    runs = pd.concat([pd.read_csv(f"{s}/run_metrics.csv") for s in seeds
                      if pathlib.Path(f"{s}/meta.json").exists()], ignore_index=True)
    wins = pd.concat([pd.read_csv(f"{s}/window_metrics.csv") for s in seeds
                      if pathlib.Path(f"{s}/meta.json").exists()], ignore_index=True)
    metas = [json.load(open(f"{s}/meta.json")) for s in seeds if pathlib.Path(f"{s}/meta.json").exists()]
    return runs, wins, metas


def group_of(df):
    return np.where(df.split == "test_shift", "physical shift",
                    np.where(df.kind == "composite", "composite",
                             np.where(df.family.isin(KNOWN), "known", "zero-day")))


def benchmark_tables(runs, wins, far=1e-3):
    SUMMARY.mkdir(parents=True, exist_ok=True)
    tag = f"{far:g}"
    a = runs[runs.kind.isin(["attack", "composite"])].copy()
    a["group"] = group_of(a)
    groups = ["known", "zero-day", "composite", "physical shift"]
    per_seed = a.groupby(["seed", "group", "method"])[f"detected@{tag}"].mean().reset_index()

    t1 = pd.DataFrame(index=pd.Index(ORDER, name="Method"))
    for g in groups:
        t1[g] = [fmt(*mean_ci(per_seed[(per_seed.group == g) & (per_seed.method == m)][f"detected@{tag}"]))
                 for m in ORDER]
    # false alarms
    fa = runs[runs.family.isin(["honest", "loss_fault"])].copy()
    fa["cat"] = np.where(fa.split == "test_shift", "honest (shift)", fa.family)
    fa = fa[fa.split.isin(["test", "test_shift"])]
    ps = fa.groupby(["seed", "cat", "method"])[f"any_alarm@{tag}"].mean().reset_index()
    for c, lab in (("honest", "FA honest"), ("loss_fault", "FA loss fault"),
                   ("honest (shift)", "FA honest (shift)")):
        t1[lab] = [fmt(*mean_ci(ps[(ps.cat == c) & (ps.method == m)][f"any_alarm@{tag}"])) for m in ORDER]
    t1.to_csv(SUMMARY / f"table_detection_far{tag}.csv")
    to_latex(t1, SUMMARY / f"table_detection_far{tag}.tex",
             rf"Percentage of attacked links detected, and of attack-free links with at least one "
             rf"alarm (FA), at a per-window false-alarm rate of {far:g} set on validation links. "
             rf"Mean $\pm$ 95\% CI over held-out seeds.", f"tab:detection{tag.replace('.', '')}")

    # AUROC by group
    w = wins.copy()
    w["group2"] = np.where(w.split == "test_shift", "physical shift", w.group)
    ps = w.groupby(["seed", "group2", "method"])["AUROC"].mean().reset_index()
    t2 = pd.DataFrame(index=pd.Index(ORDER, name="Method"))
    for g in groups:
        t2[g] = [fmt(*mean_ci(ps[(ps.group2 == g) & (ps.method == m)]["AUROC"]), pct=False) for m in ORDER]
    t2.to_csv(SUMMARY / "table_auroc.csv")
    to_latex(t2, SUMMARY / "table_auroc.tex",
             r"Window-level AUROC (attack-active windows vs.\ windows of attack-free links). "
             r"Mean $\pm$ 95\% CI over held-out seeds.", "tab:auroc")

    # per family
    fam = a[a.split == "test"].groupby(["seed", "family", "method"])[f"detected@{tag}"].mean().reset_index()
    fams = [f for f in F.FAMILY_LABEL if f in fam.family.unique()]
    t3 = pd.DataFrame(index=pd.Index(ORDER, name="Method"))
    for f in fams:
        t3[F.FAMILY_LABEL[f]] = [fmt(*mean_ci(fam[(fam.family == f) & (fam.method == m)][f"detected@{tag}"]))
                                 for m in ORDER]
    t3.to_csv(SUMMARY / f"table_per_family_far{tag}.csv")

    # delays and key exposure on runs detected by Q-TRACE
    dd = a[a.split == "test"]
    t4 = dd.groupby("method").agg(median_delay=(f"delay@{tag}", "median"),
                                  mean_key_exposed_bits=(f"key_exposed@{tag}", "mean")).reindex(ORDER)
    t4.to_csv(SUMMARY / f"table_delay_key_far{tag}.csv")

    # theory: detection as a function of the Lorden bound
    th = a[(a.kind == "attack") & np.isfinite(a.d_star)].copy()
    th["bound"] = np.log(1 / far) / (np.maximum(th.d_star, 1e-300) * 1e7)
    bins = [0, 1, 10, 100, np.inf]
    th["bound_bin"] = pd.cut(th.bound, bins, labels=["<1", "1-10", "10-100", ">100"])
    t5 = th.pivot_table(index="method", columns="bound_bin", values=f"detected@{tag}",
                        aggfunc="mean", observed=False).reindex(ORDER)
    t5["n_runs"] = None
    t5.loc[ORDER[0], "n_runs"] = str(th[th.method == ORDER[0]].bound_bin.value_counts().sort_index().to_dict())
    t5.to_csv(SUMMARY / f"table_detection_vs_bound_far{tag}.csv")
    return t1, t2, t3, t4, t5


def benchmark_figures(runs, metas):
    F.fig_detection_vs_strength(runs, FIG / "fig3_detection_vs_strength")
    F.fig_delay_vs_bound(runs, FIG / "fig4_delay_vs_lorden_bound")
    F.fig_group_summary(runs, FIG / "fig5_detection_summary",
                        methods=[m for m in ORDER if m in runs.method.unique()])
    F.fig_false_alarms(runs, FIG / "fig6_false_alarms",
                       methods=[m for m in ORDER if m in runs.method.unique()])
    F.fig_attribution(runs, FIG / "fig7_physical_attribution")
    ex = sorted(glob.glob(str(BENCH / "seed_*" / "examples.npz")))
    if ex:
        seed = int(pathlib.Path(ex[0]).parent.name.split("_")[1])
        thr = next(m["thresholds"] for m in metas if m["seed"] == seed)
        thr = {k: v["0.001"] for k, v in thr.items()}
        F.fig_example_traces(ex[0], thr, FIG / "fig2_example_traces")


def hardware_outputs(path):
    bob, eve, design, meta = load_raw(path)
    rehearsal = meta["mode"] != "hardware"
    fig_dir = FIG / ("rehearsal" if rehearsal else "hardware")
    sum_dir = SUMMARY / ("rehearsal" if rehearsal else "hardware")
    sum_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(path).stem
    n1, n = A.count_tables(bob, design)
    tests, disp = A.link_epoch_tests(n1, n, design)
    det = A.detection_table(tests)
    curve = A.detection_vs_rounds(bob, design, alpha=1e-3)
    pred = A.predicted_power(bob, design, sorted(curve.shots.unique()))
    r2d = A.rounds_to_detect(curve)
    info = A.eve_information(bob, eve, design)
    leak = A.leaked_before_detection(info, r2d)
    for name, df in (("link_epoch_tests", tests), ("detection", det), ("detection_vs_rounds", curve),
                     ("predicted_power", pred), ("eve_information", info), ("leakage", leak)):
        df.to_csv(sum_dir / f"{stem}_{name}.csv", index=False)
    json.dump(dict(dispersion=disp, backend=meta["backend"], mode=meta["mode"],
                   shots=int(bob.shape[1]), links=int(bob.shape[2]), log=meta.get("log")),
              open(sum_dir / f"{stem}_meta.json", "w"), indent=1, default=str)
    # condition-matched analysis (reference = honest monitoring epochs)
    mt = A.matched_tests(n1, n, design)
    mdet = A.detection_table(mt)
    mcurve = A.matched_detection_vs_rounds(bob, design, alpha=1e-3)
    mr2d = A.rounds_to_detect(mcurve)
    mleak = A.leaked_before_detection(info, mr2d)
    for name, df in (("matched_tests", mt), ("matched_detection", mdet),
                     ("matched_detection_vs_rounds", mcurve), ("matched_leakage", mleak)):
        df.to_csv(sum_dir / f"{stem}_{name}.csv", index=False)
    F.fig_hw_detection(det, fig_dir / f"{stem}_detection_commissioning_reference")
    F.fig_hw_detection(mdet, fig_dir / f"{stem}_detection_matched_reference")
    probes = [p for p in mdet.probe if "0.150" in p or "0.250" in p or "0.030" in p][:3]
    F.fig_hw_rounds(mcurve, pred.iloc[0:0], fig_dir / f"{stem}_rounds_matched", probes)
    F.fig_hw_condition_shift(n1, n, design, fig_dir / f"{stem}_condition_shift")
    F.fig_hw_physics(mdet, info, fig_dir / f"{stem}_physics")
    return mdet, info, mleak, disp


def main():
    runs, wins, metas = load_benchmark()
    print(f"benchmark: {runs.seed.nunique()} seeds, {len(runs) // runs.method.nunique()} scored runs")
    for far in (1e-3, 1e-2):
        t1, *_ = benchmark_tables(runs, wins, far)
    print(t1.to_string())
    benchmark_figures(runs, metas)
    for p in sorted(glob.glob(str(ROOT / "results" / "hardware" / "*.npz"))):
        det, info, leak, disp = hardware_outputs(p)
        print(f"\n{pathlib.Path(p).name}: dispersion {disp}")
        print(det.to_string(index=False))


if __name__ == "__main__":
    main()
