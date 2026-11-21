"""Generate numbers.tex and tables/*.tex for the manuscript from the Q-TRACE results.

    python build_numbers.py --repo ../Q-TRACE

Every quantitative statement in the manuscript is a macro defined here, so the
paper can be regenerated from the repository without hand-typed numbers.
"""

import argparse
import glob
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="..")
args = ap.parse_args()
REPO = pathlib.Path(args.repo)
OUT = pathlib.Path(__file__).resolve().parent
TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)

KNOWN = {"intercept_resend", "pns", "blinding"}
ORDER = ["Q-TRACE", "XGBoost (supervised)", "QBER CUSUM", "Decoy CUSUM", "QBER static",
         "Autoencoder", "Isolation Forest", "One-class SVM", "Quantum-kernel OCSVM"]
SHORT = {"Q-TRACE": "Q-TRACE", "XGBoost (supervised)": "XGBoost (sup.)", "QBER CUSUM": "QBER CUSUM",
         "Decoy CUSUM": "Decoy CUSUM", "QBER static": "QBER static", "Autoencoder": "Autoencoder",
         "Isolation Forest": "Isolation forest", "One-class SVM": "One-class SVM",
         "Quantum-kernel OCSVM": "QK one-class SVM"}
PROSE = {"XGBoost (supervised)": "supervised XGBoost", "Quantum-kernel OCSVM": "quantum-kernel one-class SVM",
         "QBER static": "static QBER threshold", "QBER CUSUM": "QBER CUSUM", "Decoy CUSUM": "decoy-state CUSUM",
         "Autoencoder": "autoencoder", "Isolation Forest": "isolation forest", "One-class SVM": "one-class SVM"}
FAM = {"intercept_resend": "Intercept-resend", "pns": "PNS", "blinding": "Blinding",
       "time_shift": "Time-shift", "trojan_horse": "Trojan horse", "rng_bias": "RNG bias",
       "pns+blinding": "PNS + blinding", "time_shift+trojan_horse": "Time-shift + Trojan",
       "intercept_resend+rng_bias": "IR + RNG bias"}
macros = {}


def M(name, value):
    assert name.isalpha(), name
    macros[name] = value


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    return x.mean(), t_dist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def pm(m, h, scale=100, dec=1):
    if not np.isfinite(m):
        return "--"
    if not np.isfinite(h):
        return f"{scale * m:.{dec}f}"
    return f"{scale * m:.{dec}f}\\,$\\pm$\\,{scale * h:.{dec}f}"


def table(path, header, rows, colspec, bold_first=True, pre_header=None):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + list(pre_header or []) + \
        [" & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
            continue
        cells = list(r)
        if bold_first and cells[0] == "Q-TRACE":
            cells = [r"\textbf{Q-TRACE}"] + [r"\textbf{" + c + "}" for c in cells[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ benchmark
seeds = sorted(glob.glob(str(REPO / "results/benchmark/seed_*/meta.json")))
seed_dirs = [pathlib.Path(s).parent for s in seeds]
runs = pd.concat([pd.read_csv(d / "run_metrics.csv") for d in seed_dirs], ignore_index=True)
wins = pd.concat([pd.read_csv(d / "window_metrics.csv") for d in seed_dirs], ignore_index=True)
metas = [json.load(open(s)) for s in seeds]
M("NSeeds", str(len(seeds)))
M("SeedList", ", ".join(str(m["seed"]) for m in metas))
n_runs = int(sum(m["n_runs"] for m in metas))
M("NRunsTotal", f"{n_runs:,}".replace(",", "{,}"))
n_scored = len(runs) // runs.method.nunique()
M("NRunsScored", f"{n_scored:,}".replace(",", "{,}"))
M("NWindowsTotal", f"{n_runs * 200:,}".replace(",", "{,}"))
M("WallPerSeed", f"{np.mean([m['wall_time_s'] for m in metas]) / 60:.0f}")

a = runs[runs.kind.isin(["attack", "composite"])].copy()
a["group"] = np.where(a.split == "test_shift", "shift", np.where(a.kind == "composite", "composite",
                      np.where(a.family.isin(KNOWN), "known", "zeroday")))
GROUPS = ["known", "zeroday", "composite", "shift"]
GNAME = {"known": "Known", "zeroday": "ZeroDay", "composite": "Comp", "shift": "Shift"}

for far, tag in ((1e-3, "0.001"), (1e-2, "0.01")):
    col = f"detected@{tag}"
    ps = a.groupby(["seed", "group", "method"])[col].mean().reset_index()
    fa = runs[runs.family.isin(["honest", "loss_fault"]) & runs.split.isin(["test", "test_shift"])].copy()
    fa["cat"] = np.where(fa.split == "test_shift", "shift", fa.family)
    pf = fa.groupby(["seed", "cat", "method"])[f"any_alarm@{tag}"].mean().reset_index()
    rows = []
    stats = {}
    for m in ORDER:
        r = [SHORT[m]]
        for g in GROUPS:
            mm, hh = ci(ps[(ps.group == g) & (ps.method == m)][col])
            stats[(m, g)] = mm
            r.append(pm(mm, hh))
        for c in ("honest", "loss_fault", "shift"):
            mm, hh = ci(pf[(pf.cat == c) & (pf.method == m)][f"any_alarm@{tag}"])
            stats[(m, "fa_" + c)] = mm
            r.append(pm(mm, hh))
        rows.append(r)
    suffix = "" if far == 1e-3 else "Lo"
    table(f"detection_far{tag.replace('.', 'p')}.tex",
          ["Method", "Known", "Zero-day", "Composite", "Long links", "FA honest", "FA loss", "FA long"],
          rows, "lccccccc")
    for g in GROUPS:
        m_, h_ = ci(ps[(ps.group == g) & (ps.method == "Q-TRACE")][col])
        M(f"Det{GNAME[g]}QT{suffix}", f"{100 * m_:.1f}")
        best = max((x for x in ORDER if x != "Q-TRACE"), key=lambda x: stats[(x, g)])
        M(f"Det{GNAME[g]}Best{suffix}", f"{100 * stats[(best, g)]:.1f}")
        M(f"Det{GNAME[g]}BestName{suffix}", PROSE[best])
    for c, nm in (("honest", "Honest"), ("loss_fault", "Loss"), ("shift", "Long")):
        M(f"FA{nm}QT{suffix}", f"{100 * stats[('Q-TRACE', 'fa_' + c)]:.1f}")
        for m in ORDER[1:]:
            key = "".join(ch for ch in SHORT[m] if ch.isalnum())
            M(f"FA{nm}{key}{suffix}", f"{100 * stats[(m, 'fa_' + c)]:.1f}")
    for m in ORDER[1:]:
        key = "".join(ch for ch in SHORT[m] if ch.isalnum())
        for g in GROUPS:
            M(f"Det{GNAME[g]}{key}{suffix}", f"{100 * stats[(m, g)]:.1f}")

# AUROC
w = wins.copy()
w["g"] = np.where(w.split == "test_shift", "shift", w.group.replace({"zero-day": "zeroday"}))
ps = w.groupby(["seed", "g", "method"])["AUROC"].mean().reset_index()
rows = []
for m in ORDER:
    r = [SHORT[m]]
    for g in GROUPS:
        mm, hh = ci(ps[(ps.g == g) & (ps.method == m)]["AUROC"])
        r.append(pm(mm, hh, scale=1, dec=3))
        if m == "Q-TRACE":
            M(f"Auc{GNAME[g]}QT", f"{mm:.3f}")
        if m == "XGBoost (supervised)":
            M(f"Auc{GNAME[g]}XGB", f"{mm:.3f}")
        if m == "Quantum-kernel OCSVM":
            M(f"Auc{GNAME[g]}QK", f"{mm:.3f}")
    rows.append(r)
table("auroc.tex", ["Method", "Known", "Zero-day", "Composite", "Long links"], rows, "lcccc")

# per family (test split)
col = "detected@0.001"
fam = a[a.split == "test"].groupby(["seed", "family", "method"])[col].mean().reset_index()
fams = [f for f in FAM if f in fam.family.unique()]
rows = []
for f in fams:
    r = [FAM[f]]
    vals = {}
    for m in ORDER:
        mm, hh = ci(fam[(fam.family == f) & (fam.method == m)][col])
        vals[m] = mm
        r.append(f"{100 * mm:.1f}")
    best = max(vals, key=vals.get)
    r = [r[0]] + [(r"\textbf{" + v + "}" if ORDER[i] == best else v) for i, v in enumerate(r[1:])]
    rows.append(r)
    key = "".join(ch for ch in FAM[f] if ch.isalnum())
    M(f"Fam{key}QT", f"{100 * vals['Q-TRACE']:.1f}")
    M(f"Fam{key}XGB", f"{100 * vals['XGBoost (supervised)']:.1f}")
    M(f"Fam{key}QC", f"{100 * vals['QBER CUSUM']:.1f}")
hdr = ["Family"] + [r"\rotatebox{70}{" + SHORT[m] + "}" for m in ORDER]
table("per_family.tex", hdr, rows, "l" + "c" * len(ORDER), bold_first=False)

# detection versus Lorden bound
th = a[(a.kind == "attack") & np.isfinite(a.d_star)].copy()
th["bound"] = np.log(1e3) / (np.maximum(th.d_star, 1e-300) * 1e7)
bins = [0, 1, 10, 100, np.inf]
labels = ["$<1$", "1--10", "10--100", "$>100$"]
th["bin"] = pd.cut(th.bound, bins, labels=labels)
rows = []
for m in ORDER:
    r = [SHORT[m]]
    for lab in labels:
        r.append(f"{100 * th[(th.method == m) & (th.bin == lab)][col].mean():.1f}")
    rows.append(r)
counts = th[th.method == "Q-TRACE"].bin.value_counts()
rows.append("MID")
rows.append(["Attack runs"] + [str(int(counts.get(lab, 0))) for lab in labels])
table("vs_bound.tex", ["Method"] + [f"bound {lab}" for lab in labels], rows, "lcccc")
for lab, key in zip(labels, ["LtOne", "OneTen", "TenHundred", "GtHundred"]):
    M(f"Bound{key}QT", f"{100 * th[(th.method == 'Q-TRACE') & (th.bin == lab)][col].mean():.1f}")
    M(f"Bound{key}XGB", f"{100 * th[(th.method == 'XGBoost (supervised)') & (th.bin == lab)][col].mean():.1f}")
    M(f"Bound{key}QC", f"{100 * th[(th.method == 'QBER CUSUM') & (th.bin == lab)][col].mean():.1f}")
    M(f"Bound{key}N", str(int(counts.get(lab, 0))))
zero_dstar = th[(th.method == "Q-TRACE")]
M("NBlindRuns", str(int((zero_dstar.d_star < 1e-14).sum())))

# median delay among detected runs, and key exposure
dd = a[(a.split == "test")]
rows = []
for m in ORDER:
    x = dd[(dd.method == m) & dd["detected@0.001"]]
    med = np.median(x["delay@0.001"]) if len(x) else np.nan
    rows.append([SHORT[m], f"{100 * dd[dd.method == m]['detected@0.001'].mean():.1f}",
                 f"{med:.0f}" if np.isfinite(med) else "--",
                 f"{dd[dd.method == m]['key_exposed@0.001'].mean() / 1e6:.2f}"])
    if m in ("Q-TRACE", "QBER CUSUM", "XGBoost (supervised)"):
        key = {"Q-TRACE": "QT", "QBER CUSUM": "QC", "XGBoost (supervised)": "XGB"}[m]
        M(f"MedDelay{key}", f"{med:.0f}")
        M(f"KeyExp{key}", f"{dd[dd.method == m]['key_exposed@0.001'].mean() / 1e6:.2f}")
table("delay_key.tex", ["Method", "Runs detected (\\%)", "Median delay (windows)", "Key exposed (Mbit)"],
      rows, "lccc")

# attribution
qt = a[(a.method == "Q-TRACE") & a["implicated@0.001"].notna() & (a.split == "test")]
for f in ("pns", "blinding", "intercept_resend", "rng_bias", "time_shift"):
    vc = qt[qt.family == f]["implicated@0.001"].value_counts(normalize=True)
    key = "".join(ch for ch in FAM[f] if ch.isalnum())
    if len(vc):
        M(f"Attr{key}Top", vc.index[0].replace("_", r"\_"))
        M(f"Attr{key}Share", f"{100 * vc.iloc[0]:.0f}")

# ------------------------------------------------------------------ hardware
hw = sorted(glob.glob(str(REPO / "results/summary/hardware/*_matched_detection.csv")))[-1]
stem = pathlib.Path(hw).stem.replace("_matched_detection", "")
S = REPO / "results/summary/hardware"
md = pd.read_csv(S / f"{stem}_matched_detection.csv")
pre = pd.read_csv(S / f"{stem}_detection.csv")
info = pd.read_csv(S / f"{stem}_eve_information.csv")
leak = pd.read_csv(S / f"{stem}_matched_leakage.csv")
meta = json.load(open(S / f"{stem}_meta.json"))
tests_pre = pd.read_csv(S / f"{stem}_link_epoch_tests.csv")
M("HwBackend", meta["backend"].replace("_", r"\_"))
M("HwLinks", str(meta["links"]))
M("HwShots", f"{meta['shots']:,}".replace(",", "{,}"))
qs = {e["stage"]: e.get("quantum_seconds") for e in meta["log"]}
M("HwPilotSec", f"{qs.get('pilot', float('nan')):.0f}")
M("HwMainSec", f"{qs.get('main', float('nan')):.0f}")
M("HwTotalSec", f"{qs.get('pilot', 0) + qs.get('main', 0):.0f}")
M("HwRounds", f"{64 * meta['shots'] * meta['links'] / 1e6:.1f}")
h_pre = pre[pre.kind == "honest"].iloc[0]
M("HwPreFAQT", f"{100 * h_pre['Q-TRACE@0.001']:.0f}")
M("HwPreFAQP", f"{100 * h_pre['QBER (pooled)@0.001']:.0f}")
M("HwPreFAQB", f"{100 * h_pre['QBER (per basis)@0.001']:.0f}")
M("HwHonestN", str(int(h_pre["n"])))

# honest condition shift (per basis)
raw = np.load(REPO / f"results/hardware/{stem}.npz", allow_pickle=False)
import sys
sys.path.insert(0, str(REPO / "src"))
from qtrace.hardware import analysis as A  # noqa: E402
from qtrace.hardware.design import ExperimentDesign  # noqa: E402
design = ExperimentDesign.from_json(str(raw["design"]))
n1, n = A.count_tables(raw["bob"], design)
off = design.n_cal_passes
sched = np.array(design.schedule)
shift = {}
for b, nm in ((0, "Z"), (1, "X")):
    mask = lambda a_, al, be, b=b: al == be == b
    d = []
    for i in range(design.n_links):
        ec, tc = A._error_counts(n1[:off, i].sum(0), n[:off, i].sum(0), mask)
        hon = [off + e for e in range(design.n_epochs) if sched[e, i] < 0]
        em, tm = A._error_counts(n1[hon, i].sum(0), n[hon, i].sum(0), mask)
        d.append(em / tm - ec / tc)
    shift[nm] = np.array(d)
    M(f"HwShift{nm}", f"{100 * np.mean(d):.2f}")
    M(f"HwShift{nm}Pos", str(int((np.array(d) > 0).sum())))
e0, t0 = A._error_counts(n1[0], n[0], lambda a_, al, be: al == be)
e1, t1 = A._error_counts(n1[1], n[1], lambda a_, al, be: al == be)
mant, expo = f"{np.abs(np.mean(e1 / t1 - e0 / t0)):.1e}".split("e")
M("HwPassDiff", rf"{mant}\times 10^{{{int(expo)}}}")
meta_raw = json.loads(str(raw["meta"]))
cal = pd.DataFrame(meta_raw.get("calibration") or [])
qcal = []
for i in range(design.n_links):
    ec, tc = A._error_counts(n1[:off, i].sum(0), n[:off, i].sum(0), lambda a_, al, be: al == be)
    qcal.append(ec / tc)
M("HwQberCalMed", f"{100 * np.median(qcal):.2f}")
M("HwQberCalMin", f"{100 * np.min(qcal):.2f}")
M("HwQberCalMax", f"{100 * np.max(qcal):.2f}")
disp = meta["dispersion"]
M("HwPhi", f"{disp['phi']:.2f}")
M("HwPhiQ", f"{disp['phi_qber']:.3f}")

# matched detection table
rows = []
for _, r in md.iterrows():
    if r.kind == "honest":
        label = "Honest (no probe)"
    elif r.kind == "probe":
        label = rf"$\theta={r.strength:.2f}$"
    else:
        label = rf"$\gamma={r.strength:.2f}$"
    rows.append((r.kind, r.strength, [label, str(int(r.n)), f"{100 * r.ideal_qber:.2f}",
                                     f"{100 * r.measured_qber_excess:.2f}",
                                     f"{r['Q-TRACE@0.001']:.2f}", f"{r['QBER (per basis)@0.001']:.2f}",
                                     f"{r['QBER (pooled)@0.001']:.2f}", f"{r['Omnibus only@0.001']:.2f}"]))
order = {"honest": 0, "probe": 1, "ad": 2}
rows.sort(key=lambda x: (order[x[0]], x[1]))
body = []
for k, (kind, _, r) in enumerate(rows):
    if k > 0 and kind != rows[k - 1][0]:
        body.append("MID")
    body.append(r)
table("hw_matched.tex", ["Probe", "$n$", r"$\Delta Q_{\rm ideal}$ (\%)", r"$\Delta Q_{\rm meas}$ (\%)",
                         "Q-TRACE", "QBER/basis", "QBER pooled", "Omnibus"], body, "lccccccc",
      bold_first=False)
for _, r in md.iterrows():
    if r.kind == "probe":
        key = "ABCDEF"[[0.15, 0.25, 0.40, 0.60, 0.90].index(round(r.strength, 2))] if round(r.strength, 2) in (0.15, 0.25, 0.40, 0.60, 0.90) else "F"
        M(f"HwTheta{key}QT", f"{int(round(r['Q-TRACE@0.001'] * r.n))}")
        M(f"HwTheta{key}QB", f"{int(round(r['QBER (per basis)@0.001'] * r.n))}")
        M(f"HwTheta{key}QP", f"{int(round(r['QBER (pooled)@0.001'] * r.n))}")
        M(f"HwTheta{key}N", f"{int(r.n)}")
        M(f"HwTheta{key}Meas", f"{100 * r.measured_qber_excess:.2f}")
        M(f"HwTheta{key}Ideal", f"{100 * r.ideal_qber:.2f}")
h_m = md[md.kind == "honest"].iloc[0]
M("HwMatchedFAQT", f"{int(round(h_m['Q-TRACE@0.001'] * h_m.n))}")

# pre-registered table
rows = []
for _, r in pre.iterrows():
    label = ("Honest (no probe)" if r.kind == "honest" else
             rf"$\theta={r.strength:.2f}$" if r.kind == "probe" else rf"$\gamma={r.strength:.2f}$")
    rows.append((r.kind, r.strength, [label, str(int(r.n)), f"{100 * r.measured_qber_excess:.2f}",
                                     f"{r['Q-TRACE@0.001']:.2f}", f"{r['QBER (per basis)@0.001']:.2f}",
                                     f"{r['QBER (pooled)@0.001']:.2f}"]))
rows.sort(key=lambda x: (order[x[0]], x[1]))
body = []
for k, (kind, _, r) in enumerate(rows):
    if k > 0 and kind != rows[k - 1][0]:
        body.append("MID")
    body.append(r)
table("hw_preregistered.tex", ["Probe", "$n$", r"$\Delta Q_{\rm meas}$ (\%)", "Q-TRACE", "QBER/basis",
                               "QBER pooled"], body, "lccccc", bold_first=False)

# leakage table
pv = leak.pivot_table(index="probe", columns="test", values=["shots_to_detect", "bits_leaked_before_alarm"])
rows = []
for probe in [p for p in md.probe if p != "honest"]:
    kind = "probe" if probe.startswith("probe") else "ad"
    s_ = float(probe.split("=")[1].rstrip(")"))
    label = rf"$\theta={s_:.2f}$" if kind == "probe" else rf"$\gamma={s_:.2f}$"
    r = [label]
    for t in ("Q-TRACE", "QBER (per basis)", "QBER (pooled)"):
        r.append(f"{pv.loc[probe, ('shots_to_detect', t)]:.0f}")
    for t in ("Q-TRACE", "QBER (per basis)", "QBER (pooled)"):
        r.append(f"{pv.loc[probe, ('bits_leaked_before_alarm', t)]:.1f}")
    rows.append((kind, s_, r))
rows.sort(key=lambda x: (order[x[0]], x[1]))
body = []
for k, (kind, _, r) in enumerate(rows):
    if k > 0 and kind != rows[k - 1][0]:
        body.append("MID")
    body.append(r)
table("hw_leakage.tex", ["Probe", "Q-TRACE", "QBER/basis", "QBER pooled", "Q-TRACE", "QBER/basis",
                         "QBER pooled"], body, "lcccccc", bold_first=False,
      pre_header=[r" & \multicolumn{3}{c}{Shots per circuit to detection} & "
                  r"\multicolumn{3}{c}{Bits leaked before alarm} \\",
                  r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"])
p15 = "probe(theta=0.150)"
for t, k in (("Q-TRACE", "QT"), ("QBER (per basis)", "QB"), ("QBER (pooled)", "QP"), ("Omnibus only", "OM")):
    M(f"HwShotsWeak{k}", f"{pv.loc[p15, ('shots_to_detect', t)]:.0f}")
    M(f"HwLeakWeak{k}", f"{pv.loc[p15, ('bits_leaked_before_alarm', t)]:.1f}")
M("HwShotsRatio", f"{pv.loc[p15, ('shots_to_detect', 'QBER (pooled)')] / pv.loc[p15, ('shots_to_detect', 'Q-TRACE')]:.1f}")

# Eve information table
rows = []
for _, r in info.iterrows():
    label = rf"$\theta={r.strength:.2f}$" if r.kind == "probe" else rf"$\gamma={r.strength:.2f}$"
    rows.append((r.kind, r.strength, [label, f"{r.I_matched:.3f}", f"{r.I_matched_ideal:.3f}",
                                     f"{r.guess_error:.3f}", f"{r.I_sifted:.3f}"]))
rows.sort(key=lambda x: (order[x[0]], x[1]))
body = []
for k, (kind, _, r) in enumerate(rows):
    if k > 0 and kind != rows[k - 1][0]:
        body.append("MID")
    body.append(r)
table("hw_info.tex", ["Probe", r"$I_{\rm match}$ (meas.)", r"$I_{\rm match}$ (ideal)", r"$P_{\rm guess\,err}$",
                      r"$I_{\rm sift}$ (meas.)"], body, "lcccc", bold_first=False)
pi2 = info[(info.kind == "probe") & (info.strength > 1.5)].iloc[0]
M("HwIRInfo", f"{pi2.I_matched:.3f}")
M("HwIRGuess", f"{pi2.guess_error:.3f}")
p04 = info[(info.kind == "probe") & (abs(info.strength - 0.4) < 1e-6)].iloc[0]
M("HwInfoFourMeas", f"{p04.I_matched:.3f}")
M("HwInfoFourIdeal", f"{p04.I_matched_ideal:.3f}")

# link calibration summary
lt = pd.DataFrame(meta_raw.get("link_table") or [])
if not cal.empty:
    rows = []
    for colname, label, sc, unit in (
            ("two_qubit_error", "CZ error (link pair)", 100, "\\%"),
            ("readout_error_data", "Readout error, data qubit", 100, "\\%"),
            ("readout_error_ancilla", "Readout error, ancilla", 100, "\\%"),
            ("T1_us_data", "$T_1$, data qubit", 1, "$\\mu$s"),
            ("T2_us_data", "$T_2$, data qubit", 1, "$\\mu$s"),
            ("T1_us_ancilla", "$T_1$, ancilla", 1, "$\\mu$s"),
            ("T2_us_ancilla", "$T_2$, ancilla", 1, "$\\mu$s")):
        src = lt if colname in lt.columns else cal
        if colname not in src.columns:
            continue
        v = src[colname].astype(float) * sc
        dec = 2 if sc == 100 else 0
        rows.append([label + f" ({unit})", f"{v.median():.{dec}f}", f"{v.min():.{dec}f}", f"{v.max():.{dec}f}"])
    table("hw_calibration.tex", ["Quantity", "Median", "Min", "Max"], rows, "lccc", bold_first=False)
    M("HwTwoMed", f"{100 * lt.two_qubit_error.median():.2f}")
    M("HwTtwoMed", f"{cal.T2_us_data.median():.0f}")

# ------------------------------------------------------------------ write
with open(OUT / "numbers.tex", "w") as f:
    f.write("% generated by build_numbers.py (do not edit by hand)\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros; tables: {sorted(p.name for p in TAB.glob('*.tex'))}")
