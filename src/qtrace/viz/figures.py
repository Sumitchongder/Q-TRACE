"""Publication figures (IEEE/Springer column widths, vector output)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

SINGLE, DOUBLE = 3.5, 7.16

STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
}

COLORS = {
    "Q-TRACE": "#C0392B", "QBER static": "#7F8C8D", "QBER CUSUM": "#2E86C1",
    "Decoy CUSUM": "#17A589", "Isolation Forest": "#B9770E", "One-class SVM": "#AF7AC5",
    "Autoencoder": "#D68910", "Quantum-kernel OCSVM": "#6C3483",
    "XGBoost (supervised)": "#1F618D",
    "QBER (pooled)": "#2E86C1", "QBER (per basis)": "#17A589", "Omnibus only": "#7F8C8D",
}
MARKERS = {"Q-TRACE": "o", "QBER CUSUM": "s", "Decoy CUSUM": "D", "Autoencoder": "^",
           "XGBoost (supervised)": "v", "Quantum-kernel OCSVM": "P", "QBER static": "x",
           "Isolation Forest": "<", "One-class SVM": ">", "QBER (pooled)": "s",
           "QBER (per basis)": "D", "Omnibus only": "x"}
FAMILY_LABEL = {
    "intercept_resend": "Intercept-resend", "pns": "Photon-number splitting",
    "blinding": "Detector blinding", "time_shift": "Time-shift", "trojan_horse": "Trojan horse",
    "rng_bias": "RNG bias", "pns+blinding": "PNS + blinding",
    "time_shift+trojan_horse": "Time-shift + Trojan", "intercept_resend+rng_bias": "IR + RNG bias",
}
STRENGTH_LABEL = {
    "intercept_resend": "intercepted fraction", "pns": "attacked fraction",
    "blinding": "blinded fraction", "time_shift": "shifted fraction",
    "trojan_horse": "back-scatter per gate", "rng_bias": r"bias amplitude $\delta$",
}


def setup():
    plt.rcParams.update(STYLE)


def clean_log_axis(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, numticks=6))
    a.set_minor_formatter(NullFormatter())


def save(fig, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    h = t_dist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


# -------------------------------------------------------------- simulation
def fig_detection_vs_strength(runs: pd.DataFrame, path, far=1e-3,
                              methods=("Q-TRACE", "Decoy CUSUM", "QBER CUSUM",
                                       "Autoencoder", "XGBoost (supervised)",
                                       "Quantum-kernel OCSVM")):
    setup()
    fams = ["intercept_resend", "pns", "blinding", "time_shift", "trojan_horse", "rng_bias"]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.6), sharey=True)
    col = f"detected@{far:g}"
    for ax, fam in zip(axes.ravel(), fams):
        d = runs[(runs.split == "test") & (runs.family == fam)]
        if d.empty:
            continue
        edges = np.quantile(np.log10(d.strength), np.linspace(0, 1, 6))
        centers = 10 ** (0.5 * (edges[1:] + edges[:-1]))
        for m in methods:
            dm = d[d.method == m]
            b = np.clip(np.digitize(np.log10(dm.strength), edges[1:-1]), 0, 4)
            y = [dm[col][b == k].mean() for k in range(5)]
            ax.plot(centers, y, marker=MARKERS.get(m, "o"), color=COLORS[m], label=m,
                    lw=1.6 if m == "Q-TRACE" else 1.0, zorder=3 if m == "Q-TRACE" else 2)
        ax.set_xscale("log")
        clean_log_axis(ax)
        zd = fam in ("time_shift", "trojan_horse", "rng_bias")
        ax.set_title(FAMILY_LABEL[fam] + (" (zero-day)" if zd else " (known)"))
        ax.set_xlabel(STRENGTH_LABEL[fam])
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[:, 0]:
        ax.set_ylabel("detection probability")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, 0.0))
    save(fig, path)


def fig_delay_vs_bound(runs: pd.DataFrame, path, far=1e-3,
                       methods=("Q-TRACE", "QBER CUSUM", "XGBoost (supervised)")):
    """Windows of post-onset data needed for the first alarm versus the Lorden bound.

    The bound constrains the *mean* delay of any detector at the given
    false-alarm rate; single runs may fall below it. Binned medians make the
    comparison with the bound explicit.
    """
    setup()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    d = runs[(runs.kind == "attack") & np.isfinite(runs.d_star) & (runs.d_star > 0)].copy()
    d["bound"] = np.log(1 / far) / (d.d_star * 1e7)
    col = f"delay@{far:g}"
    bins = np.logspace(-3, 3, 13)
    for m in methods:
        dm = d[(d.method == m) & d[col].notna()]
        ax.scatter(dm.bound, dm[col] + 1, s=5, alpha=0.35, color=COLORS[m], linewidths=0,
                   marker=MARKERS.get(m, "o"))
        idx = np.digitize(dm.bound, bins)
        xs, ys = [], []
        for k in np.unique(idx):
            sel = idx == k
            if sel.sum() >= 3:
                xs.append(np.exp(np.log(dm.bound[sel]).mean()))
                ys.append(np.median(dm[col][sel] + 1))
        ax.plot(xs, ys, marker=MARKERS.get(m, "o"), color=COLORS[m], label=f"{m} (median)",
                lw=1.6 if m == "Q-TRACE" else 1.0)
    xs = np.logspace(-3, 3, 200)
    ax.plot(xs, np.maximum(1, xs), "k--", lw=0.8, label="Lorden bound (mean delay)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    clean_log_axis(ax, "x")
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(0.8, 300)
    ax.set_xlabel(r"$\log(1/\alpha)\,/\,(N D^\ast)$ [windows]")
    ax.set_ylabel("windows to first alarm")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


def fig_group_summary(runs: pd.DataFrame, path, far=1e-3, methods=None):
    setup()
    methods = methods or [m for m in COLORS if m in runs.method.unique()]
    col = f"detected@{far:g}"
    d = runs[runs.kind.isin(["attack", "composite"])].copy()
    known = {"intercept_resend", "pns", "blinding"}
    d["group"] = np.where(d.split == "test_shift", "physical shift",
                          np.where(d.kind == "composite", "composite",
                                   np.where(d.family.isin(known), "known", "zero-day")))
    groups = ["known", "zero-day", "composite", "physical shift"]
    per_seed = d.groupby(["seed", "group", "method"])[col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.3))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for g in groups:
            x = per_seed[(per_seed.group == g) & (per_seed.method == m)][col]
            vals.append(x.mean())
            ci = _ci(x)
            lo.append(x.mean() - ci[0] if np.isfinite(ci[0]) else 0)
            hi.append(ci[1] - x.mean() if np.isfinite(ci[1]) else 0)
        pos = np.arange(len(groups)) + (j - len(methods) / 2 + 0.5) * w
        ax.bar(pos, vals, w * 0.92, color=COLORS[m], label=m, yerr=[lo, hi],
               error_kw=dict(lw=0.6, capsize=1.2), edgecolor="black" if m == "Q-TRACE" else "none",
               linewidth=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(fr"runs detected at FAR $={far:g}$")
    ax.set_ylim(0, 1)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    save(fig, path)


def fig_false_alarms(runs: pd.DataFrame, path, far=1e-3, methods=None):
    setup()
    methods = methods or [m for m in COLORS if m in runs.method.unique()]
    col = f"any_alarm@{far:g}"
    cats = [("honest", "honest links"), ("loss_fault", "benign loss faults"),
            ("device_fault", "device fault (alarm desired)")]
    d = runs[runs.split == "test"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.0))
    w = 0.8 / len(methods)
    vmax = 0.0
    for j, m in enumerate(methods):
        vals = [d[(d.family == c) & (d.method == m)][col].mean() for c, _ in cats]
        vmax = max(vmax, np.nanmax(vals))
        pos = np.arange(len(cats)) + (j - len(methods) / 2 + 0.5) * w
        ax.bar(pos, vals, w * 0.92, color=COLORS[m], label=m,
               edgecolor="black" if m == "Q-TRACE" else "none", linewidth=0.6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[1] for c in cats])
    ax.set_ylabel("fraction of runs with an alarm")
    ax.set_ylim(0, max(0.1, 1.2 * vmax))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout()
    save(fig, path)


def fig_attribution(runs: pd.DataFrame, path, far=1e-3):
    setup()
    col = f"implicated@{far:g}"
    d = runs[(runs.method == "Q-TRACE") & runs[col].notna() & (runs.split == "test")]
    params = ["e_Z", "e_X", "p_d0", "p_d1", "kappa", "rho", "delta_e_decoy", "double", "eps_X"]
    plabel = [r"$e_Z$", r"$e_X$", r"$p_{d0}$", r"$p_{d1}$", r"$\kappa$", r"$\rho$",
              r"$\Delta e_{\rm dec}$", "double", r"$\epsilon_X$"]
    # attribution is defined over device parameters; RNG bias acts on the setting
    # allocation and is excluded (its alarms come from the setting channel)
    fams = [f for f in FAMILY_LABEL if f in d.family.unique() and "rng_bias" not in f]
    m = np.zeros((len(fams), len(params)))
    for i, f in enumerate(fams):
        vc = d[d.family == f][col].value_counts(normalize=True)
        for j, p in enumerate(params):
            m[i, j] = vc.get(p, 0.0)
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.72, 2.7))
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(plabel, rotation=0, fontsize=7)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in fams])
    for i in range(len(fams)):
        for j in range(len(params)):
            if m[i, j] >= 0.1:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("share of alarms")
    ax.set_xlabel("implicated device parameter at first alarm")
    fig.tight_layout()
    save(fig, path)


def fig_example_traces(examples_npz, thresholds, path, families=("pns", "blinding", "rng_bias"),
                       methods=("Q-TRACE", "Decoy CUSUM", "QBER CUSUM", "Autoencoder")):
    setup()
    z = np.load(examples_npz)
    fig, axes = plt.subplots(1, len(families), figsize=(DOUBLE, 1.9), sharey=True)
    for ax, fam in zip(axes, families):
        k = f"{fam}__0__"
        lab = z[k + "label"]
        t = np.arange(len(lab))
        on = np.argmax(lab) if lab.any() else None
        for m in methods:
            s = z[k + "score__" + m]
            thr = thresholds[m]
            ax.plot(t, s / thr, color=COLORS[m], lw=1.4 if m == "Q-TRACE" else 0.9, label=m)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        if on is not None:
            ax.axvspan(on, len(lab), color="0.92", zorder=0)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"{FAMILY_LABEL[fam]} (strength {float(z[k + 'strength']):.2g})")
        ax.set_xlabel("monitoring window")
    axes[0].set_ylabel("score / threshold")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- hardware
def fig_hw_detection(det: pd.DataFrame, path, alpha=1e-3):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharey=True)
    for ax, kind, xlab in ((axes[0], "probe", r"probe angle $\theta$"),
                           (axes[1], "ad", r"damping $\gamma$")):
        d = det[det.kind == kind].sort_values("strength")
        for t in ("Q-TRACE", "QBER (per basis)", "QBER (pooled)"):
            ax.plot(d.strength, d[f"{t}@{alpha:g}"], marker=MARKERS[t], color=COLORS[t], label=t,
                    lw=1.6 if t == "Q-TRACE" else 1.0)
        ax.set_xlabel(xlab)
        ax.set_ylim(-0.03, 1.03)
    h = det[det.kind == "honest"]
    if len(h):
        axes[0].set_title(f"entangling probe (honest FAR: Q-TRACE {h[f'Q-TRACE@{alpha:g}'].iloc[0]:.3f})")
    axes[1].set_title("amplitude-damping probe")
    axes[0].set_ylabel(fr"link-epochs detected ($\alpha={alpha:g}$)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, path)


def fig_hw_rounds(curve: pd.DataFrame, pred: pd.DataFrame, path, probes):
    setup()
    fig, axes = plt.subplots(1, len(probes), figsize=(DOUBLE, 2.0), sharey=True)
    for ax, pr in zip(np.atleast_1d(axes), probes):
        for t in ("Q-TRACE", "QBER (per basis)", "QBER (pooled)"):
            d = curve[(curve.probe == pr) & (curve.test == t)].sort_values("shots")
            ax.plot(d.shots, d.detect, marker=MARKERS[t], color=COLORS[t], label=t)
        p = pred[pred.probe == pr].sort_values("shots")
        ax.plot(p.shots, p.power, "k:", lw=0.9, label=r"Q-TRACE (noncentral $\chi^2$)")
        ax.set_xscale("log")
        ax.set_title(pr, fontsize=7)
        ax.set_xlabel("shots per circuit")
    np.atleast_1d(axes)[0].set_ylabel("detection probability")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=5.5)
    fig.tight_layout()
    save(fig, path)


def fig_hw_information(info: pd.DataFrame, det: pd.DataFrame, path):
    setup()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    m = info.merge(det[["probe", "measured_qber_excess"]], on="probe")
    for kind, mk, lab in (("probe", "o", "entangling probe"), ("ad", "s", "amplitude damping")):
        d = m[m.kind == kind].sort_values("strength")
        ax.plot(d.measured_qber_excess, d.I_sifted, mk + "-", color=COLORS["Q-TRACE"] if kind == "probe" else "#1F618D",
                label=f"{lab} (hardware)")
        ax.plot(d.measured_qber_excess, d.I_sifted_ideal, mk, mfc="none",
                color="0.4", label=f"{lab} (ideal device)")
    ax.set_xlabel("measured QBER increase")
    ax.set_ylabel("Eve's information [bits / sifted bit]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


def fig_hw_condition_shift(n1, n, design, path):
    """Honest links: commissioning versus honest-monitoring QBER, per basis."""
    from qtrace.hardware import analysis as A
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
    off = design.n_cal_passes
    sched = np.array(design.schedule)
    for ax, b, name in ((axes[0], 0, "Z"), (axes[1], 1, "X")):
        mask = lambda a, al, be, b=b: al == be == b
        xs, ys = [], []
        for i in range(design.n_links):
            ec, tc = A._error_counts(n1[:off, i].sum(0), n[:off, i].sum(0), mask)
            hon = [off + e for e in range(design.n_epochs) if sched[e, i] < 0]
            em, tm = A._error_counts(n1[hon, i].sum(0), n[hon, i].sum(0), mask)
            xs.append(ec / tc)
            ys.append(em / tm)
        xs, ys = np.array(xs), np.array(ys)
        lim = [0, max(xs.max(), ys.max()) * 1.08]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.scatter(xs, ys, s=10, color=COLORS["Q-TRACE"] if b else "#1F618D", alpha=0.8, linewidths=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{name}-basis QBER, commissioning")
        ax.set_ylabel(f"{name}-basis QBER, honest monitoring")
        ax.set_title(f"{name} basis: mean shift {np.mean(ys - xs):+.4f}")
    fig.tight_layout()
    save(fig, path)


def fig_hw_physics(det_matched: pd.DataFrame, info: pd.DataFrame, path):
    """(a) measured versus ideal QBER increase; (b) Eve's information versus QBER increase."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    d = det_matched[det_matched.kind != "honest"]
    ax = axes[0]
    for kind, mk, lab, col in (("probe", "o", "entangling probe", COLORS["Q-TRACE"]),
                               ("ad", "s", "amplitude damping", "#1F618D")):
        dd = d[d.kind == kind]
        ax.loglog(dd.ideal_qber, dd.measured_qber_excess.clip(lower=1e-4), mk, color=col, label=lab)
    xs = np.logspace(-3, -0.5, 20)
    ax.loglog(xs, xs, "k--", lw=0.7, label="ideal device")
    ax.set_xlabel("ideal QBER increase")
    ax.set_ylabel("measured QBER increase")
    ax.legend(loc="upper left")
    ax = axes[1]
    m = info.merge(det_matched[["probe", "measured_qber_excess"]], on="probe")
    for kind, mk, lab, col in (("probe", "o", "entangling probe", COLORS["Q-TRACE"]),
                               ("ad", "s", "amplitude damping", "#1F618D")):
        dd = m[m.kind == kind].sort_values("strength")
        ax.plot(dd.measured_qber_excess, dd.I_sifted, mk + "-", color=col, label=f"{lab} (hardware)")
        ax.plot(dd.measured_qber_excess, dd.I_sifted_ideal, mk, mfc="none", color="0.45",
                label=f"{lab} (ideal)")
    ax.set_xlabel("measured QBER increase")
    ax.set_ylabel("Eve's information [bit per sifted bit]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)
