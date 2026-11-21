"""Analysis of the multiplexed BB84 hardware experiment.

Per (link, monitoring epoch) three tests compare the epoch with the link's
own commissioning data:

``Q-TRACE``
    The hardware analogue of the simulation detector: physically
    structured channels plus an omnibus channel, Bonferroni-combined,
    ``p = min(1, 3 min(p_err, p_asym, p_omni))``:

    * ``err``: two-sided change of the Z- and X-basis error rates, chi2_2;
    * ``asym``: change of the bit asymmetry ``P(1|a=0) - P(0|a=1)`` in each
      basis (non-unital disturbance, e.g. amplitude damping), chi2_2;
    * ``omni``: full-table two-sample G test over all eight settings
      (sifted and unsifted, both bit values), ``G / phi ~ chi2_8``.

    Dispersions (drift between circuits beyond shot noise) are estimated
    from the two commissioning passes and pooled over links.
``QBER (pooled)``
    One-sided two-proportion test on the sifted QBER, the quantity every
    deployed system monitors.
``QBER (per basis)``
    The stronger variant: one-sided tests on the Z- and X-basis QBERs,
    Bonferroni-combined.

Further outputs: detection probability versus rounds (shot subsampling)
with the noncentral-chi-square prediction, Eve's mutual information with
Alice's bit measured on her ancilla, and the information leaked to Eve
before the first alarm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, ncx2, norm

from qtrace.hardware.design import SETTINGS, ExperimentDesign


# --------------------------------------------------------------------- tables
def count_tables(bob, design: ExperimentDesign, n_shots=None, rng=None):
    """Bob=1 counts and totals per (pass, link, setting).

    ``n_shots`` subsamples the shots of *monitoring* circuits only
    (commissioning is always used in full). Returns ``n1, n`` with shape
    ``(n_passes, L, 8)``.
    """
    nc, s, nl = bob.shape
    n_pass = nc // 8
    n1 = np.zeros((n_pass, nl, 8))
    n = np.zeros((n_pass, nl, 8))
    for c in range(nc):
        p = c // 8
        b = bob[c]
        if n_shots is not None and p >= design.n_cal_passes and n_shots < s:
            idx = (rng.choice(s, n_shots, replace=False) if rng is not None
                   else np.arange(n_shots))
            b = b[idx]
        for i in range(nl):
            k = (i + c) % 8
            n1[p, i, k] += b[:, i].sum()
            n[p, i, k] += len(b)
    return n1, n


def _g_two_sample(x1, n1, x2, n2):
    """Likelihood-ratio statistic for equal binomial proportions (elementwise)."""
    def term(x, n, p):
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(x > 0, x * np.log(x / (n * p)), 0.0)
            b = np.where(n - x > 0, (n - x) * np.log((n - x) / (n * (1 - p))), 0.0)
        return a + b
    pp = np.clip((x1 + x2) / (n1 + n2), 1e-12, 1 - 1e-12)
    return 2 * (term(x1, n1, pp) + term(x2, n2, pp))


def _error_counts(n1, n, settings_mask):
    """Sifted errors and rounds from Bob=1 counts for the selected settings."""
    err = np.zeros(n1.shape[:-1])
    tot = np.zeros(n1.shape[:-1])
    for k, (a, al, be) in enumerate(SETTINGS):
        if not settings_mask(a, al, be):
            continue
        e = n1[..., k] if a == 0 else n[..., k] - n1[..., k]
        err += e
        tot += n[..., k]
    return err, tot


def _rate(e, t):
    return e / np.maximum(t, 1)


def _rate_var(e, t):
    r = _rate(e, t)
    return np.clip(r * (1 - r), 1e-9, None) / np.maximum(t, 1)


def _qber_z(err_m, tot_m, err_c, tot_c):
    qm, qc = err_m / tot_m, err_c / tot_c
    qp = (err_m + err_c) / (tot_m + tot_c)
    se = np.sqrt(np.clip(qp * (1 - qp), 1e-12, None) * (1 / tot_m + 1 / tot_c))
    return (qm - qc) / se


def dispersion(n1, n, design):
    """Pooled excess dispersion from the two commissioning passes (>= 1)."""
    if design.n_cal_passes < 2:
        return 1.0, 1.0, 1.0
    g = _g_two_sample(n1[0], n[0], n1[1], n[1]).sum(axis=-1)          # (L,)
    phi = max(1.0, float(np.mean(g) / 8.0))
    sifted = lambda a, al, be: al == be
    zs = []
    for mask in (sifted, lambda a, al, be: al == be == 0, lambda a, al, be: al == be == 1):
        e0, t0 = _error_counts(n1[0], n[0], mask)
        e1, t1 = _error_counts(n1[1], n[1], mask)
        zs.append(_qber_z(e0, t0, e1, t1))
    phi_q = max(1.0, float(np.mean(zs[0] ** 2)))
    phi_qb = max(1.0, float(np.mean(np.r_[zs[1], zs[2]] ** 2)))
    return phi, phi_q, phi_qb


def link_epoch_tests(n1, n, design, phis=None):
    """p-values of the three tests for every (link, monitoring epoch)."""
    phi, phi_q, phi_qb = phis or dispersion(n1, n, design)
    cal = slice(0, design.n_cal_passes)
    c1, cn = n1[cal].sum(axis=0), n[cal].sum(axis=0)
    rows = []
    sifted = lambda a, al, be: al == be
    zmask = lambda a, al, be: al == be == 0
    xmask = lambda a, al, be: al == be == 1
    ec, tc = _error_counts(c1, cn, sifted)
    ecz, tcz = _error_counts(c1, cn, zmask)
    ecx, tcx = _error_counts(c1, cn, xmask)
    for e in range(design.n_epochs):
        p = design.n_cal_passes + e
        g = _g_two_sample(n1[p], n[p], c1, cn).sum(axis=-1)
        em, tm = _error_counts(n1[p], n[p], sifted)
        emz, tmz = _error_counts(n1[p], n[p], zmask)
        emx, tmx = _error_counts(n1[p], n[p], xmask)
        zq = _qber_z(em, tm, ec, tc) / np.sqrt(phi_q)
        zz = _qber_z(emz, tmz, ecz, tcz) / np.sqrt(phi_qb)
        zx = _qber_z(emx, tmx, ecx, tcx) / np.sqrt(phi_qb)
        za = []
        for b in (0, 1):
            m0 = lambda a, al, be, b=b: al == be == b and a == 0
            m1 = lambda a, al, be, b=b: al == be == b and a == 1
            e0m, t0m = _error_counts(n1[p], n[p], m0)
            e1m, t1m = _error_counts(n1[p], n[p], m1)
            e0c, t0c = _error_counts(c1, cn, m0)
            e1c, t1c = _error_counts(c1, cn, m1)
            q, v = _rate, _rate_var
            dlt = (q(e1m, t1m) - q(e0m, t0m)) - (q(e1c, t1c) - q(e0c, t0c))
            za.append(dlt / np.sqrt(v(e1m, t1m) + v(e0m, t0m) + v(e1c, t1c) + v(e0c, t0c)))
        s_asym = (za[0] ** 2 + za[1] ** 2) / phi_qb
        s_err = zz ** 2 + zx ** 2
        p_err = chi2.sf(s_err, 2)
        p_asym = chi2.sf(s_asym, 2)
        p_omni = chi2.sf(g / phi, 8)
        p_qt = np.minimum(1.0, 3 * np.minimum(np.minimum(p_err, p_asym), p_omni))
        for i in range(design.n_links):
            j = design.schedule[e][i]
            probe = design.probes[j] if j >= 0 else None
            rows.append(dict(
                link=i, epoch=e, probe=probe.label if probe else "honest",
                kind=probe.kind if probe else "honest",
                strength=probe.strength if probe else 0.0,
                ideal_qber=probe.ideal_qber() if probe else 0.0,
                qber_cal=ec[i] / tc[i], qber_epoch=em[i] / tm[i],
                G=g[i], p_qtrace=p_qt[i], p_err=p_err[i], p_asym=p_asym[i], p_omni=p_omni[i],
                p_qber=norm.sf(zq[i]),
                p_qber_basis=min(1.0, 2 * min(norm.sf(zz[i]), norm.sf(zx[i]))),
            ))
    return pd.DataFrame(rows), dict(phi=phi, phi_qber=phi_q, phi_qber_basis=phi_qb)


def _block_pvalues(x1, xn, c1, cn, phis):
    """p-values of every test for test counts ``(x1, xn)`` against reference ``(c1, cn)``.

    All arrays have shape ``(L, 8)``; returns a dict of length-``L`` arrays.
    """
    phi, phi_q, phi_qb = phis
    sifted = lambda a, al, be: al == be
    zmask = lambda a, al, be: al == be == 0
    xmask = lambda a, al, be: al == be == 1
    g = _g_two_sample(x1, xn, c1, cn).sum(axis=-1)
    em, tm = _error_counts(x1, xn, sifted)
    ec, tc = _error_counts(c1, cn, sifted)
    zq = _qber_z(em, tm, ec, tc) / np.sqrt(phi_q)
    zz = _qber_z(*_error_counts(x1, xn, zmask), *_error_counts(c1, cn, zmask)) / np.sqrt(phi_qb)
    zx = _qber_z(*_error_counts(x1, xn, xmask), *_error_counts(c1, cn, xmask)) / np.sqrt(phi_qb)
    za = []
    for b in (0, 1):
        m0 = lambda a, al, be, b=b: al == be == b and a == 0
        m1 = lambda a, al, be, b=b: al == be == b and a == 1
        e0m, t0m = _error_counts(x1, xn, m0)
        e1m, t1m = _error_counts(x1, xn, m1)
        e0c, t0c = _error_counts(c1, cn, m0)
        e1c, t1c = _error_counts(c1, cn, m1)
        dlt = (_rate(e1m, t1m) - _rate(e0m, t0m)) - (_rate(e1c, t1c) - _rate(e0c, t0c))
        za.append(dlt / np.sqrt(_rate_var(e1m, t1m) + _rate_var(e0m, t0m)
                                + _rate_var(e1c, t1c) + _rate_var(e0c, t0c)))
    p_err = chi2.sf(zz ** 2 + zx ** 2, 2)
    p_asym = chi2.sf((za[0] ** 2 + za[1] ** 2) / phi_qb, 2)
    p_omni = chi2.sf(g / phi, 8)
    return dict(G=g, qber_epoch=em / tm, qber_ref=ec / tc,
                p_qtrace=np.minimum(1.0, 3 * np.minimum(np.minimum(p_err, p_asym), p_omni)),
                p_err=p_err, p_asym=p_asym, p_omni=p_omni, p_qber=norm.sf(zq),
                p_qber_basis=np.minimum(1.0, 2 * np.minimum(norm.sf(zz), norm.sf(zx))))


def matched_tests(n1, n, design, phis=None):
    """Condition-matched analysis: reference = the link's honest *monitoring* epochs.

    Commissioning circuits contain no probe gates, so their timing and the
    concurrent two-qubit activity on the chip differ from monitoring
    circuits. Here each link's reference is built from its honest
    monitoring epochs, which run under identical conditions. Attacked
    epochs are tested against the pooled honest epochs of the same link;
    the false-alarm rate is measured by testing each honest epoch against
    the link's *other* honest epoch(s).
    """
    phis = phis or dispersion(n1, n, design)
    sched = np.array(design.schedule)                     # (E, L)
    nl = design.n_links
    off = design.n_cal_passes
    rows = []

    def add(res, link, epoch, probe, ref_kind):
        rows.append(dict(link=link, epoch=epoch, probe=probe.label if probe else "honest",
                         kind=probe.kind if probe else "honest",
                         strength=probe.strength if probe else 0.0,
                         ideal_qber=probe.ideal_qber() if probe else 0.0, reference=ref_kind,
                         **{k: float(v) for k, v in res.items()}))

    for i in range(nl):
        honest = [e for e in range(design.n_epochs) if sched[e, i] < 0]
        c1 = n1[[off + e for e in honest], i].sum(0)
        cn = n[[off + e for e in honest], i].sum(0)
        for e in range(design.n_epochs):
            j = sched[e, i]
            if j >= 0:
                res = _block_pvalues(n1[off + e, i][None], n[off + e, i][None], c1[None], cn[None], phis)
                add({k: v[0] for k, v in res.items()}, i, e, design.probes[j], "honest monitoring epochs")
            else:
                others = [h for h in honest if h != e]
                r1 = n1[[off + h for h in others], i].sum(0)
                rn = n[[off + h for h in others], i].sum(0)
                res = _block_pvalues(n1[off + e, i][None], n[off + e, i][None], r1[None], rn[None], phis)
                add({k: v[0] for k, v in res.items()}, i, e, None, "other honest epoch")
    return pd.DataFrame(rows)


def matched_detection_vs_rounds(bob, design, grid=None, n_rep=5, alpha=1e-3, seed=0):
    """Detection probability vs shots per circuit for the condition-matched analysis.

    Only the tested epoch is subsampled; references use all shots.
    """
    rng = np.random.default_rng(seed)
    s = bob.shape[1]
    grid = grid or [g for g in (50, 100, 200, 400, 800, 1600, 3200) if g < s] + [s]
    n1f, nf = count_tables(bob, design)
    phis = dispersion(n1f, nf, design)
    sched = np.array(design.schedule)
    off = design.n_cal_passes
    out = []
    for ns in grid:
        for rep in range(n_rep if ns < s else 1):
            n1s, ns_ = count_tables(bob, design, n_shots=ns, rng=rng)
            for i in range(design.n_links):
                honest = [e for e in range(design.n_epochs) if sched[e, i] < 0]
                for e in range(design.n_epochs):
                    j = sched[e, i]
                    ref = honest if j >= 0 else [h for h in honest if h != e]
                    c1 = n1f[[off + h for h in ref], i].sum(0)[None]
                    cn = nf[[off + h for h in ref], i].sum(0)[None]
                    res = _block_pvalues(n1s[off + e, i][None], ns_[off + e, i][None], c1, cn, phis)
                    p = design.probes[j] if j >= 0 else None
                    for name, col in TESTS.items():
                        out.append(dict(shots=ns, rep=rep, probe=p.label if p else "honest",
                                        kind=p.kind if p else "honest",
                                        strength=p.strength if p else 0.0, test=name,
                                        detect=float(res[col][0] < alpha)))
    d = pd.DataFrame(out)
    return d.groupby(["shots", "probe", "kind", "strength", "test"], as_index=False)["detect"].mean()


TESTS = {"Q-TRACE": "p_qtrace", "QBER (pooled)": "p_qber", "QBER (per basis)": "p_qber_basis",
         "Omnibus only": "p_omni"}


def detection_table(df, alphas=(1e-3, 1e-2)):
    ref_col = "qber_cal" if "qber_cal" in df.columns else "qber_ref"
    out = []
    for probe, g in df.groupby("probe", sort=False):
        row = dict(probe=probe, kind=g.kind.iloc[0], strength=g.strength.iloc[0],
                   ideal_qber=g.ideal_qber.iloc[0], n=len(g),
                   measured_qber_excess=float((g.qber_epoch - g[ref_col]).mean()))
        for name, col in TESTS.items():
            for a in alphas:
                row[f"{name}@{a:g}"] = float((g[col] < a).mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["kind", "strength"])


# ----------------------------------------------------- rounds to detection
def detection_vs_rounds(bob, design, grid=None, n_rep=5, alpha=1e-3, seed=0):
    """Detection probability vs shots per circuit, by probe and test."""
    rng = np.random.default_rng(seed)
    s = bob.shape[1]
    grid = grid or [g for g in (50, 100, 200, 400, 800, 1600, 3200, 6400) if g < s] + [s]
    n1f, nf = count_tables(bob, design)
    phis = dispersion(n1f, nf, design)
    rows = []
    for ns in grid:
        for rep in range(n_rep if ns < s else 1):
            n1, n = count_tables(bob, design, n_shots=ns, rng=rng)
            df, _ = link_epoch_tests(n1, n, design, phis)
            for probe, g in df.groupby("probe", sort=False):
                for name, col in TESTS.items():
                    rows.append(dict(shots=ns, rep=rep, probe=probe, kind=g.kind.iloc[0],
                                     strength=g.strength.iloc[0], test=name,
                                     detect=float((g[col] < alpha).mean())))
    d = pd.DataFrame(rows)
    return d.groupby(["shots", "probe", "kind", "strength", "test"], as_index=False)["detect"].mean()


def predicted_power(bob, design, shots_grid, alpha=1e-3):  # omnibus channel
    """Noncentral-chi-square prediction of Q-TRACE power from the measured tables."""
    n1, n = count_tables(bob, design)
    phi, _, _ = dispersion(n1, n, design)
    cal = slice(0, design.n_cal_passes)
    pc = n1[cal].sum(0) / n[cal].sum(0)                  # (L, 8)
    nc = n[cal].sum(0)
    s_full = bob.shape[1]
    crit = chi2.isf(alpha, 8)
    rows = []
    for e in range(design.n_epochs):
        p = design.n_cal_passes + e
        pm = n1[p] / n[p]
        for i in range(design.n_links):
            j = design.schedule[e][i]
            label = design.probes[j].label if j >= 0 else "honest"
            pbar = np.clip(pc[i], 1e-6, 1 - 1e-6)
            for ns in shots_grid:
                nm = n[p, i] * ns / s_full
                lam = np.sum((pm[i] - pc[i]) ** 2 / (pbar * (1 - pbar)) * nm / (1 + nm / nc[i]))
                rows.append(dict(probe=label, shots=ns, power=float(ncx2.sf(crit, 8, lam / phi))))
    return pd.DataFrame(rows).groupby(["probe", "shots"], as_index=False)["power"].mean()


def rounds_to_detect(curve, level=0.5):
    """Shots per circuit at which detection probability first reaches ``level`` (log-interp)."""
    out = []
    for (probe, test), g in curve.groupby(["probe", "test"]):
        g = g.sort_values("shots")
        x, y = np.log(g.shots.values), g.detect.values
        hit = np.where(y >= level)[0]
        if len(hit) == 0:
            v = np.nan
        elif hit[0] == 0:
            v = float(np.exp(x[0]))
        else:
            k = hit[0]
            v = float(np.exp(x[k - 1] + (level - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1])))
        out.append(dict(probe=probe, test=test, shots_to_detect=v))
    return pd.DataFrame(out)


# ------------------------------------------------------------ Eve's information
def _mi(x, y):
    """Plug-in mutual information (bits) with Miller-Madow bias correction."""
    n = len(x)
    if n == 0:
        return np.nan
    joint = np.zeros((2, 2))
    np.add.at(joint, (x, y), 1)
    p = joint / n
    px, py = p.sum(1), p.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = lambda q: -np.nansum(np.where(q > 0, q * np.log2(q), 0.0))
    mi = h(px) + h(py) - h(p.reshape(-1))
    k = lambda q: np.count_nonzero(q)
    corr = (k(p.reshape(-1)) - k(px) - k(py) + 1) / (2 * n * np.log(2))
    return float(max(mi - corr, 0.0))


def eve_information(bob, eve, design):
    """Eve's information on sifted key bits, per probe.

    Eve learns the bases at sifting, so she conditions on whether her probe
    basis matched Alice's. Returns per-probe ``I_matched`` (bits per round
    where her basis matched), ``I_sifted`` (bits per sifted round) and her
    guessing error on matched rounds.
    """
    nc, s, nl = bob.shape
    acc = {}
    for c in range(nc):
        for i in range(nl):
            p = design.probe(c, i)
            if p.kind == "honest":
                continue
            a, al, be = design.setting(c, i)
            if al != be:
                continue
            if p.kind == "probe":
                match = design.eve_basis[c][i] == al
            else:
                match = al == 0
            d = acc.setdefault(p.label, dict(kind=p.kind, strength=p.strength,
                                             m_a=[], m_e=[], n_sift=0, n_match=0))
            d["n_sift"] += s
            if match:
                d["n_match"] += s
                d["m_a"].append(np.full(s, a, dtype=np.uint8))
                d["m_e"].append(eve[c, :, i])
    rows = []
    for label, d in acc.items():
        a = np.concatenate(d["m_a"])
        e = np.concatenate(d["m_e"])
        mi = _mi(a, e)
        err = float(np.mean(a != e))
        frac = d["n_match"] / d["n_sift"]
        if d["kind"] == "probe":
            th = d["strength"]
            pe = (1 - np.sin(th)) / 2
            ideal = 1 - (-(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe)) if 0 < pe < 1 else 0.0)
        else:
            g = d["strength"]
            # Z channel: a=1 -> e=1 w.p. g ; a=0 -> e=0
            pe1 = 0.5 * g
            hb = lambda q: 0.0 if q in (0, 1) else -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
            ideal = hb(pe1) - 0.5 * hb(g)
        rows.append(dict(probe=label, kind=d["kind"], strength=d["strength"],
                         I_matched=mi, I_sifted=mi * frac, guess_error=min(err, 1 - err),
                         I_matched_ideal=ideal, I_sifted_ideal=ideal * frac))
    return pd.DataFrame(rows).sort_values(["kind", "strength"])


def leaked_before_detection(info, r2d, sifted_settings_per_pass=4):
    """Bits of sifted key Eve learns before the median first alarm."""
    m = r2d.merge(info[["probe", "I_sifted"]], on="probe")
    m["sifted_rounds_to_detect"] = m.shots_to_detect * sifted_settings_per_pass
    m["bits_leaked_before_alarm"] = m.sifted_rounds_to_detect * m.I_sifted
    return m
