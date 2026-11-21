"""QKD-TRACE v2 benchmark protocol.

One *seed* of the benchmark generates an independent population of links
and runs, processes them in a streaming fashion (only compact per-window
scores and features are kept), fits every method, sets every threshold on
attack-free validation links, and evaluates on held-out test links.

Splits
------
train  : honest links (+ known-attack links, used only by the supervised
         baseline).
val    : honest links, half with benign loss faults; used for thresholds.
test   : honest, benign-fault and device-fault links; the three known
         attack families; the three zero-day families; three compositions;
         and a physical-shift regime (60-120 km, low-efficiency receivers)
         never seen in training.

Labels are per window: a window is positive iff an attack is active in it.
Pre-onset windows of attack runs are honest and are labelled negative.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from qtrace.attacks.models import Composite, make_attack
from qtrace.baselines.detectors import (SupervisedXGB, UnsupervisedSuite, industry_scores,
                                        link_features)
from qtrace.detect.detectability import detectability_exponent, mean_state
from qtrace.detect.detector import ChannelNull, DetectorConfig, QTraceDetector
from qtrace.detect.score_test import PHYS_NAMES
from qtrace.physics.decoy import summarise_counts
from qtrace.twin.simulator import DecoyBB84Twin, Fault, Range, TwinConfig, sample_link

KNOWN = ("intercept_resend", "pns", "blinding")
ZERO_DAY = ("time_shift", "trojan_horse", "rng_bias")
COMPOSITES = (("pns", "blinding"), ("time_shift", "trojan_horse"),
              ("intercept_resend", "rng_bias"))
METHODS = ("Q-TRACE", "QBER static", "QBER CUSUM", "Decoy CUSUM", "Isolation Forest",
           "One-class SVM", "Autoencoder", "Quantum-kernel OCSVM", "XGBoost (supervised)")


@dataclass
class BenchmarkConfig:
    n_windows: int = 200
    n_cal: int = 60
    pulses_per_window: int = 10_000_000
    onset_range: tuple = (20, 100)            # relative to the end of commissioning
    afterpulse_spec_error: float = 0.10
    runs: dict = field(default_factory=lambda: {
        "train_honest": 60, "train_attack": 30,
        "val_honest": 50, "val_loss_fault": 25,
        "test_honest": 50, "test_loss_fault": 25, "test_device_fault": 15,
        "test_attack": 35, "test_composite": 20,
        "shift_honest": 25, "shift_attack": 15,
    })


def sample_attack(name, onset, rng):
    """Draw an attack instance with physically motivated strength ranges."""
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if name == "intercept_resend":
        return make_attack(name, onset=onset, fraction=lu(0.002, 0.2)), "fraction"
    if name in ("pns", "blinding"):
        return make_attack(name, onset=onset, fraction=lu(0.01, 1.0)), "fraction"
    if name == "time_shift":
        return make_attack(name, onset=onset, fraction=lu(0.02, 1.0),
                           mismatch=rng.uniform(0.1, 0.5)), "fraction"
    if name == "trojan_horse":
        return make_attack(name, onset=onset, backscatter=lu(1e-8, 3e-6)), "backscatter"
    if name == "rng_bias":
        return make_attack(name, onset=onset, delta=lu(5e-5, 2e-3),
                           period=rng.uniform(10, 80), phase=rng.uniform(0, 2 * np.pi)), "delta"
    raise KeyError(name)


def shifted_twin_config():
    cfg = TwinConfig()
    cfg.length_km = Range(60.0, 120.0)
    cfg.eta_bob = Range(0.05, 0.20)
    return cfg


def build_run_specs(cfg: BenchmarkConfig, rng):
    specs = []
    r = cfg.runs
    lo, hi = cfg.onset_range

    def onset():
        return int(cfg.n_cal + rng.integers(lo, hi))

    def add(split, family, n, kind, regime="nominal", **kw):
        for _ in range(n):
            specs.append(dict(split=split, family=family, kind=kind, regime=regime,
                              onset=onset(), **kw))

    add("train", "honest", r["train_honest"], "honest")
    for a in KNOWN:
        add("train", a, r["train_attack"], "attack")
    add("val", "honest", r["val_honest"], "honest")
    add("val", "loss_fault", r["val_loss_fault"], "fault")
    add("test", "honest", r["test_honest"], "honest")
    add("test", "loss_fault", r["test_loss_fault"], "fault")
    add("test", "device_fault", r["test_device_fault"], "fault")
    for a in KNOWN + ZERO_DAY:
        add("test", a, r["test_attack"], "attack")
    for c in COMPOSITES:
        add("test", "+".join(c), r["test_composite"], "composite", components=c)
    add("test_shift", "honest", r["shift_honest"], "honest", regime="shifted")
    for a in KNOWN:
        add("test_shift", a, r["shift_attack"], "attack", regime="shifted")
    return specs


def process_run(spec, cfg: BenchmarkConfig, twins, detector: QTraceDetector, rng):
    twin = twins[spec["regime"]]
    link = sample_link(twin.cfg, rng)
    attack, fault, strength = None, None, np.nan
    if spec["kind"] == "attack":
        attack, key = sample_attack(spec["family"], spec["onset"], rng)
        strength = getattr(attack, key)
    elif spec["kind"] == "composite":
        parts = [sample_attack(c, spec["onset"], rng)[0] for c in spec["components"]]
        attack = Composite(parts)
    elif spec["kind"] == "fault":
        if spec["family"] == "loss_fault":
            kind = "loss_step" if rng.random() < 0.6 else "loss_ramp"
            fault = Fault(kind, spec["onset"], rng.uniform(0.5, 3.0))
        else:
            fault = Fault("darkcount_rise", spec["onset"], rng.uniform(1.5, 5.0))
    res = twin.simulate(link, cfg.n_windows, rng, attack=attack, fault=fault,
                        pulses_per_window=cfg.pulses_per_window)

    ap_spec = link.afterpulse * (1 + cfg.afterpulse_spec_error * rng.uniform(-1, 1))
    _, channels, z, cal, eta = detector.score_run(res.counts, cfg.n_cal, link.mu_signal, ap_spec,
                                                  link.p_intensity)
    pz = detector.parameter_zscores(cal)
    summ = summarise_counts(res.counts, link.mu_signal, link.mu_decoy)
    X, _, _ = link_features(summ, cfg.n_cal)
    ind = industry_scores(summ, cfg.n_cal)

    d_star = np.nan
    if attack is not None and spec["kind"] == "attack":
        st = mean_state(res.state, slice(cfg.n_cal, None))
        tv = (spec["onset"] + np.arange(int(np.ceil(attack.period))) if spec["family"] == "rng_bias"
              else None)
        d_star, _ = detectability_exponent(st, link.afterpulse, copy.deepcopy(attack), rng,
                                           t_values=tv)

    active = res.active[cfg.n_cal:]
    if spec["kind"] in ("honest",) or spec["family"] == "loss_fault":
        label = np.zeros_like(active)
    else:
        label = active.astype(bool)
    return dict(
        spec=spec, link=link.to_dict(), strength=strength, d_star=d_star,
        attack=attack.describe() if attack is not None else None,
        channels={k: v.astype(np.float32) for k, v in channels.items()},
        param_z=pz.astype(np.float32),
        X=X.astype(np.float32), industry={k: v.astype(np.float32) for k, v in ind.items()},
        label=label, active=active, key_rate=summ["key_rate"][cfg.n_cal:].astype(np.float32),
        fit=cal.fit_info,
    )


def threshold_at_far(scores, far):
    return float(np.quantile(scores, 1.0 - far))


def first_alarm(score, thr, start):
    hits = np.where(score[start:] > thr)[0]
    return int(hits[0]) if len(hits) else None


def run_seed(seed: int, cfg: BenchmarkConfig, det_cfg: DetectorConfig | None = None,
             fars=(1e-3, 1e-2), log=print):
    rng = np.random.default_rng(seed)
    twins = {"nominal": DecoyBB84Twin(TwinConfig()), "shifted": DecoyBB84Twin(shifted_twin_config())}
    for t in twins.values():
        t.cfg.pulses_per_window = cfg.pulses_per_window
    detector = QTraceDetector(det_cfg)
    specs = build_run_specs(cfg, rng)
    runs = []
    t0 = time.time()
    for i, spec in enumerate(specs):
        runs.append(process_run(spec, cfg, twins, detector, rng))
        if (i + 1) % 50 == 0:
            log(f"[seed {seed}] {i + 1}/{len(specs)} runs  ({time.time() - t0:.0f}s)")

    # ---- fit methods on training data
    train_honest = [r for r in runs if r["spec"]["split"] == "train" and r["spec"]["kind"] == "honest"]
    null = ChannelNull().fit([r["channels"] for r in train_honest])
    X_h = np.vstack([r["X"] for r in train_honest])
    unsup = UnsupervisedSuite(seed=seed).fit(X_h)
    tr = [r for r in runs if r["spec"]["split"] == "train"]
    sup = SupervisedXGB(seed=seed).fit(np.vstack([r["X"] for r in tr]),
                                       np.concatenate([r["label"] for r in tr]).astype(int))

    for r in runs:
        if r["spec"]["split"] == "train":
            continue
        s = {"Q-TRACE": null.transform(r["channels"])}
        s.update(r["industry"])
        s.update(unsup.score(r["X"]))
        s["XGBoost (supervised)"] = sup.score(r["X"])
        r["scores"] = {k: np.asarray(v, float) for k, v in s.items()}

    # ---- thresholds on validation honest + benign-fault links
    val = [r for r in runs if r["spec"]["split"] == "val"]
    thresholds = {m: {far: threshold_at_far(np.concatenate([r["scores"][m] for r in val]), far)
                      for far in fars} for m in METHODS}

    # ---- metrics
    rows, run_rows = [], []
    for r in runs:
        if r["spec"]["split"] == "train":
            continue
        sp = r["spec"]
        onset_rel = sp["onset"] - cfg.n_cal
        for m in METHODS:
            sc = r["scores"][m]
            entry = dict(seed=seed, split=sp["split"], family=sp["family"], kind=sp["kind"],
                         regime=sp["regime"], method=m, strength=r["strength"],
                         d_star=r["d_star"], onset=onset_rel,
                         length_km=r["link"]["length_km"])
            for far in fars:
                thr = thresholds[m][far]
                pre = first_alarm(sc[:onset_rel], thr, 0) if onset_rel > 0 else None
                fa = first_alarm(sc, thr, onset_rel)
                entry[f"detected@{far:g}"] = fa is not None
                entry[f"delay@{far:g}"] = fa if fa is not None else np.nan
                entry[f"pre_onset_alarm@{far:g}"] = pre is not None
                if m == "Q-TRACE" and fa is not None:
                    t_al = onset_rel + fa
                    entry[f"implicated@{far:g}"] = PHYS_NAMES[int(np.argmax(np.abs(r["param_z"][t_al])))]
                entry[f"any_alarm@{far:g}"] = bool((sc > thr).any())
                entry[f"window_alarm_rate@{far:g}"] = float((sc > thr).mean())
                post = r["label"].astype(bool)
                if post.any():
                    kr = r["key_rate"][post]
                    det = sc[post] > thr
                    cut = np.argmax(det) if det.any() else len(det)
                    entry[f"key_exposed@{far:g}"] = float(kr[:cut].sum() * cfg.pulses_per_window)
            run_rows.append(entry)

    families = sorted({r["spec"]["family"] for r in runs if r["spec"]["split"] in ("test", "test_shift")
                       and r["spec"]["kind"] in ("attack", "composite")})
    for split in ("test", "test_shift"):
        neg_runs = [r for r in runs if r["spec"]["split"] == split and r["spec"]["family"] == "honest"]
        for fam in families:
            pos_runs = [r for r in runs if r["spec"]["split"] == split and r["spec"]["family"] == fam]
            if not pos_runs:
                continue
            for m in METHODS:
                neg = np.concatenate([r["scores"][m] for r in neg_runs])
                pos = np.concatenate([r["scores"][m][r["label"].astype(bool)] for r in pos_runs])
                y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
                s = np.r_[neg, pos]
                row = dict(seed=seed, split=split, family=fam, method=m,
                           AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s),
                           group=("known" if fam in KNOWN else "zero-day" if fam in ZERO_DAY
                                  else "composite"))
                for far in fars:
                    row[f"TPR@{far:g}"] = float((pos > thresholds[m][far]).mean())
                rows.append(row)
    window_df = pd.DataFrame(rows)
    run_df = pd.DataFrame(run_rows)
    meta = dict(seed=seed, n_runs=len(runs), thresholds=thresholds, null=null.to_dict(),
                elapsed_s=time.time() - t0,
                fit_success_rate=float(np.mean([r["fit"]["success"] for r in runs])))
    return window_df, run_df, meta, runs
