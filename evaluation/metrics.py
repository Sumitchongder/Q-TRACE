"""
Q-TRACE :: evaluation.metrics
================================
Standard detection metrics plus the two Q-TRACE-specific operational
metrics from the research plan: Secret-Key Retention Rate (SKRR, Section
18) and the composite risk-weighted objective J (Section 19).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve


def detection_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {}
    try:
        out["AUROC"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["AUROC"] = np.nan
    try:
        out["AUPRC"] = average_precision_score(y_true, y_score)
    except ValueError:
        out["AUPRC"] = np.nan
    y_pred = (y_score >= threshold).astype(int)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_score) if len(np.unique(y_true)) > 1 else ([0], [0], [0])
    out["FAR"] = float(fpr[np.argmin(np.abs(np.array(tpr) - 0.95))]) if len(fpr) > 1 else np.nan
    return out


def secret_key_retention_rate(skr_usable, skr_baseline):
    """
    SKRR = K_usable_after_detection / K_usable_baseline  (Section 18).
    Both arguments are arrays of secure-key-rate estimates (skr_est column
    from the telemetry, see qkd_simulator.physics.secure_key_rate_asymptotic);
    "usable_after_detection" should be pre-masked to zero wherever the
    detector gated key distillation (see gate_key_distillation below).
    """
    baseline_total = np.sum(np.clip(skr_baseline, 0, None))
    usable_total = np.sum(np.clip(skr_usable, 0, None))
    if baseline_total <= 0:
        return np.nan
    return float(usable_total / baseline_total)


def gate_key_distillation(skr_est, risk_score, tau_threat):
    """
    Zero out usable key rate wherever the risk engine declares THREAT
    (risk_score >= tau_threat, Section 20). Windows below threshold keep
    their full estimated key contribution.
    """
    gated = np.array(skr_est, dtype=float).copy()
    gated[np.asarray(risk_score) >= tau_threat] = 0.0
    return gated


def find_threshold_for_recall(risk_score, y_true, target_recall):
    """
    Find the gating threshold tau such that (approximately) `target_recall`
    fraction of ACTUAL attack windows would be gated.

    Motivation: the top-decile threshold used elsewhere (Section 20's
    tau_1/tau_2 scheme, `np.percentile(risk, 90)`) is a percentile of the
    ENTIRE evaluation population, not of the attack class specifically. On
    an evaluation split where attacks are the majority class (as our
    zero-day split is, by construction of the dataset generator -- three
    attack families plus one normal class means attacks outnumber normal
    windows roughly 3:1), "top 10% of everything" is a much more
    conservative cutoff than it sounds, because most of that population is
    already attack traffic. This function instead lets you directly pick
    the sensitivity you want to report (e.g. "the threshold that catches
    80% of attacks") and see what SKRR costs at THAT operating point,
    rather than being confined to a single, population-dependent default.

    Parameters
    ----------
    risk_score : array-like, the score used to threshold (higher = more
        suspicious). Length must match y_true.
    y_true : array-like of {0, 1}, 1 = attack window, 0 = normal window.
    target_recall : float in (0, 1], desired fraction of attack windows
        to be gated.

    Returns
    -------
    tau : float, the threshold value such that risk_score >= tau gates
        approximately `target_recall` of the attack-class windows. Found
        as the (1 - target_recall) quantile of the attack-class scores.
    """
    y_true = np.asarray(y_true).astype(int)
    risk_score = np.asarray(risk_score)
    attack_scores = risk_score[y_true == 1]
    if len(attack_scores) == 0:
        return np.nan
    quantile = np.clip(1.0 - target_recall, 0.0, 1.0)
    return float(np.quantile(attack_scores, quantile))


def gating_effectiveness(y_true, risk_score, tau_threat):
    """
    Companion metric to SKRR: SKRR alone can be misleading, because a
    detector that gates NOTHING trivially retains 100% of key while also
    catching 0% of attacks. This function reports, for a given gating
    strategy, what fraction of ACTUAL attack windows it caught versus
    missed, and what fraction of LEGITIMATE (normal) windows it wrongly
    gated (over-blocking cost). Read alongside SKRR: a high SKRR paired
    with a high `attack_missed_rate` means the strategy is "retaining key"
    only because it isn't detecting anything, not because it is
    selectively sparing legitimate traffic while catching attacks.

    Parameters
    ----------
    y_true : array-like of {0, 1}, 1 = attack window, 0 = normal window.
    risk_score : array-like, same length as y_true. For a static
        threshold strategy (e.g. QBER > 11%), pass the raw QBER (or other)
        signal directly and set tau_threat to that strategy's cutoff.
    tau_threat : float, the gating threshold: windows with
        risk_score >= tau_threat are gated (key distillation withheld).

    Returns
    -------
    dict with:
        attack_detected_rate : fraction of attack windows gated (= recall
            of the gating decision itself, NOT the underlying detector's
            AUROC -- this is recall AT the specific operating threshold
            actually used to gate key).
        attack_missed_rate   : 1 - attack_detected_rate.
        normal_gated_rate    : fraction of NORMAL windows wrongly gated
            (i.e. legitimate key needlessly discarded).
        n_attack_windows, n_normal_windows : counts, for context.
    """
    y_true = np.asarray(y_true).astype(int)
    gated_mask = np.asarray(risk_score) >= tau_threat

    attack_mask = y_true == 1
    normal_mask = y_true == 0
    n_attack = int(attack_mask.sum())
    n_normal = int(normal_mask.sum())

    attack_detected_rate = float(gated_mask[attack_mask].mean()) if n_attack > 0 else np.nan
    normal_gated_rate = float(gated_mask[normal_mask].mean()) if n_normal > 0 else np.nan

    return {
        "attack_detected_rate": attack_detected_rate,
        "attack_missed_rate": 1.0 - attack_detected_rate if not np.isnan(attack_detected_rate) else np.nan,
        "normal_gated_rate": normal_gated_rate,
        "n_attack_windows": n_attack,
        "n_normal_windows": n_normal,
    }


def qtrace_risk_score(anomaly_score, physics_inconsistency, temporal_deviation,
                       uncertainty, w=(0.4, 0.25, 0.2, 0.15)):
    """
    Composite Q-TRACE risk score R_t = w1*A + w2*P + w3*T + w4*U
    (Section 20). All four inputs should already be min-max normalized to
    [0, 1] over the evaluation set before calling this function.
    """
    w1, w2, w3, w4 = w
    return (w1 * np.asarray(anomaly_score) + w2 * np.asarray(physics_inconsistency)
            + w3 * np.asarray(temporal_deviation) + w4 * np.asarray(uncertainty))


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def security_weighted_objective(fn_rate, fp_rate, skrr, detection_latency,
                                 alpha=5.0, beta=1.0, gamma=2.0, delta=0.5):
    """
    J = alpha*FN + beta*FP + gamma*(1-SKRR) + delta*T_d   (Section 19).
    alpha >> beta by default: missing a real attack costs more than a
    modest false alarm.
    """
    return alpha * fn_rate + beta * fp_rate + gamma * (1 - skrr) + delta * detection_latency


class PhysicsConsistencyModel:
    """
    A properly FITTED physical-consistency residual (replaces the earlier
    stateless `physics_inconsistency_score` function, which had two bugs:
    it multiplied its QBER term by 0.0 -- a dead stub that contributed
    nothing -- and it computed its "expected" baseline from whatever data
    was being scored, including the very attack rows it was supposed to
    flag, which actively suppressed the residual signal on any split
    where attacks make up a large fraction of rows.

    This version fits robust linear relationships delta_Y ~ f(covariates)
    and delta_E ~ g(covariates) ONCE on training-split HEALTHY ("normal")
    rows only, then scores new data as a residual normalized by the
    training-time residual scale (a z-score-like measure), so scale is
    stable and comparable across splits regardless of each split's own
    attack/normal composition.

    Covariates are physical quantities that legitimately explain healthy
    variation in delta_Y/delta_E (channel loss, detector efficiency,
    source intensities), NOT quantities we want the residual to be
    sensitive to (we deliberately do NOT include QBER_signal as a
    covariate for delta_Y's model, since QBER moving independently of
    what channel/detector conditions would predict is itself part of the
    anomaly signal for several attack families).
    """

    def __init__(self):
        self.dy_model = None
        self.de_model = None
        self.dy_resid_std = 1.0
        self.de_resid_std = 1.0
        self._covariate_cols = ["channel_loss_db", "detector_efficiency", "mu_signal", "mu_decoy"]

    def _design_matrix(self, df):
        X = df[self._covariate_cols].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, df_normal):
        from sklearn.linear_model import HuberRegressor

        Xd = df_normal[self._covariate_cols].to_numpy(dtype=float)
        dY = df_normal["delta_Y"].to_numpy(dtype=float)
        dE = df_normal["delta_E"].to_numpy(dtype=float)

        self.dy_model = HuberRegressor().fit(Xd, dY)
        dy_resid = dY - self.dy_model.predict(Xd)
        self.dy_resid_std = max(np.std(dy_resid), 1e-6)

        self.de_model = HuberRegressor().fit(Xd, dE)
        de_resid = dE - self.de_model.predict(Xd)
        self.de_resid_std = max(np.std(de_resid), 1e-6)
        return self

    def score(self, df):
        Xd = df[self._covariate_cols].to_numpy(dtype=float)
        dY = df["delta_Y"].to_numpy(dtype=float)
        dE = df["delta_E"].to_numpy(dtype=float)

        dy_z = np.abs(dY - self.dy_model.predict(Xd)) / self.dy_resid_std
        de_z = np.abs(dE - self.de_model.predict(Xd)) / self.de_resid_std
        return dy_z + de_z


def physics_inconsistency_score(df, feature_cols=("delta_Y", "delta_E", "QBER_signal")):
    """
    DEPRECATED: kept only so any old imports don't hard-crash. This
    stateless version has the two bugs described in PhysicsConsistencyModel's
    docstring (dead QBER term, self-referential baseline) and MUST NOT be
    used for reported results. Use PhysicsConsistencyModel (fit once on
    training-normal data, then .score()) instead -- this is what
    models.qtrace_detector.QTraceDetector now does internally.
    """
    import warnings
    warnings.warn(
        "physics_inconsistency_score() is deprecated and methodologically "
        "unsound (dead QBER coefficient, baseline computed from the scored "
        "data itself). Use PhysicsConsistencyModel instead.",
        DeprecationWarning, stacklevel=2,
    )
    dY = df["delta_Y"].to_numpy()
    dE = df["delta_E"].to_numpy()
    pred_dY = np.median(dY)
    resid = np.abs(dY - pred_dY) + np.abs(dE)
    return resid
