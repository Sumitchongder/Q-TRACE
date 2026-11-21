"""Q-TRACE sequential detector.

For every telemetry window the detector

1. profiles the transmittance ``eta_t`` under the calibrated honest model
   (the closest honest explanation of the window);
2. forms Freeman-Tukey residuals ``r = sqrt(n) + sqrt(n+1) - sqrt(4E+1)``
   for every click cell ``(k, a, alpha, beta, outcome)`` and for the
   setting allocation (which carries the signature of biased random-number
   generators);
3. whitens them with the per-component mean and spread observed during
   commissioning, which absorbs static model mismatch the physics does not
   describe (residual afterpulsing error, intensity offsets, ...);
4. accumulates the whitened residuals ``z_t`` over several trailing
   horizons ``M`` in two channels:

   * persistent shift ``S_M = ||sum z||^2 / (M (1 + M / W_cal))``,
     sensitive to deviations of fixed sign (decoy-relation breaks, error
     excess, detector-balance changes);
   * energy ``E_M = sum ||z||^2``, sensitive to deviations whose sign
     changes in time (modulated attacks).

   Each channel is centred and scaled by its chi-square reference
   ``(x - M D) / sqrt(2 M D)`` and the alarm statistic is the maximum
   over channels and horizons.

No attack data is used anywhere: the detector is defined entirely by the
honest model and the link's own commissioning windows. Thresholds are set
on attack-free validation links for a target per-window false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qtrace.detect.honest_model import HonestParams, fit_honest, model_probs, profile_eta
from qtrace.detect.score_test import (ScoreTest, accumulated_statistic, dispersion_matrix,
                                     local_statistic)
from qtrace.physics.optics import setting_probabilities

CLICK_OUTCOMES = (1, 2, 3)

# physically meaningful residual groups used for alarm attribution
GROUP_NAMES = ("signal", "decoy", "vacuum", "sifted_error", "sifted_correct", "unsifted",
               "double_click", "settings")


def freeman_tukey(n, e):
    return np.sqrt(n) + np.sqrt(n + 1.0) - np.sqrt(4.0 * e + 1.0)


def _group_masks():
    """Boolean masks over the 96-dimensional residual vector."""
    k, a, al, be, o = np.meshgrid(np.arange(3), np.arange(2), np.arange(2), np.arange(2),
                                  np.arange(3), indexing="ij")
    o = o + 1  # click outcomes 1..3
    sifted = al == be
    err = sifted & (((a == 0) & (o == 2)) | ((a == 1) & (o == 1)))
    cor = sifted & (((a == 0) & (o == 1)) | ((a == 1) & (o == 2)))
    flat = lambda m: m.reshape(-1)
    pad = np.zeros(24, dtype=bool)
    masks = {
        "signal": flat(k == 0), "decoy": flat(k == 1), "vacuum": flat(k == 2),
        "sifted_error": flat(err), "sifted_correct": flat(cor),
        "unsifted": flat(~sifted & (o != 3)), "double_click": flat(o == 3),
    }
    masks = {g: np.concatenate([m, pad]) for g, m in masks.items()}
    masks["settings"] = np.concatenate([np.zeros(72, dtype=bool), np.ones(24, dtype=bool)])
    return masks


GROUP_MASKS = _group_masks()


@dataclass
class DetectorConfig:
    horizons: tuple = (1, 4, 16)
    shrinkage: float = 10.0
    std_floor: float = 0.5
    use_omnibus: bool = True
    use_score: bool = True
    use_settings: bool = True
    use_local: bool = True
    local_horizons: tuple = (4, 16, 32)
    local_reference: int = 48


@dataclass
class CalibratedLink:
    params: HonestParams
    bias: np.ndarray
    scale: np.ndarray
    n_cal: int
    fit_info: dict = field(default_factory=dict)
    score_bias_phys: np.ndarray | None = None
    score_bias_set: np.ndarray | None = None
    phi_phys: np.ndarray | None = None
    phi_set: np.ndarray | None = None
    z_cal: np.ndarray | None = None
    u_cal: np.ndarray | None = None
    i_cal: np.ndarray | None = None


class QTraceDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.score_test = ScoreTest()

    # ------------------------------------------------------------------ residuals
    def residuals(self, counts, params: HonestParams):
        counts = np.asarray(counts, dtype=float)
        eta, _ = profile_eta(counts, params)
        p = model_probs(params, eta)
        n_set = counts.sum(axis=-1)                                   # (W, 3,2,2,2)
        e_cells = n_set[..., None] * p
        r_cells = freeman_tukey(counts[..., 1:], e_cells[..., 1:])     # (W, 3,2,2,2,3)
        total = n_set.sum(axis=(-4, -3, -2, -1))
        p_set = setting_probabilities(np.asarray(params.p_k, float))
        e_set = total[:, None, None, None, None] * p_set
        r_set = freeman_tukey(n_set, e_set)
        w = counts.shape[0]
        return np.concatenate([r_cells.reshape(w, -1), r_set.reshape(w, -1)], axis=1), eta

    # ---------------------------------------------------------------- calibration
    def calibrate(self, counts_cal, mu_s, afterpulse_spec=0.0, p_k=(0.70, 0.20, 0.10)):
        params, info = fit_honest(counts_cal, mu_s, afterpulse_spec, p_k)
        r, _ = self.residuals(counts_cal, params)
        w = r.shape[0]
        bias = r.mean(axis=0)
        var = r.var(axis=0, ddof=1)
        k = self.cfg.shrinkage
        scale = np.sqrt((w * var + k * 1.0) / (w + k))
        scale = np.maximum(scale, self.cfg.std_floor)
        link = CalibratedLink(params=params, bias=bias, scale=scale, n_cal=w, fit_info=info)
        eta, _ = profile_eta(np.asarray(counts_cal, float), params)
        up, ip, us, is_ = self.score_test.window_scores(counts_cal, params, eta)
        link.score_bias_phys = up.mean(axis=0)
        link.score_bias_set = us.mean(axis=0)
        link.phi_phys = dispersion_matrix(up - link.score_bias_phys, ip)
        link.phi_set = dispersion_matrix(us - link.score_bias_set, is_)
        link.z_cal = (r - bias) / scale
        link.u_cal, link.i_cal = up, ip
        return link

    # ----------------------------------------------------------------- monitoring
    def whitened(self, counts, link: CalibratedLink):
        r, eta = self.residuals(counts, link.params)
        return (r - link.bias) / link.scale, eta

    def statistics(self, z, n_cal):
        """Per-window alarm statistic from whitened residuals ``z`` (W, D)."""
        w, d = z.shape
        cs = np.vstack([np.zeros((1, d)), np.cumsum(z, axis=0)])
        en = np.concatenate([[0.0], np.cumsum(np.sum(z ** 2, axis=1))])
        idx = np.arange(1, w + 1)
        channels = {}
        for m_max in self.cfg.horizons:
            m = np.minimum(idx, m_max)
            s_vec = cs[idx] - cs[idx - m]
            shift = np.sum(s_vec ** 2, axis=1) / (m * (1.0 + m / n_cal))
            energy = en[idx] - en[idx - m]
            channels[f"shift_{m_max}"] = (shift - d) / np.sqrt(2 * d)
            channels[f"energy_{m_max}"] = (energy - m * d) / np.sqrt(2 * m * d)
        stack = np.vstack(list(channels.values()))
        return stack.max(axis=0), channels

    def attribution(self, z, t, horizon=16):
        """Share of the persistent-shift statistic carried by each physical group."""
        lo = max(0, t - horizon + 1)
        s = z[lo:t + 1].sum(axis=0) ** 2
        tot = s.sum() + 1e-300
        return {g: float(s[GROUP_MASKS[g]].sum() / tot) for g in GROUP_NAMES}

    def local_shift(self, z, link: CalibratedLink):
        """Moving-reference version of the omnibus persistent-shift channel."""
        za = np.vstack([link.z_cal, z])
        n, d = za.shape
        cs = np.vstack([np.zeros((1, d)), np.cumsum(za, axis=0)])
        g = np.arange(link.n_cal + 1, n + 1)
        out = {}
        for m in self.cfg.local_horizons:
            lo = np.maximum(g - m, 0)
            mm = g - lo
            lr = np.maximum(lo - self.cfg.local_reference, 0)
            rr = np.maximum(lo - lr, 1)
            dv = (cs[g] - cs[lo]) - (mm / rr)[:, None] * (cs[lo] - cs[lr])
            s = np.sum(dv ** 2, axis=1) / (mm * (1 + mm / rr))
            out[f"local_shift_{m}"] = (s - d) / np.sqrt(2 * d)
        return out

    def score_channels(self, counts, link: CalibratedLink, eta):
        """Physics score-test channels for monitoring windows."""
        up, ip, us, is_ = self.score_test.window_scores(counts, link.params, eta)
        self.last_scores = (up, ip, us, is_)
        ch = {}
        if self.cfg.use_local and self.cfg.use_score:
            ua = np.vstack([link.u_cal, up])
            ia = np.concatenate([link.i_cal, ip])
            for m, v in local_statistic(ua, ia, link.n_cal, self.cfg.local_horizons,
                                        self.cfg.local_reference, link.phi_phys).items():
                ch[f"local_physics_{m}"] = v
        if self.cfg.use_score:
            for m, v in accumulated_statistic(up, ip, self.cfg.horizons, link.score_bias_phys,
                                              link.n_cal, link.phi_phys).items():
                ch[f"physics_{m}"] = v
        if self.cfg.use_settings:
            for m, v in accumulated_statistic(us, is_, self.cfg.horizons, link.score_bias_set,
                                              link.n_cal, link.phi_set).items():
                ch[f"settings_{m}"] = v
        return ch

    def parameter_zscores(self, link: CalibratedLink, horizon=16):
        """Per-window z-score of each physical parameter's apparent displacement.

        ``delta = I_M^{-1} U_M`` estimated over the trailing ``horizon``
        windows (moving reference), standardised by its sampling error.
        The parameter with the largest ``|z|`` at alarm time is the
        physical constraint the anomaly implicates.
        """
        up, ip = self.last_scores[0], self.last_scores[1]
        ua = np.vstack([link.u_cal, up])
        ia = np.concatenate([link.i_cal, ip])
        n, q = ua.shape
        cu = np.vstack([np.zeros((1, q)), np.cumsum(ua, axis=0)])
        ci = np.concatenate([np.zeros((1, q, q)), np.cumsum(ia, axis=0)])
        g = np.arange(link.n_cal + 1, n + 1)
        lo = np.maximum(g - horizon, 0)
        mm = g - lo
        lr = np.maximum(lo - self.cfg.local_reference, 0)
        rr = np.maximum(lo - lr, 1)
        U = (cu[g] - cu[lo]) - (mm / rr)[:, None] * (cu[lo] - cu[lr])
        I = ci[g] - ci[lo]
        I = I + 1e-9 * np.trace(I, axis1=1, axis2=2)[:, None, None] * np.eye(q)
        cov = np.linalg.inv(I)
        delta = np.einsum("wij,wj->wi", cov, U)
        sd = np.sqrt(np.einsum("wii->wi", cov) * (1 + mm / rr)[:, None]
                     * np.diag(link.phi_phys)[None, :])
        return delta / sd

    def all_channels(self, counts_monitor, link: CalibratedLink):
        z, eta = self.whitened(counts_monitor, link)
        ch = {}
        if self.cfg.use_omnibus:
            _, ch = self.statistics(z, link.n_cal)
            if self.cfg.use_local:
                ch.update(self.local_shift(z, link))
        ch.update(self.score_channels(counts_monitor, link, eta))
        return ch, z, eta

    def score_run(self, counts, n_cal, mu_s, afterpulse_spec=0.0, p_k=(0.70, 0.20, 0.10)):
        """Calibrate on the first ``n_cal`` windows and compute every channel
        for the remaining (monitoring) windows.

        Returns ``(raw_max_score, channels, z, calibrated_link, eta_hat)``.
        """
        link = self.calibrate(counts[:n_cal], mu_s, afterpulse_spec, p_k)
        ch, z, eta = self.all_channels(np.asarray(counts[n_cal:], float), link)
        raw = np.vstack(list(ch.values())).max(axis=0)
        return raw, ch, z, link, eta


class ChannelNull:
    """Per-channel standardisation fitted on attack-free training links only.

    Slow honest nuisance processes (polarisation drift, temperature-driven
    background) are temporally correlated, so long-horizon channels have a
    wider honest distribution than their chi-square reference. Each channel
    is therefore rescaled by its own honest-run median and upper-tail
    spread before the maximum is taken; this equalises the false-alarm
    contribution of every channel. Only honest data are used.
    """

    def __init__(self, tail_quantile=0.999):
        self.q = tail_quantile
        self.loc = {}
        self.scale = {}

    def fit(self, channel_dicts):
        keys = channel_dicts[0].keys()
        for k in keys:
            v = np.concatenate([c[k] for c in channel_dicts])
            med = np.median(v)
            self.loc[k] = float(med)
            self.scale[k] = float(max(np.quantile(v, self.q) - med, 1e-6))
        return self

    def transform(self, channels):
        stack = np.vstack([(channels[k] - self.loc[k]) / self.scale[k] for k in self.loc])
        return stack.max(axis=0)

    def to_dict(self):
        return {"tail_quantile": self.q, "loc": self.loc, "scale": self.scale}
