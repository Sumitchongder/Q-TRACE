"""The honest-device manifold and its calibration.

The honest model of a link is the family of outcome distributions
``p(psi, eta)`` produced by an attack-free decoy-state BB84 link with

* static device parameters ``psi = (e_Z, e_X, p_d0, p_d1, kappa, rho)``
  (misalignment per basis, per-detector background, relative detector
  efficiency, decoy-to-signal intensity ratio), fitted once per link on
  commissioning windows; and
* a free per-window transmittance ``eta_t`` (nuisance), profiled out in
  every window.

Treating ``eta_t`` as a nuisance encodes the physical fact that loss is not
a security-relevant deviation in decoy-state QKD: arbitrary channel loss
is already priced into the decoy-state key-rate bound. Changes that cannot
be explained by *any* transmittance (broken decoy relations, altered
error structure, detector-balance anomalies, biased settings) are what
remains to be detected.

The signal intensity ``mu_s`` is taken from the transmitter's monitor; it
is degenerate with ``eta_t`` and needs no estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from qtrace.physics.optics import outcome_probs_poisson, setting_probabilities

_EPS = 1e-300


@dataclass
class HonestParams:
    e_z: float
    e_x: float
    pd0: float
    pd1: float
    kappa: float
    rho: float
    mu_s: float
    afterpulse: float = 0.0
    p_k: tuple = (0.70, 0.20, 0.10)

    @property
    def mu(self):
        return np.array([self.mu_s, self.rho * self.mu_s, 0.0])

    def to_vector(self):
        return np.array([logit(self.e_z), logit(self.e_x), np.log(self.pd0), np.log(self.pd1),
                         np.log(self.kappa), np.log(self.rho)])

    @classmethod
    def from_vector(cls, v, mu_s, afterpulse=0.0, p_k=(0.70, 0.20, 0.10)):
        return cls(e_z=float(expit(v[0])), e_x=float(expit(v[1])), pd0=float(np.exp(v[2])),
                   pd1=float(np.exp(v[3])), kappa=float(np.exp(v[4])), rho=float(np.exp(v[5])),
                   mu_s=mu_s, afterpulse=afterpulse, p_k=tuple(p_k))


def model_probs(params: HonestParams, eta):
    """Honest outcome probabilities, including datasheet afterpulsing.

    Afterpulsing adds a background to each detector proportional to its
    mean click probability; ``params.afterpulse`` is the manufacturer's
    specification, not a fitted quantity.
    """
    eta = np.asarray(eta, dtype=float)
    mu = np.broadcast_to(params.mu, eta.shape + (3,))
    p = outcome_probs_poisson(mu, eta, params.e_z, params.e_x, params.pd0, params.pd1,
                              params.kappa)
    if params.afterpulse > 0:
        from qtrace.twin.simulator import add_background, detector_click_rates
        w = setting_probabilities(np.asarray(params.p_k, float))
        r = detector_click_rates(p, w)
        p = add_background(p, params.afterpulse * r[..., 0], params.afterpulse * r[..., 1])
    return p


def _loglik_windows(counts, probs):
    return np.sum(counts * np.log(np.maximum(probs, _EPS)), axis=(-5, -4, -3, -2, -1))


def initial_eta(counts, params: HonestParams):
    """Moment estimate of the transmittance from the signal click rate."""
    c = counts[..., 0, :, :, :, :]
    n = c.sum(axis=(-4, -3, -2, -1))
    clicks = c[..., 1:].sum(axis=(-4, -3, -2, -1))
    g = np.clip(clicks / np.maximum(n, 1), 1e-12, 0.99)
    return -np.log1p(-g) / (params.mu_s * 0.5 * (1 + params.kappa))


def profile_eta(counts, params: HonestParams, n_iter=12, h=1e-4):
    """Per-window MLE of ``eta`` under ``params`` (vectorised Newton on log eta).

    Returns ``(eta_hat, loglik_at_eta_hat)``.
    """
    x = np.log(initial_eta(counts, params))
    for _ in range(n_iter):
        l0 = _loglik_windows(counts, model_probs(params, np.exp(x)))
        lp = _loglik_windows(counts, model_probs(params, np.exp(x + h)))
        lm = _loglik_windows(counts, model_probs(params, np.exp(x - h)))
        g = (lp - lm) / (2 * h)
        H = (lp - 2 * l0 + lm) / h ** 2
        step = np.where(H < 0, -g / np.minimum(H, -1e-12), np.sign(g) * 0.05)
        x = x + np.clip(step, -0.5, 0.5)
    eta = np.exp(x)
    return eta, _loglik_windows(counts, model_probs(params, eta))


def moment_estimates(counts, mu_s, afterpulse=0.0, p_k=(0.70, 0.20, 0.10)):
    """Closed-form starting point for the calibration fit."""
    c = np.asarray(counts, dtype=float).sum(axis=0)
    vac = c[2]
    n_vac = vac.sum(axis=-1).sum()
    rate = c[..., 1:].sum() / max(c.sum(), 1)
    pd0 = max((vac[..., 1] + vac[..., 3]).sum() / max(n_vac, 1) - afterpulse * rate / 2, 1e-8)
    pd1 = max((vac[..., 2] + vac[..., 3]).sum() / max(n_vac, 1) - afterpulse * rate / 2, 1e-8)
    errs = []
    for b in (0, 1):
        cell = c[0, :, b, b, :]
        clicks = cell[..., 1:].sum()
        err = cell[0, 2] + cell[1, 1] + 0.5 * (cell[0, 3] + cell[1, 3])
        errs.append(np.clip(err / max(clicks, 1), 1e-4, 0.3))
    mism = c[0][:, [0, 1], [1, 0], :].sum(axis=(0, 1))
    kappa = np.clip((mism[2] + mism[3]) / max(mism[1] + mism[3], 1), 0.3, 3.0)
    g = c[:2, ..., 1:].sum(axis=(1, 2, 3, 4)) / np.maximum(c[:2].sum(axis=(1, 2, 3, 4)), 1)
    rho = np.clip(np.log1p(-g[1]) / np.log1p(-g[0]), 0.02, 0.9)
    return HonestParams(e_z=errs[0], e_x=errs[1], pd0=pd0, pd1=pd1, kappa=kappa, rho=rho,
                        mu_s=mu_s, afterpulse=afterpulse, p_k=tuple(p_k))


def fit_honest(counts, mu_s, afterpulse=0.0, p_k=(0.70, 0.20, 0.10), maxiter=200):
    """Maximum-likelihood calibration of ``psi`` on commissioning windows.

    The per-window transmittance is profiled inside every objective
    evaluation, so the optimisation runs over six parameters only.
    """
    counts = np.asarray(counts, dtype=float)
    p0 = moment_estimates(counts, mu_s, afterpulse, p_k)
    scale = counts.sum()

    def objective(v):
        p = HonestParams.from_vector(v, mu_s, afterpulse, p_k)
        _, ll = profile_eta(counts, p, n_iter=8)
        return -ll.sum() / scale

    res = minimize(objective, p0.to_vector(), method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-10, "eps": 1e-6})
    params = HonestParams.from_vector(res.x, mu_s, afterpulse, p_k)
    return params, {"success": bool(res.success), "nit": int(res.nit),
                    "neg_loglik_per_pulse": float(res.fun)}
