"""Count-level optical model of a decoy-state BB84 receiver.

Every transmitted pulse is labelled by a *setting*
``(k, a, alpha, beta)``: intensity class ``k`` in {signal, decoy, vacuum},
Alice's bit ``a``, Alice's basis ``alpha`` and Bob's basis ``beta``, and
produces one of four receiver *outcomes*: no click, a click on D0 only, a
click on D1 only, or a double click. The model returns the outcome
probabilities for every setting, arranged as an array whose trailing axes
are ``(K, A, ALPHA, BETA, O) = (3, 2, 2, 2, 4)``.

Two-detector threshold receivers are treated exactly. Given the
probability ``S_j`` that detector ``j`` stays silent and the joint silence
probability ``S_01``::

    P(none) = S_01
    P(D0)   = S_1 - S_01
    P(D1)   = S_0 - S_01
    P(both) = 1 - S_0 - S_1 + S_01

For a photon-number-resolved input with distribution ``P(n)`` and an
arrival probability-generating function ``G_n(z) = E[z^{n'} | n]`` (``n'``
photons reaching Bob), and per-arriving-photon detection probabilities
``pi_j`` (routing x detector efficiency)::

    S_j  = (1 - p_dj) * sum_n P(n) G_n(1 - pi_j)
    S_01 = (1 - p_d0)(1 - p_d1) * sum_n P(n) G_n(1 - pi_0 - pi_1)

For a Poissonian source and a memoryless lossy channel of transmittance
``t`` this reduces to ``exp(-mu t pi_j)`` (Ma, Qi, Zhao, Lo, PRA 72,
012326, 2005). The generating-function form is what lets attacks that act
on the photon-number distribution (photon-number splitting) be written
without approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

N_INTENSITIES = 3
N_OUTCOMES = 4
SETTING_SHAPE = (3, 2, 2, 2)
CELL_SHAPE = SETTING_SHAPE + (N_OUTCOMES,)
N_SETTINGS = int(np.prod(SETTING_SHAPE))
N_CELLS = N_SETTINGS * N_OUTCOMES

INTENSITY_NAMES = ("signal", "decoy", "vacuum")
OUTCOME_NAMES = ("none", "D0", "D1", "double")
BASIS_NAMES = ("Z", "X")


def routing_weights(e_z, e_x):
    """Fraction of light routed to each detector for every ``(a, alpha, beta)``.

    Matching bases send the photon to the detector encoding Alice's bit
    with probability ``1 - e_beta`` (``e_beta`` is the optical
    misalignment of Bob's basis); mismatched bases split evenly.

    Returns an array of shape ``batch + (2, 2, 2, 2)`` indexed
    ``[a, alpha, beta, detector]``.
    """
    e_z = np.asarray(e_z, dtype=float)
    e_x = np.asarray(e_x, dtype=float)
    batch = np.broadcast(e_z, e_x).shape
    w = np.full(batch + (2, 2, 2, 2), 0.5)
    for b, e in enumerate((e_z, e_x)):
        for a in (0, 1):
            w[..., a, b, b, a] = 1.0 - e
            w[..., a, b, b, 1 - a] = e
    return w


def silence_to_outcomes(s0, s1, s01):
    """Convert silence probabilities to the four outcome probabilities."""
    p_none = s01
    p_d0 = s1 - s01
    p_d1 = s0 - s01
    p_dbl = 1.0 - s0 - s1 + s01
    out = np.stack([p_none, p_d0, p_d1, p_dbl], axis=-1)
    return np.clip(out, 0.0, 1.0)


def outcome_probs_poisson(mu, eta, e_z, e_x, pd0, pd1, kappa):
    """Honest outcome probabilities for a phase-randomised coherent source.

    Parameters
    ----------
    mu : array (..., 3)
        Mean photon number of each intensity class at Alice's output.
    eta : array (...)
        Overall transmittance to the reference detector D0 (channel x
        receiver optics x D0 efficiency).
    e_z, e_x : array (...)
        Optical misalignment per basis.
    pd0, pd1 : array (...)
        Per-gate background (dark-count) probability of each detector.
    kappa : array (...)
        Efficiency of D1 relative to D0.

    Returns
    -------
    array (..., 3, 2, 2, 2, 4)
    """
    mu = np.asarray(mu, dtype=float)
    eta = np.asarray(eta, dtype=float)[..., None]
    w = routing_weights(e_z, e_x)                       # (..., a, al, be, det)
    kappa = np.asarray(kappa, dtype=float)
    d = np.stack([np.ones_like(kappa), kappa], axis=-1)  # (..., det)
    pi = w * d[..., None, None, None, :]                 # (..., a, al, be, det)
    lam = (mu * eta)[..., :, None, None, None, None] * pi[..., None, :, :, :, :]
    pd0 = np.asarray(pd0, dtype=float)[..., None, None, None, None]
    pd1 = np.asarray(pd1, dtype=float)[..., None, None, None, None]
    s0 = (1.0 - pd0) * np.exp(-lam[..., 0])
    s1 = (1.0 - pd1) * np.exp(-lam[..., 1])
    s01 = s0 * s1
    return silence_to_outcomes(s0, s1, s01)


def poisson_pmf(mu, n_max):
    """Poisson pmf over ``n = 0..n_max`` broadcast over ``mu``; shape ``mu.shape + (n_max+1,)``."""
    mu = np.asarray(mu, dtype=float)[..., None]
    n = np.arange(n_max + 1)
    with np.errstate(divide="ignore"):
        logp = n * np.log(np.where(mu > 0, mu, 1.0)) - mu - gammaln(n + 1)
    p = np.exp(logp)
    p = np.where(mu > 0, p, (n == 0).astype(float))
    return p


def outcome_probs_photon_resolved(mu, arrival_gf, eta_b, e_z, e_x, pd0, pd1, kappa,
                                  n_max=14):
    """Outcome probabilities for an arbitrary photon-number-dependent channel.

    ``arrival_gf(n, z)`` must return ``G_n(z)`` broadcast over a batch: it
    receives ``n`` with shape ``(n_max+1,)`` and ``z`` with shape
    ``batch + (a, al, be)`` and returns ``batch + (a, al, be, n_max+1)``.
    ``eta_b`` is Bob's receiver efficiency (optics x D0 efficiency); the
    channel transmittance is carried by ``arrival_gf``.
    """
    mu = np.asarray(mu, dtype=float)
    pn = poisson_pmf(mu, n_max)                          # (..., 3, n)
    w = routing_weights(e_z, e_x)
    kappa = np.asarray(kappa, dtype=float)
    d = np.stack([np.ones_like(kappa), kappa], axis=-1)
    eta_b = np.asarray(eta_b, dtype=float)
    pi = w * d[..., None, None, None, :] * eta_b[..., None, None, None, None]
    n = np.arange(n_max + 1)
    g0 = arrival_gf(n, 1.0 - pi[..., 0])                 # (..., a, al, be, n)
    g1 = arrival_gf(n, 1.0 - pi[..., 1])
    g01 = arrival_gf(n, 1.0 - pi[..., 0] - pi[..., 1])
    pn_b = pn[..., :, None, None, None, :]               # (..., 3, 1,1,1, n)
    e0 = np.sum(pn_b * g0[..., None, :, :, :, :], axis=-1)
    e1 = np.sum(pn_b * g1[..., None, :, :, :, :], axis=-1)
    e01 = np.sum(pn_b * g01[..., None, :, :, :, :], axis=-1)
    pd0 = np.asarray(pd0, dtype=float)[..., None, None, None, None]
    pd1 = np.asarray(pd1, dtype=float)[..., None, None, None, None]
    s0 = (1.0 - pd0) * e0
    s1 = (1.0 - pd1) * e1
    s01 = (1.0 - pd0) * (1.0 - pd1) * e01
    return silence_to_outcomes(s0, s1, s01)


def lossy_channel_gf(t):
    """Arrival generating function of a memoryless channel with transmittance ``t``."""
    t = np.asarray(t, dtype=float)

    def gf(n, z):
        base = 1.0 - t[..., None, None, None] + t[..., None, None, None] * z
        return base[..., None] ** n
    return gf


def setting_probabilities(p_k, p_a1=0.5, p_alpha_z=0.5, p_beta_z=0.5):
    """Joint probability of each setting, shape ``batch + (3, 2, 2, 2)``."""
    p_k = np.asarray(p_k, dtype=float)
    p_a1 = np.asarray(p_a1, dtype=float)
    pa = np.stack([1.0 - p_a1, p_a1], axis=-1)
    p_alpha_z = np.asarray(p_alpha_z, dtype=float)
    pal = np.stack([p_alpha_z, 1.0 - p_alpha_z], axis=-1)
    p_beta_z = np.asarray(p_beta_z, dtype=float)
    pbe = np.stack([p_beta_z, 1.0 - p_beta_z], axis=-1)
    return (p_k[..., :, None, None, None] * pa[..., None, :, None, None]
            * pal[..., None, None, :, None] * pbe[..., None, None, None, :])
