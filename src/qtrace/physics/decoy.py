"""Decoy-state parameter estimation and asymptotic key rate.

Implements the vacuum + weak-decoy bounds of Ma, Qi, Zhao and Lo
(PRA 72, 012326, 2005) and the GLLP key rate with the single-photon gain
``Q_1 = Y_1 mu exp(-mu)``. Key is distilled from the Z basis; the
single-photon phase error is bounded with the X-basis decoy statistics
(efficient BB84 with basis-dependent sifting, Lo, Chau, Ardehali 2005).

Also provides :func:`summarise_counts`, which maps a raw outcome table
into the scalar telemetry a deployed QKD node exports (gains, QBERs,
double-click rate, detector balance, sifting statistics). These scalars
feed the classical baselines; the Q-TRACE detector itself works on the
full table.
"""

from __future__ import annotations

import numpy as np

SIGNAL, DECOY, VACUUM = 0, 1, 2
Z, X = 0, 1


def h2(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def _clicks_and_errors(counts):
    """Sifted clicks and errors per ``(k, basis)`` from an outcome table.

    Double clicks are assigned a uniformly random bit (squashing), so they
    contribute half an error in expectation.
    """
    c = np.asarray(counts, dtype=float)
    n_pulses = np.zeros(c.shape[:-5] + (3, 2))
    clicks = np.zeros_like(n_pulses)
    errors = np.zeros_like(n_pulses)
    for b in (Z, X):
        for a in (0, 1):
            cell = c[..., :, a, b, b, :]
            n_pulses[..., b] += cell.sum(axis=-1)
            clicks[..., b] += cell[..., 1] + cell[..., 2] + cell[..., 3]
            wrong = 2 if a == 0 else 1
            errors[..., b] += cell[..., wrong] + 0.5 * cell[..., 3]
    return n_pulses, clicks, errors


def gains_and_qbers(counts):
    n, clicks, errors = _clicks_and_errors(counts)
    gain = clicks / np.maximum(n, 1.0)
    qber = errors / np.maximum(clicks, 1.0)
    return gain, qber


def decoy_bounds(q_mu, q_nu, y0, e_nu, mu, nu, e0=0.5):
    """Lower bound on Y1 and upper bound on e1 (vacuum + one weak decoy)."""
    q_mu, q_nu, y0, e_nu = map(np.asarray, (q_mu, q_nu, y0, e_nu))
    mu, nu = np.asarray(mu, float), np.asarray(nu, float)
    y1_l = (mu / (mu * nu - nu ** 2)) * (
        q_nu * np.exp(nu) - q_mu * np.exp(mu) * nu ** 2 / mu ** 2
        - (mu ** 2 - nu ** 2) / mu ** 2 * y0)
    y1_l = np.clip(y1_l, 0.0, 1.0)
    e1_u = (e_nu * q_nu * np.exp(nu) - e0 * y0) / np.maximum(y1_l * nu, 1e-300)
    e1_u = np.where(y1_l > 0, np.clip(e1_u, 0.0, 0.5), 0.5)
    return y1_l, e1_u


def key_rate(counts, mu, nu, f_ec=1.16):
    """Asymptotic secret-key rate per Z-basis signal pulse (bits/pulse).

    Returns ``(R, y1_l, e1_u)``; ``R`` is clipped at zero.
    """
    gain, qber = gains_and_qbers(counts)
    q_mu = gain[..., SIGNAL, Z]
    e_mu = qber[..., SIGNAL, Z]
    q_nu = gain[..., DECOY, X]
    e_nu = qber[..., DECOY, X]
    y0 = 0.5 * (gain[..., VACUUM, Z] + gain[..., VACUUM, X])
    y1_l, e1_u = decoy_bounds(q_mu, q_nu, y0, e_nu, mu, nu)
    q1 = y1_l * mu * np.exp(-mu)
    rate = q1 * (1.0 - h2(e1_u)) - f_ec * q_mu * h2(e_mu)
    return np.clip(rate, 0.0, None), y1_l, e1_u


def summarise_counts(counts, mu, nu):
    """Scalar telemetry per window (dict of 1-D arrays)."""
    c = np.asarray(counts, dtype=float)
    gain, qber = gains_and_qbers(c)
    rate, y1_l, e1_u = key_rate(c, mu, nu)
    total = c.sum(axis=(-5, -4, -3, -2, -1))
    clicks_any = c[..., 1:].sum(axis=(-5, -4, -3, -2, -1))
    dbl = c[..., 3].sum(axis=(-4, -3, -2, -1))
    single = c[..., 1:3].sum(axis=(-4, -3, -2, -1))
    d1_frac_mismatch = (c[..., :2, :, [0, 1], [1, 0], 2].sum(axis=(-3, -2, -1))
                        / np.maximum(c[..., :2, :, [0, 1], [1, 0], 1:3].sum(axis=(-4, -3, -2, -1)), 1.0))
    settings = c.sum(axis=-1)
    p_a1 = settings[..., :, 1, :, :].sum(axis=(-3, -2, -1)) / np.maximum(total, 1.0)
    p_alpha_z = settings[..., :, :, 0, :].sum(axis=(-3, -2, -1)) / np.maximum(total, 1.0)
    out = {
        "gain_s_Z": gain[..., SIGNAL, Z], "gain_s_X": gain[..., SIGNAL, X],
        "gain_d_Z": gain[..., DECOY, Z], "gain_d_X": gain[..., DECOY, X],
        "gain_v": 0.5 * (gain[..., VACUUM, Z] + gain[..., VACUUM, X]),
        "qber_s_Z": qber[..., SIGNAL, Z], "qber_s_X": qber[..., SIGNAL, X],
        "qber_d_Z": qber[..., DECOY, Z], "qber_d_X": qber[..., DECOY, X],
        "gain_ratio_d_s": gain[..., DECOY, Z] / np.maximum(gain[..., SIGNAL, Z], 1e-12),
        "double_rate_s": dbl[..., SIGNAL] / np.maximum(single[..., SIGNAL] + dbl[..., SIGNAL], 1.0),
        "double_rate_d": dbl[..., DECOY] / np.maximum(single[..., DECOY] + dbl[..., DECOY], 1.0),
        "d1_fraction_unsifted": d1_frac_mismatch,
        "click_rate": clicks_any / np.maximum(total, 1.0),
        "p_a1": p_a1, "p_alpha_z": p_alpha_z,
        "y1_l": y1_l, "e1_u": e1_u, "key_rate": rate,
    }
    return out


TELEMETRY_FEATURES = (
    "gain_s_Z", "gain_s_X", "gain_d_Z", "gain_d_X", "gain_v",
    "qber_s_Z", "qber_s_X", "qber_d_Z", "qber_d_X", "gain_ratio_d_s",
    "double_rate_s", "double_rate_d", "d1_fraction_unsifted", "click_rate",
    "p_a1", "p_alpha_z", "y1_l", "e1_u", "key_rate",
)
