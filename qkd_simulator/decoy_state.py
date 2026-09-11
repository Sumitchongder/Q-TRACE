"""
Q-TRACE :: qkd_simulator.decoy_state
======================================
Vacuum + weak decoy-state estimation of the single-photon yield Y1 and
single-photon error rate e1, following Ma, Qi, Zhao, Lo (PRA 72, 012326
(2005)). Poisson photon-number statistics are assumed for the source.

These estimators are what let Q-TRACE compute physically meaningful
observables (Y1_lb, e1_ub) that feed both the physical-consistency loss
and the simplified secure-key-rate calculation.
"""

import numpy as np
from scipy.stats import poisson


def _poisson_weight(n, mu):
    return poisson.pmf(n, mu)


def estimate_Y1_lower_bound(Y_signal, Y_decoy, Y_vacuum, mu_signal, mu_decoy, n_max=6):
    """
    Standard 2-intensity (signal + one decoy, vacuum as second decoy)
    lower bound on the single-photon yield:

        Y1_lb = (mu_s / (mu_s*mu_d - mu_d^2)) *
                [ Y_decoy*exp(mu_d) - Y_signal*exp(mu_s)*(mu_d/mu_s)^2
                  - (mu_s^2 - mu_d^2)/mu_s^2 * Y_vacuum ]

    Clipped to be non-negative; this is the practical form used widely in
    decoy-state QKD implementation papers.
    """
    mu_s, mu_d = mu_signal, mu_decoy
    denom = mu_s * mu_d - mu_d ** 2
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)

    term1 = Y_decoy * np.exp(mu_d)
    term2 = Y_signal * np.exp(mu_s) * (mu_d / mu_s) ** 2
    term3 = ((mu_s ** 2 - mu_d ** 2) / mu_s ** 2) * Y_vacuum

    Y1_lb = (mu_s / denom) * (term1 - term2 - term3)
    return np.clip(Y1_lb, 0.0, 1.0)


def estimate_e1_upper_bound(E_signal, E_decoy, Y_signal, Y_decoy, Y1_lb,
                             mu_signal, mu_decoy):
    """
    Upper bound on the single-photon error rate e1, following the standard
    decoy-state relation:

        e1 * Y1 <= [ E_decoy*Y_decoy*exp(mu_d) - E_signal*Y_signal*exp(mu_s) ] /
                    (mu_d - mu_s)   (sign handled via absolute value)

    Divided by Y1_lb (clipped away from zero) to obtain e1_ub.
    """
    mu_s, mu_d = mu_signal, mu_decoy
    denom = (mu_d - mu_s)
    denom = np.where(np.abs(denom) < 1e-9, -1e-9, denom)

    numer = E_decoy * Y_decoy * np.exp(mu_d) - E_signal * Y_signal * np.exp(mu_s)
    e1Y1 = numer / denom
    e1Y1 = np.abs(e1Y1)

    Y1_safe = np.clip(Y1_lb, 1e-6, None)
    e1_ub = np.clip(e1Y1 / Y1_safe, 0.0, 0.5)
    return e1_ub


def q1_lower_bound(Y1_lb, mu_signal):
    """Single-photon detection probability lower bound Q1_lb = Y1_lb*mu*exp(-mu)."""
    return Y1_lb * mu_signal * np.exp(-mu_signal)
