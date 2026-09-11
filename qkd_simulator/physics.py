"""
Q-TRACE :: qkd_simulator.physics
==================================
Closed-form decoy-state BB84 physics used to generate expected yields and
QBER for a given physical parameter vector theta. These are the standard
GLLP / Lo-Ma-Chen decoy-state relations used throughout the practical-QKD
literature (Lo, Ma, Chen, PRL 2005; Ma et al., PRA 2005).

Notation
--------
For a weak coherent pulse of mean photon number mu, transmitted through a
channel with transmittance eta_ch and detected with efficiency eta_d, and
background/dark-count probability Y0, the click probability (yield) is:

    Y_mu = 1 - (1 - Y0) * exp(-mu * eta_ch * eta_d)

The overall detection efficiency eta = eta_ch * eta_d, where

    eta_ch = 10^(-alpha_db_per_km * L / 10)

The observed QBER for intensity mu is modelled as:

    E_mu = [ Y0 * 0.5 + e_d * (Y_mu - Y0) ] / Y_mu

where e_d is the intrinsic (optical) error rate of the detection apparatus,
and the factor 0.5 reflects that dark counts contribute random outcomes.

This module returns *expected* (mean) values; Poissonian/binomial sampling
noise is added downstream in generator.py to emulate a finite number of
detection events per telemetry window.
"""

import numpy as np


def channel_transmittance(alpha_db_per_km: np.ndarray, length_km: np.ndarray) -> np.ndarray:
    """Fiber channel transmittance eta_ch = 10^(-alpha*L/10)."""
    return 10.0 ** (-(alpha_db_per_km * length_km) / 10.0)


def yield_mu(mu: np.ndarray, eta_ch: np.ndarray, eta_d: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """Overall detection yield for intensity mu (Eq. GLLP)."""
    eta = eta_ch * eta_d
    return 1.0 - (1.0 - Y0) * np.exp(-mu * eta)


def qber_mu(mu: np.ndarray, eta_ch: np.ndarray, eta_d: np.ndarray,
            Y0: np.ndarray, e_d: np.ndarray) -> np.ndarray:
    """Expected QBER for intensity mu."""
    Y = yield_mu(mu, eta_ch, eta_d, Y0)
    Y = np.clip(Y, 1e-12, None)
    err = Y0 * 0.5 + e_d * (Y - Y0)
    E = err / Y
    return np.clip(E, 0.0, 0.5)


def apply_timing_jitter_penalty(qber: np.ndarray, sigma_t_ps: np.ndarray,
                                 gate_width_ps: float = 500.0) -> np.ndarray:
    """
    Timing jitter widens the effective detection gate mismatch, adding a
    small multiplicative penalty to QBER and yield loss. Modelled as a
    Gaussian-overlap degradation factor.
    """
    penalty = 1.0 - np.exp(-0.5 * (sigma_t_ps / gate_width_ps) ** 2)
    return np.clip(qber + 0.05 * penalty, 0.0, 0.5)


def apply_phase_noise_penalty(qber: np.ndarray, sigma_phi_rad: np.ndarray) -> np.ndarray:
    """
    Phase noise in phase-encoded BB84 directly reduces interference
    visibility V = exp(-sigma_phi^2 / 2), which maps to an added QBER term
    (1 - V) / 2.
    """
    visibility = np.exp(-0.5 * sigma_phi_rad ** 2)
    added = (1.0 - visibility) / 2.0
    return np.clip(qber + added, 0.0, 0.5)


def secure_key_rate_asymptotic(Q_signal: np.ndarray, E_signal: np.ndarray,
                                Y1_lb: np.ndarray, e1_ub: np.ndarray,
                                q: float = 0.5, f_ec: float = 1.16) -> np.ndarray:
    """
    Simplified asymptotic GLLP secure key rate lower bound per signal pulse:

        R >= q * { -Q_mu * f_ec * h2(E_mu) + Q1_lb * [1 - h2(e1_ub)] }

    where Q1_lb is (approximately) the single-photon detection probability
    lower bound Y1_lb * mu * exp(-mu), h2 is binary entropy. This is a
    pedagogical simplification of Lo-Ma-Chen; sufficient for relative
    comparisons of secret-key retention under attack, not for certified
    security proofs (see finite_key.py for the finite-key correction used
    in the SKRR experiments).
    """
    def h2(x):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    Q1_lb = np.clip(Y1_lb, 0.0, None)
    term1 = Q_signal * f_ec * h2(E_signal)
    term2 = Q1_lb * (1.0 - h2(e1_ub))
    R = q * (term2 - term1)
    return R
