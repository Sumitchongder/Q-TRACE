"""
Q-TRACE :: qkd_simulator.bb84
===============================
The QKD Digital Twin: a decoy-state BB84 simulator that produces
time-resolved telemetry windows under healthy operation, and which can be
composed with pluggable attack modules (see attacks/base.py) to inject
implementation-level eavesdropping effects.

Design
------
1. Sample a physical parameter vector theta from DecoyBB84Config ranges
   (one draw per simulated "run" / trajectory).
2. For each time window t = 0..T-1:
     a. Compute instantaneous channel transmittance & detector efficiency
        (including slow drift).
     b. Compute expected signal/decoy yields and QBERs from closed-form
        physics.
     c. If an Attack object is supplied, let it perturb the *expected*
        yields/QBERs (attacks act on the physical channel, not the
        finished telemetry, to keep them physically grounded).
     d. Draw finite-statistics (binomial) samples for a finite number of
        pulses per window, to emulate realistic shot noise.
     e. Estimate decoy-state Y1_lb / e1_ub, and package the full
        telemetry feature vector.

This produces one row of telemetry per time window; a full "run" is a
trajectory of T windows sharing the same theta (and same attack, if any).
"""

from dataclasses import asdict
import numpy as np
import pandas as pd

from .config import DecoyBB84Config
from .channel import FiberChannel
from .detector import Detector
from . import physics
from . import decoy_state as ds


class DecoyBB84Simulator:
    def __init__(self, config: DecoyBB84Config = None, rng=None):
        self.cfg = config if config is not None else DecoyBB84Config()
        self.rng = rng if rng is not None else np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    def sample_theta(self):
        """Draw one physical parameter vector theta for a simulated run."""
        c = self.cfg
        theta = dict(
            mu_signal=c.mu_signal.sample(self.rng),
            mu_decoy=c.mu_decoy.sample(self.rng),
            mu_vacuum=c.mu_vacuum,
            alpha_db_per_km=c.fiber_atten_db_per_km.sample(self.rng),
            length_km=c.link_length_km.sample(self.rng),
            eta_d=c.detector_efficiency.sample(self.rng),
            Y0=c.dark_count_rate.sample(self.rng),
            e_d=c.intrinsic_error_rate.sample(self.rng),
            sigma_t=c.timing_jitter_ps.sample(self.rng),
            sigma_phi=c.phase_noise_rad.sample(self.rng),
            temp_drift=c.temperature_drift_amp.sample(self.rng),
            vib_amp=c.vibration_amp.sample(self.rng),
        )
        return theta

    # ------------------------------------------------------------------
    def simulate_run(self, n_windows: int, theta: dict = None, attack=None,
                      run_id: str = "run0", attack_family: str = "normal",
                      zero_day: bool = False, physical_regime: str = "nominal"):
        """
        Simulate one trajectory of n_windows telemetry samples.

        Parameters
        ----------
        theta : dict, optional
            Physical parameter vector. If None, sampled fresh.
        attack : attacks.base.Attack instance or None
            Pluggable attack object. If None, healthy (Level 0) operation.
        run_id, attack_family, zero_day, physical_regime :
            Metadata columns carried through to the output DataFrame,
            used later for run-level / attack-level dataset splitting.

        Returns
        -------
        pandas.DataFrame with one row per time window.
        """
        if theta is None:
            theta = self.sample_theta()

        channel = FiberChannel(theta["alpha_db_per_km"], theta["length_km"],
                                theta["temp_drift"], theta["vib_amp"], rng=self.rng)
        detector = Detector(theta["eta_d"], theta["Y0"], theta["e_d"],
                             theta["sigma_t"], drift_amp=0.1 * theta["temp_drift"],
                             rng=self.rng)

        t_index = np.arange(n_windows)
        eta_ch = channel.transmittance(t_index)
        eta_d = detector.efficiency(t_index)

        mu_s = np.full(n_windows, theta["mu_signal"])
        mu_d = np.full(n_windows, theta["mu_decoy"])
        mu_v = np.full(n_windows, theta["mu_vacuum"])
        Y0 = np.full(n_windows, theta["Y0"])
        e_d = np.full(n_windows, theta["e_d"])
        sigma_t = np.full(n_windows, theta["sigma_t"])
        sigma_phi = np.full(n_windows, theta["sigma_phi"])

        # ---- Expected (noiseless) physics ----
        Y_signal = physics.yield_mu(mu_s, eta_ch, eta_d, Y0)
        Y_decoy = physics.yield_mu(mu_d, eta_ch, eta_d, Y0)
        Y_vacuum = physics.yield_mu(mu_v, eta_ch, eta_d, Y0)

        E_signal = physics.qber_mu(mu_s, eta_ch, eta_d, Y0, e_d)
        E_decoy = physics.qber_mu(mu_d, eta_ch, eta_d, Y0, e_d)

        E_signal = physics.apply_timing_jitter_penalty(E_signal, sigma_t)
        E_decoy = physics.apply_timing_jitter_penalty(E_decoy, sigma_t)
        E_signal = physics.apply_phase_noise_penalty(E_signal, sigma_phi)
        E_decoy = physics.apply_phase_noise_penalty(E_decoy, sigma_phi)

        state = dict(
            Y_signal=Y_signal, Y_decoy=Y_decoy, Y_vacuum=Y_vacuum,
            E_signal=E_signal, E_decoy=E_decoy,
            eta_ch=eta_ch, eta_d=eta_d, theta=theta, t_index=t_index,
        )

        # ---- Attack perturbation (acts on expected physics) ----
        if attack is not None:
            state = attack.apply(state, self.rng)

        Y_signal, Y_decoy, Y_vacuum = state["Y_signal"], state["Y_decoy"], state["Y_vacuum"]
        E_signal, E_decoy = state["E_signal"], state["E_decoy"]

        # ---- Finite-statistics sampling noise (binomial shot noise) ----
        n_pulses = self.cfg.pulses_per_window
        n_signal_pulses = np.maximum(1, (n_pulses * self.cfg.p_signal)).astype(int)
        n_decoy_pulses = np.maximum(1, (n_pulses * self.cfg.p_decoy)).astype(int)

        Y_signal_obs = self.rng.binomial(n_signal_pulses, np.clip(Y_signal, 0, 1)) / n_signal_pulses
        Y_decoy_obs = self.rng.binomial(n_decoy_pulses, np.clip(Y_decoy, 0, 1)) / n_decoy_pulses

        # error counts are a fraction of detected (clicked) events
        clicks_signal = np.maximum(1, (Y_signal_obs * n_signal_pulses)).astype(int)
        clicks_decoy = np.maximum(1, (Y_decoy_obs * n_decoy_pulses)).astype(int)
        E_signal_obs = self.rng.binomial(clicks_signal, np.clip(E_signal, 0, 1)) / clicks_signal
        E_decoy_obs = self.rng.binomial(clicks_decoy, np.clip(E_decoy, 0, 1)) / clicks_decoy

        # ---- Decoy-state estimation ----
        Y1_lb = ds.estimate_Y1_lower_bound(Y_signal_obs, Y_decoy_obs, Y_vacuum, mu_s, mu_d)
        e1_ub = ds.estimate_e1_upper_bound(E_signal_obs, E_decoy_obs, Y_signal_obs,
                                            Y_decoy_obs, Y1_lb, mu_s, mu_d)
        q1_lb = ds.q1_lower_bound(Y1_lb, mu_s)
        skr_est = physics.secure_key_rate_asymptotic(Y_signal_obs, E_signal_obs, Y1_lb, e1_ub)

        # ---- Derived / auxiliary features ----
        delta_Y = Y_signal_obs - Y_decoy_obs
        delta_E = E_signal_obs - E_decoy_obs
        detection_rate_signal = Y_signal_obs
        detection_rate_decoy = Y_decoy_obs

        p = np.clip(E_signal_obs, 1e-9, 1 - 1e-9)
        binary_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        channel_loss_db = theta["alpha_db_per_km"] * theta["length_km"]

        df = pd.DataFrame({
            "t": t_index,
            "run_id": run_id,
            "QBER_signal": E_signal_obs,
            "QBER_decoy": E_decoy_obs,
            "Y_signal": Y_signal_obs,
            "Y_decoy": Y_decoy_obs,
            "delta_Y": delta_Y,
            "delta_E": delta_E,
            "detection_rate_signal": detection_rate_signal,
            "detection_rate_decoy": detection_rate_decoy,
            "entropy": binary_entropy,
            "channel_loss_db": channel_loss_db,
            "detector_efficiency": eta_d,
            "dark_count_rate": Y0,
            "timing_jitter_ps": sigma_t,
            "phase_noise_rad": sigma_phi,
            "Y1_lb": Y1_lb,
            "e1_ub": e1_ub,
            "Q1_lb": q1_lb,
            "skr_est": skr_est,
            "mu_signal": mu_s,
            "mu_decoy": mu_d,
            "attack_family": attack_family,
            "zero_day": zero_day,
            "physical_regime": physical_regime,
        })
        df.attrs["theta"] = theta
        return df
