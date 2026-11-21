"""Count-level digital twin of a deployed decoy-state BB84 link.

Each simulated *run* is one physical link observed for ``n_windows``
telemetry windows of ``pulses_per_window`` pulses. The first
``n_calibration`` windows are attack-free commissioning data. The twin
deliberately contains effects that the Q-TRACE honest model does **not**
contain, so that detector performance is always measured under model
mismatch:

* detector afterpulsing (background proportional to the click rate),
* window-to-window intensity jitter of each intensity class,
* slow polarisation drift of the optical misalignment,
* temperature-driven dark-count modulation,
* channel-transmittance random walk and diurnal modulation.

Benign channel faults (loss steps, loss ramps) and device faults
(dark-count rise) can be injected into honest runs to measure false-alarm
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from qtrace.physics.optics import (outcome_probs_poisson, setting_probabilities,
                                   SETTING_SHAPE)


@dataclass
class Range:
    low: float
    high: float

    def sample(self, rng, log=False):
        if log:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))


@dataclass
class TwinConfig:
    """Sampling ranges for link parameters and nuisance processes."""
    mu_signal: Range = field(default_factory=lambda: Range(0.40, 0.60))
    decoy_ratio: Range = field(default_factory=lambda: Range(0.15, 0.30))
    p_intensity: tuple = (0.70, 0.20, 0.10)
    length_km: Range = field(default_factory=lambda: Range(5.0, 60.0))
    atten_db_km: Range = field(default_factory=lambda: Range(0.18, 0.22))
    eta_bob: Range = field(default_factory=lambda: Range(0.10, 0.50))
    e_misalign: Range = field(default_factory=lambda: Range(0.005, 0.02))
    dark_count: Range = field(default_factory=lambda: Range(2e-7, 5e-6))
    kappa: Range = field(default_factory=lambda: Range(0.85, 1.15))
    afterpulse: Range = field(default_factory=lambda: Range(0.0, 0.03))
    intensity_jitter: Range = field(default_factory=lambda: Range(0.002, 0.01))
    transmittance_rw_sigma: Range = field(default_factory=lambda: Range(0.001, 0.006))
    diurnal_amp: Range = field(default_factory=lambda: Range(0.0, 0.04))
    polarisation_rw_sigma: Range = field(default_factory=lambda: Range(0.0, 2e-4))
    darkcount_temp_amp: Range = field(default_factory=lambda: Range(0.0, 0.10))
    pulses_per_window: int = 10_000_000


@dataclass
class LinkParams:
    mu_signal: float
    mu_decoy: float
    p_intensity: tuple
    length_km: float
    atten_db_km: float
    eta_bob: float
    e_z: float
    e_x: float
    pd0: float
    pd1: float
    kappa: float
    afterpulse: float
    intensity_jitter: float
    transmittance_rw_sigma: float
    diurnal_amp: float
    diurnal_period: float
    polarisation_rw_sigma: float
    darkcount_temp_amp: float

    @property
    def eta_channel(self) -> float:
        return 10.0 ** (-self.atten_db_km * self.length_km / 10.0)

    def to_dict(self):
        return asdict(self)


def sample_link(cfg: TwinConfig, rng) -> LinkParams:
    mu_s = cfg.mu_signal.sample(rng)
    e0 = cfg.e_misalign.sample(rng)
    return LinkParams(
        mu_signal=mu_s,
        mu_decoy=mu_s * cfg.decoy_ratio.sample(rng),
        p_intensity=tuple(cfg.p_intensity),
        length_km=cfg.length_km.sample(rng),
        atten_db_km=cfg.atten_db_km.sample(rng),
        eta_bob=cfg.eta_bob.sample(rng),
        e_z=e0 * rng.uniform(0.8, 1.25),
        e_x=e0 * rng.uniform(0.8, 1.25),
        pd0=cfg.dark_count.sample(rng, log=True),
        pd1=cfg.dark_count.sample(rng, log=True),
        kappa=cfg.kappa.sample(rng),
        afterpulse=cfg.afterpulse.sample(rng),
        intensity_jitter=cfg.intensity_jitter.sample(rng),
        transmittance_rw_sigma=cfg.transmittance_rw_sigma.sample(rng),
        diurnal_amp=cfg.diurnal_amp.sample(rng),
        diurnal_period=float(rng.uniform(150.0, 600.0)),
        polarisation_rw_sigma=cfg.polarisation_rw_sigma.sample(rng),
        darkcount_temp_amp=cfg.darkcount_temp_amp.sample(rng),
    )


@dataclass
class WindowState:
    """Per-window physical state (all arrays have leading dimension W)."""
    mu: np.ndarray            # (W, 3)
    eta_ch: np.ndarray        # (W,)
    eta_b: np.ndarray         # (W,)
    e_z: np.ndarray
    e_x: np.ndarray
    pd0: np.ndarray
    pd1: np.ndarray
    kappa: np.ndarray
    p_k: np.ndarray           # (W, 3)
    p_a1: np.ndarray
    p_alpha_z: np.ndarray
    p_beta_z: np.ndarray
    t: np.ndarray

    @property
    def eta(self):
        return self.eta_ch * self.eta_b

    def honest_probs(self):
        return outcome_probs_poisson(self.mu, self.eta, self.e_z, self.e_x,
                                     self.pd0, self.pd1, self.kappa)

    def setting_probs(self):
        return setting_probabilities(self.p_k, self.p_a1, self.p_alpha_z, self.p_beta_z)


def add_background(probs, q0, q1):
    """Add independent extra click probability ``q_j`` to detector ``j``."""
    q0 = np.asarray(q0, float)[..., None, None, None, None]
    q1 = np.asarray(q1, float)[..., None, None, None, None]
    s01 = probs[..., 0]
    s1 = probs[..., 0] + probs[..., 1]     # D1 silent
    s0 = probs[..., 0] + probs[..., 2]     # D0 silent
    s0n = s0 * (1 - q0)
    s1n = s1 * (1 - q1)
    s01n = s01 * (1 - q0) * (1 - q1)
    out = np.stack([s01n, s1n - s01n, s0n - s01n, 1 - s0n - s1n + s01n], axis=-1)
    return np.clip(out, 0.0, 1.0)


def detector_click_rates(probs, setting_probs):
    """Average click probability of each detector over all settings, (W, 2)."""
    p_d0 = probs[..., 1] + probs[..., 3]
    p_d1 = probs[..., 2] + probs[..., 3]
    w = setting_probs
    r0 = np.sum(w * p_d0, axis=(-4, -3, -2, -1))
    r1 = np.sum(w * p_d1, axis=(-4, -3, -2, -1))
    return np.stack([r0, r1], axis=-1)


@dataclass
class Fault:
    """Non-adversarial event injected into an honest run."""
    kind: str                 # "loss_step" | "loss_ramp" | "darkcount_rise"
    onset: int
    magnitude: float

    def apply(self, state: WindowState):
        m = state.t >= self.onset
        if self.kind == "loss_step":
            state.eta_ch = np.where(m, state.eta_ch * 10 ** (-self.magnitude / 10), state.eta_ch)
        elif self.kind == "loss_ramp":
            ramp = np.clip((state.t - self.onset) / 100.0, 0, 1) * self.magnitude
            state.eta_ch = state.eta_ch * 10 ** (-ramp / 10)
        elif self.kind == "darkcount_rise":
            state.pd0 = np.where(m, state.pd0 * self.magnitude, state.pd0)
            state.pd1 = np.where(m, state.pd1 * self.magnitude, state.pd1)
        else:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        return state


@dataclass
class RunResult:
    counts: np.ndarray        # (W, 3, 2, 2, 2, 4) int64
    active: np.ndarray        # (W,) bool, attack (or fault) active
    state: WindowState
    link: LinkParams
    meta: dict


class DecoyBB84Twin:
    """Generates raw outcome tables for one link."""

    def __init__(self, config: Optional[TwinConfig] = None):
        self.cfg = config or TwinConfig()

    def window_states(self, link: LinkParams, n_windows: int, rng) -> WindowState:
        t = np.arange(n_windows)
        rw = np.zeros(n_windows)
        for i in range(1, n_windows):
            rw[i] = 0.995 * rw[i - 1] + link.transmittance_rw_sigma * rng.standard_normal()
        phase = rng.uniform(0, 2 * np.pi)
        diurnal = link.diurnal_amp * np.sin(2 * np.pi * t / link.diurnal_period + phase)
        eta_ch = np.clip(link.eta_channel * np.exp(rw + diurnal), 1e-9, 1.0)

        pol = np.zeros((n_windows, 2))
        for i in range(1, n_windows):
            pol[i] = 0.98 * pol[i - 1] + link.polarisation_rw_sigma * rng.standard_normal(2)
        e_z = np.clip(link.e_z + pol[:, 0], 1e-4, 0.2)
        e_x = np.clip(link.e_x + pol[:, 1], 1e-4, 0.2)

        temp = 1 + link.darkcount_temp_amp * np.sin(2 * np.pi * t / link.diurnal_period + phase + 0.7)
        jitter = 1 + link.intensity_jitter * rng.standard_normal((n_windows, 2))
        mu = np.column_stack([link.mu_signal * jitter[:, 0], link.mu_decoy * jitter[:, 1],
                              np.zeros(n_windows)])
        ones = np.ones(n_windows)
        return WindowState(
            mu=mu, eta_ch=eta_ch, eta_b=link.eta_bob * ones, e_z=e_z, e_x=e_x,
            pd0=link.pd0 * temp, pd1=link.pd1 * temp, kappa=link.kappa * ones,
            p_k=np.tile(np.asarray(link.p_intensity, float), (n_windows, 1)),
            p_a1=0.5 * ones, p_alpha_z=0.5 * ones, p_beta_z=0.5 * ones, t=t)

    def simulate(self, link: LinkParams, n_windows: int, rng, attack=None,
                 fault: Optional[Fault] = None, pulses_per_window: Optional[int] = None,
                 meta: Optional[dict] = None) -> RunResult:
        n_pulses = pulses_per_window or self.cfg.pulses_per_window
        state = self.window_states(link, n_windows, rng)
        active = np.zeros(n_windows, dtype=bool)
        if fault is not None:
            state = fault.apply(state)
            active = state.t >= fault.onset

        probs = state.honest_probs()
        set_p = state.setting_probs()
        if attack is not None:
            probs, set_p, active = attack.apply(state, probs, set_p, rng)

        # afterpulsing: extra background proportional to each detector's click rate
        rates = detector_click_rates(probs, set_p)
        probs = add_background(probs, link.afterpulse * rates[:, 0], link.afterpulse * rates[:, 1])

        set_flat = set_p.reshape(n_windows, -1)
        set_flat = set_flat / set_flat.sum(axis=1, keepdims=True)
        n_set = rng.multinomial(n_pulses, set_flat)                     # (W, 24)
        pv = probs.reshape(n_windows, -1, 4)
        pv = pv / pv.sum(axis=-1, keepdims=True)
        counts = rng.multinomial(n_set, pv)                             # (W, 24, 4)
        counts = counts.reshape((n_windows,) + SETTING_SHAPE + (4,))
        return RunResult(counts=counts.astype(np.int64), active=active, state=state,
                         link=link, meta=dict(meta or {}))
