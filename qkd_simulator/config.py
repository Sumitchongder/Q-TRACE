"""
Q-TRACE :: qkd_simulator.config
================================
Central configuration for the decoy-state BB84 QKD digital twin.

All physical parameters are defined as *ranges* (not fixed points) so that
the simulator can sample entire operating regimes rather than a single
static configuration. This is essential for the "physical distribution
shift" and "zero-day" experiments described in the Q-TRACE research plan.

Units
-----
- Channel loss (alpha)     : dB/km
- Link length (L)          : km
- Detector efficiency      : dimensionless [0, 1]
- Dark count rate (Y0)     : probability per pulse (dimensionless)
- Timing jitter (sigma_t)  : picoseconds (ps)
- Phase noise (sigma_phi)  : radians
- Intrinsic error (e_d)    : dimensionless [0, 1]
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ParamRange:
    """A closed interval [low, high] used for uniform sampling."""
    low: float
    high: float

    def sample(self, rng, size=None):
        return rng.uniform(self.low, self.high, size=size)


@dataclass
class DecoyBB84Config:
    # ---- Source intensities (mean photon number per pulse) ----
    mu_signal: ParamRange = field(default_factory=lambda: ParamRange(0.45, 0.65))
    mu_decoy: ParamRange = field(default_factory=lambda: ParamRange(0.08, 0.18))
    mu_vacuum: float = 0.0  # vacuum state, fixed by definition

    # ---- State preparation probabilities ----
    p_signal: float = 0.7
    p_decoy: float = 0.25
    p_vacuum: float = 0.05

    # ---- Channel ----
    fiber_atten_db_per_km: ParamRange = field(default_factory=lambda: ParamRange(0.18, 0.22))
    link_length_km: ParamRange = field(default_factory=lambda: ParamRange(2.0, 25.0))

    # ---- Detector ----
    detector_efficiency: ParamRange = field(default_factory=lambda: ParamRange(0.10, 0.70))
    dark_count_rate: ParamRange = field(default_factory=lambda: ParamRange(1e-7, 5e-6))
    intrinsic_error_rate: ParamRange = field(default_factory=lambda: ParamRange(0.005, 0.02))

    # ---- Timing / phase imperfections ----
    timing_jitter_ps: ParamRange = field(default_factory=lambda: ParamRange(20.0, 120.0))
    phase_noise_rad: ParamRange = field(default_factory=lambda: ParamRange(0.01, 0.08))

    # ---- Environmental drift (slow multiplicative modulation) ----
    temperature_drift_amp: ParamRange = field(default_factory=lambda: ParamRange(0.0, 0.03))
    vibration_amp: ParamRange = field(default_factory=lambda: ParamRange(0.0, 0.02))

    # ---- Pulses ----
    pulses_per_window: int = 200_000   # pulses aggregated into one telemetry sample
    window_stride_s: float = 1.0       # nominal time between telemetry samples

    # ---- Random seed ----
    seed: int = 42


DEFAULT_CONFIG = DecoyBB84Config()
