"""
Q-TRACE :: qkd_simulator.detector
===================================
Single-photon detector model: efficiency, dark counts, timing jitter,
and slow efficiency drift (aging / thermal effects on SPADs / SNSPDs).
"""

import numpy as np


class Detector:
    def __init__(self, efficiency, dark_count_rate, intrinsic_error_rate,
                 timing_jitter_ps, drift_amp=0.0, rng=None):
        self.eta_d0 = efficiency
        self.Y0 = dark_count_rate
        self.e_d = intrinsic_error_rate
        self.sigma_t = timing_jitter_ps
        self.drift_amp = drift_amp
        self.rng = rng if rng is not None else np.random.default_rng()
        self._phase = self.rng.uniform(0, 2 * np.pi)

    def efficiency(self, t_index: np.ndarray) -> np.ndarray:
        """Detector efficiency with slow aging/thermal drift."""
        drift = 1.0 + self.drift_amp * np.sin(0.005 * t_index + self._phase)
        return np.clip(self.eta_d0 * drift, 1e-6, 1.0)
