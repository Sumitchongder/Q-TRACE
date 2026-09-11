"""
Q-TRACE :: qkd_simulator.channel
==================================
Fiber channel model with environmental drift (temperature, vibration).
"""

import numpy as np
from .physics import channel_transmittance


class FiberChannel:
    """
    A fiber-optic quantum channel with slow environmental drift applied on
    top of the nominal attenuation. Drift is modelled as a low-frequency
    sinusoid + Gaussian noise, representing thermal expansion / vibration
    induced transmittance fluctuations seen in real deployed links.
    """

    def __init__(self, alpha_db_per_km, length_km,
                 temperature_drift_amp=0.0, vibration_amp=0.0, rng=None):
        self.alpha = alpha_db_per_km
        self.length = length_km
        self.temp_amp = temperature_drift_amp
        self.vib_amp = vibration_amp
        self.rng = rng if rng is not None else np.random.default_rng()
        self._phase = self.rng.uniform(0, 2 * np.pi)

    def transmittance(self, t_index: np.ndarray) -> np.ndarray:
        """Instantaneous transmittance at (vectorized) time index t_index."""
        base = channel_transmittance(self.alpha, self.length)
        drift = 1.0 + self.temp_amp * np.sin(0.01 * t_index + self._phase)
        vib_noise = self.vib_amp * self.rng.standard_normal(size=np.shape(t_index))
        eta_ch = np.clip(base * drift + vib_noise * base, 1e-9, 1.0)
        return eta_ch
