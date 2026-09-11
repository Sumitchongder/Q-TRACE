"""
Q-TRACE :: attacks.rng_manipulation
=====================================
RNG-manipulation attack: Eve exploits weak or partially predictable
quantum/pseudo-random number generation used for basis/state selection,
introducing a subtle periodic bias rather than pure white-noise
fluctuation. Modelled as a low-frequency sinusoidal modulation of the
error rate -- a signature that is easy for spectral/temporal models to
catch but easy for scalar (non-temporal) detectors to miss, motivating
the temporal-block feature representation used in Q-TRACE.
"""

import numpy as np
from .base import Attack


class RNGManipulationAttack(Attack):
    name = "rng_manipulation"

    def __init__(self, period=15, **kwargs):
        super().__init__(**kwargs)
        self.period = period

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength
        t = state["t_index"]

        periodic_bias = 0.01 * s * np.sin(2 * np.pi * t / self.period)
        state["E_signal"] = state["E_signal"] + periodic_bias * mask
        state["E_decoy"] = state["E_decoy"] + periodic_bias * mask
        return state
