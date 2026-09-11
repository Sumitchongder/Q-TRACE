"""
Q-TRACE :: attacks.time_shift
===============================
Time-shift attack: Eve exploits the time-dependent efficiency mismatch
between Bob's two detectors by shifting photon arrival times, biasing
which detector (and hence which bit value) is more likely to click. This
manifests as an asymmetric, jitter-correlated error inflation that grows
with the device's own timing_jitter_ps (the attack "hides" inside
legitimate jitter), making it a good stress test for physics-informed vs.
purely statistical detectors.
"""

import numpy as np
from .base import Attack


class TimeShiftAttack(Attack):
    name = "time_shift"

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength

        jitter_norm = state["theta"]["sigma_t"] / 120.0  # normalize to config max
        bias = 0.03 * s * jitter_norm * mask

        state["E_signal"] = state["E_signal"] + bias
        state["E_decoy"] = state["E_decoy"] + bias * 0.6
        return state
