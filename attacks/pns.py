"""
Q-TRACE :: attacks.pns
========================
Photon-Number-Splitting (PNS) attack.

Physical rationale: Eve deterministically splits off and stores one photon
from multi-photon pulses (n>=2), forwarding the rest lossless. Because
PNS acts selectively on the photon-number distribution, it breaks the
expected mu-dependence of the yield -- i.e. it perturbs the *relationship*
between Y_signal and Y_decoy (delta_Y) -- while introducing negligible
extra QBER (this is precisely why decoy-state analysis, and QNu's ARMOS
signal/decoy-yield monitoring, was designed to catch it).
"""

import numpy as np
from .base import Attack


class PNSAttack(Attack):
    name = "pns"

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength

        # PNS suppresses the *signal* yield more than decoy (higher mu ->
        # more multi-photon pulses available to split), distorting delta_Y
        # without materially raising QBER.
        suppression_signal = 1.0 - 0.12 * s * mask
        suppression_decoy = 1.0 - 0.03 * s * mask

        state["Y_signal"] = state["Y_signal"] * suppression_signal
        state["Y_decoy"] = state["Y_decoy"] * suppression_decoy

        # tiny QBER perturbation from imperfect splitting optics
        state["E_signal"] = state["E_signal"] + 0.002 * s * mask
        return state
