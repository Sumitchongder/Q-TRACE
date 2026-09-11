"""
Q-TRACE :: attacks.blinding
=============================
Detector-blinding (fake-state) attack: Eve blinds Bob's SPADs with strong
CW light, forcing them into a linear regime, then triggers deterministic
clicks with tailored bright pulses. Hallmark symptom: the yield loses its
expected dependence on mu (blinded detectors click almost independently
of the true weak-coherent-pulse intensity), so Y_signal and Y_decoy
converge toward a common blinded click probability, collapsing delta_Y.
"""

import numpy as np
from .base import Attack


class BlindingAttack(Attack):
    name = "blinding"

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength

        blinded_click_prob = 0.03  # Eve's controlled deterministic click rate
        target = blinded_click_prob

        state["Y_signal"] = state["Y_signal"] * (1 - s * mask) + target * s * mask
        state["Y_decoy"] = state["Y_decoy"] * (1 - s * mask) + target * s * mask

        # blinding can be done nearly error-free by a competent attacker
        state["E_signal"] = state["E_signal"] * (1 - 0.3 * s * mask)
        state["E_decoy"] = state["E_decoy"] * (1 - 0.3 * s * mask)
        return state
