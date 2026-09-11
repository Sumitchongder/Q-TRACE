"""
Q-TRACE :: attacks.intercept_resend
=====================================
Intercept-Resend (IR) attack: Eve measures each intercepted pulse in a
randomly chosen basis and resends a freshly prepared state. For pulses
where Eve's basis mismatches Alice/Bob's, this introduces a 25% error
rate on the intercepted fraction (standard BB84 IR result), applied
identically to signal and decoy pulses since Eve cannot distinguish them.
"""

import numpy as np
from .base import Attack


class InterceptResendAttack(Attack):
    name = "intercept_resend"

    def __init__(self, intercept_fraction=0.5, **kwargs):
        super().__init__(**kwargs)
        self.intercept_fraction = intercept_fraction

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength
        f = self.intercept_fraction

        added_qber = 0.25 * f * s * mask
        state["E_signal"] = state["E_signal"] + added_qber
        state["E_decoy"] = state["E_decoy"] + added_qber

        # resend introduces slight excess loss (imperfect re-preparation)
        state["Y_signal"] = state["Y_signal"] * (1.0 - 0.01 * f * s * mask)
        state["Y_decoy"] = state["Y_decoy"] * (1.0 - 0.01 * f * s * mask)
        return state
