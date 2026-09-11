"""
Q-TRACE :: attacks.trojan_horse
=================================
Trojan-Horse attack: Eve injects bright light into Alice's (or Bob's)
apparatus and analyzes back-reflected light to infer modulator settings.
Symptom in telemetry: a small excess back-reflected detection/yield
signature correlated with signal pulses (the injected probe co-propagates
with signal timing), with comparatively little QBER change -- making it
one of the harder attacks to catch via QBER-only monitoring, and the
motivation for ETSI's dedicated Trojan-horse implementation-security
work item.
"""

import numpy as np
from .base import Attack


class TrojanHorseAttack(Attack):
    name = "trojan_horse"

    def apply(self, state, rng):
        mask = self._onset_mask(state["t_index"])
        s = self.strength

        # excess back-reflection inflates the *signal* yield slightly
        # (extra spurious clicks from injected-light echoes)
        state["Y_signal"] = state["Y_signal"] * (1.0 + 0.04 * s * mask)

        # small added timing-correlated error from probe interference
        state["E_signal"] = state["E_signal"] + 0.004 * s * mask
        return state
