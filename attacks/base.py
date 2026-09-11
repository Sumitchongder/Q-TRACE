"""
Q-TRACE :: attacks.base
=========================
Common interface for all attack modules. An Attack perturbs the *expected*
physical state (yields, QBERs) computed by the digital twin, before finite
-statistics sampling noise is applied. This keeps every attack physically
grounded: it modifies the same quantities a real eavesdropper would
actually influence (click probabilities, error rates), rather than
tampering with the final telemetry table directly.

state dict keys (see qkd_simulator.bb84.DecoyBB84Simulator.simulate_run):
    Y_signal, Y_decoy, Y_vacuum : ndarray, expected yields
    E_signal, E_decoy           : ndarray, expected QBERs
    eta_ch, eta_d               : ndarray, channel / detector efficiency
    theta                       : dict, physical parameter vector
    t_index                     : ndarray, window time indices
"""

from abc import ABC, abstractmethod
import numpy as np


class Attack(ABC):
    name = "base_attack"

    def __init__(self, onset_frac=0.4, strength=1.0, rng=None):
        """
        onset_frac : fraction into the run at which the attack begins
                     (attacks are not necessarily present from t=0, which
                     is itself an important realism feature for a
                     detection-latency metric).
        strength   : dimensionless severity multiplier in [0, 1+], allows
                     generating both subtle and blatant instances of the
                     same attack family.
        """
        self.onset_frac = onset_frac
        self.strength = strength
        self.rng = rng if rng is not None else np.random.default_rng()

    def _onset_mask(self, t_index):
        onset_t = int(self.onset_frac * len(t_index))
        mask = np.zeros(len(t_index), dtype=float)
        mask[onset_t:] = 1.0
        return mask

    @abstractmethod
    def apply(self, state: dict, rng) -> dict:
        """Return a new state dict with attack-perturbed quantities."""
        raise NotImplementedError


class CompositeAttack(Attack):
    """Applies two (or more) attacks in sequence -- used for the
    compositional zero-day evaluation (Section 11 of the research plan)."""
    name = "composite"

    def __init__(self, attacks, **kwargs):
        super().__init__(**kwargs)
        self.attacks = attacks
        self.name = "+".join(a.name for a in attacks)

    def apply(self, state, rng):
        for a in self.attacks:
            state = a.apply(state, rng)
        return state
