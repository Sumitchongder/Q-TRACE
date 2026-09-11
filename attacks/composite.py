"""
Q-TRACE :: attacks.composite
==============================
Convenience factories for compositional zero-day attacks (Section 11):
combinations of two base attacks never presented together during training.
"""

from .base import CompositeAttack
from .pns import PNSAttack
from .intercept_resend import InterceptResendAttack
from .trojan_horse import TrojanHorseAttack
from .blinding import BlindingAttack
from .time_shift import TimeShiftAttack
from .rng_manipulation import RNGManipulationAttack

_REGISTRY = {
    "pns": PNSAttack,
    "intercept_resend": InterceptResendAttack,
    "trojan_horse": TrojanHorseAttack,
    "blinding": BlindingAttack,
    "time_shift": TimeShiftAttack,
    "rng_manipulation": RNGManipulationAttack,
}

# canonical compositions used in the compositional zero-day experiment (E4)
CANONICAL_COMPOSITIONS = [
    ("pns", "blinding"),
    ("trojan_horse", "time_shift"),
    ("pns", "time_shift"),
    ("intercept_resend", "rng_manipulation"),
]


def make_attack(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def make_composite(name_a, name_b, onset_frac=0.4, strength=1.0, rng=None):
    a = make_attack(name_a, onset_frac=onset_frac, strength=strength, rng=rng)
    b = make_attack(name_b, onset_frac=onset_frac, strength=strength, rng=rng)
    return CompositeAttack([a, b], onset_frac=onset_frac, strength=strength, rng=rng)
