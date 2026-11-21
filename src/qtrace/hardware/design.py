"""Design of the multiplexed BB84 hardware experiment.

``L`` independent BB84 links run in parallel on one processor; link ``i``
uses a data qubit (Alice -> channel -> Bob) and a neighbouring ancilla
qubit owned by Eve. Every circuit realises one setting ``(a, alpha,
beta)`` per link; settings rotate through all eight combinations in a
Latin-square pattern (link ``i`` in circuit ``c`` uses setting
``(i + c) mod 8``), so every link sees every setting once per *pass* of
eight circuits and every circuit contains all settings.

Epochs
------
* commissioning: ``n_cal_passes`` passes, every link honest;
* monitoring: ``n_epochs`` epochs of one pass each; every link is honest in
  ``n_honest_per_link`` of them and attacked with a distinct probe in the
  others, according to a balanced, seeded schedule.

Probes (Eve's interaction on the channel)
-----------------------------------------
``probe(theta)``
    Eve couples her ancilla with
    ``RZX(theta)`` (learns the Z-basis bit, disturbs X) or ``RXX(theta)``
    (learns the X-basis bit, disturbs Z), balanced so that in every pass she
    matches Alice's basis on exactly half of the sifted settings (a purely
    random choice leaves a 1/16 chance that an epoch is never disturbed).
    Ancilla read in the Y basis,
    which is Helstrom-optimal for both. Ideal disturbance in the attacked
    basis is ``sin^2(theta/2)`` and Eve's guessing error there is
    ``(1 - sin theta)/2``; ``theta = pi/2`` is intercept-resend.
``ad(gamma)``
    Amplitude-damping coupling ``CRY(2 asin sqrt gamma)`` + ``CX`` that
    decays ``|1> -> |0>`` with probability ``gamma``: bit-asymmetric
    errors, the qubit analogue of detector-efficiency-mismatch attacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

SETTINGS = [(a, al, be) for a in (0, 1) for al in (0, 1) for be in (0, 1)]  # 8, basis 0=Z 1=X


@dataclass(frozen=True)
class Probe:
    kind: str            # "honest" | "probe" | "ad"
    strength: float = 0.0

    @property
    def label(self):
        if self.kind == "honest":
            return "honest"
        if self.kind == "probe":
            return f"probe(theta={self.strength:.3f})"
        return f"ad(gamma={self.strength:.3f})"

    def ideal_qber(self):
        """Ideal sifted QBER added by the probe, averaged over bases and bits."""
        if self.kind == "probe":
            return 0.5 * np.sin(self.strength / 2) ** 2
        if self.kind == "ad":
            g = self.strength
            return 0.5 * (g / 2) + 0.5 * (1 - np.sqrt(1 - g)) / 2
        return 0.0


DEFAULT_PROBES = (
    [Probe("probe", t) for t in (0.15, 0.25, 0.40, 0.60, 0.90, float(np.pi / 2))]
    + [Probe("ad", g) for g in (0.03, 0.06, 0.12, 0.25)]
)


@dataclass
class ExperimentDesign:
    n_links: int
    n_cal_passes: int
    n_epochs: int
    probes: list
    schedule: list                      # [epoch][link] -> probe index (-1 honest)
    eve_basis: list                     # [circuit][link] -> 0 (RZX) / 1 (RXX)
    seed: int
    physical_pairs: list = field(default_factory=list)   # [(data, ancilla)]

    @property
    def n_circuits(self):
        return 8 * (self.n_cal_passes + self.n_epochs)

    def epoch_of_circuit(self, c):
        """-1 for commissioning circuits, else monitoring epoch index."""
        p = c // 8
        return -1 if p < self.n_cal_passes else p - self.n_cal_passes

    def setting(self, c, link):
        return SETTINGS[(link + c) % 8]

    def probe(self, c, link) -> Probe:
        e = self.epoch_of_circuit(c)
        if e < 0:
            return Probe("honest")
        j = self.schedule[e][link]
        return Probe("honest") if j < 0 else self.probes[j]

    def to_json(self):
        d = asdict(self)
        d["probes"] = [asdict(p) for p in self.probes]
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        d["probes"] = [Probe(**p) for p in d["probes"]]
        d["physical_pairs"] = [tuple(p) for p in d.get("physical_pairs", [])]
        return cls(**d)


def build_design(n_links=40, n_cal_passes=2, n_epochs=6, n_honest_per_link=2,
                 probes=DEFAULT_PROBES, seed=2026) -> ExperimentDesign:
    """Balanced, seeded attack schedule.

    Each link is attacked in ``n_epochs - n_honest_per_link`` epochs with
    distinct probes; probe usage is balanced across the whole experiment
    by dealing probe indices from a shuffled, cyclically repeated deck.
    """
    rng = np.random.default_rng(seed)
    probes = list(probes)
    n_att = n_epochs - n_honest_per_link
    deck = []
    while len(deck) < n_links * n_att:
        deck.extend(rng.permutation(len(probes)).tolist())
    schedule = -np.ones((n_epochs, n_links), dtype=int)
    k = 0
    for link in range(n_links):
        epochs = rng.choice(n_epochs, size=n_att, replace=False)
        chosen = []
        for e in sorted(epochs):
            # avoid repeating a probe on the same link
            j = deck[k]
            tries = 0
            while j in chosen and tries < len(deck):
                deck.append(deck.pop(k))
                j = deck[k]
                tries += 1
            chosen.append(j)
            schedule[e, link] = j
            k += 1
    # Eve's probe basis: balanced so that in every pass exactly one of the two
    # sifted settings of each Alice basis is attacked in the matching basis
    # (disturbing the other), with a random per-(pass, link) assignment.
    n_circ = 8 * (n_cal_passes + n_epochs)
    eve_basis = np.zeros((n_circ, n_links), dtype=int)
    for p in range(n_cal_passes + n_epochs):
        flip = rng.integers(0, 2, size=(n_links, 2))
        for c in range(8 * p, 8 * p + 8):
            for link in range(n_links):
                a, al, be = SETTINGS[(link + c) % 8]
                if al == be:
                    eve_basis[c, link] = al ^ (a ^ flip[link, al])
                else:
                    eve_basis[c, link] = rng.integers(0, 2)
    return ExperimentDesign(n_links=n_links, n_cal_passes=n_cal_passes, n_epochs=n_epochs,
                            probes=probes, schedule=schedule.tolist(),
                            eve_basis=eve_basis.tolist(), seed=seed)
