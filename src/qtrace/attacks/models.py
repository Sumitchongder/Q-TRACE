"""Implementation attacks on decoy-state BB84, written at the level of the
receiver outcome table.

Every attack acts on a fraction ``fraction`` of the pulses sent from its
``onset`` window onward and is derived from its physical mechanism:

``intercept_resend``
    Eve measures in a random basis and resends the state she found
    (ideal resend: photon statistics preserved). Sifted error becomes
    ``e/2 + 1/4`` on intercepted pulses.
``pns``
    Photon-number splitting with a lossless channel (Brassard et al.,
    PRL 85, 1330, 2000): Eve stores one photon of every multi-photon pulse,
    forwards the rest losslessly and suppresses single photons so the
    signal gain matches the honest gain. Written exactly with the
    photon-number-resolved receiver model.
``blinding``
    Faked-state attack on blinded APDs (Lydersen et al., Nat. Photon. 4,
    686, 2010). Eve detects Alice's pulse, resends a bright trigger in her
    basis; Bob's blinded detectors click deterministically when bases
    agree, never double-click and show no dark counts. Eve tunes her
    detection efficiency to reproduce the signal gain and her error rate
    to reproduce the honest QBER ("stealthy" variant).
``time_shift``
    Efficiency-mismatch exploitation (Qi et al., QIC 7, 73, 2007; Zhao et
    al., PRA 78, 042333, 2008). Each attacked pulse is shifted to a time
    where one detector's efficiency is reduced to ``r``; Eve compensates
    the lost gain by lowering channel loss.
``trojan_horse``
    Bright-light probing of Alice's modulators (Gisin et al., PRA 73,
    022320, 2006). Passive with respect to Bob except for optional
    back-scattered background ``backscatter`` (per-gate click probability
    added to both detectors); ``backscatter = 0`` is telemetry-blind.
``rng_bias``
    A partially predictable bit generator at Alice: ``P(a=1)`` is
    modulated as ``1/2 + delta sin(2 pi t / period)``.

Composite attacks superpose the first-order perturbations of their
components.
"""

from __future__ import annotations

import numpy as np

from qtrace.physics.optics import (outcome_probs_poisson, outcome_probs_photon_resolved,
                                   setting_probabilities)


def _bisect(fun, lo, hi, n_iter=50):
    """Vectorised bisection for monotone increasing ``fun`` with root in [lo, hi]."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = fun(mid)
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return 0.5 * (lo + hi)


def _signal_click_prob(probs):
    """Mean probability of any click for signal pulses, (W,)."""
    return 1.0 - probs[:, 0, ..., 0].mean(axis=(-3, -2, -1))


class Attack:
    name = "attack"

    def __init__(self, onset: int, fraction: float = 1.0):
        self.onset = int(onset)
        self.fraction = float(fraction)

    def attacked_probs(self, state, honest, rng):
        return honest

    def modify_settings(self, state, set_p):
        return set_p

    def perturbation(self, state, honest, set_p, rng):
        att = self.attacked_probs(state, honest, rng)
        return self.fraction * (att - honest), self.modify_settings(state, set_p) - set_p

    def apply(self, state, probs, set_p, rng):
        m = state.t >= self.onset
        dp, ds = self.perturbation(state, probs, set_p, rng)
        mb = m[:, None, None, None, None, None]
        probs = np.where(mb, np.clip(probs + dp, 0.0, 1.0), probs)
        set_p = np.where(m[:, None, None, None, None], set_p + ds, set_p)
        return probs, set_p, m

    def describe(self):
        d = {"attack": self.name, "onset": self.onset, "fraction": self.fraction}
        d.update({k: v for k, v in vars(self).items() if k not in ("onset", "fraction")})
        return d


class InterceptResend(Attack):
    name = "intercept_resend"

    def attacked_probs(self, state, honest, rng):
        e_z = state.e_z / 2 + 0.25
        e_x = state.e_x / 2 + 0.25
        return outcome_probs_poisson(state.mu, state.eta, e_z, e_x, state.pd0, state.pd1,
                                     state.kappa)


class PhotonNumberSplitting(Attack):
    name = "pns"

    @staticmethod
    def _gf(t1, block):
        def gf(n, z):
            t1b = t1[:, None, None, None, None]
            bb = block[:, None, None, None, None]
            zz = z[..., None]
            g1 = 1 - t1b + t1b * zz
            gm = (1 - bb) * zz ** np.maximum(n - 1, 0) + bb
            return np.where(n == 0, 1.0, np.where(n == 1, g1, gm))
        return gf

    def _probs(self, state, t1, block):
        return outcome_probs_photon_resolved(state.mu, self._gf(t1, block), state.eta_b,
                                             state.e_z, state.e_x, state.pd0, state.pd1,
                                             state.kappa)

    def attacked_probs(self, state, honest, rng):
        target = _signal_click_prob(honest)
        w = len(target)
        zero, one = np.zeros(w), np.ones(w)
        g_multi = _signal_click_prob(self._probs(state, zero, zero))
        need_block = g_multi > target
        t1 = _bisect(lambda x: _signal_click_prob(self._probs(state, x, zero)) - target, zero, one)
        blk = _bisect(lambda x: target - _signal_click_prob(self._probs(state, zero, x)), zero, one)
        t1 = np.where(need_block, 0.0, t1)
        blk = np.where(need_block, blk, 0.0)
        self.last_t1, self.last_block = t1, blk
        return self._probs(state, t1, blk)


class Blinding(Attack):
    name = "blinding"

    def attacked_probs(self, state, honest, rng):
        g_s = _signal_click_prob(honest)
        mu_s = state.mu[:, 0]
        eta_e = -np.log(np.clip(1 - 2 * g_s, 1e-12, 1)) / mu_s
        # reproduce the honest sifted QBER on signal pulses
        qber = []
        for b in (0, 1):
            cell = honest[:, 0, :, b, b, :]                      # (W, a, outcome)
            clicks = cell[..., 1:].sum(axis=(-2, -1))
            err = cell[:, 0, 2] + cell[:, 1, 1] + 0.5 * (cell[:, 0, 3] + cell[:, 1, 3])
            qber.append(err / np.maximum(clicks, 1e-300))
        e_eve = np.clip(0.5 * (qber[0] + qber[1]), 0, 0.5)
        p_e = 1 - np.exp(-state.mu * eta_e[:, None])                  # (W, 3)
        out = np.zeros_like(honest)
        for a in (0, 1):
            for al in (0, 1):
                for be in (0, 1):
                    click = p_e / 2
                    if al == be:
                        pc = click * (1 - e_eve[:, None])
                        pw = click * e_eve[:, None]
                        d_a, d_w = (1, 2) if a == 0 else (2, 1)
                        out[:, :, a, al, be, d_a] = pc
                        out[:, :, a, al, be, d_w] = pw
                    else:
                        out[:, :, a, al, be, 1] = click / 2
                        out[:, :, a, al, be, 2] = click / 2
                    out[:, :, a, al, be, 0] = 1 - click
        return out


class TimeShift(Attack):
    name = "time_shift"

    def __init__(self, onset, fraction=1.0, mismatch=0.25):
        super().__init__(onset, fraction)
        self.mismatch = float(mismatch)

    def attacked_probs(self, state, honest, rng):
        r = self.mismatch
        g = np.minimum(2.0 / (1.0 + r), 1.0 / state.eta_ch)
        eta = state.eta * g
        # shift favouring D0: D1 efficiency scaled by r; favouring D1: D0 scaled by r
        p_fav0 = outcome_probs_poisson(state.mu, eta, state.e_z, state.e_x, state.pd0, state.pd1,
                                       state.kappa * r)
        p_fav1 = outcome_probs_poisson(state.mu, eta * r, state.e_z, state.e_x, state.pd0,
                                       state.pd1, state.kappa / r)
        return 0.5 * (p_fav0 + p_fav1)


class TrojanHorse(Attack):
    name = "trojan_horse"

    def __init__(self, onset, fraction=1.0, backscatter=0.0, mu_leak=1e-6):
        super().__init__(onset, fraction)
        self.backscatter = float(backscatter)
        self.mu_leak = float(mu_leak)

    def attacked_probs(self, state, honest, rng):
        from qtrace.twin.simulator import add_background
        q = np.full(len(state.t), self.backscatter)
        return add_background(honest, q, q)


class RNGBias(Attack):
    name = "rng_bias"

    def __init__(self, onset, fraction=1.0, delta=5e-4, period=40.0, phase=0.0):
        super().__init__(onset, fraction)
        self.delta, self.period, self.phase = float(delta), float(period), float(phase)

    def modify_settings(self, state, set_p):
        p_a1 = 0.5 + self.delta * np.sin(2 * np.pi * state.t / self.period + self.phase)
        return setting_probabilities(state.p_k, p_a1, state.p_alpha_z, state.p_beta_z)

    def perturbation(self, state, honest, set_p, rng):
        return np.zeros_like(honest), self.modify_settings(state, set_p) - set_p


class Composite(Attack):
    def __init__(self, attacks):
        onset = min(a.onset for a in attacks)
        super().__init__(onset, 1.0)
        self.attacks = list(attacks)
        self.name = "+".join(a.name for a in attacks)

    def perturbation(self, state, honest, set_p, rng):
        dp = np.zeros_like(honest)
        ds = np.zeros_like(set_p)
        for a in self.attacks:
            m = (state.t >= a.onset)
            p, s = a.perturbation(state, honest, set_p, rng)
            dp += p * m[:, None, None, None, None, None]
            ds += s * m[:, None, None, None, None]
        return dp, ds

    def describe(self):
        return {"attack": self.name, "components": [a.describe() for a in self.attacks]}


REGISTRY = {
    "intercept_resend": InterceptResend,
    "pns": PhotonNumberSplitting,
    "blinding": Blinding,
    "time_shift": TimeShift,
    "trojan_horse": TrojanHorse,
    "rng_bias": RNGBias,
}


def make_attack(name: str, **kwargs) -> Attack:
    if name not in REGISTRY:
        raise KeyError(f"unknown attack {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
