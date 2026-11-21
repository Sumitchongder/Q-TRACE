"""Information-theoretic detectability of implementation attacks.

For an attack that changes the per-pulse distribution of settings and
outcomes from the honest family ``{P_eta}`` to ``Q``, the *detectability
exponent* is the information divergence from ``Q`` to the closest honest
explanation::

    D* = min_eta  KL(Q || P_eta)        [nats per pulse]

``D*`` is a property of the attack and the link alone, not of any
detector. Two consequences are used in the paper:

* **Universal delay bound.** For any sequential change detector whose mean
  time between false alarms is at least ``1 / alpha`` windows, the
  worst-case mean detection delay obeys (Lorden, Ann. Math. Stat. 42, 1897,
  1971; Lai, IEEE Trans. Inf. Theory 44, 2917, 1998)::

        delay >~ log(1 / alpha) / (N D*)      windows of N pulses.

  No classical, deep-learning or quantum model operating on the same
  telemetry can beat this bound.
* **Telemetry-blind attacks.** If ``D* = 0`` the attacked statistics are
  reproduced exactly by some honest link, and no telemetry-based monitor
  can detect the attack at any sample size. Such attacks must be excluded
  by device-level countermeasures, not by monitoring.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from qtrace.twin.simulator import WindowState, add_background, detector_click_rates


def mean_state(state: WindowState, sl=slice(None)) -> WindowState:
    """Collapse a window state to a single representative window."""
    f = lambda a: np.asarray(a[sl]).mean(axis=0, keepdims=True)
    return WindowState(mu=f(state.mu), eta_ch=f(state.eta_ch), eta_b=f(state.eta_b),
                       e_z=f(state.e_z), e_x=f(state.e_x), pd0=f(state.pd0), pd1=f(state.pd1),
                       kappa=f(state.kappa), p_k=f(state.p_k), p_a1=f(state.p_a1),
                       p_alpha_z=f(state.p_alpha_z), p_beta_z=f(state.p_beta_z),
                       t=np.array([10 ** 9]))


def _with_afterpulse(probs, set_p, afterpulse):
    r = detector_click_rates(probs, set_p)
    return add_background(probs, afterpulse * r[:, 0], afterpulse * r[:, 1])


def joint_distribution(state: WindowState, afterpulse: float, attack=None, rng=None):
    """Per-pulse joint distribution over (setting, outcome) for one window."""
    probs = state.honest_probs()
    set_p = state.setting_probs()
    if attack is not None:
        probs, set_p, _ = attack.apply(state, probs, set_p, rng or np.random.default_rng(0))
    probs = _with_afterpulse(probs, set_p, afterpulse)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    set_p = set_p / set_p.sum(axis=(-4, -3, -2, -1), keepdims=True)
    return set_p[..., None] * probs


def kl(q, p):
    q = np.clip(q, 1e-300, None)
    p = np.clip(p, 1e-300, None)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def detectability_exponent(state: WindowState, afterpulse: float, attack, rng=None,
                           t_values=None):
    """``D*`` in nats per pulse for ``attack`` on the link state ``state`` (one window).

    For attacks whose effect is modulated in time, pass the window indices
    ``t_values`` spanning one modulation period; the divergence is then
    averaged over them.
    """
    t_values = [10 ** 9] if t_values is None else list(t_values)
    qs = [joint_distribution(WindowState(**{**state.__dict__, "t": np.array([t])}),
                             afterpulse, attack, rng) for t in t_values]
    eta0 = float(state.eta_ch[0])

    def obj(log_scale):
        s = WindowState(**{**state.__dict__, "eta_ch": np.array([eta0 * np.exp(log_scale)])})
        p = joint_distribution(s, afterpulse)
        return float(np.mean([kl(q, p) for q in qs]))

    res = minimize_scalar(obj, bounds=(-1.5, 1.5), method="bounded",
                          options={"xatol": 1e-7})
    return max(float(res.fun), 0.0), float(eta0 * np.exp(res.x))


def lorden_delay_bound(d_star, pulses_per_window, alpha):
    """Lower bound on mean detection delay (windows) at false-alarm rate ``alpha``."""
    with np.errstate(divide="ignore"):
        return np.log(1.0 / alpha) / (np.asarray(d_star) * pulses_per_window)
