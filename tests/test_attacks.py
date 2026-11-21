import numpy as np

from qtrace.attacks.models import (Blinding, InterceptResend, PhotonNumberSplitting, RNGBias,
                                   TimeShift, TrojanHorse)
from qtrace.detect.detectability import detectability_exponent, mean_state
from qtrace.physics.decoy import gains_and_qbers
from qtrace.twin.simulator import DecoyBB84Twin, sample_link


def _setup(seed=1):
    rng = np.random.default_rng(seed)
    tw = DecoyBB84Twin()
    link = sample_link(tw.cfg, rng)
    st = tw.window_states(link, 20, rng)
    return rng, link, st


def _signal_gain(p):
    return 1 - p[:, 0, ..., 0].mean(axis=(-3, -2, -1))


def test_pns_and_blinding_preserve_signal_gain():
    rng, link, st = _setup()
    h = st.honest_probs()
    for atk in (PhotonNumberSplitting(0, 1.0), Blinding(0, 1.0)):
        p = atk.attacked_probs(st, h, rng)
        assert np.allclose(_signal_gain(p), _signal_gain(h), rtol=1e-3)


def test_blinding_removes_double_clicks_and_vacuum_clicks():
    rng, link, st = _setup()
    p = Blinding(0, 1.0).attacked_probs(st, st.honest_probs(), rng)
    assert np.allclose(p[..., 3], 0)
    assert np.allclose(p[:, 2, ..., 0], 1)


def test_intercept_resend_raises_qber_by_quarter_of_fraction():
    rng, link, st = _setup()
    h = st.honest_probs()
    p, _, _ = InterceptResend(0, 0.2).apply(st, h.copy(), st.setting_probs(), rng)
    n = 1e9 * st.setting_probs()[..., None] * p
    _, q = gains_and_qbers(n)
    n0 = 1e9 * st.setting_probs()[..., None] * h
    _, q0 = gains_and_qbers(n0)
    assert np.allclose(q[:, 0, 0] - q0[:, 0, 0], 0.2 * (0.25 - q0[:, 0, 0] / 2), atol=2e-3)


def test_filtered_trojan_horse_is_telemetry_blind():
    rng, link, st = _setup()
    ms = mean_state(st)
    d0, _ = detectability_exponent(ms, link.afterpulse, TrojanHorse(0, 1.0, backscatter=0.0))
    d1, _ = detectability_exponent(ms, link.afterpulse, TrojanHorse(0, 1.0, backscatter=1e-6))
    assert d0 < 1e-15 and d1 > 0


def test_detectability_increases_with_strength():
    rng, link, st = _setup()
    ms = mean_state(st)
    ds = [detectability_exponent(ms, link.afterpulse, TimeShift(0, f, 0.25))[0]
          for f in (0.05, 0.2, 0.8)]
    assert ds[0] < ds[1] < ds[2]


def test_rng_bias_changes_only_settings():
    rng, link, st = _setup()
    h = st.honest_probs()
    p, s, _ = RNGBias(0, 1.0, delta=1e-3, period=8).apply(st, h.copy(), st.setting_probs(), rng)
    assert np.allclose(p, h)
    assert not np.allclose(s, st.setting_probs())
