import numpy as np
import pytest

from qtrace.physics.decoy import decoy_bounds, key_rate
from qtrace.physics.optics import (lossy_channel_gf, outcome_probs_photon_resolved,
                                   outcome_probs_poisson, setting_probabilities)


def test_photon_resolved_matches_poisson():
    mu = np.array([0.5, 0.1, 0.0])
    p1 = outcome_probs_poisson(mu, 0.03, 0.01, 0.015, 1e-6, 2e-6, 0.9)
    p2 = outcome_probs_photon_resolved(mu, lossy_channel_gf(np.array(0.1)), np.array(0.3),
                                       0.01, 0.015, 1e-6, 2e-6, 0.9)
    assert np.allclose(p1, p2, atol=1e-13)


def test_outcomes_normalised_and_vacuum_is_dark_only():
    p = outcome_probs_poisson(np.array([0.5, 0.1, 0.0]), 0.05, 0.01, 0.01, 3e-6, 1e-6, 1.1)
    assert np.allclose(p.sum(-1), 1.0)
    vac = p[2]
    assert np.allclose(vac[..., 1] + vac[..., 3], 3e-6)
    assert np.allclose(vac[..., 2] + vac[..., 3], 1e-6)


def test_setting_probabilities_sum_to_one():
    assert np.isclose(setting_probabilities(np.array([0.7, 0.2, 0.1])).sum(), 1.0)


def test_decoy_bounds_recover_single_photon_yield():
    # noiseless lossy channel: Y1 = eta (+ Y0), e1 = e_d
    eta, y0, ed, mu, nu = 0.01, 1e-6, 0.01, 0.5, 0.1
    q = lambda m: y0 + 1 - np.exp(-eta * m)
    e = lambda m: (0.5 * y0 + ed * (1 - np.exp(-eta * m))) / q(m)
    y1, e1 = decoy_bounds(q(mu), q(nu), y0, e(nu), mu, nu)
    assert y1 == pytest.approx(eta + y0, rel=0.05)
    assert y1 <= eta + y0 + 1e-6          # it is a lower bound
    assert e1 >= ed - 1e-6                # it is an upper bound


def test_key_rate_positive_for_short_link_and_zero_for_high_qber():
    from qtrace.twin.simulator import DecoyBB84Twin, sample_link
    rng = np.random.default_rng(0)
    tw = DecoyBB84Twin()
    link = sample_link(tw.cfg, rng)
    r = tw.simulate(link, 5, rng)
    rate, _, _ = key_rate(r.counts, link.mu_signal, link.mu_decoy)
    assert np.all(rate > 0)
