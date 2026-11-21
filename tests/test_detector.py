import numpy as np

from qtrace.attacks.models import InterceptResend
from qtrace.detect.detector import QTraceDetector
from qtrace.detect.honest_model import fit_honest
from qtrace.twin.simulator import DecoyBB84Twin, sample_link


def test_calibration_recovers_device_parameters():
    rng = np.random.default_rng(3)
    tw = DecoyBB84Twin()
    link = sample_link(tw.cfg, rng)
    r = tw.simulate(link, 40, rng)
    p, info = fit_honest(r.counts, link.mu_signal, link.afterpulse, link.p_intensity)
    assert info["success"]
    assert abs(p.kappa / link.kappa - 1) < 0.02
    assert abs(p.rho / (link.mu_decoy / link.mu_signal) - 1) < 0.03
    assert abs(p.e_z / link.e_z - 1) < 0.15


def test_strong_attack_detected_quickly_and_honest_stays_quiet():
    rng = np.random.default_rng(4)
    tw = DecoyBB84Twin()
    det = QTraceDetector()
    link = sample_link(tw.cfg, rng)
    honest = tw.simulate(link, 120, rng)
    raw_h, *_ = det.score_run(honest.counts, 60, link.mu_signal, link.afterpulse)
    attacked = tw.simulate(link, 120, rng, attack=InterceptResend(90, 0.1))
    raw_a, *_ = det.score_run(attacked.counts, 60, link.mu_signal, link.afterpulse)
    assert raw_a[35:].max() > 5 * max(raw_h.max(), 1.0)
