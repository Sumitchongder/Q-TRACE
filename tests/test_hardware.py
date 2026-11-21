import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qtrace.hardware import analysis as A
from qtrace.hardware.circuits import build_circuit, select_links, transpile_design
from qtrace.hardware.design import ExperimentDesign, build_design
from qtrace.hardware.runner import bitstrings_to_array


def test_schedule_is_balanced_and_links_disjoint():
    be = FakeMarrakesh()
    pairs, _ = select_links(be, 40)
    flat = [q for p in pairs for q in p]
    assert len(flat) == len(set(flat))
    d = build_design(n_links=len(pairs))
    s = np.array(d.schedule)
    assert ((s >= 0).sum(axis=0) == 4).all()                  # 4 attacked epochs per link
    counts = np.bincount(s[s >= 0], minlength=len(d.probes))
    assert counts.max() - counts.min() <= 2
    for link in range(d.n_links):                              # no repeated probe on a link
        used = s[:, link][s[:, link] >= 0]
        assert len(used) == len(set(used))


def test_design_roundtrip_and_transpile_without_swaps():
    be = FakeMarrakesh()
    pairs, _ = select_links(be, 6)
    d = build_design(n_links=len(pairs))
    d.physical_pairs = pairs
    d2 = ExperimentDesign.from_json(d.to_json())
    assert d2.schedule == d.schedule and d2.physical_pairs == d.physical_pairs
    isa = transpile_design(d, be, circuits=[build_circuit(d, c) for c in (0, 20, 40)])
    assert all("swap" not in c.count_ops() for c in isa)


def test_bitstring_conversion_order():
    arr = bitstrings_to_array(["001", "100"], 3)
    assert arr.tolist() == [[1, 0, 0], [0, 0, 1]]


def test_analysis_on_ideal_synthetic_data():
    """Ideal noiseless links: probes at theta=pi/2 must be detected, honest not."""
    rng = np.random.default_rng(0)
    d = build_design(n_links=20, seed=1)
    nc, shots = d.n_circuits, 1000
    bob = np.zeros((nc, shots, d.n_links), dtype=np.uint8)
    for c in range(nc):
        for i in range(d.n_links):
            a, al, be = d.setting(c, i)
            p = d.probe(c, i)
            flip = 0.01
            if al != be:
                bob[c, :, i] = rng.integers(0, 2, shots)
                continue
            if p.kind == "probe" and d.eve_basis[c][i] != al:
                flip = flip + np.sin(p.strength / 2) ** 2 * (1 - 2 * flip)
            bob[c, :, i] = a ^ (rng.random(shots) < flip)
    n1, n = A.count_tables(bob, d)
    tests, _ = A.link_epoch_tests(n1, n, d)
    det = A.detection_table(tests)
    ir = det[det.probe.str.contains("1.571")]
    assert ir["Q-TRACE@0.001"].iloc[0] == 1.0
    assert det[det.kind == "honest"]["Q-TRACE@0.001"].iloc[0] <= 0.05
