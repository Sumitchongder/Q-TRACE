import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

from qtrace.baselines.detectors import zz_statevectors


def test_analytic_feature_map_matches_qiskit():
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.pi, np.pi, (4, 5))
    fm = zz_feature_map(5, reps=2, entanglement="linear")
    ref = np.array([Statevector(fm.assign_parameters(xi)).data for xi in x])
    fid = np.abs(np.sum(ref.conj() * zz_statevectors(x, 2), axis=1))
    assert np.allclose(fid, 1.0, atol=1e-10)
