"""
Q-TRACE :: quantum.quantum_kernel
====================================
Fidelity quantum kernel (Section 15): encodes the PCA-reduced telemetry
into a ZZ-feature-map circuit, computes pairwise state-fidelity kernel
values, and feeds the resulting Gram matrix to a classical SVM decision
layer. Runs on Qiskit's AerSimulator by default; the same feature map is
reused unmodified by the IBM Runtime script for real-hardware execution
(quantum/ibm_runtime_job.py), which is the key reproducibility property
this module was designed around.
"""

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from sklearn.svm import SVC


def build_feature_map(n_qubits: int, reps: int = 2, entanglement: str = "linear"):
    """ZZ feature map: standard, hardware-efficient encoding used
    throughout the quantum-kernel QML literature."""
    return zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)


def build_quantum_kernel(n_qubits: int, reps: int = 2, sampler=None,
                          max_circuits_per_job: int = 300):
    """
    Build a FidelityQuantumKernel using the ComputeUncompute fidelity
    primitive over a StatevectorSampler by default (fast, exact simulation
    -- appropriate for the classical-vs-quantum comparison table). Pass a
    different `sampler` (e.g. an AerSampler with a noise model, or an IBM
    Runtime SamplerV2) to reuse this identical circuit construction on
    noisy or real hardware.
    """
    fmap = build_feature_map(n_qubits, reps=reps)
    if sampler is None:
        sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity,
                                    max_circuits_per_job=max_circuits_per_job)
    return kernel


def fit_quantum_kernel_svm(X_train_reduced, y_train, n_qubits, reps=2,
                            sampler=None, C=2.0, subsample=250):
    """
    Fit an SVM on the precomputed quantum kernel Gram matrix.

    NOTE on scale: quantum kernel evaluation costs O(n^2) circuit
    executions. `subsample` keeps this tractable on a simulator; for the
    real hardware run, quantum/ibm_runtime_job.py uses a much smaller,
    fixed-size batch chosen to fit an 8-minute QPU budget.
    """
    if len(X_train_reduced) > subsample:
        idx = np.random.default_rng(42).choice(len(X_train_reduced), subsample, replace=False)
        X_train_reduced = X_train_reduced[idx]
        y_train = y_train[idx]

    kernel = build_quantum_kernel(n_qubits, reps=reps, sampler=sampler)
    K_train = kernel.evaluate(x_vec=X_train_reduced)

    svm = SVC(kernel="precomputed", C=C, probability=True, random_state=42)  # sklearn>=1.9: see CHANGELOG note in README re: probability deprecation
    svm.fit(K_train, y_train)
    return {"svm": svm, "kernel": kernel, "X_train_reduced": X_train_reduced}


def predict_proba_quantum_kernel(fitted, X_test_reduced, subsample=250):
    if len(X_test_reduced) > subsample:
        idx = np.random.default_rng(0).choice(len(X_test_reduced), subsample, replace=False)
        X_test_reduced = X_test_reduced[idx]
        keep_idx = idx
    else:
        keep_idx = np.arange(len(X_test_reduced))

    kernel = fitted["kernel"]
    K_test = kernel.evaluate(x_vec=X_test_reduced, y_vec=fitted["X_train_reduced"])
    proba = fitted["svm"].predict_proba(K_test)[:, 1]
    return proba, keep_idx
