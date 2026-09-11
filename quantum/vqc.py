"""
Q-TRACE :: quantum.vqc
========================
Variational Quantum Classifier (Section 16): a small parameterized
circuit U(x, theta) = U_ansatz(theta) . U_feature_map(x), trained via a
classical optimizer against a cross-entropy loss on measurement outcomes.
Deliberately kept small (4-8 qubits, shallow ansatz) since the research
question is about representational quality under data/distribution
scarcity, not raw qubit count.
"""

import numpy as np
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA


def build_vqc(n_qubits: int, feature_reps: int = 1, ansatz_reps: int = 2,
              sampler=None, maxiter: int = 150, callback=None):
    fmap = zz_feature_map(feature_dimension=n_qubits, reps=feature_reps)
    ansatz = real_amplitudes(num_qubits=n_qubits, reps=ansatz_reps)
    if sampler is None:
        sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=maxiter)
    vqc = VQC(feature_map=fmap, ansatz=ansatz, optimizer=optimizer,
              sampler=sampler, callback=callback)
    return vqc


def fit_vqc(X_train_reduced, y_train, n_qubits, subsample=400, **kwargs):
    if len(X_train_reduced) > subsample:
        idx = np.random.default_rng(42).choice(len(X_train_reduced), subsample, replace=False)
        X_train_reduced = X_train_reduced[idx]
        y_train = y_train[idx]
    vqc = build_vqc(n_qubits, **kwargs)
    vqc.fit(X_train_reduced, y_train)
    return vqc


def predict_proba_vqc(vqc, X_reduced):
    # VQC exposes predict(); for a probability-like score we use the
    # underlying neural network's forward pass (softmax over 2 classes).
    probs = vqc.neural_network.forward(X_reduced, vqc.weights)
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    # fall back to hard predictions if only a single output is returned
    return vqc.predict(X_reduced).astype(float)
