"""
Q-TRACE :: quantum.feature_reduction
=======================================
Reduces the high-dimensional temporal-block feature vector down to a
small number of qubits (4-8, per Section 16 of the research plan) via
PCA, fit ONLY on the training split to avoid leakage. This dimensionality
reduction step is what makes the quantum circuits tractable both on
simulators and on real superconducting hardware within a small QPU time
budget.
"""

import numpy as np
from sklearn.decomposition import PCA


class QuantumFeatureReducer:
    def __init__(self, n_qubits=6, whiten=True):
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits, whiten=whiten, random_state=42)
        self._angle_scale = None

    def fit(self, X_flat_train):
        z = self.pca.fit_transform(X_flat_train)
        # scale PCA components into a sensible rotation-angle range [-pi, pi]
        self._angle_scale = np.percentile(np.abs(z), 95, axis=0)
        self._angle_scale[self._angle_scale < 1e-9] = 1.0
        return self

    def transform(self, X_flat):
        z = self.pca.transform(X_flat)
        z = np.clip(z / self._angle_scale, -1.0, 1.0) * np.pi
        return z.astype(np.float64)

    def fit_transform(self, X_flat_train):
        return self.fit(X_flat_train).transform(X_flat_train)
