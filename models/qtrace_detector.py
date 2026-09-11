"""
Q-TRACE :: models.qtrace_detector
====================================
The full Q-TRACE detector (Sections 12-21): combines

    A_t : anomaly score       (autoencoder reconstruction error, secure-manifold distance)
    P_t : physics inconsistency (evaluation.metrics.PhysicsConsistencyModel,
          fit once on healthy training data)
    T_t : temporal deviation  (LSTM/sequence-model anomaly score, or windowed variance fallback)
    U_t : model uncertainty   (disagreement across an ensemble of component detectors)

into the composite risk score R_t (evaluation.metrics.qtrace_risk_score),
and supports component ablation ("Q-TRACE - physics", "- quantum", etc,
Section 38) by simply zeroing the corresponding weight.
"""

import numpy as np
from models.anomaly_baselines import SklearnAutoencoder
from evaluation.metrics import qtrace_risk_score, minmax, PhysicsConsistencyModel


class QTraceDetector:
    def __init__(self, use_physics=True, use_quantum=True, use_temporal=True,
                 use_uncertainty=True, quantum_module=None, quantum_reducer=None,
                 n_qubits=4):
        self.use_physics = use_physics
        self.use_quantum = use_quantum
        self.use_temporal = use_temporal
        self.use_uncertainty = use_uncertainty
        self.quantum_module = quantum_module   # fitted quantum-kernel dict, or None
        self.quantum_reducer = quantum_reducer
        self.n_qubits = n_qubits
        self.autoencoder = None
        self.physics_model = None

    def fit(self, X_train_normal, df_train_normal=None):
        """
        df_train_normal : the raw per-row telemetry DataFrame (channel_loss_db,
            detector_efficiency, mu_signal, mu_decoy, delta_Y, delta_E, ...),
            row-aligned with X_train_normal, restricted to HEALTHY ("normal")
            rows only. Required when use_physics=True -- see
            evaluation.metrics.PhysicsConsistencyModel for why this must be
            fit once on training-normal data rather than recomputed on
            whatever split is later scored.
        """
        self.autoencoder = SklearnAutoencoder(max_iter=300).fit(X_train_normal)
        if self.use_physics:
            if df_train_normal is None:
                raise ValueError(
                    "QTraceDetector(use_physics=True).fit() requires "
                    "df_train_normal (the aligned healthy-only telemetry rows) "
                    "to fit the physics-consistency model. Pass use_physics=False "
                    "if you don't have it, or see evaluation.data_utils.align_raw_df."
                )
            self.physics_model = PhysicsConsistencyModel().fit(df_train_normal)
        return self

    def _temporal_deviation(self, X):
        """Fallback temporal-deviation signal: rolling variance across the
        temporal block dimension, per sample (cheap, dependency-free proxy
        for a trained sequence model's hidden-state drift)."""
        return X.var(axis=1).mean(axis=1)

    def score(self, X, df_meta, quantum_proba_lookup=None):
        """
        Returns the composite risk score R_t in [0, 1] for each row of X.

        quantum_proba_lookup : optional 1-D array aligned with X giving
        precomputed quantum-kernel probabilities (since quantum scoring is
        expensive, it is typically computed once for a subsample and
        passed in here rather than recomputed per ablation variant).
        """
        A = minmax(self.autoencoder.anomaly_score(X))

        if self.use_physics:
            P = minmax(self.physics_model.score(df_meta))
        else:
            P = np.zeros(len(X))

        if self.use_temporal:
            T = minmax(self._temporal_deviation(X))
        else:
            T = np.zeros(len(X))

        if self.use_quantum and quantum_proba_lookup is not None:
            Q = minmax(quantum_proba_lookup)
        else:
            Q = np.zeros(len(X))

        if self.use_uncertainty:
            # ensemble disagreement proxy: spread across the active signals
            stacked = np.vstack([s for s, active in
                                  [(A, True), (P, self.use_physics),
                                   (T, self.use_temporal), (Q, self.use_quantum)]
                                  if active])
            U = minmax(stacked.std(axis=0)) if stacked.shape[0] > 1 else np.zeros(len(X))
        else:
            U = np.zeros(len(X))

        # fold quantum score into the "anomaly" channel of the weighted sum
        # when present, else weight collapses gracefully to the ablation.
        combined_anomaly = A if not self.use_quantum else minmax(0.5 * A + 0.5 * Q)

        R = qtrace_risk_score(combined_anomaly, P, T, U)
        return R
