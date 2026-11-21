"""Baseline detectors evaluated under the same protocol as Q-TRACE.

All baselines receive the same information as Q-TRACE: the scalar
telemetry of :func:`qtrace.physics.decoy.summarise_counts`, standardised
per link against that link's own commissioning windows, augmented with a
trailing mean over the same horizon Q-TRACE uses. Every baseline's alarm
threshold is set on the same attack-free validation links for the same
per-window false-alarm rate.

Industry monitors
    ``qber_static``  : QBER of signal Z-basis pulses (static 11 % cut-off).
    ``qber_cusum``   : one-sided CUSUM on the standardised signal QBER.
    ``decoy_cusum``  : two-sided CUSUM on the decoy-state estimates
                       (Y1 lower bound, e1 upper bound, decoy/signal gain
                       ratio): the decoy-state consistency check most
                       deployed systems perform, made sequential.
Unsupervised (honest data only)
    Isolation Forest, one-class SVM, MLP autoencoder, and a one-class SVM
    on a fidelity quantum kernel.
Supervised
    XGBoost trained on the known attack families.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from qtrace.physics.decoy import TELEMETRY_FEATURES

TREND_HORIZON = 16


def link_features(summary: dict, n_cal: int, horizon: int = TREND_HORIZON):
    """Per-link standardised telemetry for monitoring windows.

    Returns ``(X, raw)`` where ``X`` has ``2 * n_features`` columns
    (current window and trailing mean) and ``raw`` holds the
    unstandardised features of the monitoring windows.
    """
    f = np.column_stack([summary[k] for k in TELEMETRY_FEATURES]).astype(float)
    mu = f[:n_cal].mean(axis=0)
    sd = f[:n_cal].std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12 * (np.abs(mu) + 1e-30), sd, 1.0)
    z = (f - mu) / sd
    zc = z[n_cal:]
    cs = np.vstack([np.zeros((1, zc.shape[1])), np.cumsum(zc, axis=0)])
    idx = np.arange(1, len(zc) + 1)
    m = np.minimum(idx, horizon)
    trend = (cs[idx] - cs[idx - m]) / m[:, None]
    return np.hstack([zc, trend]), f[n_cal:], z[:n_cal]


def _cusum(x, k=0.5):
    s = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = max(0.0, acc + v - k)
        s[i] = acc
    return s


def industry_scores(summary: dict, n_cal: int):
    """Scores of the three industry monitors for the monitoring windows."""
    idx = {k: i for i, k in enumerate(TELEMETRY_FEATURES)}
    f = np.column_stack([summary[k] for k in TELEMETRY_FEATURES]).astype(float)
    mu = f[:n_cal].mean(axis=0)
    sd = np.maximum(f[:n_cal].std(axis=0, ddof=1), 1e-12)
    z = ((f - mu) / sd)[n_cal:]
    q = f[n_cal:, idx["qber_s_Z"]]
    cus_q = _cusum(z[:, idx["qber_s_Z"]])
    dec = np.vstack([_cusum(-z[:, idx["y1_l"]]), _cusum(z[:, idx["e1_u"]]),
                     _cusum(z[:, idx["gain_ratio_d_s"]]), _cusum(-z[:, idx["gain_ratio_d_s"]])])
    return {"QBER static": q, "QBER CUSUM": cus_q, "Decoy CUSUM": dec.max(axis=0)}


class Autoencoder:
    def __init__(self, hidden=32, bottleneck=6, seed=0):
        self.model = MLPRegressor(hidden_layer_sizes=(hidden, bottleneck, hidden),
                                  max_iter=400, early_stopping=True, random_state=seed)

    def fit(self, X):
        self.model.fit(X, X)
        return self

    def score(self, X):
        return np.mean((X - self.model.predict(X)) ** 2, axis=1)


# ----------------------------------------------------------------- quantum kernel
def zz_statevectors(x, reps=2):
    """Exact statevectors of the linear-entanglement ZZ feature map.

    Matches ``qiskit.circuit.library.zz_feature_map(n, reps, entanglement="linear")``
    (verified in ``tests/test_quantum_kernel.py``). Each repetition applies
    ``H^{(x)n}`` followed by the diagonal phase
    ``exp(i [sum_i 2 x_i z_i + sum_{i} 2 (pi - x_i)(pi - x_{i+1}) (z_i xor z_{i+1})])``.
    """
    x = np.atleast_2d(x)
    n_s, n = x.shape
    dim = 2 ** n
    idx = np.arange(dim)
    z = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)    # (dim, n)
    phase = 2 * x @ z.T                                                 # (n_s, dim)
    for i in range(n - 1):
        xor = np.abs(z[:, i] - z[:, i + 1])
        phase += 2 * np.outer((np.pi - x[:, i]) * (np.pi - x[:, i + 1]), xor)
    diag = np.exp(1j * phase)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hn = h
    for _ in range(n - 1):
        hn = np.kron(hn, h)
    psi = np.zeros((n_s, dim), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = psi @ hn.T
        psi = psi * diag
    return psi


class QuantumKernelOCSVM:
    """One-class SVM on the fidelity kernel ``|<phi(x)|phi(x')>|^2``."""

    def __init__(self, n_qubits=6, reps=2, nu=0.05, n_train=400, seed=0):
        self.n_qubits, self.reps, self.nu, self.n_train = n_qubits, reps, nu, n_train
        self.seed = seed

    def _encode(self, X):
        z = self.pca.transform(X) / self.angle_scale
        return np.clip(z, -1, 1) * np.pi

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.pca = PCA(n_components=self.n_qubits, random_state=self.seed).fit(X)
        z = self.pca.transform(X)
        self.angle_scale = np.maximum(np.quantile(np.abs(z), 0.99, axis=0), 1e-9)
        sel = rng.choice(len(X), size=min(self.n_train, len(X)), replace=False)
        self.psi_train = zz_statevectors(self._encode(X[sel]), self.reps)
        k = np.abs(self.psi_train.conj() @ self.psi_train.T) ** 2
        self.svm = OneClassSVM(kernel="precomputed", nu=self.nu).fit(k)
        return self

    def score(self, X, batch=4096):
        out = []
        for i in range(0, len(X), batch):
            psi = zz_statevectors(self._encode(X[i:i + batch]), self.reps)
            k = np.abs(psi.conj() @ self.psi_train.T) ** 2
            out.append(-self.svm.decision_function(k))
        return np.concatenate(out)


# ------------------------------------------------------------------ registry
class UnsupervisedSuite:
    """Fits every honest-data-only baseline on pooled commissioning features."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X_honest):
        rng = np.random.default_rng(self.seed)
        sub = X_honest[rng.choice(len(X_honest), size=min(4000, len(X_honest)), replace=False)]
        self.models = {
            "Isolation Forest": IsolationForest(n_estimators=300, random_state=self.seed).fit(X_honest),
            "One-class SVM": OneClassSVM(nu=0.05, gamma="scale").fit(sub),
            "Autoencoder": Autoencoder(seed=self.seed).fit(X_honest),
            "Quantum-kernel OCSVM": QuantumKernelOCSVM(seed=self.seed).fit(X_honest),
        }
        return self

    def score(self, X):
        out = {}
        for name, m in self.models.items():
            if name == "Isolation Forest":
                out[name] = -m.score_samples(X)
            elif name == "One-class SVM":
                out[name] = -m.decision_function(X)
            else:
                out[name] = m.score(X)
        return out


class SupervisedXGB:
    def __init__(self, seed=0):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric="logloss", random_state=seed, n_jobs=1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]
