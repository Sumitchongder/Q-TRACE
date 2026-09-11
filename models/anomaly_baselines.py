"""
Q-TRACE :: models.anomaly_baselines
======================================
Unsupervised anomaly-detection baselines trained ONLY on normal (healthy)
telemetry, matching the "secure manifold learning" philosophy of Q-TRACE
itself (Section 13), but without the physics-informed / quantum
machinery. This is the fairest possible unsupervised comparison group.

Includes a numpy/sklearn autoencoder (via MLPRegressor bottleneck trick)
so the pipeline runs without requiring PyTorch; if torch is available,
`TorchAutoencoder` is used automatically for a proper deep autoencoder
(see models/sequence_models.py for the torch-availability flag).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor

from .classical_baselines import flatten

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def fit_isolation_forest(X_train_normal, **kwargs):
    params = dict(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    params.update(kwargs)
    model = IsolationForest(**params)
    model.fit(flatten(X_train_normal))
    return model


def anomaly_score_isolation_forest(model, X):
    # sklearn: higher score_samples = more normal -> flip sign for "risk"
    return -model.score_samples(flatten(X))


def fit_one_class_svm(X_train_normal, subsample=4000, **kwargs):
    Xf = flatten(X_train_normal)
    if len(Xf) > subsample:
        idx = np.random.default_rng(42).choice(len(Xf), subsample, replace=False)
        Xf = Xf[idx]
    params = dict(kernel="rbf", nu=0.05, gamma="scale")
    params.update(kwargs)
    model = OneClassSVM(**params)
    model.fit(Xf)
    return model


def anomaly_score_one_class_svm(model, X):
    return -model.decision_function(flatten(X))


class SklearnAutoencoder:
    """MLP-based autoencoder (bottleneck MLPRegressor(X -> X)); a pragmatic
    dependency-light stand-in for a deep autoencoder. Anomaly score is the
    reconstruction MSE."""

    def __init__(self, bottleneck=8, hidden=64, max_iter=400, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, bottleneck, hidden),
            activation="relu", max_iter=max_iter, random_state=random_state,
            early_stopping=True, n_iter_no_change=15,
        )
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train_normal):
        Xf = flatten(X_train_normal)
        self.mean_ = Xf.mean(axis=0)
        self.std_ = Xf.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Xn = (Xf - self.mean_) / self.std_
        self.model.fit(Xn, Xn)
        return self

    def anomaly_score(self, X):
        Xf = flatten(X)
        Xn = (Xf - self.mean_) / self.std_
        recon = self.model.predict(Xn)
        return np.mean((Xn - recon) ** 2, axis=1)


if TORCH_AVAILABLE:
    class TorchAutoencoder(nn.Module):
        """Proper deep autoencoder used automatically when torch is
        installed (e.g. in the user's `qgss` conda environment)."""

        def __init__(self, input_dim, bottleneck=8, hidden=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden), nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    def fit_torch_autoencoder(X_train_normal, epochs=60, lr=1e-3, batch_size=128):
        Xf = flatten(X_train_normal).astype(np.float32)
        mean_, std_ = Xf.mean(0), Xf.std(0)
        std_[std_ < 1e-9] = 1.0
        Xn = (Xf - mean_) / std_

        model = TorchAutoencoder(Xn.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X_t = torch.tensor(Xn)

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                batch = X_t[idx]
                opt.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
        model.eval()
        return model, mean_, std_

    def torch_autoencoder_anomaly_score(model, mean_, std_, X):
        Xf = flatten(X).astype(np.float32)
        Xn = (Xf - mean_) / std_
        with torch.no_grad():
            recon = model(torch.tensor(Xn)).numpy()
        return np.mean((Xn - recon) ** 2, axis=1)
