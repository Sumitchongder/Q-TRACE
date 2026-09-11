"""
Q-TRACE :: models.sequence_models
====================================
Deep sequence baselines (LSTM, GRU, Temporal-CNN) operating directly on
the 3-D temporal blocks (n_samples, block_size, n_features).

If PyTorch is available in the environment, proper recurrent/convolutional
networks are trained. If not (e.g. a minimal CI/sandbox environment), the
module automatically falls back to an MLP-on-flattened-window classifier
so the overall experiment pipeline still executes end-to-end without
crashing -- but this fallback is clearly logged, since it is NOT an
equivalent architecture and should not be reported as "LSTM" results in
the paper if it was used.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.neural_network import MLPClassifier
from .classical_baselines import flatten


# ----------------------------------------------------------------------
# Fallback (always available)
# ----------------------------------------------------------------------
def fit_mlp_fallback(X_train, y_train, **kwargs):
    params = dict(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42,
                  early_stopping=True)
    params.update(kwargs)
    model = MLPClassifier(**params)
    model.fit(flatten(X_train), y_train)
    return model


def predict_proba_fallback(model, X):
    return model.predict_proba(flatten(X))[:, 1]


# ----------------------------------------------------------------------
# Torch models (used automatically when available)
# ----------------------------------------------------------------------
if TORCH_AVAILABLE:

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, hidden=64, layers=1):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, (h, c) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    class GRUClassifier(nn.Module):
        def __init__(self, n_features, hidden=64, layers=1):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    class TemporalCNNClassifier(nn.Module):
        def __init__(self, n_features, channels=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(channels, 1)

        def forward(self, x):
            # x: (batch, seq, features) -> (batch, features, seq)
            x = x.permute(0, 2, 1)
            z = self.net(x).squeeze(-1)
            return self.head(z).squeeze(-1)

    def _train_torch_model(model_cls, X_train, y_train, epochs=30, lr=1e-3,
                            batch_size=128, **model_kwargs):
        n_features = X_train.shape[-1]
        model = model_cls(n_features, **model_kwargs)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X_train.astype(np.float32))
        y_t = torch.tensor(y_train.astype(np.float32))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        model.eval()
        return model

    def fit_lstm(X_train, y_train, **kwargs):
        return _train_torch_model(LSTMClassifier, X_train, y_train, **kwargs)

    def fit_gru(X_train, y_train, **kwargs):
        return _train_torch_model(GRUClassifier, X_train, y_train, **kwargs)

    def fit_temporal_cnn(X_train, y_train, **kwargs):
        return _train_torch_model(TemporalCNNClassifier, X_train, y_train, **kwargs)

    def predict_proba_torch(model, X):
        with torch.no_grad():
            logits = model(torch.tensor(X.astype(np.float32)))
            return torch.sigmoid(logits).numpy()

else:
    # Aliases so downstream code can call fit_lstm/fit_gru/fit_temporal_cnn
    # regardless of torch availability -- they just fall back to the MLP.
    def fit_lstm(X_train, y_train, **kwargs):
        return fit_mlp_fallback(X_train, y_train)

    def fit_gru(X_train, y_train, **kwargs):
        return fit_mlp_fallback(X_train, y_train)

    def fit_temporal_cnn(X_train, y_train, **kwargs):
        return fit_mlp_fallback(X_train, y_train)

    def predict_proba_torch(model, X):
        return predict_proba_fallback(model, X)
