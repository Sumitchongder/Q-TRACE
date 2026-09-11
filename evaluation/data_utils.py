"""
Q-TRACE :: evaluation.data_utils
==================================
Shared utilities for loading the QKD-TRACE dataset and preparing
(features, labels) for both classical and quantum models. Centralizing
this avoids subtle train/test mismatches between baselines, which would
invalidate any head-to-head comparison table.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "QBER_signal", "QBER_decoy", "Y_signal", "Y_decoy", "delta_Y", "delta_E",
    "detection_rate_signal", "detection_rate_decoy", "entropy",
    "channel_loss_db", "detector_efficiency", "dark_count_rate",
    "timing_jitter_ps", "phase_noise_rad", "Y1_lb", "e1_ub", "Q1_lb", "skr_est",
]


def load_telemetry(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    """1 = any attack present, 0 = normal (healthy) operation."""
    return (df["attack_family"] != "normal").astype(int).to_numpy()


def split_frames(df: pd.DataFrame):
    """Return the five canonical splits as separate DataFrames."""
    return {
        "train": df[df.split == "train"].reset_index(drop=True),
        "val": df[df.split == "val"].reset_index(drop=True),
        "test_zeroday": df[df.split == "test_zeroday"].reset_index(drop=True),
        "test_physical_shift": df[df.split == "test_physical_shift"].reset_index(drop=True),
        "test_composite": df[df.split == "test_composite"].reset_index(drop=True),
    }


def align_raw_df(df_raw, meta):
    """
    Re-index a raw telemetry DataFrame (e.g. the `train`/`test_zeroday`
    frames returned by split_frames) to match the row order of a
    windowed-sample metadata frame produced by make_windows(). Both share
    (run_id, t) as a natural key. This is needed anywhere a component
    (e.g. the physics-consistency model) needs the ORIGINAL per-row
    physical columns (delta_Y, QBER_signal, channel_loss_db, ...) aligned
    1:1 with a windowed X/y array, rather than the windowed features
    themselves.
    """
    return (df_raw.set_index(["run_id", "t"])
            .loc[list(zip(meta.run_id, meta.t))]
            .reset_index())
    """Minimal, dependency-free standard scaler (fit on train only) so
    every model (classical and quantum) uses an identical, leakage-free
    normalization pipeline."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_windows(df: pd.DataFrame, block_size: int, feature_cols=FEATURE_COLS):
    """
    Build flattened temporal-block features per run (Section 8), returned
    as a 2-D array of shape (n_samples, block_size * n_features) so any
    scikit-learn-style model can consume them directly; sequence models
    (LSTM) reshape back to (n_samples, block_size, n_features).
    """
    X_list, y_list, meta = [], [], []
    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("t")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        labels = (g["attack_family"] != "normal").astype(int).to_numpy()
        padded = np.pad(arr, ((block_size - 1, 0), (0, 0)), mode="edge")
        for i in range(len(arr)):
            window = padded[i:i + block_size]
            X_list.append(window)
            y_list.append(labels[i])
            meta.append({
                "run_id": run_id, "t": g["t"].iloc[i],
                "attack_family": g["attack_family"].iloc[i],
                "zero_day": g["zero_day"].iloc[i] if "zero_day" in g.columns else False,
                "physical_regime": g["physical_regime"].iloc[i] if "physical_regime" in g.columns else "nominal",
                "split": g["split"].iloc[i] if "split" in g.columns else "unspecified",
            })
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    meta = pd.DataFrame(meta)
    return X, y, meta
