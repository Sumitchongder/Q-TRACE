"""
Q-TRACE :: models.classical_baselines
========================================
Tree-based and kernel classical baselines: XGBoost, Random Forest, SVM.
These operate on the *flattened* temporal-block features
(block_size * n_features per sample) so they see the same temporal
context as the sequence/quantum models, keeping the comparison fair.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def flatten(X):
    """(n, block_size, n_features) -> (n, block_size*n_features)."""
    return X.reshape(X.shape[0], -1)


def fit_xgboost(X_train, y_train, **kwargs):
    params = dict(n_estimators=300, max_depth=5, learning_rate=0.08,
                  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                  n_jobs=-1, random_state=42)
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    model.fit(flatten(X_train), y_train)
    return model


def fit_random_forest(X_train, y_train, **kwargs):
    params = dict(n_estimators=400, max_depth=10, n_jobs=-1, random_state=42)
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(flatten(X_train), y_train)
    return model


def fit_svm(X_train, y_train, subsample=4000, **kwargs):
    """SVM with RBF kernel; optionally subsample for tractable training
    time on large temporal-block feature vectors."""
    Xf = flatten(X_train)
    if len(Xf) > subsample:
        idx = np.random.default_rng(42).choice(len(Xf), subsample, replace=False)
        Xf, y_train = Xf[idx], y_train[idx]
    params = dict(kernel="rbf", C=4.0, gamma="scale", probability=True, random_state=42)
    params.update(kwargs)
    model = SVC(**params)
    model.fit(Xf, y_train)
    return model


def fit_logistic_regression(X_train, y_train, **kwargs):
    params = dict(max_iter=2000, C=1.0, random_state=42)
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(flatten(X_train), y_train)
    return model


def predict_proba(model, X):
    Xf = flatten(X)
    proba = model.predict_proba(Xf)
    return proba[:, 1]
