"""
Q-TRACE :: evaluation.run_experiments
========================================
End-to-end experiment orchestration. Trains every baseline in Section 14
plus the quantum models (Section 15-16) plus the full Q-TRACE detector
(Sections 12-21), evaluates on every held-out split (known/E1-E2,
zero-day, physical-shift, compositional), and writes:

    results/comparison_table.csv       (Section 17's Table)
    results/skrr_table.csv             (Section 18)
    results/ablation_table.csv         (Section 38)
    figures/fig5_roc_zeroday.{png,pdf}
    figures/fig7_skrr.{png,pdf}
    figures/fig8_quantum_vs_classical.{png,pdf}

Run:
    python -m evaluation.run_experiments --data dataset/qkd_trace_v1/qkd_trace_telemetry.csv \
        --block-size 20 --n-qubits 4 --out results

For a fast correctness check (not for reported results), pass --smoke to
sharply reduce subsample sizes and VQC iterations.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from evaluation.data_utils import load_telemetry, make_windows, align_raw_df
from evaluation.metrics import (detection_metrics, secret_key_retention_rate,
                                 gate_key_distillation, gating_effectiveness,
                                 find_threshold_for_recall, minmax)
from evaluation.plotting import plot_roc_curves, plot_skrr_bars, plot_bar_comparison

from models.classical_baselines import (fit_xgboost, fit_random_forest, fit_svm,
                                         fit_logistic_regression, predict_proba, flatten)
from models.anomaly_baselines import (fit_isolation_forest, anomaly_score_isolation_forest,
                                       fit_one_class_svm, anomaly_score_one_class_svm,
                                       SklearnAutoencoder)
from models.sequence_models import fit_lstm, fit_gru, fit_temporal_cnn, predict_proba_torch, TORCH_AVAILABLE
from models.sequence_models import predict_proba_fallback

from quantum.feature_reduction import QuantumFeatureReducer
from quantum.quantum_kernel import fit_quantum_kernel_svm, predict_proba_quantum_kernel
from quantum.vqc import fit_vqc, predict_proba_vqc

from models.qtrace_detector import QTraceDetector


def _seq_predict(model, X):
    return predict_proba_torch(model, X) if TORCH_AVAILABLE else predict_proba_fallback(model, X)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = load_telemetry(args.data)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test_zd = df[df.split == "test_zeroday"]
    test_shift = df[df.split == "test_physical_shift"]
    test_comp = df[df.split == "test_composite"]

    bs = args.block_size
    X_train, y_train, meta_train = make_windows(train, block_size=bs)
    X_val, y_val, meta_val = make_windows(val, block_size=bs)
    X_zd, y_zd, meta_zd = make_windows(test_zd, block_size=bs)
    X_shift, y_shift, meta_shift = make_windows(test_shift, block_size=bs)
    X_comp, y_comp, meta_comp = make_windows(test_comp, block_size=bs)

    qsub = 20 if args.smoke else args.quantum_subsample
    vqc_iter = 10 if args.smoke else args.vqc_maxiter

    print(f"[data] train={X_train.shape} val={X_val.shape} zeroday={X_zd.shape} "
          f"shift={X_shift.shape} composite={X_comp.shape}")

    # ---------------- classical supervised baselines ----------------
    print("[models] fitting classical supervised baselines...")
    clf_models = {
        "XGBoost": fit_xgboost(X_train, y_train),
        "RandomForest": fit_random_forest(X_train, y_train),
        "LogisticRegression": fit_logistic_regression(X_train, y_train),
        "SVM": fit_svm(X_train, y_train, subsample=min(3000, len(X_train))),
    }
    seq_models = {
        "LSTM": fit_lstm(X_train, y_train),
        "GRU": fit_gru(X_train, y_train),
        "TemporalCNN": fit_temporal_cnn(X_train, y_train),
    }

    # ---------------- unsupervised anomaly baselines ----------------
    print("[models] fitting unsupervised anomaly baselines...")
    Xn = X_train[y_train == 0]
    # aligned raw telemetry rows for the healthy-only training samples --
    # needed to fit the physics-consistency model on TRAINING data only
    # (see evaluation.metrics.PhysicsConsistencyModel docstring for why
    # this must never be computed from the split being scored).
    aligned_train_df = align_raw_df(train, meta_train)
    aligned_train_normal_df = aligned_train_df[y_train == 0].reset_index(drop=True)
    iso = fit_isolation_forest(Xn)
    ocsvm = fit_one_class_svm(Xn, subsample=min(3000, len(Xn)))
    ae = SklearnAutoencoder(max_iter=300).fit(Xn)

    # ---------------- quantum models ----------------
    print("[models] fitting quantum kernel + VQC...")
    reducer = QuantumFeatureReducer(n_qubits=args.n_qubits)
    Xtr_red = reducer.fit_transform(flatten(X_train))

    qk_fitted = fit_quantum_kernel_svm(Xtr_red, y_train, n_qubits=args.n_qubits, subsample=qsub)
    vqc_model = fit_vqc(Xtr_red, y_train, n_qubits=args.n_qubits, subsample=qsub, maxiter=vqc_iter)

    # ---------------- Q-TRACE full detector ----------------
    print("[models] fitting Q-TRACE composite detector...")
    qtrace = QTraceDetector(n_qubits=args.n_qubits).fit(Xn, aligned_train_normal_df)

    # =====================================================================
    # Evaluation across splits
    # =====================================================================
    eval_splits = {
        "known (val)": (X_val, y_val, meta_val, val),
        "zero_day": (X_zd, y_zd, meta_zd, test_zd),
        "physical_shift": (X_shift, y_shift, meta_shift, test_shift),
        "compositional": (X_comp, y_comp, meta_comp, test_comp),
    }

    rows = []
    roc_curves_zeroday = {}
    q_lookup_by_split = {}  # keyed by split_name, filled below; used later for SKRR/ablation

    for split_name, (X, y, meta, df_raw) in eval_splits.items():
        preds = {}
        preds["XGBoost"] = predict_proba(clf_models["XGBoost"], X)
        preds["RandomForest"] = predict_proba(clf_models["RandomForest"], X)
        preds["LogisticRegression"] = predict_proba(clf_models["LogisticRegression"], X)
        preds["SVM"] = predict_proba(clf_models["SVM"], X)
        preds["LSTM"] = _seq_predict(seq_models["LSTM"], X)
        preds["GRU"] = _seq_predict(seq_models["GRU"], X)
        preds["TemporalCNN"] = _seq_predict(seq_models["TemporalCNN"], X)
        preds["IsolationForest"] = minmax(anomaly_score_isolation_forest(iso, X))
        preds["OneClassSVM"] = minmax(anomaly_score_one_class_svm(ocsvm, X))
        preds["Autoencoder"] = minmax(ae.anomaly_score(X))

        # quantum: evaluate on a fixed-size subsample for tractability.
        # IMPORTANT: both QuantumKernel and VQC must use an IDENTICAL,
        # RANDOMLY-drawn subsample -- NOT a naive X_red[:qsub] slice.
        # make_windows() groups rows contiguously by run_id, so a plain
        # slice of the first `qsub` rows can land entirely inside one or
        # two runs of the same class (this previously produced VQC AUROC
        # = NaN on every split, because y_true ended up single-class).
        X_red = reducer.transform(flatten(X))
        if len(X_red) > qsub:
            subsample_idx = np.random.default_rng(7).choice(len(X_red), qsub, replace=False)
        else:
            subsample_idx = np.arange(len(X_red))

        q_proba, keep_idx = predict_proba_quantum_kernel(
            qk_fitted, X_red[subsample_idx], subsample=qsub)
        keep_idx = subsample_idx[keep_idx]  # map back to original row indices
        preds["QuantumKernel"] = (q_proba, keep_idx)  # subsampled; handled specially below

        vqc_proba = predict_proba_vqc(vqc_model, X_red[subsample_idx])
        preds["VQC"] = (vqc_proba, subsample_idx)

        # Q-TRACE composite (reuses quantum kernel proba on the same subsample)
        q_lookup_full = np.full(len(X), np.nan)
        q_lookup_full[keep_idx] = q_proba
        # fill missing (non-subsampled) rows with the mean quantum score
        q_lookup_full[np.isnan(q_lookup_full)] = np.nanmean(q_proba)
        q_lookup_by_split[split_name] = q_lookup_full
        preds["Q-TRACE"] = qtrace.score(X, df_raw, quantum_proba_lookup=q_lookup_full)

        for name, p in preds.items():
            if isinstance(p, tuple):
                proba, idx = p
                yt = y[idx]
                m = detection_metrics(yt, proba)
            else:
                m = detection_metrics(y, p)
            m.update({"model": name, "split": split_name})
            rows.append(m)

            if split_name == "zero_day":
                yt = y[idx] if isinstance(p, tuple) else y
                score = p[0] if isinstance(p, tuple) else p
                if len(np.unique(yt)) > 1:
                    fpr, tpr, _ = roc_curve(yt, score)
                    roc_curves_zeroday[name] = (fpr, tpr, m["AUROC"])

    table = pd.DataFrame(rows)[["split", "model", "AUROC", "AUPRC", "F1", "FAR"]]
    table.to_csv(os.path.join(args.out, "comparison_table.csv"), index=False)
    print("\n=== Comparison table (all splits) ===")
    print(table.to_string(index=False))

    # ---------------- Figure 5: zero-day ROC curves ----------------
    plot_roc_curves(roc_curves_zeroday, "Zero-Day Detection ROC",
                     "figures/fig5_roc_zeroday")

    # ---------------- Figure 8: quantum vs classical bar ----------------
    zd_table = table[table.split == "zero_day"].set_index("model")
    quantum_vs_classical = {
        "AUROC": {m: zd_table.loc[m, "AUROC"] for m in
                  ["XGBoost", "LSTM", "Autoencoder", "QuantumKernel", "VQC", "Q-TRACE"]
                  if m in zd_table.index},
    }
    plot_bar_comparison(list(quantum_vs_classical["AUROC"].keys()), quantum_vs_classical,
                         "AUROC", "Quantum vs. Classical Representations (Zero-Day Test)",
                         "figures/fig8_quantum_vs_classical")

    # =====================================================================
    # Secret-Key Retention Rate (Section 18) on the zero-day split
    # =====================================================================
    print("\n[SKRR] computing secret-key retention under Q-TRACE gating...")
    skrr_results = {}
    gating_results = {}  # companion metric: attack_detected/missed_rate, normal_gated_rate
    baseline_skr = test_zd["skr_est"].to_numpy()
    q_lookup_zd = q_lookup_by_split["zero_day"]

    for name, score_or_tuple in [("Q-TRACE (top-decile)", qtrace.score(X_zd, test_zd,
                                    quantum_proba_lookup=q_lookup_zd))]:
        risk = minmax(score_or_tuple)
        tau = np.percentile(risk, 90)  # top-decile of the WHOLE population flagged as THREAT
        gated = gate_key_distillation(baseline_skr[:len(risk)], risk, tau)
        skrr_results[name] = secret_key_retention_rate(gated, baseline_skr[:len(risk)])
        gating_results[name] = gating_effectiveness(y_zd[:len(risk)], risk, tau)

        # Recall-targeted operating points: on a split where attacks are
        # the MAJORITY class (as our zero-day split is, by dataset
        # construction), "top 10% of the whole population" is a far more
        # conservative cutoff than it sounds -- see find_threshold_for_recall's
        # docstring. Report SKRR at explicit, interpretable sensitivity
        # targets as well, so the sensitivity/key-retention tradeoff is
        # visible rather than confined to one arbitrary operating point.
        for target_recall in (0.5, 0.8, 0.95):
            recall_name = f"Q-TRACE ({int(target_recall*100)}% recall target)"
            tau_r = find_threshold_for_recall(risk, y_zd[:len(risk)], target_recall)
            gated_r = gate_key_distillation(baseline_skr[:len(risk)], risk, tau_r)
            skrr_results[recall_name] = secret_key_retention_rate(gated_r, baseline_skr[:len(risk)])
            gating_results[recall_name] = gating_effectiveness(y_zd[:len(risk)], risk, tau_r)

    # static-threshold baseline: gate purely on raw QBER_signal > 11%
    qber = test_zd["QBER_signal"].to_numpy()
    static_gate = (qber > 0.11).astype(float)
    gated_static = baseline_skr * (1 - static_gate)
    skrr_results["Static QBER Threshold"] = secret_key_retention_rate(gated_static, baseline_skr)
    gating_results["Static QBER Threshold"] = gating_effectiveness(y_zd, qber, 0.11)

    # no-detection baseline (upper bound, ignores attacks entirely -- by
    # construction this strategy never gates ANY window, so its
    # attack_detected_rate is trivially 0.0 and attack_missed_rate is
    # trivially 1.0; included explicitly rather than left implicit, since
    # SKRR alone (=1.0) could otherwise be misread as "this strategy is
    # fine" rather than "this strategy does not attempt detection".
    skrr_results["No Detection (baseline)"] = 1.0
    gating_results["No Detection (baseline)"] = {
        "attack_detected_rate": 0.0, "attack_missed_rate": 1.0,
        "normal_gated_rate": 0.0,
        "n_attack_windows": int((y_zd == 1).sum()), "n_normal_windows": int((y_zd == 0).sum()),
    }

    pd.Series(skrr_results).to_csv(os.path.join(args.out, "skrr_table.csv"))
    plot_skrr_bars(skrr_results, "Secret-Key Retention Under Zero-Day Attacks",
                    "figures/fig7_skrr")
    print(skrr_results)

    # companion table: SKRR alongside what each strategy actually caught
    gating_table = pd.DataFrame(gating_results).T
    gating_table["SKRR"] = pd.Series(skrr_results)
    gating_table = gating_table[["SKRR", "attack_detected_rate", "attack_missed_rate",
                                  "normal_gated_rate", "n_attack_windows", "n_normal_windows"]]
    gating_table.to_csv(os.path.join(args.out, "skrr_gating_effectiveness_table.csv"))
    print("\n=== SKRR + gating effectiveness (companion table) ===")
    print(gating_table.to_string())

    # =====================================================================
    # Ablation study (Section 38)
    # =====================================================================
    print("\n[ablation] running Q-TRACE component ablation...")
    ablation_rows = []
    configs = {
        "Q-TRACE (full)": dict(use_physics=True, use_quantum=True, use_temporal=True, use_uncertainty=True),
        "Q-TRACE - physics": dict(use_physics=False, use_quantum=True, use_temporal=True, use_uncertainty=True),
        "Q-TRACE - quantum": dict(use_physics=True, use_quantum=False, use_temporal=True, use_uncertainty=True),
        "Q-TRACE - temporal": dict(use_physics=True, use_quantum=True, use_temporal=False, use_uncertainty=True),
        "Q-TRACE - uncertainty": dict(use_physics=True, use_quantum=True, use_temporal=True, use_uncertainty=False),
    }
    for cfg_name, cfg in configs.items():
        det = QTraceDetector(n_qubits=args.n_qubits, **cfg).fit(Xn, aligned_train_normal_df)
        score = det.score(X_zd, test_zd, quantum_proba_lookup=q_lookup_zd)
        m = detection_metrics(y_zd, score)
        m["config"] = cfg_name
        ablation_rows.append(m)
    ablation_table = pd.DataFrame(ablation_rows)[["config", "AUROC", "AUPRC", "F1", "FAR"]]
    ablation_table.to_csv(os.path.join(args.out, "ablation_table.csv"), index=False)
    print(ablation_table.to_string(index=False))

    print(f"\nAll results written to '{args.out}/' and figures to 'figures/'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/qkd_trace_v1/qkd_trace_telemetry.csv")
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--quantum-subsample", type=int, default=250)
    ap.add_argument("--vqc-maxiter", type=int, default=120)
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true",
                     help="Fast correctness check with tiny subsamples; do not use for reported results.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
