"""
Q-TRACE :: evaluation.plotting
=================================
Matplotlib styling and figure helpers aimed at Springer/Nature-family
journal conventions: serif fonts, muted colorblind-safe palette, no
gridlines on ROC/PR curves, consistent 300 dpi export, vector PDF +
raster PNG side by side.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "XGBoost": "#4C72B0", "RandomForest": "#DD8452", "SVM": "#55A868",
    "LogisticRegression": "#C44E52", "LSTM": "#8172B2", "GRU": "#937860",
    "TemporalCNN": "#DA8BC3", "Autoencoder": "#8C8C8C", "IsolationForest": "#CCB974",
    "OneClassSVM": "#64B5CD", "QuantumKernel": "#B71C1C", "VQC": "#0D47A1",
    "Q-TRACE": "#1B5E20",
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png")
    fig.savefig(path_no_ext + ".pdf")
    plt.close(fig)


def plot_roc_curves(curves: dict, title, out_path):
    """curves: {model_name: (fpr, tpr, auroc)}. Legend is placed outside the
    axes (common Springer/Nature convention for >6 series) to avoid
    overlapping the curves themselves."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, (fpr, tpr, auroc) in curves.items():
        color = PALETTE.get(name, None)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    save_fig(fig, out_path)


def plot_bar_comparison(labels, values_dict, ylabel, title, out_path):
    """values_dict: {metric_name: {model_name: value}}"""
    set_style()
    models = list(next(iter(values_dict.values())).keys())
    n_metrics = len(values_dict)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(models))
    width = 0.8 / n_metrics
    for i, (metric, vals) in enumerate(values_dict.items()):
        heights = [vals[m] for m in models]
        ax.bar(x + i * width, heights, width, label=metric)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    save_fig(fig, out_path)


def plot_heatmap(matrix, x_labels, y_labels, xlabel, ylabel, title, out_path, cmap="viridis"):
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f"{v:.1f}" for v in x_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([f"{v:.1f}" for v in y_labels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Zero-Day Detection Recall")
    save_fig(fig, out_path)


def plot_skrr_bars(skrr_dict, title, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    names = list(skrr_dict.keys())
    vals = [skrr_dict[n] for n in names]
    colors = [PALETTE.get(n, "#777777") for n in names]
    ax.bar(names, vals, color=colors)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_ylabel("Secret-Key Retention Rate (SKRR)")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    save_fig(fig, out_path)
