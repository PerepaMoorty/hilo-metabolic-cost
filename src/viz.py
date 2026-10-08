"""Plots of experiment results (read from the CSVs in results/tables)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONDITIONS_PER_DAY, NOISE_FLOOR_MSE

FEATURE_SET_LABELS = {"all": "All features", "step_emg": "Step + EMG", "step": "Step only", "emg": "EMG only"}
PROTOCOL_STYLE = {"kfold": ("#2a78b5", "Random 5-fold (paper)"), "lodo": ("#d4622a", "Leave-one-day-out")}
MODEL_COLORS = {"constant": "#9e9e9e", "lasso": "#2a78b5", "ridge": "#7fb2dd", "nn": "#d4622a"}


def plot_neuron_sweep(sweep: pd.DataFrame, subject: str):
    """MSE vs hidden-layer size for each feature set and both CV protocols."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (fs, label) in zip(axes.ravel(), FEATURE_SET_LABELS.items()):
        part = sweep[(sweep.subject == subject) & (sweep.feature_set == fs)]
        for protocol, (color, plabel) in PROTOCOL_STYLE.items():
            p = part[part.protocol == protocol]
            ax.plot(p.neurons, p.mse, "o-", color=color, label=plabel)
        ax.axhline(NOISE_FLOOR_MSE, color="k", ls=":", lw=1, label="Noise floor")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Hidden neurons")
    for ax in axes[:, 0]:
        ax.set_ylabel("CV MSE")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{subject}: neural network size vs cross-validated MSE")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison: pd.DataFrame, subject: str):
    """Grouped bars: MSE of each model on each feature set, one panel per protocol."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    models = [m for m in MODEL_COLORS if m in comparison.model.unique()]
    width = 0.8 / len(models)
    x = np.arange(len(FEATURE_SET_LABELS))
    for ax, (protocol, (_, plabel)) in zip(axes, PROTOCOL_STYLE.items()):
        part = comparison[(comparison.subject == subject) & (comparison.protocol == protocol)]
        for i, m in enumerate(models):
            vals = [part[(part.model == m) & (part.feature_set == fs)].mse.mean() for fs in FEATURE_SET_LABELS]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, color=MODEL_COLORS[m], label=m)
        ax.axhline(NOISE_FLOOR_MSE, color="k", ls=":", lw=1)
        ax.set_xticks(x, FEATURE_SET_LABELS.values())
        ax.set_title(plabel)
    axes[0].set_ylabel("CV MSE (lower is better)")
    axes[0].legend()
    fig.suptitle(f"{subject}: model comparison")
    fig.tight_layout()
    return fig


def plot_lodo_predictions(predictions: pd.DataFrame, subject: str):
    """Measured vs predicted metabolic cost when each day is held out in turn."""
    part = predictions[predictions.subject == subject].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(part.y_true, "k.-", lw=1, label="Measured")
    for col, color in [("pred_lasso", MODEL_COLORS["lasso"]), ("pred_nn", MODEL_COLORS["nn"])]:
        ax.plot(part[col], "-", color=color, lw=1.4, label=col.replace("pred_", "Predicted "))
    for b in np.flatnonzero(np.diff(part.day)) + 0.5:
        ax.axvline(b, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(f"Condition (dashed lines separate days of ~{CONDITIONS_PER_DAY})")
    ax.set_ylabel("Metabolic cost (norm.)")
    ax.set_title(f"{subject}: leave-one-day-out predictions (all features)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_selection_frequency(frequency: pd.DataFrame, top: int = 15):
    """How often each feature was chosen by nested forward selection."""
    subjects = frequency.subject.unique()
    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 5))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subjects):
        part = frequency[(frequency.subject == s) & (frequency.protocol == "lodo")]
        part = part.sort_values("frequency", ascending=False).head(top)
        ax.barh(part.feature, part.frequency, color=MODEL_COLORS["nn"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of held-out days where selected")
        ax.set_title(f"{s}: forward-selection frequency (top {top})")
    fig.tight_layout()
    return fig


def plot_pca_sweep(pca: pd.DataFrame):
    """MSE vs number of principal components, with cumulative explained variance."""
    subjects = pca.subject.unique()
    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 4))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subjects):
        part = pca[pca.subject == s]
        for protocol, (color, plabel) in PROTOCOL_STYLE.items():
            p = part[part.protocol == protocol]
            ax.plot(p.components, p.mse, "o-", color=color, label=plabel)
        ax.set_xlabel("Principal components")
        ax.set_ylabel("CV MSE")
        ax.set_title(f"{s}: PCA + NN")
        ev = ax.twinx()
        p = part[part.protocol == "kfold"]
        ev.plot(p.components, p.explained_variance, "k:", label="Explained variance")
        ev.set_ylim(0, 1.05)
        ev.set_ylabel("Cumulative explained variance")
        ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig
