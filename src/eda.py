"""Exploratory plots of the raw (unscaled) dataset."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONDITIONS_PER_DAY, N_DAYS

GROUP_COLORS = {"step": "#2a78b5", "emg": "#d4622a", "control": "#5a9e45"}


def _group_of(name: str) -> str:
    if name.startswith("emg"):
        return "emg"
    if name.startswith("ctrl"):
        return "control"
    return "step"


def plot_metabolic_by_day(datasets: dict):
    """Metabolic cost of every condition in walking order, one panel per subject.

    Shows the day-to-day drift (the subject is still learning to walk with the
    exoskeleton), which is why leave-one-day-out CV is the harder test.
    """
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3.2 * len(datasets)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, datasets.items()):
        position = np.arange(len(d.y))
        for k in range(N_DAYS):
            m = d.day == k
            ax.plot(position[m], d.y[m], "o-", ms=3, lw=1, label=f"Day {k + 1}")
            ax.hlines(d.y[m].mean(), position[m].min(), position[m].max(), colors="k", linestyles="--", lw=1)
        ax.set_title(f"{name}: normalised metabolic cost per condition (dashed = day mean)")
        ax.set_ylabel("Metabolic cost (norm.)")
    axes[-1].set_xlabel(f"Condition index (each day = {CONDITIONS_PER_DAY} conditions)")
    axes[0].legend(ncol=N_DAYS, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def day_summary(datasets: dict) -> pd.DataFrame:
    """Mean and standard deviation of metabolic cost per subject and day."""
    rows = []
    for name, d in datasets.items():
        for k in range(N_DAYS):
            yk = d.y[d.day == k]
            rows.append({"subject": name, "day": k + 1, "n": len(yk), "mean": yk.mean(), "std": yk.std()})
    return pd.DataFrame(rows).round(3)


def feature_target_correlation(data) -> pd.Series:
    """Pearson correlation of each feature with metabolic cost."""
    df = pd.DataFrame(data.X, columns=data.feature_names)
    return df.corrwith(pd.Series(data.y)).rename("corr")


def plot_feature_correlations(datasets: dict):
    """Bar chart of feature-vs-target correlation, coloured by feature group."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, datasets.items()):
        corr = feature_target_correlation(d)
        colors = [GROUP_COLORS[_group_of(n)] for n in corr.index]
        ax.barh(corr.index, corr.values, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{name}: correlation with metabolic cost")
        ax.set_xlabel("Pearson r")
        ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[-1].legend(handles, GROUP_COLORS.keys(), loc="lower right")
    fig.tight_layout()
    return fig


def plot_feature_heatmap(data):
    """Feature-feature correlation matrix, to show redundancy (e.g. left vs right leg)."""
    corr = np.corrcoef(data.X, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(data.feature_names)), data.feature_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(data.feature_names)), data.feature_names, fontsize=7)
    ax.set_title(f"{data.subject}: feature correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
