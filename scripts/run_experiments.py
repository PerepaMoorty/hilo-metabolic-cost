"""Run every experiment and write tables + figures to results/.

Usage (from the repository root):
    python -m scripts.run_experiments           # full run, ~5-10 minutes
    python -m scripts.run_experiments --quick   # smoke test, ~1 minute
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import BASELINES  # noqa: E402
from src.config import FEATURE_GROUPS, FIGURES_DIR, TABLES_DIR  # noqa: E402
from src.data import load_all  # noqa: E402
from src.evaluation import PROTOCOLS, cross_validate_model  # noqa: E402
from src.feature_selection import forward_selection, pca_sweep  # noqa: E402
from src.models import make_tuned_nn, neuron_sweep  # noqa: E402
from src import viz  # noqa: E402

warnings.filterwarnings("ignore")

# Table I of Krimsky & Ng (2018): random k-fold MSE, (lasso, nn)
PAPER_TABLE = {
    ("S1", "all"): (0.0089, 0.0089), ("S1", "step_emg"): (0.0089, 0.0104),
    ("S1", "step"): (0.0200, 0.0138), ("S1", "emg"): (0.0158, 0.0118),
    ("S2", "all"): (0.0176, 0.0301), ("S2", "step_emg"): (0.0168, 0.0313),
    ("S2", "step"): (0.0217, 0.0233), ("S2", "emg"): (0.0203, 0.0199),
}
PAPER_CONSTANT = {"S1": 0.0657, "S2": 0.0465}


def log(msg, start):
    print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)


def run_model_comparison(datasets, n_repeats, nn_kwargs, start):
    rows, predictions = [], []
    for subject, data in datasets.items():
        for fs in FEATURE_GROUPS:
            d = data.subset(fs)
            for protocol in PROTOCOLS:
                models = {name: factory() for name, factory in BASELINES.items()}
                models["nn"] = make_tuned_nn(**nn_kwargs)
                preds = {}
                for name, model in models.items():
                    res = cross_validate_model(model, d, protocol, n_repeats=n_repeats)
                    rows.append({"subject": subject, "feature_set": fs, "protocol": protocol, "model": name,
                                 "mse": res.mse, "mse_std": res.mse_std, "rmse": res.rmse, "r2": res.r2})
                    preds[name] = res
                if fs == "all" and protocol == "lodo":
                    for i in range(d.n_samples):
                        predictions.append({"subject": subject, "day": int(d.day[i]), "y_true": d.y[i],
                                            "pred_lasso": preds["lasso"].y_pred[i], "pred_nn": preds["nn"].y_pred[i]})
                log(f"compare {subject} {fs:8s} {protocol:5s} "
                    + " ".join(f"{m}={r.mse:.4f}" for m, r in preds.items()), start)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def add_paper_columns(comparison):
    def paper(row):
        if row.protocol != "kfold":
            return None
        if row.model == "constant":
            return PAPER_CONSTANT[row.subject]
        if row.model in ("lasso", "nn"):
            return PAPER_TABLE[(row.subject, row.feature_set)][0 if row.model == "lasso" else 1]
        return None
    comparison["paper_mse"] = comparison.apply(paper, axis=1)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny grids and 1 repeat, for smoke testing")
    args = parser.parse_args()

    n_repeats = 1 if args.quick else 3
    neurons = (1, 2, 4) if args.quick else range(1, 11)
    nn_kwargs = {"hidden_grid": (2, 4), "alpha_grid": (1.0,)} if args.quick else {}
    pca_components = (2, 8, 29) if args.quick else (1, 2, 3, 5, 8, 12, 16, 20, 25, 29)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    datasets = load_all()

    # 1. Baselines vs tuned neural network, 4 feature sets x 2 protocols
    comparison, predictions = run_model_comparison(datasets, n_repeats, nn_kwargs, start)
    comparison = add_paper_columns(comparison)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(TABLES_DIR / "lodo_predictions.csv", index=False)

    # 2. Neural-network size sweep (paper Fig. 1, plus LODO)
    sweeps = []
    for subject, data in datasets.items():
        for fs in FEATURE_GROUPS:
            for protocol in PROTOCOLS:
                s = neuron_sweep(data.subset(fs), protocol, neurons=neurons, n_repeats=n_repeats)
                sweeps.append(s.assign(subject=subject, feature_set=fs, protocol=protocol))
        log(f"neuron sweep {subject} done", start)
    sweep = pd.concat(sweeps, ignore_index=True)
    sweep.to_csv(TABLES_DIR / "neuron_sweep.csv", index=False)

    # 3. Nested forward selection and PCA on all features
    freq_rows, sel_rows, pca_rows = [], [], []
    for subject, data in datasets.items():
        for protocol in PROTOCOLS:
            mse, freq = forward_selection(data, protocol)
            sel_rows.append({"subject": subject, "protocol": protocol, "mse": mse,
                             "features_always_selected": int((freq == 1).sum())})
            freq_rows.append(freq.rename_axis("feature").reset_index().assign(subject=subject, protocol=protocol))
            pca_rows.append(pca_sweep(data, protocol, components=pca_components).assign(subject=subject, protocol=protocol))
            log(f"selection {subject} {protocol}: forward-selection MSE={mse:.4f}", start)
    frequency = pd.concat(freq_rows, ignore_index=True)
    pca = pd.concat(pca_rows, ignore_index=True)
    frequency.to_csv(TABLES_DIR / "selection_frequency.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(TABLES_DIR / "selection_summary.csv", index=False)
    pca.to_csv(TABLES_DIR / "pca_sweep.csv", index=False)

    # 4. Figures
    figures = {}
    for subject in datasets:
        figures[f"neuron_sweep_{subject}"] = viz.plot_neuron_sweep(sweep, subject)
        figures[f"model_comparison_{subject}"] = viz.plot_model_comparison(comparison, subject)
        figures[f"lodo_predictions_{subject}"] = viz.plot_lodo_predictions(predictions, subject)
    figures["selection_frequency"] = viz.plot_selection_frequency(frequency)
    figures["pca_sweep"] = viz.plot_pca_sweep(pca)
    for name, fig in figures.items():
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
        plt.close(fig)

    summary = comparison[comparison.feature_set == "all"].pivot_table(
        index=["subject", "model"], columns="protocol", values="mse").round(4)
    print("\nMSE with all features:\n", summary)
    log(f"done - tables in {TABLES_DIR}, figures in {FIGURES_DIR}", start)


if __name__ == "__main__":
    main()
