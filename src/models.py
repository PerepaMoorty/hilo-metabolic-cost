"""Shallow neural-network regressor and its hyper-parameter tuning.

The reference study used MATLAB's ``feedforwardnet`` with one tanh hidden
layer, a linear output and Bayesian-regularisation training (``trainbr``).
Our equivalent in scikit-learn:

* ``MLPRegressor`` with one hidden layer and ``tanh`` activation
  (the output layer of MLPRegressor is always linear),
* an L2 weight penalty ``alpha`` in place of Bayesian regularisation
  (both shrink the weights to stop a tiny, noisy dataset being over-fitted),
* the quasi-Newton ``lbfgs`` solver, which converges well on small datasets,
* inputs AND target standardised using the training folds only.
"""
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.evaluation import cross_validate_model

NEURON_SWEEP = range(1, 11)          # hidden sizes plotted, as in the paper's Fig. 1
HIDDEN_UNIT_GRID = (1, 2, 3, 4, 6, 8)  # hidden sizes searched by the tuned model
ALPHA_GRID = (0.1, 1.0, 10.0)        # L2 strengths searched by the tuned model


def make_nn(hidden_units: int = 3, alpha: float = 1.0, seed: int = RANDOM_STATE):
    """Standardise -> 1-hidden-layer tanh MLP -> linear output (target also standardised)."""
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        activation="tanh",
        solver="lbfgs",
        alpha=alpha,
        max_iter=3000,
        random_state=seed,
    )
    net = Pipeline([("scale", StandardScaler()), ("mlp", mlp)])
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def make_tuned_nn(seed: int = RANDOM_STATE, hidden_grid=HIDDEN_UNIT_GRID, alpha_grid=ALPHA_GRID):
    """NN whose hidden size and L2 strength are chosen by an inner 4-fold CV.

    When this object is itself cross-validated, the inner search only ever
    sees the outer training folds (nested CV), so the reported error is not
    biased by picking the best configuration on the test data.
    """
    grid = {
        "regressor__mlp__hidden_layer_sizes": [(h,) for h in hidden_grid],
        "regressor__mlp__alpha": list(alpha_grid),
    }
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=seed)
    return GridSearchCV(make_nn(seed=seed), grid, scoring="neg_mean_squared_error", cv=inner_cv, n_jobs=1)


def neuron_sweep(data, protocol: str, neurons=NEURON_SWEEP, alpha: float = 1.0, n_repeats: int = 3) -> pd.DataFrame:
    """Cross-validated MSE for each hidden-layer size (reproduces the paper's Fig. 1 curves)."""
    rows = []
    for h in neurons:
        res = cross_validate_model(make_nn(h, alpha), data, protocol, n_repeats=n_repeats)
        rows.append({"neurons": h, "mse": res.mse, "mse_std": res.mse_std})
    return pd.DataFrame(rows)
