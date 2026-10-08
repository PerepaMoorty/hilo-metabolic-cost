"""Dimensionality reduction: forward stepwise selection and PCA, both nested inside CV.

The reference study ran selection/PCA on the full dataset before testing,
which leaks test information into the chosen features. Here every selector
is fitted on the training folds only and then applied to the held-out fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import cross_validate_model, make_splitter
from src.models import make_nn

PCA_COMPONENTS = (1, 2, 3, 5, 8, 12, 16, 20, 25, 29)


def make_forward_selector_model(tol: float = 1e-4, hidden_units: int = 3, alpha: float = 1.0):
    """Standardise -> greedy forward selection (scored with a fast ridge model) -> NN.

    Features are added one at a time while the inner-CV MSE improves by more
    than ``tol``.
    """
    selector = SequentialFeatureSelector(
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
        n_features_to_select="auto",
        tol=tol,
        direction="forward",
        scoring="neg_mean_squared_error",
        cv=5,
    )
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", selector),
        ("model", make_nn(hidden_units, alpha)),
    ])


def forward_selection(data, protocol: str, tol: float = 1e-4):
    """Run nested forward selection.

    Returns ``(mse, frequency)`` where ``frequency`` is the fraction of outer
    folds in which each feature was selected - a stable measure of importance
    that answers "which signals matter?" without trusting one greedy run.
    """
    splitter = make_splitter(protocol)
    X, y = data.X, data.y
    y_pred = np.empty_like(y)
    counts = np.zeros(X.shape[1])
    n_folds = 0
    groups = data.day if protocol == "lodo" else None
    for train, test in splitter.split(X, y, groups):
        model = clone(make_forward_selector_model(tol=tol)).fit(X[train], y[train])
        y_pred[test] = model.predict(X[test])
        counts += model.named_steps["select"].get_support()
        n_folds += 1
    frequency = pd.Series(counts / n_folds, index=data.feature_names, name="frequency")
    return float(mean_squared_error(y, y_pred)), frequency.sort_values(ascending=False)


def make_pca_model(n_components: int, hidden_units: int = 3, alpha: float = 1.0):
    """Standardise -> PCA(n_components) -> NN."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model", make_nn(hidden_units, alpha)),
    ])


def pca_sweep(data, protocol: str, components=PCA_COMPONENTS, n_repeats: int = 1) -> pd.DataFrame:
    """Cross-validated MSE versus number of principal components.

    ``explained_variance`` is computed on the full dataset purely for
    reporting; the models themselves fit PCA on training folds only.
    """
    components = [k for k in components if k <= data.X.shape[1]]
    full_pca = PCA().fit(StandardScaler().fit_transform(data.X))
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    rows = []
    for k in components:
        res = cross_validate_model(make_pca_model(k), data, protocol, n_repeats=n_repeats)
        rows.append({"components": k, "mse": res.mse, "explained_variance": float(cumulative[k - 1])})
    return pd.DataFrame(rows)
