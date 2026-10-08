"""Baseline regressors: constant prediction and regularised linear models."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same regularisation range the reference study searched over
LASSO_ALPHAS = np.logspace(-5, -1, 25)
RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def make_constant():
    """Always predicts the training-set mean. Its MSE equals the target variance."""
    return DummyRegressor(strategy="mean")


def make_lasso():
    """Standardise -> LASSO, with the L1 strength chosen by an inner 5-fold CV."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=50_000)),
    ])


def make_ridge():
    """Standardise -> Ridge, with the L2 strength chosen by efficient leave-one-out CV."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])


BASELINES = {
    "constant": make_constant,
    "lasso": make_lasso,
    "ridge": make_ridge,
}
