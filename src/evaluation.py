"""Cross-validation protocols and metrics shared by every model.

Two protocols are compared throughout the project:

* ``kfold`` - shuffled 5-fold CV over all rows, repeated with different seeds.
  This mirrors the protocol of Krimsky & Ng (2018) and lets us compare
  numbers with their paper.
* ``lodo``  - leave-one-day-out CV: train on 4 walking days, test on the
  unseen 5th day. This is the stricter test the paper proposes as future
  work, because conditions from the same day are strongly correlated.

Every model passed in here must be a full sklearn estimator/pipeline that
contains its own scaler, so scaling is learned on the training folds only
(the reference implementation scaled the whole dataset before splitting).
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

from src.config import RANDOM_STATE

PROTOCOLS = ("kfold", "lodo")


@dataclass
class EvalResult:
    protocol: str
    mse: float
    mse_std: float
    r2: float
    y_pred: np.ndarray
    fold_mse: list = field(default_factory=list)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


def make_splitter(protocol: str, seed: int = RANDOM_STATE, n_splits: int = 5):
    if protocol == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if protocol == "lodo":
        return LeaveOneGroupOut()
    raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def _fold_mse(y, y_pred, splitter, X, groups):
    return [float(mean_squared_error(y[test], y_pred[test])) for _, test in splitter.split(X, y, groups)]


def cross_validate_model(model, data, protocol: str, n_repeats: int = 3, n_jobs: int = -1) -> EvalResult:
    """Out-of-fold MSE / R^2 of ``model`` on ``data`` (a SubjectData).

    For ``kfold`` the split is repeated ``n_repeats`` times with different
    seeds and the MSE is averaged; ``y_pred`` comes from the first repeat.
    For ``lodo`` the split is deterministic, so it is run once and
    ``fold_mse`` holds the error on each held-out day.
    """
    X, y = data.X, data.y
    groups = data.day if protocol == "lodo" else None  # KFold ignores (and warns about) groups
    repeats = n_repeats if protocol == "kfold" else 1

    mses, first_pred, first_folds = [], None, None
    for r in range(repeats):
        splitter = make_splitter(protocol, seed=RANDOM_STATE + r)
        y_pred = cross_val_predict(clone(model), X, y, groups=groups, cv=splitter, n_jobs=n_jobs)
        mses.append(mean_squared_error(y, y_pred))
        if first_pred is None:
            first_pred = y_pred
            first_folds = _fold_mse(y, y_pred, splitter, X, groups)

    spread = np.std(mses) if repeats > 1 else np.std(first_folds)
    return EvalResult(
        protocol=protocol,
        mse=float(np.mean(mses)),
        mse_std=float(spread),
        r2=float(r2_score(y, first_pred)),
        y_pred=first_pred,
        fold_mse=first_folds,
    )
