"""Loading the processed HILO dataset into a convenient per-subject structure."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import CONDITIONS_PER_DAY, DATA_PATH, FEATURE_GROUPS, N_DAYS, SUBJECTS


@dataclass(frozen=True)
class SubjectData:
    """Features, target and day labels for one subject.

    X     : (n_samples, n_features) feature matrix
    y     : (n_samples,) normalised metabolic cost
    day   : (n_samples,) walking day index 0..4, used for leave-one-day-out CV
    feature_names : column names of X
    """

    subject: str
    X: np.ndarray
    y: np.ndarray
    day: np.ndarray
    feature_names: list

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    def subset(self, group: str) -> "SubjectData":
        """Return a copy that keeps only one feature group ('all', 'step_emg', 'step', 'emg')."""
        cols = FEATURE_GROUPS[group]
        return replace(self, X=self.X[:, cols], feature_names=[self.feature_names[c] for c in cols])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["day"] = self.day
        df["metabolic_cost"] = self.y
        return df


def load_subject(subject: str, path=DATA_PATH, drop_invalid: bool = True) -> SubjectData:
    """Load one subject ('S1' or 'S2') from processed_data.mat.

    Rows are stored day by day (36 conditions per day), so the day label is
    recovered from the row position. A normalised metabolic cost below zero
    means less energy than quiet standing, which is physically impossible
    while walking, so those rows are treated as sensor errors and dropped.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}; expected one of {list(SUBJECTS)}")
    mat = loadmat(path)
    prefix = SUBJECTS[subject]
    X = np.asarray(mat[f"{prefix}_data"], dtype=float)
    y = np.asarray(mat[f"{prefix}_metabolics"], dtype=float).ravel()
    names = [str(n[0]) for n in mat["data_labels"].ravel()]

    expected = N_DAYS * CONDITIONS_PER_DAY
    if X.shape[0] != expected:
        raise ValueError(f"Expected {expected} rows for {subject}, found {X.shape[0]}")
    day = np.repeat(np.arange(N_DAYS), CONDITIONS_PER_DAY)

    if drop_invalid:
        keep = y >= 0
        X, y, day = X[keep], y[keep], day[keep]

    return SubjectData(subject=subject, X=X, y=y, day=day, feature_names=names)


def load_all(**kwargs) -> dict:
    """Load every subject: {'S1': SubjectData, 'S2': SubjectData}."""
    return {s: load_subject(s, **kwargs) for s in SUBJECTS}
