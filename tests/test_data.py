import numpy as np
import pytest

from src.config import FEATURE_GROUPS
from src.data import load_all, load_subject


def test_shapes_and_days():
    s1 = load_subject("S1")
    assert s1.X.shape == (180, 29)
    assert len(s1.feature_names) == 29
    assert np.bincount(s1.day).tolist() == [36] * 5


def test_invalid_target_dropped_for_s2():
    s2 = load_subject("S2")
    raw = load_subject("S2", drop_invalid=False)
    assert s2.n_samples == 179 and raw.n_samples == 180
    assert (s2.y >= 0).all()


def test_feature_group_subset():
    s1 = load_subject("S1")
    for name, cols in FEATURE_GROUPS.items():
        sub = s1.subset(name)
        assert sub.X.shape == (180, len(cols))
        assert sub.feature_names[0] == s1.feature_names[cols[0]]
    assert s1.subset("emg").feature_names[0] == "emg1"


def test_load_all_and_unknown_subject():
    assert set(load_all()) == {"S1", "S2"}
    with pytest.raises(ValueError):
        load_subject("S3")
