import numpy as np
from sklearn.dummy import DummyRegressor

from src.data import load_subject
from src.evaluation import cross_validate_model, make_splitter


def test_lodo_holds_out_one_whole_day():
    d = load_subject("S1")
    folds = list(make_splitter("lodo").split(d.X, d.y, d.day))
    assert len(folds) == 5
    for train, test in folds:
        assert len(np.unique(d.day[test])) == 1
        assert not set(d.day[train]) & set(d.day[test])


def test_constant_model_matches_target_variance():
    d = load_subject("S1")
    res = cross_validate_model(DummyRegressor(), d, "kfold", n_repeats=2)
    assert abs(res.mse - d.y.var()) < 0.01
    assert len(res.y_pred) == d.n_samples


def test_lodo_reports_one_error_per_day():
    d = load_subject("S2")
    res = cross_validate_model(DummyRegressor(), d, "lodo")
    assert len(res.fold_mse) == 5
    assert res.rmse > 0
