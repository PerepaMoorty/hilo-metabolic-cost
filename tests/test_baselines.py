from src.baselines import BASELINES, make_constant, make_lasso
from src.data import load_subject
from src.evaluation import cross_validate_model


def test_all_baselines_fit_and_predict():
    d = load_subject("S1")
    for name, factory in BASELINES.items():
        pred = factory().fit(d.X, d.y).predict(d.X)
        assert pred.shape == d.y.shape, name


def test_lasso_beats_constant_on_kfold():
    d = load_subject("S1")
    lasso = cross_validate_model(make_lasso(), d, "kfold", n_repeats=1)
    const = cross_validate_model(make_constant(), d, "kfold", n_repeats=1)
    assert lasso.mse < const.mse / 2
