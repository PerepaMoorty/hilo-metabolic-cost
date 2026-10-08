import numpy as np

from src.data import load_subject
from src.models import make_nn, make_tuned_nn, neuron_sweep


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.tanh(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=n)
    return X, y


def test_nn_learns_simple_function():
    X, y = _toy()
    model = make_nn(hidden_units=4, alpha=0.1).fit(X[:100], y[:100])
    mse = np.mean((model.predict(X[100:]) - y[100:]) ** 2)
    assert mse < 0.1 * y.var()


def test_tuned_nn_picks_from_grid():
    X, y = _toy()
    search = make_tuned_nn(hidden_grid=(1, 3), alpha_grid=(1.0,)).fit(X, y)
    assert search.best_params_["regressor__mlp__hidden_layer_sizes"] in [(1,), (3,)]


def test_neuron_sweep_on_real_data():
    d = load_subject("S1").subset("step")
    sweep = neuron_sweep(d, "lodo", neurons=(1, 2), n_repeats=1)
    assert list(sweep.neurons) == [1, 2]
    assert (sweep.mse > 0).all()
