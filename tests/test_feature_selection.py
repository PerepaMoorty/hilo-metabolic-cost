from src.data import load_subject
from src.feature_selection import forward_selection, pca_sweep


def test_forward_selection_frequencies():
    d = load_subject("S1").subset("step")
    mse, freq = forward_selection(d, "lodo")
    assert mse > 0
    assert len(freq) == 9
    assert freq.between(0, 1).all()


def test_pca_sweep_explained_variance():
    d = load_subject("S1").subset("step")
    pca = pca_sweep(d, "kfold", components=(2, 9))
    assert list(pca.components) == [2, 9]
    assert pca.explained_variance.iloc[-1] > 0.99
