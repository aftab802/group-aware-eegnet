import numpy as np

from cybex_eeg.augment import shift_noise_augment
from cybex_eeg.normalize import epoch_global_zscore, per_subject_zscore


def test_per_subject_zscore_uses_each_subject_only(mi_data):
    X, _, g = mi_data
    Xn = per_subject_zscore(X, g)
    for s in np.unique(g):
        m = Xn[g == s]
        assert np.allclose(m.mean(axis=(0, 2)), 0, atol=1e-4)
        assert np.allclose(m.std(axis=(0, 2)), 1, atol=1e-3)
    # changing another subject's data must not change this subject's output
    X2 = X.copy(); X2[g == 2] *= 100
    assert np.allclose(per_subject_zscore(X2, g)[g == 1], Xn[g == 1])


def test_epoch_global_zscore():
    X = np.random.default_rng(1).normal(5, 3, (4, 8, 50))
    Z = epoch_global_zscore(X)
    assert np.allclose(Z.reshape(4, -1).mean(1), 0, atol=1e-5)
    assert np.allclose(Z.reshape(4, -1).std(1), 1, atol=1e-4)


def test_augment_keeps_originals_and_triples(mi_data):
    X, y, _ = mi_data
    Xa, ya = shift_noise_augment(X, y, n_augments=2)
    assert Xa.shape[0] == 3 * X.shape[0] and (ya[:len(y)] == y).all()
    assert np.array_equal(Xa[:len(X)], X)
