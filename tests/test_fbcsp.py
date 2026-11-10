import numpy as np
from sklearn.linear_model import LogisticRegression

from cybex_eeg.fbcsp import FilterBankCSP
from cybex_eeg.splits import subject_split


def test_shape_and_separability(mi_data):
    X, y, g = mi_data
    s = subject_split(g, y, test_size=0.25, val_size=None)
    csp = FilterBankCSP(sfreq=160.0).fit(X[s["train"]], y[s["train"]])
    F = csp.transform(X)
    assert F.shape == (len(X), 18) and np.isfinite(F).all()
    acc = LogisticRegression().fit(F[s["train"]], y[s["train"]]).score(F[s["test"]], y[s["test"]])
    assert acc > 0.8


def test_transform_never_refits(mi_data):
    X, y, _ = mi_data
    csp = FilterBankCSP().fit(X[:80], y[:80])
    before = [w.copy() for w in csp.filters_]
    csp.transform(X[80:])
    assert all(np.array_equal(a, b) for a, b in zip(before, csp.filters_))


def test_filters_depend_only_on_training_trials(mi_data):
    X, y, _ = mi_data
    a = FilterBankCSP().fit(X[:80], y[:80]).transform(X[:5])
    X2 = X.copy(); X2[80:] = 0  # wipe the would-be test trials
    b = FilterBankCSP().fit(X2[:80], y[:80]).transform(X2[:5])
    assert np.allclose(a, b)
