"""Filter-bank Common Spatial Patterns as a scikit-learn transformer.

Spatial filters are estimated in ``fit`` from the training epochs only and are
frozen afterwards; ``transform`` never refits, which is the leakage safeguard
the CSP+CNN paper relies on.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_BANDS = ((8.0, 12.0), (13.0, 20.0), (21.0, 30.0))


def _bandpass(X, lo, hi, sfreq, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def _mean_cov(X):
    covs = []
    for x in X:
        c = x @ x.T
        covs.append(c / np.trace(c))
    return np.mean(covs, axis=0)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Per-band CSP with ``n_pairs`` filter pairs; outputs log-variance features.

    With the default three bands and three pairs this gives 18 features.
    X: (trials, channels, samples); y: binary labels.
    """

    def __init__(self, bands=DEFAULT_BANDS, n_pairs: int = 3, sfreq: float = 160.0, reg: float = 1e-10):
        self.bands = bands
        self.n_pairs = n_pairs
        self.sfreq = sfreq
        self.reg = reg

    def fit(self, X, y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("FilterBankCSP is defined for binary problems.")
        self.classes_ = classes
        self.filters_ = []
        for lo, hi in self.bands:
            Xb = _bandpass(X, lo, hi, self.sfreq)
            ca, cb = _mean_cov(Xb[y == classes[0]]), _mean_cov(Xb[y == classes[1]])
            n = ca.shape[0]
            evals, evecs = eigh(ca, ca + cb + self.reg * np.eye(n))
            order = np.argsort(evals)
            pick = np.r_[order[:self.n_pairs], order[-self.n_pairs:]]
            self.filters_.append(evecs[:, pick].T)
        self.n_features_out_ = len(self.bands) * 2 * self.n_pairs
        return self

    def transform(self, X):
        feats = []
        for (lo, hi), W in zip(self.bands, self.filters_):
            Z = np.einsum("fc,ncs->nfs", W, _bandpass(X, lo, hi, self.sfreq))
            var = Z.var(axis=-1)
            feats.append(np.log(var / var.sum(axis=1, keepdims=True)))
        return np.concatenate(feats, axis=1).astype(np.float32)
